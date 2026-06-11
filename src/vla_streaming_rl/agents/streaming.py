# SPDX-License-Identifier: MIT
import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

from vla_streaming_rl.agents.step_result import StepResult
from vla_streaming_rl.networks.infer_result import InferResult
from vla_streaming_rl.optimizers.adam_et import AdamET
from vla_streaming_rl.replay_buffer import ReplayBuffer
from vla_streaming_rl.reward_processor import RewardProcessor
from vla_streaming_rl.self_forcing.goal_predictor import WorldModelGoalPredictor
from vla_streaming_rl.utils import create_reward_image


class StreamingAgent:
    def __init__(
        self,
        *,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        network: nn.Module,
        normalizing_by_return: bool,
        max_grad_norm: float,
        use_done: bool,
        accumulation_steps: int,
        seq_len: int,
        horizon: int,
        use_eligibility_trace: bool,
        learning_rate: float,
        gamma: float,
        et_lambda: float,
        buffer_device: str,
        max_new_tokens: int,
        max_prompt_tokens: int,
        pad_token_id: int,
        goal_predictor: WorldModelGoalPredictor,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.observation_space = observation_space

        # action properties
        self.action_space = action_space
        self.action_dim = np.prod(action_space.shape)
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_scale = (action_space.high - action_space.low) / 2.0
        self.action_bias = (action_space.high + action_space.low) / 2.0
        self.reward_processor = RewardProcessor("scaling", 1.0)
        self.normalizing_by_return = normalizing_by_return

        self.max_grad_norm = max_grad_norm
        self.use_done = use_done
        self.accumulation_steps = accumulation_steps
        self._accumulation_count = 0

        # Sequence observation management
        self.seq_len = seq_len
        self.horizon = horizon

        # Action chunking state
        self.action_chunk = None  # (horizon, action_dim) - current action chunk
        self.chunk_step = 0  # current step within chunk

        self.network = network
        self.rnn_state = self.network.init_state().to(self.device)

        self.use_eligibility_trace = bool(use_eligibility_trace)
        self.critic_optimizer = None
        if self.use_eligibility_trace:
            critic_params = list(self.network.value_head.parameters())
            critic_param_ids = {id(p) for p in critic_params}
            other_params = [p for p in self.network.parameters() if id(p) not in critic_param_ids]
            self.critic_optimizer = AdamET(
                critic_params,
                lr=learning_rate,
                gamma=gamma,
                et_lambda=et_lambda,
            )
            self.optimizer = optim.AdamW(other_params, lr=learning_rate, weight_decay=0.1)
        else:
            self.optimizer = optim.AdamW(
                self.network.parameters(), lr=learning_rate, weight_decay=0.1
            )

        obs_z_shape = tuple(self.network.image_processor.output_shape)
        self.rb = ReplayBuffer(
            size=self.seq_len + self.horizon,
            seq_len=self.seq_len + self.horizon,
            obs_shape=observation_space.shape,
            obs_z_shape=obs_z_shape,
            rnn_state_shape=self.rnn_state.squeeze(0).shape,
            action_shape=action_space.shape,
            output_device=self.device,
            storage_device=torch.device(buffer_device),
            max_new_tokens=max_new_tokens,
            max_prompt_tokens=max_prompt_tokens,
            pad_token_id=pad_token_id,
        )

        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_action_token_ids = []
        self._episode_reset = False

        # Next-frame prediction visualization state. ``_last_pred_image`` is the
        # latest predicted frame, kept so the "prediction" panel persists across
        # cached-chunk steps; ``_fresh_pred_image`` is the prediction that is
        # exactly one step old, so its loss is logged only once (when it can be
        # compared against the observation it predicted). The placeholder keeps
        # the panel a fixed shape before the first prediction exists, honoring
        # the stable-panel contract.
        obs_c, obs_h, obs_w = observation_space.shape
        self._pred_placeholder = np.zeros((obs_h, obs_w, obs_c), dtype=np.float32)
        self._last_pred_image: np.ndarray | None = None
        self._fresh_pred_image: np.ndarray | None = None
        # Same persist (panel) / fresh (loss-once) split for the reward
        # prediction. ``create_reward_image`` is a fixed size regardless of its
        # inputs, so the reward panel needs no placeholder.
        self._last_pred_reward: float | None = None
        self._fresh_pred_reward: float | None = None

        self.goal_predictor = goal_predictor

    def _prepare_step(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        metrics: dict,
        panels: dict,
        task_prompt: str,
    ) -> None:
        episode_done = terminated or truncated
        if episode_done:
            self.action_chunk = None
            self.chunk_step = 0
            self.prev_action_token_ids = []
            self._episode_reset = self.use_done

        action_norm = np.linalg.norm(self.prev_action)
        if not self.normalizing_by_return:
            self.reward_processor.update(reward)
        metrics["action_norm"] = action_norm
        metrics["processed_reward"] = self.reward_processor.normalize(torch.tensor(reward)).item()

        # Validate the previous inference's next-frame prediction against this
        # observation. The loss is logged only while the prediction is fresh
        # (one step old); the panel persists the latest prediction so it stays
        # visible across cached-chunk steps.
        obs_hwc = obs.transpose(1, 2, 0)
        if self._fresh_pred_image is not None:
            metrics["losses/pred_image_loss"] = float(
                np.mean(np.abs(self._fresh_pred_image - obs_hwc))
            )
            self._fresh_pred_image = None
        panels["prediction"] = (
            self._last_pred_image if self._last_pred_image is not None else self._pred_placeholder
        )
        if self._fresh_pred_reward is not None:
            metrics["losses/pred_reward_loss"] = abs(self._fresh_pred_reward - reward)
            self._fresh_pred_reward = None
        pred_reward = self._last_pred_reward if self._last_pred_reward is not None else 0.0
        panels["reward"] = create_reward_image(pred_reward, reward)

        obs_tensor = torch.from_numpy(obs).to(self.device)
        with torch.inference_mode():
            obs_z = self.network.image_processor.encode(obs_tensor.unsqueeze(0))
            obs_z = obs_z.squeeze(0)
        normalized_action = (self.prev_action - self.action_bias) / self.action_scale
        task_prompt_token_ids = self.network.tokenize_task_prompt(task_prompt)
        self.rb.add(
            obs_tensor,
            obs_z,
            reward,
            episode_done if self.use_done else False,
            self.rnn_state.squeeze(0),
            torch.from_numpy(normalized_action).to(self.device),
            0.0,
            0.0,
            self.prev_action_token_ids,
            task_prompt_token_ids,
        )

        goal_image = self.goal_predictor.step(obs)
        if self.goal_predictor.enabled:
            panels["goal"] = goal_image
        if episode_done:
            self.goal_predictor.reset()

    def _use_action_chunk(self, metrics: dict) -> np.ndarray:
        action = self.action_chunk[self.chunk_step]
        action = action * self.action_scale + self.action_bias
        action = np.clip(action, self.action_low, self.action_high)
        self.prev_action = action
        self.chunk_step += 1
        metrics["chunk_step"] = self.chunk_step
        return action

    def _start_new_chunk(
        self, infer_result: InferResult, metrics: dict
    ) -> tuple[np.ndarray, np.ndarray, float]:
        self.rnn_state = infer_result.rnn_state
        self.prev_action_token_ids = infer_result.action_token_ids
        metrics["value"] = infer_result.value
        next_image = infer_result.next_image
        next_reward = infer_result.next_reward

        action_chunk = infer_result.action[0].cpu().numpy()
        self.action_chunk = action_chunk
        self.chunk_step = 1

        action = action_chunk[0]
        action = action * self.action_scale + self.action_bias
        action = np.clip(action, self.action_low, self.action_high)
        self.prev_action = action
        metrics["chunk_step"] = self.chunk_step
        return action, next_image, next_reward

    @torch.inference_mode()
    def select_action(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
    ) -> StepResult:
        metrics = {}
        panels = {}
        self._prepare_step(obs, reward, terminated, truncated, metrics, panels, task_prompt)

        if self.action_chunk is not None and self.chunk_step < self.horizon:
            action = self._use_action_chunk(metrics)
            return StepResult(action=action, metrics=metrics, panels=panels)

        latest_data = self.rb.get_latest(self.seq_len)
        infer_result = self.network.infer(
            latest_data.observations,
            latest_data.obs_z,
            latest_data.actions,
            latest_data.rewards,
            self.rnn_state,
            task_prompts=[task_prompt],
        )
        action, next_image, next_reward = self._start_new_chunk(infer_result, metrics)
        self._store_prediction(next_image, next_reward, terminated or truncated)
        return StepResult(action=action, metrics=metrics, panels=panels)

    def _store_prediction(
        self, next_image: np.ndarray, next_reward: float, episode_done: bool
    ) -> None:
        """Stash the fresh next-frame / next-reward predictions for next-step
        validation and display. Drop them at an episode boundary so they are
        never compared against the next episode's first step."""
        if episode_done:
            self._last_pred_image = None
            self._fresh_pred_image = None
            self._last_pred_reward = None
            self._fresh_pred_reward = None
        else:
            self._last_pred_image = next_image
            self._fresh_pred_image = next_image
            self._last_pred_reward = next_reward
            self._fresh_pred_reward = next_reward

    def step(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
    ) -> StepResult:
        metrics = {}
        panels = {}
        self._prepare_step(obs, reward, terminated, truncated, metrics, panels, task_prompt)

        # cached action: no inference, no training
        if self.action_chunk is not None and self.chunk_step < self.horizon:
            action = self._use_action_chunk(metrics)
            return StepResult(action=action, metrics=metrics, panels=panels)

        # combined inference + training
        data = self.rb.get_latest(self.seq_len + self.horizon)
        data.rewards = self.reward_processor.normalize(data.rewards)

        result = self.network.infer_and_compute_loss(data)
        action, next_image, next_reward = self._start_new_chunk(result.infer_result, metrics)
        self._store_prediction(next_image, next_reward, terminated or truncated)

        metrics.update({f"losses/{key}": value for key, value in result.loss_result.info.items()})

        if self.use_eligibility_trace:
            # Actor: backward actor-only loss → encoder + actor grads
            actor_loss = result.et_info.actor_entropy_loss / self.accumulation_steps
            actor_loss.backward(retain_graph=True)

            # Critic: backward -V(s) → value_head grads only (detached from encoder)
            neg_value = result.et_info.neg_value / self.accumulation_steps
            neg_value.backward()
        else:
            scaled_loss = result.loss_result.loss / self.accumulation_steps
            scaled_loss.backward()

        self._accumulation_count += 1
        if self._accumulation_count % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()
            if self.use_eligibility_trace:
                self.critic_optimizer.step(delta=result.et_info.delta, reset=self._episode_reset)
                self._episode_reset = False
                self.critic_optimizer.zero_grad()
            self.optimizer.zero_grad()

        return StepResult(action=action, metrics=metrics, panels=panels)

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        return {}
