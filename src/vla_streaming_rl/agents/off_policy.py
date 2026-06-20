# SPDX-License-Identifier: MIT
import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

from vla_streaming_rl.agents.step_result import StepResult
from vla_streaming_rl.networks.interface import InferInput
from vla_streaming_rl.replay_buffer import ReplayBuffer
from vla_streaming_rl.reward_processor import RewardProcessor
from vla_streaming_rl.self_forcing.goal_predictor import WorldModelGoalPredictor
from vla_streaming_rl.utils import create_reward_image


class OffPolicyAgent:
    def __init__(
        self,
        *,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        network: nn.Module,
        normalizing_by_return: bool,
        learning_starts: int,
        batch_size: int,
        max_grad_norm: float,
        use_done: bool,
        seq_len: int,
        horizon: int,
        actor_lr: float,
        critic_lr: float,
        buffer_size: int,
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

        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.use_done = use_done

        # Sequence observation management
        self.seq_len = seq_len
        self.horizon = horizon

        # Action chunking state
        self.action_chunk = None  # (horizon, action_dim) - current action chunk
        self.chunk_step = 0  # current step within chunk

        self.network = network
        self.rnn_state = self.network.init_state().to(self.device)

        critic_params = list(self.network.value_head.parameters())
        critic_param_ids = {id(p) for p in critic_params}
        actor_params = [p for p in self.network.parameters() if id(p) not in critic_param_ids]
        self.actor_optimizer = optim.AdamW(actor_params, lr=actor_lr, weight_decay=0.0)
        self.critic_optimizer = optim.AdamW(critic_params, lr=critic_lr, weight_decay=0.0)

        obs_z_shape = tuple(self.network.image_processor.output_shape)
        self.rb = ReplayBuffer(
            size=buffer_size,
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

        # Initialize gradient norm targets

        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)

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
        # inputs, so the reward panel is always emittable with no placeholder.
        self._last_pred_reward: float | None = None
        self._fresh_pred_reward: float | None = None

        self.goal_predictor = goal_predictor

    @torch.inference_mode()
    def select_action(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
        info: dict,
    ) -> StepResult:
        # ``info`` is the env's step/reset info dict (used by env-coupled agents
        # such as SimLingo); the obs-driven off-policy agent ignores it.
        del info
        metrics = {}
        panels = {}

        # Reset chunk on episode boundary
        if terminated or truncated:
            self.action_chunk = None
            self.chunk_step = 0

        # calculate train reward
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

        # add to replay buffer
        obs_tensor = torch.from_numpy(obs).to(self.device)
        obs_z = self.network.image_processor.encode(obs_tensor.unsqueeze(0))
        obs_z = obs_z.squeeze(0)
        normalized_action = (self.prev_action - self.action_bias) / self.action_scale
        task_prompt_token_ids = self.network.tokenize_task_prompt(task_prompt)
        self.rb.add(
            obs_tensor,
            obs_z,
            reward,
            (terminated or truncated) if self.use_done else False,
            self.rnn_state.squeeze(0),
            torch.from_numpy(normalized_action).to(self.device),
            0.0,
            0.0,
            [],
            task_prompt_token_ids,
        )

        goal_image = self.goal_predictor.step(obs)
        if self.goal_predictor.enabled:
            panels["goal"] = goal_image
        if terminated or truncated:
            self.goal_predictor.reset()

        # Use cached action from chunk if available (except during random exploration)
        if (
            global_step >= self.learning_starts
            and self.action_chunk is not None
            and self.chunk_step < self.horizon
        ):
            action = self.action_chunk[self.chunk_step]
            action = action * self.action_scale + self.action_bias
            action = np.clip(action, self.action_low, self.action_high)
            self.prev_action = action
            self.chunk_step += 1
            metrics["chunk_step"] = self.chunk_step
            return StepResult(action=action, metrics=metrics, panels=panels)

        # inference - predict new action chunk
        latest_data = self.rb.get_latest(self.seq_len)
        infer_result = self.network.infer(
            InferInput(
                s_seq=latest_data.observations,
                obs_z_seq=latest_data.obs_z,
                a_seq=latest_data.actions,
                r_seq=latest_data.rewards,
                rnn_state=self.rnn_state,
                task_prompts=[task_prompt],
            )
        )
        self.rnn_state = infer_result.rnn_state
        metrics.update(infer_result.value_report)
        next_image = infer_result.next_image
        next_reward = infer_result.next_reward
        # Stash the fresh prediction for next-step validation / display. Drop it
        # at an episode boundary so it is never compared against the next
        # episode's first frame.
        if terminated or truncated:
            self._last_pred_image = None
            self._fresh_pred_image = None
            self._last_pred_reward = None
            self._fresh_pred_reward = None
        else:
            self._last_pred_image = next_image
            self._fresh_pred_image = next_image
            self._last_pred_reward = next_reward
            self._fresh_pred_reward = next_reward

        # action
        if global_step < self.learning_starts:
            action = self.action_space.sample()
            self.action_chunk = None
            self.chunk_step = 0
        else:
            # action chunk: (B, horizon, action_dim) -> (horizon, action_dim)
            action_chunk = infer_result.action[0].cpu().numpy()
            self.action_chunk = action_chunk
            self.chunk_step = 1

            # Use first action from chunk
            action = action_chunk[0]
            action = action * self.action_scale + self.action_bias
            action = np.clip(action, self.action_low, self.action_high)
        self.prev_action = action

        metrics["chunk_step"] = self.chunk_step
        return StepResult(action=action, metrics=metrics, panels=panels)

    def step(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
        info: dict,
    ) -> StepResult:
        # train, then make decision; the training metrics merge into the
        # action's StepResult.
        train_metrics = self._train(global_step)
        result = self.select_action(
            global_step, obs, reward, terminated, truncated, task_prompt, info
        )
        result.metrics.update(train_metrics)
        return result

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        return {}

    ####################
    # Internal methods #
    ####################

    def _train(self, global_step: int) -> dict:
        info_dict = {}

        if global_step < self.learning_starts:
            return info_dict
        elif global_step == self.learning_starts:
            print(f"Start training at global step {global_step}.")

        # Sample data for training using ReplayBuffer
        data = self.rb.sample(self.batch_size)

        # apply reward processing
        data.rewards = self.reward_processor.normalize(data.rewards)

        # compute loss
        loss_result = self.network.compute_loss(data)

        # add prefixes to info_dict keys
        info_dict = {f"losses/{key}": value for key, value in loss_result.info.items()}

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss_result.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return info_dict
