# SPDX-License-Identifier: MIT
import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

from vla_streaming_rl.agents.step_result import StepResult
from vla_streaming_rl.networks.interface import InferInput
from vla_streaming_rl.optimizers.adam_et import AdamET
from vla_streaming_rl.replay_buffer import ReplayBuffer
from vla_streaming_rl.reward_processor import RewardProcessor
from vla_streaming_rl.self_forcing.goal_predictor import WorldModelGoalPredictor
from vla_streaming_rl.utils import create_reward_image


class StandardAgent:
    def __init__(
        self,
        *,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        network: nn.Module,
        learning_mode: str,
        normalizing_by_return: bool,
        learning_starts: int,
        batch_size: int,
        max_grad_norm: float,
        use_done: bool,
        seq_len: int,
        horizon: int,
        use_eligibility_trace: bool,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        et_lambda: float,
        buffer_size: int,
        buffer_device: str,
        max_new_tokens: int,
        max_prompt_tokens: int,
        pad_token_id: int,
        goal_predictor: WorldModelGoalPredictor,
    ) -> None:
        if learning_mode not in ("off_policy", "streaming"):
            raise ValueError(f"Unknown learning_mode: {learning_mode!r}")
        self.learning_mode = learning_mode
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

        # Actor / critic optimizer split (critic == value head). The critic uses
        # AdamET (eligibility traces) only in streaming-trace mode, AdamW
        # otherwise; the actor is always AdamW. Off-policy uses no weight decay,
        # streaming uses 0.1 — preserving the per-mode behavior of the original
        # OffPolicyAgent / StreamingAgent.
        self.use_eligibility_trace = bool(use_eligibility_trace)
        weight_decay = 0.0 if learning_mode == "off_policy" else 0.1
        critic_params = list(self.network.value_head.parameters())
        critic_param_ids = {id(p) for p in critic_params}
        actor_params = [p for p in self.network.parameters() if id(p) not in critic_param_ids]
        self.actor_optimizer = optim.AdamW(actor_params, lr=actor_lr, weight_decay=weight_decay)
        if learning_mode == "streaming" and self.use_eligibility_trace:
            self.critic_optimizer = AdamET(
                critic_params, lr=critic_lr, gamma=gamma, et_lambda=et_lambda
            )
        else:
            self.critic_optimizer = optim.AdamW(
                critic_params, lr=critic_lr, weight_decay=weight_decay
            )

        # Off-policy keeps a large replay buffer; streaming keeps only the one
        # window it trains on (the latest seq_len + horizon transition).
        buffer_capacity = buffer_size if learning_mode == "off_policy" else seq_len + horizon
        obs_z_shape = tuple(self.network.image_processor.output_shape)
        self.rb = ReplayBuffer(
            size=buffer_capacity,
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
        # Streaming stores the VLM action chunk's token ids alongside the
        # transition; off-policy never uses them and leaves this empty.
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

    # --- agent surface -----------------------------------------------------

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
        metrics = self._store_transition(obs, reward, terminated, truncated, task_prompt, info)
        action, act_metrics = self._act(global_step, obs, task_prompt, info)
        metrics.update(act_metrics)
        return StepResult(action=action, metrics=metrics, panels=self._panels(obs, reward))

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
        if self.learning_mode == "off_policy":
            return self._step_offpolicy(
                global_step, obs, reward, terminated, truncated, task_prompt, info
            )
        return self._step_streaming(
            global_step, obs, reward, terminated, truncated, task_prompt, info
        )

    def _step_offpolicy(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
        info: dict,
    ) -> StepResult:
        # Train on a random replay batch (its own forward), then act; the
        # training metrics merge into the action's StepResult.
        train_metrics = self._train_offpolicy(global_step)
        result = self.select_action(
            global_step, obs, reward, terminated, truncated, task_prompt, info
        )
        result.metrics.update(train_metrics)
        return result

    def _step_streaming(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
        info: dict,
    ) -> StepResult:
        metrics = self._store_transition(obs, reward, terminated, truncated, task_prompt, info)
        panels = self._panels(obs, reward)

        # cached chunk: no inference, no training
        if self.action_chunk is not None and self.chunk_step < self.horizon:
            action, act_metrics = self._act(global_step, obs, task_prompt, info)
            metrics.update(act_metrics)
            return StepResult(action=action, metrics=metrics, panels=panels)

        # new chunk: a single grad-enabled forward yields both the action chunk
        # and the training loss (fused inference + training).
        data = self.rb.get_latest(self.seq_len + self.horizon)
        data.rewards = self.reward_processor.normalize(data.rewards)
        result = self.network.infer_and_compute_loss(data)

        infer_result = result.infer_result
        self.rnn_state = infer_result.rnn_state
        if self.learning_mode == "streaming":
            self.prev_action_token_ids = infer_result.action_token_ids
        metrics.update(infer_result.value_report)
        action_chunk = infer_result.action[0].cpu().numpy()
        self.action_chunk = action_chunk
        self.chunk_step = 1
        action = self._to_env_action(action_chunk[0])
        self.prev_action = action
        metrics["chunk_step"] = self.chunk_step
        self._last_pred_image = infer_result.next_image
        self._fresh_pred_image = infer_result.next_image
        self._last_pred_reward = infer_result.next_reward
        self._fresh_pred_reward = infer_result.next_reward
        metrics.update({f"losses/{key}": value for key, value in result.loss_result.info.items()})

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        if self.use_eligibility_trace:
            # Actor: backward actor-only loss → encoder + actor grads.
            result.et_info.actor_entropy_loss.backward(retain_graph=True)
            # Critic: backward -V(s) → value_head grads only (detached from encoder).
            result.et_info.neg_value.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step(delta=result.et_info.delta, reset=self._episode_reset)
            self._episode_reset = False
        else:
            result.loss_result.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        return StepResult(action=action, metrics=metrics, panels=panels)

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        del score, feedback_text
        # Between-episode cleanup of state that outlives the terminal step's own
        # processing: the next-frame / reward predictions (stashed during the
        # terminal step, dropped here so they are never validated against the next
        # episode's first frame) and the world-model goal predictor. The chunk /
        # eligibility-trace reset is done in ``_store_transition`` instead, since
        # it must take effect *before* the terminal step trains.
        self._last_pred_image = None
        self._fresh_pred_image = None
        self._last_pred_reward = None
        self._fresh_pred_reward = None
        self.goal_predictor.reset()
        return {}

    # --- training ----------------------------------------------------------

    def _train_offpolicy(self, global_step: int) -> dict:
        if global_step < self.learning_starts:
            return {}
        elif global_step == self.learning_starts:
            print(f"Start training at global step {global_step}.")

        data = self.rb.sample(self.batch_size)
        data.rewards = self.reward_processor.normalize(data.rewards)
        loss_result = self.network.compute_loss(data)
        info_dict = {f"losses/{key}": value for key, value in loss_result.info.items()}

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss_result.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        return info_dict

    # --- per-tick machinery ------------------------------------------------

    def _store_transition(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
        info: dict,
    ) -> dict:
        """Record the current timestep into the replay buffer and return the
        per-tick telemetry it produces (processed reward, action norm, and the
        previous prediction's validation losses).

        On an episode boundary it also clears the chunk and arms the
        eligibility-trace reset — both must take effect before the terminal step
        trains (a cleared chunk forces the terminal transition through the
        training path instead of the cached-action fast path)."""
        metrics = {}
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

        # Validate the previous inference's next-frame / reward prediction against
        # this observation. Logged only while the prediction is fresh (one step
        # old); the panels persist the latest prediction across cached steps.
        obs_hwc = obs.transpose(1, 2, 0)
        if self._fresh_pred_image is not None:
            metrics["losses/pred_image_loss"] = float(
                np.mean(np.abs(self._fresh_pred_image - obs_hwc))
            )
            self._fresh_pred_image = None
        if self._fresh_pred_reward is not None:
            metrics["losses/pred_reward_loss"] = abs(self._fresh_pred_reward - reward)
            self._fresh_pred_reward = None

        obs_tensor, obs_z, task_prompt_token_ids = self._preprocess(obs, info, task_prompt)
        normalized_action = (self.prev_action - self.action_bias) / self.action_scale
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
        return metrics

    def _act(
        self, global_step: int, obs: np.ndarray, task_prompt: str, info: dict
    ) -> tuple[np.ndarray, dict]:
        """Produce the env action for this tick: replay a cached chunk action
        when one is available, otherwise run inference for a fresh chunk. During
        the off-policy warmup (``global_step < learning_starts``) the action is a
        random sample, but inference still runs so the recurrent state and the
        prediction panels keep advancing."""
        del obs, info  # the obs-driven standard agent acts off the replay buffer
        metrics = {}
        warmup = self.learning_mode == "off_policy" and global_step < self.learning_starts

        if not warmup and self.action_chunk is not None and self.chunk_step < self.horizon:
            action = self._to_env_action(self.action_chunk[self.chunk_step])
            self.prev_action = action
            self.chunk_step += 1
            metrics["chunk_step"] = self.chunk_step
            return action, metrics

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
        if self.learning_mode == "streaming":
            self.prev_action_token_ids = infer_result.action_token_ids
        metrics.update(infer_result.value_report)
        action_chunk = infer_result.action[0].cpu().numpy()
        self.action_chunk = action_chunk
        self.chunk_step = 1
        action = self._to_env_action(action_chunk[0])
        self.prev_action = action
        metrics["chunk_step"] = self.chunk_step
        self._last_pred_image = infer_result.next_image
        self._fresh_pred_image = infer_result.next_image
        self._last_pred_reward = infer_result.next_reward
        self._fresh_pred_reward = infer_result.next_reward

        if warmup:
            action = self.action_space.sample()
            self.action_chunk = None
            self.chunk_step = 0
            self.prev_action = action
            metrics["chunk_step"] = self.chunk_step
        return action, metrics

    def _panels(self, obs: np.ndarray, reward: float) -> dict:
        panels = {}
        panels["prediction"] = (
            self._last_pred_image if self._last_pred_image is not None else self._pred_placeholder
        )
        pred_reward = self._last_pred_reward if self._last_pred_reward is not None else 0.0
        panels["reward"] = create_reward_image(pred_reward, reward)
        goal_image = self.goal_predictor.step(obs)
        if self.goal_predictor.enabled:
            panels["goal"] = goal_image
        return panels

    def _preprocess(self, obs: np.ndarray, info: dict, task_prompt: str) -> tuple:
        """Turn the raw observation into what the replay buffer stores this tick:
        the raw obs tensor, its encoded latent ``obs_z``, and the tokenized task
        prompt. ``info`` is unused by the obs-driven standard agent."""
        del info
        obs_tensor = torch.from_numpy(obs).to(self.device)
        with torch.inference_mode():
            obs_z = self.network.image_processor.encode(obs_tensor.unsqueeze(0)).squeeze(0)
        task_prompt_token_ids = self.network.tokenize_task_prompt(task_prompt)
        return obs_tensor, obs_z, task_prompt_token_ids

    def _to_env_action(self, net_action: np.ndarray) -> np.ndarray:
        """Map a single normalized policy action into the env's action space."""
        return np.clip(
            net_action * self.action_scale + self.action_bias, self.action_low, self.action_high
        )
