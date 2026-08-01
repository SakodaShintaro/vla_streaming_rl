# SPDX-License-Identifier: MIT
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

from vla_streaming_rl.agents.base import Agent, StepResult
from vla_streaming_rl.networks.interface import InferInput
from vla_streaming_rl.optimizers.adam_et import AdamET
from vla_streaming_rl.replay_buffer import ReplayBuffer
from vla_streaming_rl.reward_processor import RewardProcessor


class StandardAgent(Agent):
    def __init__(
        self,
        *,
        observation_space: gym.spaces.Dict,
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
        weight_decay: float,
        gamma: float,
        et_lambda: float,
        buffer_size: int,
        buffer_device: str,
        max_prompt_tokens: int,
        pad_token_id: int,
    ) -> None:
        super().__init__(learning_mode=learning_mode, horizon=horizon)
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

        # Action chunking state
        self.action_chunk = None  # (horizon, action_dim) - current action chunk
        self.chunk_step = 0  # current step within chunk

        self.network = network
        self.rnn_state = self.network.init_state().to(self.device)

        # Actor / critic optimizer split (critic == value head). The critic uses
        # AdamET (eligibility traces) only in streaming-trace mode, AdamW
        # otherwise; the actor is always AdamW.
        self.use_eligibility_trace = bool(use_eligibility_trace)
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
        self.rb = ReplayBuffer(
            size=buffer_capacity,
            seq_len=self.seq_len + self.horizon,
            obs_shape=self.network.packed_obs_shape,
            rnn_state_shape=self.rnn_state.squeeze(0).shape,
            action_shape=action_space.shape,
            output_device=self.device,
            storage_device=torch.device(buffer_device),
            max_prompt_tokens=max_prompt_tokens,
            pad_token_id=pad_token_id,
        )

        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self._episode_reset = False

    # --- agent surface -----------------------------------------------------

    def _step_streaming(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult:
        metrics = {}
        episode_done = terminated or truncated
        if episode_done:
            self.action_chunk = None
            self.chunk_step = 0
            self._episode_reset = self.use_done
        metrics["action_norm"] = np.linalg.norm(self.prev_action)
        if not self.normalizing_by_return:
            self.reward_processor.update(reward)
        metrics["processed_reward"] = self.reward_processor.normalize(torch.tensor(reward)).item()
        obs_tensor, task_prompt_token_ids = self._preprocess(obs, info)
        normalized_action = (self.prev_action - self.action_bias) / self.action_scale
        self.rb.add(
            obs_tensor,
            reward,
            episode_done if self.use_done else False,
            self.rnn_state.squeeze(0),
            torch.from_numpy(normalized_action).to(self.device),
            task_prompt_token_ids,
        )
        if self.action_chunk is not None and self.chunk_step < self.horizon:
            action = self._to_env_action(self.action_chunk[self.chunk_step])
            self.prev_action = action
            self.chunk_step += 1
            metrics["chunk_step"] = self.chunk_step
            return StepResult(action=action, metrics=metrics, panels={})

        # new chunk: a single grad-enabled forward yields both the action chunk
        # and the training loss (fused inference + training).
        data = self.rb.get_latest(self.seq_len + self.horizon)
        data.rewards = self.reward_processor.normalize(data.rewards)
        result = self.network.infer_and_compute_loss(data)

        infer_result = result.infer_result
        self.rnn_state = infer_result.rnn_state
        metrics.update(infer_result.value_report)
        action_chunk = infer_result.action[0].cpu().numpy()
        self.action_chunk = action_chunk
        self.chunk_step = 1
        action = self._to_env_action(action_chunk[0])
        self.prev_action = action
        metrics["chunk_step"] = self.chunk_step
        metrics.update(result.loss_result.info)

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

        return StepResult(action=action, metrics=metrics, panels={})

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        del score, feedback_text
        return {}

    # --- per-tick machinery ------------------------------------------------

    @torch.no_grad()
    def select_action(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult:
        metrics = {}
        episode_done = terminated or truncated
        if episode_done:
            self.action_chunk = None
            self.chunk_step = 0
            self._episode_reset = self.use_done
        metrics["action_norm"] = np.linalg.norm(self.prev_action)
        if not self.normalizing_by_return:
            self.reward_processor.update(reward)
        metrics["processed_reward"] = self.reward_processor.normalize(torch.tensor(reward)).item()
        obs_tensor, task_prompt_token_ids = self._preprocess(obs, info)
        normalized_action = (self.prev_action - self.action_bias) / self.action_scale
        self.rb.add(
            obs_tensor,
            reward,
            episode_done if self.use_done else False,
            self.rnn_state.squeeze(0),
            torch.from_numpy(normalized_action).to(self.device),
            task_prompt_token_ids,
        )

        warmup = self.learning_mode == "off_policy" and global_step < self.learning_starts

        if not warmup and self.action_chunk is not None and self.chunk_step < self.horizon:
            action = self._to_env_action(self.action_chunk[self.chunk_step])
            self.prev_action = action
            self.chunk_step += 1
            metrics["chunk_step"] = self.chunk_step
            return StepResult(action=action, metrics=metrics, panels={})

        latest_data = self.rb.get_latest(self.seq_len)
        infer_result = self.network.infer(
            InferInput(
                s_seq=latest_data.observations,
                a_seq=latest_data.actions,
                r_seq=latest_data.rewards,
                rnn_state=self.rnn_state,
                task_prompts=[obs["language"]],
            )
        )
        self.rnn_state = infer_result.rnn_state
        metrics.update(infer_result.value_report)
        action_chunk = infer_result.action[0].cpu().numpy()
        self.action_chunk = action_chunk
        self.chunk_step = 1
        action = self._to_env_action(action_chunk[0])
        self.prev_action = action
        metrics["chunk_step"] = self.chunk_step

        if warmup:
            action = self.action_space.sample()
            self.action_chunk = None
            self.chunk_step = 0
            self.prev_action = action
            metrics["chunk_step"] = self.chunk_step
        return StepResult(action=action, metrics=metrics, panels={})

    def _preprocess(self, obs: dict[str, Any], info: dict) -> tuple:
        """Turn the raw observation into what the replay buffer stores this tick:
        the network-packed obs tensor (image plus the raw scalar observations
        ``[velocity, episode_return, pass_mark]``; the network updates its running
        stats here and normalizes at unpack) and the tokenized task prompt.
        ``info`` is unused by the obs-driven standard agent."""
        del info
        image = torch.from_numpy(obs["image"]).to(self.device)
        scalar_obs = np.concatenate(
            [obs["velocity"], obs["episode_return"], obs["pass_mark"]]
        ).astype(np.float32)
        self.network.observe_scalar_obs(scalar_obs)
        packed = self.network.pack_obs(image, torch.from_numpy(scalar_obs).to(self.device))
        task_prompt_token_ids = self.network.tokenize_task_prompt(obs["language"])
        return packed, task_prompt_token_ids

    def _to_env_action(self, net_action: np.ndarray) -> np.ndarray:
        """Map a single normalized policy action into the env's action space."""
        return np.clip(
            net_action * self.action_scale + self.action_bias, self.action_low, self.action_high
        )
