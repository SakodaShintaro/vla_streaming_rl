# SPDX-License-Identifier: MIT
"""Streaming learning mode: inference and the update share one forward pass.

There is no replay: the buffer holds only the window the update reads (the
latest ``seq_len + horizon`` transitions), and every time a new action chunk is
needed the grad-enabled forward that produces it also produces the loss. With
``use_eligibility_trace`` the critic is optimized by AdamET, which carries the
trace across ticks and is reset on episode boundaries.

The learning mode is the class and the network is a constructor argument, so
this file is one half of the (learning mode) x (network) grid; the off-policy
half is ``off_policy.py``. The per-tick flow both share lives in :class:`Agent`.
"""

from typing import Any

import gymnasium as gym
import torch
from torch import nn, optim

from vla_streaming_rl.agents.base import Agent, StepResult, split_actor_critic_params
from vla_streaming_rl.networks.interface import InferInput
from vla_streaming_rl.optimizers.adam_et import AdamET
from vla_streaming_rl.reward_scaler import FixedRewardScaler, RewardScaler


class StreamingAgent(Agent):
    def __init__(
        self,
        *,
        action_space: gym.spaces.Box,
        image_names: tuple[str, ...],
        network: nn.Module,
        normalizing_by_return: bool,
        reward_scaler: RewardScaler | FixedRewardScaler,
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
        buffer_device: str,
        reset_on_episode_end: bool,
        instruction: str,
    ) -> None:
        # Actor / critic optimizer split (critic == value head). The critic uses
        # AdamET (eligibility traces) when the trace is on, AdamW otherwise; the
        # actor is always AdamW.
        actor_params, critic_params = split_actor_critic_params(network)
        super().__init__(
            action_space=action_space,
            image_names=image_names,
            network=network,
            instruction=instruction,
            actor_optimizer=optim.AdamW(actor_params, lr=actor_lr, weight_decay=weight_decay),
            critic_optimizer=(
                AdamET(critic_params, lr=critic_lr, gamma=gamma, et_lambda=et_lambda)
                if use_eligibility_trace
                else optim.AdamW(critic_params, lr=critic_lr, weight_decay=weight_decay)
            ),
            # Only the one window the update trains on is kept.
            buffer_size=seq_len + horizon,
            buffer_device=buffer_device,
            # 最新の窓だけで学習し、引くことがないので優先度は使わない
            priority_alpha=0.0,
            priority_beta=0.0,
            priority_epsilon=1.0,
            seq_len=seq_len,
            horizon=horizon,
            normalizing_by_return=normalizing_by_return,
            reward_scaler=reward_scaler,
            use_done=use_done,
            reset_on_episode_end=reset_on_episode_end,
        )
        self.max_grad_norm = max_grad_norm
        self.use_eligibility_trace = use_eligibility_trace
        # エピソードが終わったか。次の critic の更新で AdamET のトレースを切る
        self._episode_reset = False

    def step(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult:
        metrics = self._observe(obs, reward, terminated, truncated, info)
        if terminated or truncated:
            self._episode_reset = self.use_done
        if self._has_chunk_left():
            return self._continue_chunk(metrics)

        # Right after an episode boundary the newest window spans two episodes:
        # its states come from both and its chunk carries actions the agent never
        # took in the episode the state belongs to. There is nothing to learn from
        # it, but a chunk still has to be started, so the tick acts without a
        # learning step.
        if not self.rb.latest_window_is_clean(self.seq_len + self.horizon):
            return self._act(metrics)

        # new chunk: a single grad-enabled forward yields both the action chunk
        # and the training loss (fused inference + training).
        data = self.rb.get_latest(self.seq_len + self.horizon)
        data.rewards = self.reward_scaler.normalize(data.rewards)
        result = self.network.infer_and_compute_loss(data)
        metrics.update(result.loss_result.info)

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        if self.use_eligibility_trace:
            # Actor: backward actor-only loss → encoder + actor grads.
            result.et_info.actor_entropy_loss.backward(retain_graph=True)
            # Critic: backward -V(s) → value_head grads only (detached from encoder).
            result.et_info.neg_value.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step(delta=result.et_info.delta, reset=self._episode_reset)
            self._episode_reset = False
        else:
            result.loss_result.loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        return self._start_chunk(result.infer_result, metrics)

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
        """Act without learning: the trainer calls this once per episode after
        the reset, and the testbed calls it on every tick."""
        metrics = self._observe(obs, reward, terminated, truncated, info)
        if terminated or truncated:
            self._episode_reset = self.use_done
        if self._has_chunk_left():
            return self._continue_chunk(metrics)
        return self._act(metrics)

    def _act(self, metrics: dict) -> StepResult:
        """Start a chunk from the newest state window, with no learning step."""
        latest_data = self.rb.get_latest(self.seq_len)
        infer_result = self.network.infer(
            InferInput(
                s_seq=latest_data.observations,
                a_seq=latest_data.actions,
                r_seq=self.reward_scaler.normalize(latest_data.rewards),
                rnn_state=self.rnn_state,
                scalar_obs_seq=latest_data.scalar_obs,
                warmup=False,
            )
        )
        return self._start_chunk(infer_result, metrics)
