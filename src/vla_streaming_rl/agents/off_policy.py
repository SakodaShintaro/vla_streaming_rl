# SPDX-License-Identifier: MIT
"""Off-policy learning mode: act now, learn later from a large replay buffer.

Every tick is stored; once ``learning_starts`` ticks have gone by, one gradient
step on a ``batch_size`` sample of the buffer fires every ``horizon`` ticks,
before the action of that tick is chosen. Below ``learning_starts`` the network
is told it is warming up and returns uniform random actions, so the buffer
fills with something other than an untrained policy's output while the
network's recurrent state still follows the episode.

The learning mode is the class and the network is a constructor argument, so
this file is one half of the (learning mode) x (network) grid; the streaming
half is ``streaming.py``. The per-tick flow both share lives in :class:`Agent`.
"""

from typing import Any

import gymnasium as gym
import torch
from torch import nn, optim

from vla_streaming_rl.agents.base import Agent, StepResult, split_actor_critic_params
from vla_streaming_rl.networks.interface import InferInput


class OffPolicyAgent(Agent):
    def __init__(
        self,
        *,
        action_space: gym.spaces.Box,
        image_names: tuple[str, ...],
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
        weight_decay: float,
        buffer_size: int,
        buffer_device: str,
        reset_on_episode_end: bool,
        instruction: str,
    ) -> None:
        # Actor / critic optimizer split (critic == value head); both AdamW,
        # the replayed update has no trace to carry.
        actor_params, critic_params = split_actor_critic_params(network)
        super().__init__(
            action_space=action_space,
            image_names=image_names,
            network=network,
            instruction=instruction,
            actor_optimizer=optim.AdamW(actor_params, lr=actor_lr, weight_decay=weight_decay),
            critic_optimizer=optim.AdamW(critic_params, lr=critic_lr, weight_decay=weight_decay),
            buffer_size=buffer_size,
            buffer_device=buffer_device,
            seq_len=seq_len,
            horizon=horizon,
            normalizing_by_return=normalizing_by_return,
            use_done=use_done,
            reset_on_episode_end=reset_on_episode_end,
        )
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.is_learning = False

    def step(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult:
        train_metrics = {}
        if (
            global_step >= self.learning_starts
            and global_step % self.horizon == 0
            and self.rb.num_stored() >= self.batch_size + self.rb.seq_len
        ):
            if not self.is_learning:
                print(f"Start learning at global step {global_step}.")
                self.is_learning = True
            data = self.rb.sample(self.batch_size)
            data.rewards = self.reward_scaler.normalize(data.rewards)
            result = self.network.compute_loss(data)
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            result.loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            train_metrics = result.info
        step_result = self.select_action(global_step, obs, reward, terminated, truncated, info)
        step_result.metrics.update(train_metrics)
        return step_result

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
        images, metrics = self._observe(obs, reward, terminated, truncated, info)
        warmup = global_step < self.learning_starts
        if not warmup and self._has_chunk_left():
            return self._continue_chunk(metrics)

        latest_data = self.rb.get_latest(self.seq_len)
        # 学習と同じく、窓の先頭のステップで保存した状態から窓を読む
        infer_result = self.network.infer(
            InferInput(
                frames=images,
                s_seq=latest_data.observations,
                a_seq=latest_data.actions,
                r_seq=latest_data.rewards,
                rnn_state=latest_data.rnn_state[:, 0],
                scalar_obs_seq=latest_data.scalar_obs,
                instruction=self.instruction,
                warmup=warmup,
            )
        )
        return self._start_chunk(infer_result, metrics)
