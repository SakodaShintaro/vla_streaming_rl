# SPDX-License-Identifier: MIT
"""Off-policy learning mode: act now, learn later from a large replay buffer.

Every tick is stored; once ``learning_starts`` ticks have gone by, one gradient
step on a ``batch_size`` sample of the buffer fires every tick, before the
action of that tick is chosen, whatever the ``horizon`` of the action chunks. Below ``learning_starts`` the network
is told it is warming up and returns uniform random actions, so the buffer
fills with something other than an untrained policy's output while the
network's recurrent state still follows the episode.

With ``streaming_update`` the replay holds the state the heads read, saved when
each action was chosen, instead of the observation, and every tick adds the
streaming update (the newest window, through the encoder, the memory layers'
LoRA carried by RTRL) to a replayed update of the heads alone on those saved
states; one optimizer step takes the sum. Only the newest window of
observations is kept for the streaming half.

The per-tick flow (observe, run the chunk) lives in :class:`Agent`.
"""

from typing import Any

import gymnasium as gym
import torch
from torch import nn, optim

from vla_streaming_rl.agents.base import Agent, StepResult, split_actor_critic_params
from vla_streaming_rl.networks.interface import InferInput, InferResult
from vla_streaming_rl.replay_buffer import ReplayBuffer
from vla_streaming_rl.reward_scaler import FixedRewardScaler, RewardScaler


class OffPolicyAgent(Agent):
    def __init__(
        self,
        *,
        action_space: gym.spaces.Box,
        image_names: tuple[str, ...],
        network: nn.Module,
        normalizing_by_return: bool,
        reward_scaler: RewardScaler | FixedRewardScaler,
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
        priority_alpha: float,
        priority_beta: float,
        priority_epsilon: float,
        reset_on_episode_end: bool,
        streaming_update: bool,
        instruction: str,
    ) -> None:
        # Actor / critic optimizer split (critic == value head); both AdamW,
        # the replayed update has no trace to carry.
        actor_params, critic_params = split_actor_critic_params(network)
        if streaming_update:
            # 状態を保存するのは行動を選んだステップだけなので、毎ステップ選ぶ長さ 1 のチャンク
            assert horizon == 1, horizon
        window_len = seq_len + horizon
        super().__init__(
            action_space=action_space,
            image_names=image_names,
            network=network,
            instruction=instruction,
            actor_optimizer=optim.AdamW(actor_params, lr=actor_lr, weight_decay=weight_decay),
            critic_optimizer=optim.AdamW(critic_params, lr=critic_lr, weight_decay=weight_decay),
            # streaming の更新があれば、観測は streaming の更新が読む最新の窓だけ持つ
            buffer_size=window_len if streaming_update else buffer_size,
            buffer_device=buffer_device,
            priority_alpha=0.0 if streaming_update else priority_alpha,
            priority_beta=0.0 if streaming_update else priority_beta,
            priority_epsilon=1.0 if streaming_update else priority_epsilon,
            seq_len=seq_len,
            horizon=horizon,
            normalizing_by_return=normalizing_by_return,
            reward_scaler=reward_scaler,
            use_done=use_done,
            reset_on_episode_end=reset_on_episode_end,
        )
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.is_learning = False
        self.streaming_update = streaming_update
        # streaming の更新があるときのリプレイ。行ごとに、観測の代わりに行動を選んだときの状態を
        # 持つ（報酬・終端・行動・スカラーは観測のバッファの同じステップの行と同じ）。再帰の
        # 状態は使わないので幅 1 の詰め物
        self.feature_rb = (
            ReplayBuffer(
                size=buffer_size,
                seq_len=window_len,
                horizon=horizon,
                obs_shape=network.feature_shape(),
                rnn_state_shape=(1,),
                action_shape=action_space.shape,
                output_device=self.device,
                storage_device=torch.device(buffer_device),
                priority_alpha=priority_alpha,
                priority_beta=priority_beta,
                priority_epsilon=priority_epsilon,
            )
            if streaming_update
            else None
        )

    def step(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult:
        if self.streaming_update:
            return self._step_with_streaming_update(
                global_step, obs, reward, terminated, truncated, info
            )
        train_metrics = {}
        if (
            global_step >= self.learning_starts
            and self.rb.num_stored() >= self.batch_size + self.rb.seq_len
        ):
            if not self.is_learning:
                print(f"Start learning at global step {global_step}.")
                self.is_learning = True
            data = self.rb.sample(self.batch_size)
            replay_stats = self.rb.batch_stats(data)
            data.rewards = self.reward_scaler.normalize(data.rewards)
            result = self.network.compute_loss(data)
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            result.loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            self.rb.update_priorities(data.start_indices, result.td_error)
            train_metrics = {**result.info, **replay_stats}
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
        metrics = self._observe(obs, reward, terminated, truncated, info)
        warmup = global_step < self.learning_starts
        if not warmup and self._has_chunk_left():
            return self._continue_chunk(metrics)
        infer_result = self._infer(warmup)
        if self.streaming_update:
            self._store_features(infer_result.features)
        return self._start_chunk(infer_result, metrics)

    def _step_with_streaming_update(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult:
        """最新の窓での streaming の更新と、保存した状態のリプレイでのヘッドの更新を足して
        1回更新し、streaming の更新の順伝播が出した行動を実行する。エピソードをまたぐ窓や
        学習を始める前は、更新せずに推論だけする。"""
        metrics = self._observe(obs, reward, terminated, truncated, info)
        window_len = self.seq_len + self.horizon
        learning = (
            global_step >= self.learning_starts
            and self.feature_rb.num_stored() >= self.batch_size + window_len
        )
        if not (learning and self.rb.latest_window_is_clean(window_len)):
            infer_result = self._infer(global_step < self.learning_starts)
            self._store_features(infer_result.features)
            return self._start_chunk(infer_result, metrics)
        if not self.is_learning:
            print(f"Start learning at global step {global_step}.")
            self.is_learning = True
            # 学習を始める前は RTRL のトレースを毎ステップ進めていないので、ここで打ち切る
            self.network.reset_memory_trace()

        data = self.rb.get_latest(window_len)
        data.rewards = self.reward_scaler.normalize(data.rewards)
        result = self.network.infer_and_compute_loss(data)
        replay = self.feature_rb.sample(self.batch_size)
        replay_stats = self.feature_rb.batch_stats(replay)
        replay.rewards = self.reward_scaler.normalize(replay.rewards)
        replay_result = self.network.compute_feature_loss(replay)

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        (result.loss_result.loss + replay_result.loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.feature_rb.update_priorities(replay.start_indices, replay_result.td_error)

        metrics.update(result.loss_result.info)
        metrics.update({f"replay/{key}": value for key, value in replay_result.info.items()})
        metrics.update(replay_stats)
        self._store_features(result.infer_result.features)
        return self._start_chunk(result.infer_result, metrics)

    def _store_features(self, features: torch.Tensor) -> None:
        """観測のバッファの最新の行に対応する、行動を選んだときの状態 (1, state_dim) を
        リプレイに積む。報酬・終端・行動・スカラーはその行のものをそのまま写す。"""
        latest = self.rb.get_latest(1)
        self.feature_rb.add(
            features.detach()[0],
            float(latest.rewards[0, 0, 0]),
            bool(latest.dones[0, 0, 0]),
            torch.zeros(1),
            latest.actions[0, 0],
            latest.scalar_obs[0, 0].cpu().numpy(),
        )

    def _infer(self, warmup: bool) -> InferResult:
        """バッファの最新の窓から、ネットワークに推論させる。"""
        latest_data = self.rb.get_latest(self.seq_len)
        # 学習と同じく、窓の先頭のステップで保存した状態から窓を読む
        return self.network.infer(
            InferInput(
                s_seq=latest_data.observations,
                a_seq=latest_data.actions,
                r_seq=self.reward_scaler.normalize(latest_data.rewards),
                rnn_state=latest_data.rnn_state[:, 0],
                scalar_obs_seq=latest_data.scalar_obs,
                warmup=warmup,
            )
        )
