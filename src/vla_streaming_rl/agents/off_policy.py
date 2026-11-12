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
"""

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

from vla_streaming_rl.networks.interface import InferInput, InferResult
from vla_streaming_rl.replay_buffer import ReplayBuffer
from vla_streaming_rl.reward_scaler import FixedRewardScaler, RewardScaler
from vla_streaming_rl.scalar_obs import read_scalar_obs


@dataclass
class StepResult:
    """Structured return value of an agent's ``select_action`` / ``step``.

    Replaces the old free-form ``info`` dict, which mixed three unrelated
    concerns and forced the trainer to know each agent's internal keys.
    The concerns are now split:

    - ``action``: the env action to execute this tick.
    - ``metrics``: flat scalar telemetry, logged verbatim to wandb.
    - ``panels``: named RGB image panels (HxWx3) for the render strip. The
      trainer concatenates whatever panels are present, in insertion order,
      so an agent can contribute extra panels (e.g. the next-frame prediction
      or a bird's-eye trajectory view) without the trainer knowing about them.
      Stable-panel contract: an agent must emit the SAME set of panel keys with
      the SAME shapes on every step of a run (use a blank placeholder when the
      content does not exist yet, e.g. before the first prediction). The render
      frames are encoded into a single video, which requires a constant size.
    - ``texts``: named free-form text, written to the episode's ``texts.tsv``.
      What a panel shows as pixels this keeps as characters, so a chain of
      thought can be read back and searched after the run. Every agent puts the
      instruction its policy read this tick under ``"prompt"``, which is also
      what the trainer captions the render with.
    """

    action: np.ndarray
    metrics: dict[str, float]
    panels: dict[str, np.ndarray]
    texts: dict[str, str]


class OffPolicyAgent:
    """The contract the trainer drives: ``step`` on every tick while learning,
    ``select_action`` when acting without learning. ``instruction`` is the text
    the policy reads beside the image.

    毎ティックの観測をバッファに1行積み、ネットワークが出した行動チャンクを1ステップずつ
    実行する。"""

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
        critic_params = list(network.value_head.parameters())
        critic_param_ids = {id(p) for p in critic_params}
        actor_params = [p for p in network.parameters() if id(p) not in critic_param_ids]
        if streaming_update:
            # 状態を保存するのは行動を選んだステップだけなので、毎ステップ選ぶ長さ 1 のチャンク
            assert horizon == 1, horizon
        window_len = seq_len + horizon
        self.actor_optimizer = optim.AdamW(actor_params, lr=actor_lr, weight_decay=weight_decay)
        self.critic_optimizer = optim.AdamW(critic_params, lr=critic_lr, weight_decay=weight_decay)
        self.device = torch.device("cuda")
        # 観測から読んでネットワークに渡す画像の名前（IMAGE_OBS_NAMES の並び）
        self.image_names = image_names
        self.network = network
        self.instruction = instruction
        self.seq_len = seq_len
        self.horizon = horizon
        self.normalizing_by_return = normalizing_by_return
        self.use_done = use_done
        self.reset_on_episode_end = reset_on_episode_end
        self.reward_scaler = reward_scaler

        # 方策の行動は [-1, 1] に正規化した空間にあり、環境の行動空間へはこれで写す
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_scale = (action_space.high - action_space.low) / 2.0
        self.action_bias = (action_space.high + action_space.low) / 2.0

        self.rnn_state = network.init_state().to(self.device)
        self.rb = ReplayBuffer(
            # streaming の更新があれば、観測は streaming の更新が読む最新の窓だけ持つ
            size=window_len if streaming_update else buffer_size,
            seq_len=window_len,
            horizon=horizon,
            obs_shape=network.stored_image_shape(),
            rnn_state_shape=self.rnn_state.squeeze(0).shape,
            action_shape=action_space.shape,
            output_device=self.device,
            storage_device=torch.device(buffer_device),
            priority_alpha=0.0 if streaming_update else priority_alpha,
            priority_beta=0.0 if streaming_update else priority_beta,
            priority_epsilon=1.0 if streaming_update else priority_epsilon,
        )

        # 実行中の行動チャンク (horizon, action_dim) と、そのうち実行した数
        self.action_chunk: np.ndarray | None = None
        self.chunk_step = 0
        # 最後に環境へ渡した行動。エピソードをまたいで持ち越す
        self.prev_action = np.zeros(action_space.shape, dtype=np.float32)
        # the first observation of a run starts an episode
        self._previous_done = True
        # Shared representation fed to policy/value/prediction heads on the most
        # recent inference (used by scripts/collect_probe_data.py).
        self.last_features: torch.Tensor | None = None

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

    def on_episode_end(self, score: float) -> None:
        """エピソードが ``score`` で終わったときに呼ぶ。学習でも評価でも呼ぶ。"""
        if self.normalizing_by_return:
            self.reward_scaler.update(score)

    def optimizer_state_dict(self) -> dict:
        """Optimizer states to checkpoint. The trainer stores and restores this
        verbatim."""
        return {
            "actor": self.actor_optimizer.state_dict(),
            "critic": self.critic_optimizer.state_dict(),
        }

    def load_optimizer_state_dict(self, state: dict) -> None:
        self.actor_optimizer.load_state_dict(state["actor"])
        self.critic_optimizer.load_state_dict(state["critic"])

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

    def _observe(
        self, obs: dict[str, Any], reward: float, terminated: bool, truncated: bool, info: dict
    ) -> dict[str, float]:
        """このティックの観測を受け取り、エピソードの境目を処理して、バッファに1行積む。
        metrics を返す。"""
        episode_done = terminated or truncated
        # A terminal observation still belongs to the episode that ended, so the
        # recurrent state is reset on the tick after it rather than on it.
        reset_state = self._previous_done and self.reset_on_episode_end
        if self._previous_done:
            if reset_state:
                self.rnn_state = self.network.init_state().to(self.device)
                # 空の記憶はパラメータによらないので、RTRL のトレースも 0 に戻す
                self.network.reset_memory_trace()
            elif self.rb.full or self.rb.idx > 0:
                # 記憶を持ち越すなら、終わったエピソードの最後の観測は、エピソードをまたぐ窓の
                # 状態になって学習に使われない。そのブロックの分も RTRL のトレースを進めて、
                # 持ち越す記憶の状態と感度をそろえる
                last = self.rb.get_latest(self.seq_len)
                last.rewards = self.reward_scaler.normalize(last.rewards)
                self.network.advance_memory_trace(last)
            # A chunk begun on the terminal frame would otherwise run into the
            # new episode; the first action of an episode comes from its own frame.
            self.action_chunk = None
            self.chunk_step = 0
        self._previous_done = episode_done

        # What the agent trains on, against what the env reported as its score.
        shaped_reward = info["shaped_reward"]
        if not self.normalizing_by_return:
            self.reward_scaler.update(shaped_reward)
        metrics = {
            "shaped_reward": shaped_reward,
            "action_norm": float(np.linalg.norm(self.prev_action)),
            # 前のステップに実行した行動の成分ごとの値。偏りを見る
            **{f"action_{i}": float(value) for i, value in enumerate(self.prev_action)},
            "processed_reward": self.reward_scaler.normalize(torch.tensor(shaped_reward)).item(),
        }

        images = [torch.from_numpy(obs[name]).to(self.device) for name in self.image_names]
        scalar_obs = read_scalar_obs(obs)
        normalized_action = (self.prev_action - self.action_bias) / self.action_scale
        self.rb.add(
            self.network.to_stored_image(
                images, scalar_obs, normalized_action, shaped_reward, self.instruction, reset_state
            ),
            shaped_reward,
            episode_done if self.use_done else False,
            self.rnn_state.squeeze(0),
            torch.from_numpy(normalized_action).to(self.device),
            scalar_obs,
        )
        return metrics

    def _has_chunk_left(self) -> bool:
        """実行中のチャンクに、まだ実行していない行動があるか。"""
        return self.action_chunk is not None and self.chunk_step < self.horizon

    def _start_chunk(self, infer_result: InferResult, metrics: dict) -> StepResult:
        """ネットワークが出したチャンクを始め、その最初の行動を実行する。"""
        self.rnn_state = infer_result.rnn_state
        self.last_features = infer_result.features
        metrics.update(infer_result.value_report)
        self.action_chunk = infer_result.action[0].cpu().numpy()
        self.chunk_step = 0
        return self._continue_chunk(metrics)

    def _continue_chunk(self, metrics: dict) -> StepResult:
        """実行中のチャンクの次の行動を、環境の行動空間に写して実行する。"""
        action = np.clip(
            self.action_chunk[self.chunk_step] * self.action_scale + self.action_bias,
            self.action_low,
            self.action_high,
        )
        self.chunk_step += 1
        self.prev_action = action
        metrics["chunk_step"] = self.chunk_step
        return StepResult(
            action=action,
            metrics=metrics,
            panels={},
            texts={"prompt": self.instruction},
        )
