# SPDX-License-Identifier: MIT
"""Off-policy learning mode: act now, learn later from a large replay buffer.

Every tick is stored; once ``learning_starts`` ticks have gone by, one gradient
step on a ``batch_size`` sample of the buffer fires every tick, after the tick
is stored and before its action is chosen. Below ``learning_starts`` the
network is told it is warming up and returns
uniform random actions, so the buffer fills with something other than an
untrained policy's output.

With ``streaming_update`` the encoder has GDN layers with LoRA that carry
their recurrent state across ticks. The buffer still stores one representation
per tick, the image tokens out of those layers, and the replayed update is
unchanged; the encoder carries the layers' recurrent state and its RTRL
sensitivity, and the agent keeps the previous tick's output of those layers
with its graph. Every tick appends the newest window to the replayed batch as
one more sample, whose gradient reaches the LoRA through that graph, and one
optimizer step takes the loss of the whole batch. The graph was built before
the previous update, so that gradient is evaluated at the weights one update
old.
"""

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

from vla_streaming_rl.networks.vla_actor_critic_with_action_value import (
    InferInput,
)
from vla_streaming_rl.replay_buffer import ReplayBuffer
from vla_streaming_rl.reward_scaler import FixedRewardScaler, RewardScaler
from vla_streaming_rl.scalar_obs import read_scalar_obs


@dataclass
class StepResult:
    """Structured return value of the agent's ``select_action`` / ``step``.

    - ``action``: the env action to execute this tick.
    - ``metrics``: flat scalar telemetry, logged verbatim to wandb.
    - ``panels``: named RGB image panels (HxWx3) for the render strip. The
      trainer concatenates whatever panels are present, in insertion order,
      so the agent can contribute extra panels (e.g. the next-frame prediction
      or a bird's-eye trajectory view) without the trainer knowing about them.
      Stable-panel contract: the agent must emit the SAME set of panel keys with
      the SAME shapes on every step of a run (use a blank placeholder when the
      content does not exist yet, e.g. before the first prediction). The render
      frames are encoded into a single video, which requires a constant size.
    - ``texts``: named free-form text, written to the episode's ``texts.tsv``.
      What a panel shows as pixels this keeps as characters, so a chain of
      thought can be read back and searched after the run. The agent puts the
      instruction its policy read this tick under ``"prompt"``, which is also
      what the trainer captions the render with.
    """

    action: np.ndarray
    metrics: dict[str, float]
    panels: dict[str, np.ndarray]
    texts: dict[str, str]


class OffPolicyAgent:
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
        actor_lr: float,
        critic_lr: float,
        lora_lr: float,
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
        # 最適化は actor・critic（価値ヘッド）・LoRA 付きの GDN の層の LoRA の3つに分ける
        critic_params = list(network.value_head.parameters())
        # LoRA の勾配は streaming の 1 サンプルからしか来ず、ヘッドの勾配とは大きさが違うので、
        # 学習率もクリップも分ける
        self.lora_params = (
            list(network.encoder.lora_gdn_layers.parameters()) if streaming_update else []
        )
        separate_ids = {id(p) for p in critic_params + self.lora_params}
        actor_params = [p for p in network.parameters() if id(p) not in separate_ids]
        self.head_params = actor_params + critic_params
        self.actor_optimizer = optim.AdamW(actor_params, lr=actor_lr, weight_decay=weight_decay)
        self.critic_optimizer = optim.AdamW(critic_params, lr=critic_lr, weight_decay=weight_decay)
        self.lora_optimizer = (
            optim.AdamW(self.lora_params, lr=lora_lr, weight_decay=weight_decay)
            if streaming_update
            else None
        )
        self.device = torch.device("cuda")
        # 観測から読んでネットワークに渡す画像の名前（IMAGE_OBS_NAMES の並び）
        self.image_names = image_names
        self.network = network
        self.instruction = instruction
        self.seq_len = seq_len
        self.normalizing_by_return = normalizing_by_return
        self.use_done = use_done
        self.reset_on_episode_end = reset_on_episode_end
        self.reward_scaler = reward_scaler
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.is_learning = False
        self.streaming_update = streaming_update

        # 方策の行動は [-1, 1] に正規化した空間にあり、環境の行動空間へはこれで写す
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_scale = (action_space.high - action_space.low) / 2.0
        self.action_bias = (action_space.high + action_space.low) / 2.0

        self.rb = ReplayBuffer(
            size=buffer_size,
            # 状態の窓と、その後の遷移の結果の1行
            seq_len=seq_len + 1,
            obs_shape=network.stored_image_shape(),
            action_shape=action_space.shape,
            output_device=self.device,
            storage_device=torch.device(buffer_device),
            priority_alpha=priority_alpha,
            priority_beta=priority_beta,
            priority_epsilon=priority_epsilon,
        )

        # 最後に環境へ渡した行動。エピソードをまたいで持ち越す
        self.prev_action = np.zeros(action_space.shape, dtype=np.float32)
        # the first observation of a run starts an episode
        self._previous_done = True
        # このティックがエピソードの最初のティックか
        self._episode_start = True
        # 最新とその前のティックの表現。streaming_update なら LoRA 付きの GDN の層の出力で、計算
        # グラフが付いている
        self.latest_image: torch.Tensor | None = None
        self.previous_image: torch.Tensor | None = None
        # Shared representation fed to policy/value/prediction heads on the most
        # recent inference (used by scripts/collect_probe_data.py).
        self.last_features: torch.Tensor | None = None

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
        if (
            global_step >= self.learning_starts
            and self.rb.num_stored() >= self.batch_size + self.rb.seq_len
        ):
            metrics.update(self._learn(global_step))
        return self._act(global_step, metrics)

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
        return self._act(global_step, metrics)

    def on_episode_end(self, score: float) -> None:
        """エピソードが ``score`` で終わったときに呼ぶ。学習でも評価でも呼ぶ。"""
        if self.normalizing_by_return:
            self.reward_scaler.update(score)

    def optimizer_state_dict(self) -> dict:
        """Optimizer states to checkpoint. The trainer stores and restores this
        verbatim."""
        state = {
            "actor": self.actor_optimizer.state_dict(),
            "critic": self.critic_optimizer.state_dict(),
        }
        if self.streaming_update:
            state["lora"] = self.lora_optimizer.state_dict()
        return state

    def load_optimizer_state_dict(self, state: dict) -> None:
        self.actor_optimizer.load_state_dict(state["actor"])
        self.critic_optimizer.load_state_dict(state["critic"])
        if self.streaming_update:
            self.lora_optimizer.load_state_dict(state["lora"])

    ####################
    # Internal methods #
    ####################

    def _learn(self, global_step: int) -> dict[str, float]:
        """リプレイのバッチ（``streaming_update`` なら最新の窓をつないだもの）の損失で1回
        更新する。metrics を返す。"""
        window_len = self.seq_len + 1
        if not self.is_learning:
            print(f"Start learning at global step {global_step}.")
            self.is_learning = True
        data = self.rb.sample(self.batch_size)
        metrics = self.rb.batch_stats(data)
        data.rewards = self.reward_scaler.normalize(data.rewards)
        # streaming_update なら最新の窓もつなぐ。エピソードをまたぐ窓では学習しない
        if self.streaming_update and not self._episode_start:
            latest = self.rb.get_latest(window_len)
            latest.rewards = self.reward_scaler.normalize(latest.rewards)
            result = self.network.compute_replay_and_streaming_loss(
                data, self.previous_image, latest
            )
        else:
            result = self.network.compute_replay_loss(data)
        metrics.update(result.info)
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        if self.streaming_update:
            self.lora_optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        metrics["losses/head_grad_norm"] = nn.utils.clip_grad_norm_(
            self.head_params, self.max_grad_norm
        ).item()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        if self.streaming_update:
            metrics["losses/lora_grad_norm"] = nn.utils.clip_grad_norm_(
                self.lora_params, self.max_grad_norm
            ).item()
            self.lora_optimizer.step()
        self.rb.update_priorities(data.start_indices, result.td_error)
        return metrics

    def _act(self, global_step: int, metrics: dict) -> StepResult:
        """バッファの最新の窓から推論した行動を、環境の行動空間に写して実行する。"""
        warmup = global_step < self.learning_starts
        latest_data = self.rb.get_latest(self.seq_len)
        infer_result = self.network.infer(
            InferInput(
                s_seq=latest_data.observations,
                a_seq=latest_data.actions,
                r_seq=self.reward_scaler.normalize(latest_data.rewards),
                scalar_obs_seq=latest_data.scalar_obs,
                warmup=warmup,
            )
        )
        self.last_features = infer_result.features
        metrics.update(infer_result.value_report)
        action = np.clip(
            infer_result.action[0].cpu().numpy() * self.action_scale + self.action_bias,
            self.action_low,
            self.action_high,
        )
        self.prev_action = action
        return StepResult(
            action=action,
            metrics=metrics,
            panels={},
            texts={"prompt": self.instruction},
        )

    def _observe(
        self, obs: dict[str, Any], reward: float, terminated: bool, truncated: bool, info: dict
    ) -> dict[str, float]:
        """このティックの観測を受け取り、エピソードの境目を処理して、バッファに1行積む。
        metrics を返す。"""
        episode_done = terminated or truncated
        # A terminal observation still belongs to the episode that ended, so the
        # recurrent state is reset on the tick after it rather than on it.
        self._episode_start = self._previous_done
        reset_state = self._episode_start and self.reset_on_episode_end
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
        image = self.network.represent(
            images, scalar_obs, normalized_action, shaped_reward, self.instruction, reset_state
        )
        self.previous_image = self.latest_image
        self.latest_image = image
        self.rb.add(
            image.detach(),
            shaped_reward,
            episode_done if self.use_done else False,
            torch.from_numpy(normalized_action).to(self.device),
            scalar_obs,
        )
        return metrics
