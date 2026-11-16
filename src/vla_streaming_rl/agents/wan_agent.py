# SPDX-License-Identifier: MIT
"""The Wan agent: the frozen Wan2.1 VAE reads the observation frames as one
continuing video, the latest latent becomes the state, and the heads learn
off-policy from a large replay buffer.

Every tick the latent containing the current frame is stored; once ``learning_starts`` ticks
have gone by, one gradient step on a ``batch_size`` sample of windows fires
every tick, after the tick is stored and before its action is chosen. Below
``learning_starts`` the network returns uniform random actions. The VAE is
frozen, so the stored latents never go stale; the heads are trained on them.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

from vla_streaming_rl.agents.step_result import StepResult
from vla_streaming_rl.networks.wan_actor_critic_with_action_value import (
    WanActorCriticWithActionValue,
)
from vla_streaming_rl.replay_buffer import ReplayBuffer
from vla_streaming_rl.reward_scaler import FixedRewardScaler, RewardScaler
from vla_streaming_rl.scalar_obs import read_scalar_obs


class WanAgent:
    def __init__(
        self,
        *,
        action_space: gym.spaces.Box,
        image_names: tuple[str, ...],
        network: WanActorCriticWithActionValue,
        normalizing_by_return: bool,
        reward_scaler: RewardScaler | FixedRewardScaler,
        learning_starts: int,
        batch_size: int,
        max_grad_norm: float,
        use_done: bool,
        actor_lr: float,
        critic_lr: float,
        weight_decay: float,
        buffer_size: int,
        buffer_device: str,
        priority_alpha: float,
        priority_beta: float,
        priority_epsilon: float,
        instruction: str,
    ) -> None:
        # 最適化は actor（エンコーダの射影などを含む）と critic（価値ヘッド）の2つに分ける
        critic_params = list(network.value_head.parameters())
        critic_ids = {id(p) for p in critic_params}
        actor_params = [
            p for p in network.parameters() if p.requires_grad and id(p) not in critic_ids
        ]
        self.head_params = actor_params + critic_params
        self.actor_optimizer = optim.AdamW(actor_params, lr=actor_lr, weight_decay=weight_decay)
        self.critic_optimizer = optim.AdamW(critic_params, lr=critic_lr, weight_decay=weight_decay)
        self.device = torch.device("cuda")
        # 観測から読んでネットワークに渡す画像の名前（IMAGE_OBS_NAMES の並び）
        self.image_names = image_names
        self.network = network
        # 描画の見出しに出す指示の文
        self.instruction = instruction
        self.normalizing_by_return = normalizing_by_return
        self.use_done = use_done
        self.reward_scaler = reward_scaler
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.is_learning = False

        # 方策の行動は [-1, 1] に正規化した空間にあり、環境の行動空間へはこれで写す
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_scale = (action_space.high - action_space.low) / 2.0
        self.action_bias = (action_space.high + action_space.low) / 2.0

        self.rb = ReplayBuffer(
            size=buffer_size,
            # 状態の窓と、その後の遷移の結果の1行
            seq_len=network.seq_len + 1,
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
        metrics = self._observe(obs, terminated, truncated, info)
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
        metrics = self._observe(obs, terminated, truncated, info)
        return self._act(global_step, metrics)

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

    ####################
    # Internal methods #
    ####################

    def _learn(self, global_step: int) -> dict[str, float]:
        """リプレイのバッチの損失で1回更新する。metrics を返す。"""
        if not self.is_learning:
            print(f"Start learning at global step {global_step}.")
            self.is_learning = True
        data = self.rb.sample(self.batch_size)
        metrics = self.rb.batch_stats(data)
        data.rewards = self.reward_scaler.normalize(data.rewards)
        result = self.network.compute_replay_loss(data)
        metrics.update(result.info)
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        metrics["losses/head_grad_norm"] = nn.utils.clip_grad_norm_(
            self.head_params, self.max_grad_norm
        ).item()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.rb.update_priorities(data.start_indices, result.td_error)
        return metrics

    def _act(self, global_step: int, metrics: dict) -> StepResult:
        """バッファの最新の窓から推論した行動を、環境の行動空間に写して実行する。"""
        window = self.rb.get_latest(self.network.seq_len)
        window.rewards = self.reward_scaler.normalize(window.rewards)
        inference = self.network.infer(window, global_step < self.learning_starts)
        self.last_features = inference.state
        metrics.update(inference.value_report)
        action = np.clip(
            inference.action[0].cpu().numpy() * self.action_scale + self.action_bias,
            self.action_low,
            self.action_high,
        )
        self.prev_action = action
        return StepResult(
            action=action, metrics=metrics, panels={}, texts={"prompt": self.instruction}
        )

    def _observe(
        self, obs: dict[str, Any], terminated: bool, truncated: bool, info: dict
    ) -> dict[str, float]:
        """このティックの観測を VAE の流れに足して、バッファに1行積む。metrics を返す。"""
        episode_done = terminated or truncated
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
        normalized_action = (self.prev_action - self.action_bias) / self.action_scale
        self.rb.add(
            self.network.represent(images),
            shaped_reward,
            episode_done if self.use_done else False,
            torch.from_numpy(normalized_action).to(self.device),
            read_scalar_obs(obs),
        )
        return metrics
