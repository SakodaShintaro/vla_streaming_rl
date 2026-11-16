# SPDX-License-Identifier: MIT
from collections.abc import Callable
from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch import nn

from ..replay_buffer import ReplayBufferData
from .modules.prediction_head import LatentTDPredictionHead
from .modules.value_head import DistributionalValueHead
from .modules.wan_encoder import WanEncoder


@dataclass
class WanInference:
    """``infer`` の出力。"""

    action: torch.Tensor  # (1, action_dim)
    value_report: dict[str, float]  # value head diagnostics (incl. "value")
    state: torch.Tensor  # (1, state_dim) ヘッドが読んだ状態


@dataclass
class WanLoss:
    """``compute_replay_loss`` の出力。``info`` は ``名前 -> 値`` の記録で、そのまま wandb に
    出す。"""

    loss: torch.Tensor
    info: dict  # scalar name -> float
    td_error: torch.Tensor  # (B,) 窓ごとの TD 誤差。リプレイの優先度の更新に使う


class WanActorCriticWithActionValue(nn.Module):
    """凍結した Wan2.1 の VAE で観測のフレームの流れを読み、最新の潜在から状態を作る
    アクタークリティック。

    毎ティック、エンコーダがフレームを VAE の流れに足して現在のフレームを含む潜在を作り
    （``represent``）、バッファに保存する。推論も学習も保存した表現を読み戻し、VAE を通さない。
    状態はステップごとに ``to_state`` で作り、時間方向には何もしない。リプレイの窓は 2 ステップで、
    最初のステップの状態が現在の状態、最後のステップの状態が次の状態。
    """

    def __init__(
        self,
        *,
        image_shapes: list[tuple[int, ...]],
        action_space_shape: tuple[int],
        value_head_factory: Callable[[int, int], DistributionalValueHead],
        policy_head_factory,
        prediction_head_factory: Callable[..., LatentTDPredictionHead],
        critic_loss_weight: float,
        wan_config: DictConfig,
    ) -> None:
        super().__init__()
        # 表現はバッファに保存するので、バッファから読む窓は今のステップだけでよい
        self.seq_len = 1
        self.critic_loss_weight = critic_loss_weight
        device = "cuda"
        action_dim = action_space_shape[0]
        self.encoder = WanEncoder(
            image_shapes=image_shapes,
            action_dim=action_dim,
            wan_config=wan_config,
            device=device,
        )
        state_dim = self.encoder.output_dim
        self.policy_head = policy_head_factory(
            state_dim=state_dim, action_dim=action_dim, token_dim=self.encoder.token_dim
        ).to(device)
        self.value_head = value_head_factory(
            in_channels=state_dim, action_dim=action_dim, token_dim=self.encoder.token_dim
        ).to(device)
        self.prediction_head = prediction_head_factory(
            state_dim=state_dim, action_dim=action_dim
        ).to(device)

    def stored_image_shape(self) -> tuple[int, ...]:
        """バッファに保存する、画像の表現の形。"""
        return self.encoder.representation_shape()

    def represent(self, images: list[torch.Tensor]) -> torch.Tensor:
        """このティックの画像 (3, H, W) の並びを VAE の流れに足した、バッファに保存する表現。"""
        return self.encoder.represent(images)

    @torch.inference_mode()
    def infer(self, window: ReplayBufferData, warmup: bool) -> WanInference:
        """バッファの最新の窓 (1, seq_len) の最後のステップの状態から行動を出す。``warmup``
        （バッファを貯めている間）なら一様乱数の行動を返す。"""
        assert window.observations.shape[0] == 1, "Batch size must be 1 for inference"
        state = self._state(window, -1)
        action = self.policy_head.get_action(state)
        value_report = self.value_head.value_report(self.value_head(state, action).output)
        if warmup:
            action = torch.rand_like(action) * 2.0 - 1.0
        return WanInference(action=action, value_report=value_report, state=state)

    def compute_replay_loss(self, data: ReplayBufferData) -> WanLoss:
        """リプレイの窓の損失。"""
        state = self._state(data, -2)
        with torch.no_grad():
            next_state = self._state(data, -1)
        action = data.actions[:, -1]
        reward = data.rewards[:, -1]
        with torch.no_grad():
            # TD の目標は割引率ごとに、その割引率の方策の次の行動で作る。最後が主の割引率の
            # 方策の行動で、状態の予測にはそれを使う
            next_gamma_actions = self.policy_head.get_gamma_actions(next_state)
            next_action = next_gamma_actions[:, -1]
            next_q = self.value_head.gamma_values(next_state, next_gamma_actions)
        state_prediction_loss, state_prediction_info, intrinsic_reward = (
            self.prediction_head.compute_loss(state, action, next_state, next_action)
        )
        # 内発報酬は遷移の報酬に足す。TD の目標にだけ入る
        reward = reward + intrinsic_reward.view(-1, 1)
        target_value = self.value_head.compute_target_value(next_q, reward, data.dones[:, -1])
        critic_loss, critic_info, td_error = self.value_head.compute_critic_loss(
            state, action, target_value, data.weights
        )
        actor_loss, actor_info = self.policy_head.compute_actor_loss(
            state, action, value_head=self.value_head
        )
        info = {
            f"losses/{key}": value
            for key, value in {**critic_info, **actor_info, **state_prediction_info}.items()
        }
        return WanLoss(
            loss=self.critic_loss_weight * critic_loss + actor_loss + state_prediction_loss,
            info=info,
            td_error=td_error,
        )

    def _state(self, data: ReplayBufferData, step: int) -> torch.Tensor:
        """窓 (B, T, ...) の ``step`` 番目のステップの、保存した表現・前の行動・報酬・スカラーの
        観測から状態を作る。"""
        return self.encoder.to_state(
            data.observations[:, step],
            data.actions[:, step],
            data.rewards[:, step],
            data.scalar_obs[:, step],
        )
