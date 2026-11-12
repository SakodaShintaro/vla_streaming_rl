# SPDX-License-Identifier: MIT
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from ..replay_buffer import ReplayBufferData
from .modules.prediction_head import LatentTDPredictionHead
from .modules.value_head import DistributionalValueHead
from .modules.vlm_encoder import VLMEncoder


@dataclass
class InferInput:
    """``infer`` の入力。バッファの最新の窓 (B=1)。"""

    s_seq: torch.Tensor  # (B, T, *obs_shape) image observation window
    a_seq: torch.Tensor  # (B, T, action_dim)
    r_seq: torch.Tensor  # (B, T, 1)
    scalar_obs_seq: torch.Tensor  # (B, T, SCALAR_OBS_DIM)、並びは SCALAR_OBS_NAMES
    # バッファを貯めている間のステップか。このあいだは一様乱数の行動を返す
    warmup: bool


@dataclass
class InferResult:
    """``infer`` の出力。"""

    action: torch.Tensor  # (B, action_dim)
    value_report: dict[str, float]  # value head diagnostics (incl. "value")
    features: torch.Tensor  # (B, state_dim) ヘッドが読んだ状態


@dataclass
class LossResult:
    """``compute_replay_loss`` と ``compute_replay_and_streaming_loss`` の出力。``info`` は
    ヘッドの種類で項目が変わる ``名前 -> 値`` の記録で、そのまま wandb に出す。"""

    loss: torch.Tensor
    info: dict  # scalar name -> float
    td_error: torch.Tensor  # (B,) 窓ごとの TD 誤差。リプレイの優先度の更新に使う


class VLAActorCriticWithActionValue(nn.Module):
    """画像と指示の文から行動を出す VLA。

    毎ステップ、観測の画像と指示の文と前の行動・報酬・スカラーの観測の文をエンコーダ（凍結した
    VLM）に読ませて画像のトークンの表現を作り（``represent``）、バッファに保存する。推論も学習も
    保存した表現を読み戻し、VLM を通さない。表現から射影とトークンを混ぜる層で状態を作り、方策
    ヘッド・行動価値ヘッド・予測ヘッドがその状態を読む。

    ``use_lora_gdn`` なら、表現はエンコーダの LoRA 付きの GDN の層の出力で、ステップをまたぐ履歴は
    その層の再帰の状態だけが運ぶ。再帰の状態と RTRL の感度はエンコーダが持ち、バッファには
    入れない。LoRA は ``compute_replay_and_streaming_loss`` の最新のステップの分で RTRL で学習し、
    リプレイの窓の分は保存した表現から射影とヘッドを学習する。
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
        vla_config: DictConfig,
        use_lora_gdn: bool,
    ) -> None:
        super().__init__()
        # 表現はバッファに保存するので、バッファから読む窓は今のステップだけでよい
        self.seq_len = 1
        self.critic_loss_weight = critic_loss_weight
        device = "cuda"
        action_dim = action_space_shape[0]
        self.encoder = VLMEncoder(
            image_shapes=image_shapes,
            action_dim=action_dim,
            vla_config=vla_config,
            use_lora_gdn=use_lora_gdn,
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
        """バッファに保存する、画像のトークンの表現の形。"""
        return self.encoder.representation_shape()

    def represent(
        self,
        images: list[torch.Tensor],
        scalar_obs: np.ndarray,
        prev_action: np.ndarray,
        reward: float,
        instruction: str,
        reset_state: bool,
    ) -> torch.Tensor:
        return self.encoder.represent(
            images, scalar_obs, prev_action, reward, instruction, reset_state
        )

    @torch.inference_mode()
    def infer(self, data: InferInput) -> InferResult:
        assert data.s_seq.shape[0] == 1, "Batch size must be 1 for inference"
        state = self.encoder.to_state(
            data.s_seq[:, -1], data.a_seq[:, -1], data.r_seq[:, -1], data.scalar_obs_seq[:, -1]
        )
        action = self.policy_head.get_action(state)
        value_report = self.value_head.value_report(self.value_head(state, action).output)
        if data.warmup:
            # バッファを貯めている間は一様乱数の行動を返す
            action = torch.rand_like(action) * 2.0 - 1.0
        return InferResult(action=action, value_report=value_report, features=state)

    def compute_replay_loss(self, data: ReplayBufferData) -> LossResult:
        """リプレイの窓の損失。"""
        state, next_state = self._replay_states(data)
        return self._head_terms(
            state,
            next_state,
            data.actions[:, -1],
            data.rewards[:, -1],
            data.dones[:, -1],
            data.weights,
        )

    def compute_replay_and_streaming_loss(
        self, data: ReplayBufferData, image: torch.Tensor, latest: ReplayBufferData
    ) -> LossResult:
        """リプレイの窓 ``data`` と最新の窓 ``latest`` (B=1) を1つのバッチにつないだ損失。
        最新の窓の現在の状態は、前のティックで ``represent`` が返した画像のトークン ``image``
        (num_positions, hidden_size)（計算グラフ付き）から作り、勾配は LoRA 付きの GDN の層の
        LoRA まで届く。ヘッドは損失の計算のたびに自分の状態（目標の予測器、誤差の統計、温度）を
        進めるので、2つに分けず1回で計算する。TD 誤差はリプレイの窓の分だけ返す。"""
        replay_state, replay_next_state = self._replay_states(data)
        streaming_state = self.encoder.to_state(
            image.unsqueeze(0),
            latest.actions[:, -2],
            latest.rewards[:, -2],
            latest.scalar_obs[:, -2],
        )
        with torch.no_grad():
            streaming_next_state = self._state(latest, -1)
        result = self._head_terms(
            torch.cat([replay_state, streaming_state]),
            torch.cat([replay_next_state, streaming_next_state]),
            torch.cat([data.actions[:, -1], latest.actions[:, -1]]),
            torch.cat([data.rewards[:, -1], latest.rewards[:, -1]]),
            torch.cat([data.dones[:, -1], latest.dones[:, -1]]),
            torch.cat([data.weights, latest.weights]),
        )
        result.td_error = result.td_error[: data.weights.shape[0]]
        return result

    ####################
    # Internal methods #
    ####################

    def _state(self, data: ReplayBufferData, step: int) -> torch.Tensor:
        """窓 (B, T, ...) の ``step`` 番目のステップの、保存した表現・前の行動・報酬・スカラーの
        観測から状態を作る。"""
        return self.encoder.to_state(
            data.observations[:, step],
            data.actions[:, step],
            data.rewards[:, step],
            data.scalar_obs[:, step],
        )

    def _replay_states(self, data: ReplayBufferData) -> tuple[torch.Tensor, torch.Tensor]:
        """リプレイの窓の現在の状態（最初のステップ）と次の状態（最後のステップ、勾配なし）。"""
        with torch.no_grad():
            next_state = self._state(data, -1)
        return self._state(data, -2), next_state

    def _head_terms(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        weights: torch.Tensor,
    ) -> LossResult:
        """現在の状態 ``state``（勾配あり）と次の状態 ``next_state`` (B, state_dim)、遷移の行動
        (B, action_dim)、報酬と終端 (B, 1)、重要度の重み (B,) から、ヘッドの損失。"""
        with torch.no_grad():
            # TD の目標は割引率ごとに、その割引率の方策の次の行動で作る。最後が主の割引率の
            # 方策の行動で、状態の予測にはそれを使う
            next_gamma_actions = self.policy_head.get_gamma_actions(next_state)
            next_action = next_gamma_actions[:, -1]
            next_q = self.value_head.gamma_values(next_state, next_gamma_actions)
        state_prediction_loss, state_prediction_info, intrinsic_reward = (
            self.prediction_head.compute_loss(state, action, next_state, next_action)
        )
        # 内発報酬は遷移の報酬に足す。TD の目標にだけ入り、観測の文には入らない
        reward = reward + intrinsic_reward.view(-1, 1)
        target_value = self.value_head.compute_target_value(next_q, reward, done)
        critic_loss, critic_info, td_error = self.value_head.compute_critic_loss(
            state, action, target_value, weights
        )
        actor_loss, actor_info = self.policy_head.compute_actor_loss(
            state, action, value_head=self.value_head
        )
        info = {
            f"losses/{key}": value
            for key, value in {**critic_info, **actor_info, **state_prediction_info}.items()
        }
        return LossResult(
            loss=self.critic_loss_weight * critic_loss + actor_loss + state_prediction_loss,
            info=info,
            td_error=td_error,
        )
