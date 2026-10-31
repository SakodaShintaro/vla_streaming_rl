# SPDX-License-Identifier: MIT
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig

from ..replay_buffer import ReplayBufferData
from .interface import (
    EligibilityTraceInfo,
    InferInput,
    InferLossResult,
    InferResult,
    LossResult,
    NetworkInterface,
)
from .modules.prediction_head import LatentTDPredictionHead
from .modules.value_head import DistributionalValueHead
from .modules.vlm_encoder import VLMEncoder


@dataclass
class LossTerms:
    """リプレイバッチ1つ分の損失と、それを計算する途中で得た次状態での推論結果。"""

    total_loss: torch.Tensor
    actor_loss: torch.Tensor
    state_prediction_loss: torch.Tensor
    info: dict
    state: torch.Tensor
    action_chunk: torch.Tensor
    next_state: torch.Tensor
    next_action: torch.Tensor
    next_critic_output: torch.Tensor
    td_error: torch.Tensor


class VLAActorCriticWithActionValue(NetworkInterface):
    """1枚の画像と指示の文から行動を出す VLA。

    毎ステップ、観測の画像とスカラーの観測をエンコーダ（凍結した VLM）に読ませて状態を作る。
    VLM は指示の文から始まるエピソードの流れを読み続けるので、過去の観測は VLM が自分の
    キャッシュに持ち越した分として反映される。方策ヘッドと行動価値ヘッドはその状態を読む。

    VLM は凍結して使うので、表現は観測を受け取ったときに一度だけ計算してバッファに保存する。
    推論も学習もそれを読み戻し、VLM を通さない。
    """

    def __init__(
        self,
        *,
        image_shapes: list[tuple[int, ...]],
        action_space_shape: tuple[int],
        value_head_factory: Callable[[int, int], DistributionalValueHead],
        policy_head_factory,
        prediction_head_factory: Callable[..., LatentTDPredictionHead],
        horizon: int,
        critic_loss_weight: float,
        vla_config: DictConfig,
    ) -> None:
        super().__init__()
        # 表現はバッファに保存するので、バッファから読む窓は今のステップだけでよい
        self.seq_len = 1
        self.horizon = horizon
        self.critic_loss_weight = critic_loss_weight
        device = "cuda"
        action_dim = action_space_shape[0]
        self.encoder = VLMEncoder(
            image_shapes=image_shapes, action_dim=action_dim, vla_config=vla_config, device=device
        )
        state_dim = self.encoder.output_dim
        self.policy_head = policy_head_factory(
            state_dim=state_dim, action_dim=action_dim, token_dim=self.encoder.token_dim
        ).to(device)
        self.value_head = value_head_factory(
            in_channels=state_dim, action_dim=action_dim, token_dim=self.encoder.token_dim
        ).to(device)
        self.use_state_predictor = vla_config.use_state_predictor
        # 予測を切るときは予測器そのものを作らない。作ると使わない重みが optimizer に入り、
        # weight decay だけを受け続ける
        self.prediction_head = (
            prediction_head_factory(state_dim=state_dim, action_dim=action_dim).to(device)
            if self.use_state_predictor
            else None
        )
        self._dummy_state = torch.zeros(1, 1, 1)

    def init_state(self) -> torch.Tensor:
        return self._dummy_state.clone()

    def stored_image_shape(self) -> tuple[int, ...]:
        """VLM の表現の、画像のトークンの格子。射影は学習するので、その手前を
        保存する。"""
        return self.encoder.representation_shape()

    def to_stored_image(
        self,
        images: list[torch.Tensor],
        scalar_obs: np.ndarray,
        instruction: str,
        reset_state: bool,
    ) -> torch.Tensor:
        return self.encoder.represent(images, scalar_obs, instruction, reset_state)

    @torch.inference_mode()
    def infer(self, data: InferInput) -> InferResult:
        assert data.s_seq.shape[0] == 1, "Batch size must be 1 for inference"
        state = self._state(data.s_seq, data.a_seq, data.r_seq, data.scalar_obs_seq, -1)
        action = self.policy_head.get_action(state)
        value_report = self.value_head.value_report(self.value_head(state, action).output)
        if data.warmup:
            # バッファを貯めている間は一様乱数の行動を返す
            action = torch.rand_like(action) * 2.0 - 1.0
        return InferResult(
            action=action,
            value_report=value_report,
            rnn_state=self._dummy_state.clone(),
            features=state,
        )

    def compute_loss(self, data: ReplayBufferData) -> LossResult:
        terms = self._loss_terms(data)
        return LossResult(loss=terms.total_loss, info=terms.info, td_error=terms.td_error)

    def infer_and_compute_loss(self, data: ReplayBufferData) -> InferLossResult:
        terms = self._loss_terms(data)
        # -Q(s,a) for eligibility trace backward (detached from encoder)
        neg_value_detached = -self.value_head.scalar_value(
            terms.state.detach(), terms.action_chunk.detach()
        ).mean()
        infer_result = InferResult(
            action=terms.next_action,
            value_report=self.value_head.value_report(terms.next_critic_output),
            rnn_state=self._dummy_state.clone(),
            features=terms.next_state,
        )
        return InferLossResult(
            infer_result=infer_result,
            loss_result=LossResult(loss=terms.total_loss, info=terms.info, td_error=terms.td_error),
            et_info=EligibilityTraceInfo(
                actor_loss=terms.actor_loss,
                state_prediction_loss=terms.state_prediction_loss,
                neg_value=neg_value_detached,
                delta=terms.info["losses/delta"],
            ),
        )

    ####################
    # Internal methods #
    ####################

    def _state(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        scalar_obs: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """窓 (B, T, ...) の ``step`` 番目のステップの、保存した表現・前の行動・報酬・スカラーの
        観測から状態を作る。"""
        return self.encoder(
            observations[:, step], actions[:, step], rewards[:, step], scalar_obs[:, step]
        )

    def _prediction_loss(
        self,
        state: torch.Tensor,
        data: ReplayBufferData,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """将来の状態の予測の損失。予測を切っているときは予測器がないので 0。"""
        if not self.use_state_predictor:
            return torch.zeros((), device=state.device), {}
        return self.prediction_head.compute_loss(
            state,
            data.actions[:, -self.horizon :],
            next_state,
            next_action,
            data.dones[:, -self.horizon :],
        )

    def _loss_terms(self, data: ReplayBufferData) -> LossTerms:
        """リプレイバッチの損失。現在の状態はチャンクの手前のステップから、次の状態は最後の
        ステップから作る。"""
        window = (data.observations, data.actions, data.rewards, data.scalar_obs)
        with torch.no_grad():
            next_state = self._state(*window, -1)
            next_action = self.policy_head.get_action(next_state)
            next_critic_output = self.value_head(next_state, next_action).output
        target_value = self.value_head.compute_target_value(
            next_critic_output, data.rewards[:, -self.horizon :], data.dones[:, -self.horizon :]
        )
        state = self._state(*window, -self.horizon - 1)
        action_chunk = data.actions[:, -self.horizon :]  # (B, horizon, action_dim)
        critic_loss, critic_info, td_error = self.value_head.compute_critic_loss(
            state, action_chunk, target_value, data.weights
        )
        actor_loss, actor_info = self.policy_head.compute_actor_loss(
            state, action_chunk, value_head=self.value_head
        )
        state_prediction_loss, state_prediction_info = self._prediction_loss(
            state, data, next_state, next_action
        )
        info = {
            f"losses/{key}": value
            for key, value in {**critic_info, **actor_info, **state_prediction_info}.items()
        }
        return LossTerms(
            total_loss=self.critic_loss_weight * critic_loss + actor_loss + state_prediction_loss,
            actor_loss=actor_loss,
            state_prediction_loss=state_prediction_loss,
            info=info,
            state=state,
            action_chunk=action_chunk,
            next_state=next_state,
            next_action=next_action,
            next_critic_output=next_critic_output,
            td_error=td_error,
        )
