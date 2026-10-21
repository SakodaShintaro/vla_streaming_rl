# SPDX-License-Identifier: MIT
from collections.abc import Callable
from dataclasses import dataclass

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
from .modules.value_head import DistributionalValueHead
from .modules.vlm_encoder import VLMEncoder


@dataclass
class LossTerms:
    """リプレイバッチ1つ分の損失と、それを計算する途中で得た次状態での推論結果。"""

    total_loss: torch.Tensor
    actor_loss: torch.Tensor
    info: dict
    state: torch.Tensor
    action_chunk: torch.Tensor
    next_state: torch.Tensor
    next_action: torch.Tensor
    next_critic_output: torch.Tensor
    td_error: torch.Tensor


class VLAActorCriticWithActionValue(NetworkInterface):
    """1枚の画像と指示の文から行動を出す VLA。

    毎ステップ、観測の画像と指示の文をエンコーダ（VLM）に読ませて状態を作る。過去の
    観測は読まない。方策ヘッドと行動価値ヘッドはその状態を読む。

    VLM の ViT を学習するので、バッファには画像そのものを保存し、推論でも学習でも読み戻した
    画像を VLM に通して状態を作る。
    """

    def __init__(
        self,
        *,
        image_shapes: list[tuple[int, ...]],
        action_space_shape: tuple[int],
        value_head_factory: Callable[[int, int], DistributionalValueHead],
        policy_head_factory,
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
        # 指示の文は学習の間ずっと変わらない。バッファには画像だけを保存するので、最初に
        # 受け取った文を覚えておき、学習で画像を VLM に通すときにも同じ文を読ませる
        self.instruction: str | None = None
        state_dim = self.encoder.output_dim
        self.policy_head = policy_head_factory(state_dim=state_dim, action_dim=action_dim).to(
            device
        )
        self.value_head = value_head_factory(in_channels=state_dim, action_dim=action_dim).to(
            device
        )
        self._dummy_state = torch.zeros(1, 1, 1)

    def init_state(self) -> torch.Tensor:
        return self._dummy_state.clone()

    def stored_image_shape(self) -> tuple[int, ...]:
        """観測の画像を並べたもの (images_num, 3, H, W)。ViT を学習するので、表現ではなく
        画像を保存する。"""
        shapes = set(self.encoder.image_shapes)
        assert len(shapes) == 1, f"images of one step must share a shape: {shapes}"
        return (self.encoder.images_num, *shapes.pop())

    def to_stored_image(self, images: list[torch.Tensor], instruction: str) -> torch.Tensor:
        if self.instruction is None:
            self.instruction = instruction
        assert instruction == self.instruction, "the instruction must not change"
        return torch.stack(images)

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
                actor_entropy_loss=terms.actor_loss,
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
        """窓 (B, T, ...) の ``step`` 番目のステップの、保存した画像を VLM に通した表現と、
        前の行動・報酬・スカラーの観測から状態を作る。"""
        representation = self.encoder.represent(observations[:, step], self.instruction)
        return self.encoder(representation, actions[:, step], rewards[:, step], scalar_obs[:, step])

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
        info = {f"losses/{key}": value for key, value in {**critic_info, **actor_info}.items()}
        return LossTerms(
            total_loss=self.critic_loss_weight * critic_loss + actor_loss,
            actor_loss=actor_loss,
            info=info,
            state=state,
            action_chunk=action_chunk,
            next_state=next_state,
            next_action=next_action,
            next_critic_output=next_critic_output,
            td_error=td_error,
        )
