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
    """リプレイバッチ1つ分の損失と、それを計算する途中で得た状態。"""

    total_loss: torch.Tensor
    actor_loss: torch.Tensor
    info: dict
    state: torch.Tensor
    action_chunk: torch.Tensor
    td_error: torch.Tensor


class VLAActorCriticWithActionValue(NetworkInterface):
    """1枚の画像と指示の文から行動を出す VLA。

    毎ステップ、観測の画像と指示の文をエンコーダ（凍結した VLM）に読ませて状態を作る。過去の
    観測は読まない。方策ヘッドと行動価値ヘッドはその状態を読む。

    VLM は凍結して使うので、表現は毎ステップ一度だけ計算してバッファに保存する（行動は
    ``horizon`` ステップのチャンクごとに決めるが、表現はチャンクの途中のステップでも保存する）。
    学習はそれを読み戻し、VLM を通さない。
    """

    def __init__(
        self,
        *,
        images_num: int,
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
            images_num=images_num, action_dim=action_dim, vla_config=vla_config, device=device
        )
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
        """何も保存しない。画像は推論のときに ``InferInput.frames`` で受け取る。"""
        return (0,)

    def to_stored_image(self, images: list[torch.Tensor]) -> torch.Tensor:
        del images
        return torch.zeros(0)

    def stored_state_shape(self) -> tuple[int, ...]:
        """VLM の表現の、画像をプーリングした格子と末尾の位置。射影は学習するので、その手前を
        保存する。"""
        return self.encoder.representation_shape()

    @torch.inference_mode()
    def infer(self, data: InferInput) -> InferResult:
        return self._act(
            data.frames, data.instruction, data.a_seq, data.r_seq, data.scalar_obs_seq, data.warmup
        )

    def compute_loss(self, data: ReplayBufferData) -> LossResult:
        terms = self._loss_terms(data, data.states[:, -1])
        return LossResult(loss=terms.total_loss, info=terms.info, td_error=terms.td_error)

    def infer_and_compute_loss(
        self,
        data: ReplayBufferData,
        frames: list[torch.Tensor],
        instruction: str,
    ) -> InferLossResult:
        # 最後のステップはまだ行動を決めていないので、ここで表現を作る
        with torch.inference_mode():
            infer_result = self._act(
                frames, instruction, data.actions, data.rewards, data.scalar_obs, False
            )
        terms = self._loss_terms(data, infer_result.stored_state.unsqueeze(0).clone())
        # -Q(s,a) for eligibility trace backward (detached from encoder)
        neg_value_detached = -self.value_head.scalar_value(
            terms.state.detach(), terms.action_chunk.detach()
        ).mean()
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

    def _act(
        self,
        frames: list[torch.Tensor],
        instruction: str,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        scalar_obs: torch.Tensor,
        warmup: bool,
    ) -> InferResult:
        """観測の画像と指示から表現を作り、最後のステップの前の行動 (1, T, action_dim)・報酬
        (1, T, 1)・スカラーの観測 (1, T, SCALAR_OBS_DIM) の最後と合わせて行動を決める。"""
        assert scalar_obs.shape[0] == 1, "Batch size must be 1 for inference"
        representation = self.encoder.represent(frames, instruction)
        state = self.encoder(
            representation.unsqueeze(0), actions[:, -1], rewards[:, -1], scalar_obs[:, -1]
        )
        action = self.policy_head.get_action(state)
        if warmup:
            # バッファを貯めている間は一様乱数の行動を返す
            action = torch.rand_like(action) * 2.0 - 1.0
        return InferResult(
            action=action,
            value_report=self.value_head.value_report(self.value_head(state, action).output),
            rnn_state=self._dummy_state.clone(),
            features=state,
            stored_state=representation,
        )

    def _loss_terms(self, data: ReplayBufferData, next_representation: torch.Tensor) -> LossTerms:
        """リプレイバッチの損失。現在の状態はチャンクの手前のステップに保存した表現から、
        次の状態は ``next_representation``（最後のステップのもの）から作る。"""
        with torch.no_grad():
            next_state = self.encoder(
                next_representation,
                data.actions[:, -1],
                data.rewards[:, -1],
                data.scalar_obs[:, -1],
            )
            next_action = self.policy_head.get_action(next_state)
            next_output = self.value_head(next_state, next_action).output
        target_value = self.value_head.compute_target_value(
            next_output, data.rewards[:, -self.horizon :], data.dones[:, -self.horizon :]
        )
        current = -self.horizon - 1
        state = self.encoder(
            data.states[:, current],
            data.actions[:, current],
            data.rewards[:, current],
            data.scalar_obs[:, current],
        )
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
            td_error=td_error,
        )
