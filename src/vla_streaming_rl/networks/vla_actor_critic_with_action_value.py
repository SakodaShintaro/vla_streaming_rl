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
    next_memory: torch.Tensor


class VLAActorCriticWithActionValue(NetworkInterface):
    """1枚の画像と指示の文から行動を出す VLA。

    毎ステップ、観測の画像とスカラーの観測をエンコーダ（凍結した VLM）に読ませて状態を作る。
    VLM は指示の文から始まるエピソードの流れを読み続けるので、過去の観測は VLM が自分の
    キャッシュに持ち越した分として反映される。方策ヘッドと行動価値ヘッドはその状態を読む。

    VLM は凍結して使うので、表現は観測を受け取ったときに一度だけ計算してバッファに保存する。
    推論も学習もそれを読み戻し、VLM を通さない。

    エンコーダに GDN の記憶の層があれば、その再帰状態をエージェントの再帰の状態として持ち回らせる
    （バッファの各行には、そのステップを読む前の状態が入る）。lora_rtrl の LoRA は streaming の
    ``infer_and_compute_loss`` で RTRL で学習し、トレースは学習しない ``infer`` で捨てる
    （streaming ではエピソードの境目でだけ呼ばれる）。off-policy の ``compute_loss`` は保存した
    状態から1ステップ分の勾配だけを使う。
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
        # 記憶の層の状態は推論のたびに1ステップ進むので、行動のチャンクは1ステップに限る
        assert vla_config.gdn_memory == "none" or horizon == 1, horizon

    def init_state(self) -> torch.Tensor:
        return self.encoder.init_memory()

    def stored_image_shape(self) -> tuple[int, ...]:
        """VLM の表現の、画像のトークンの格子。射影は学習するので、その手前を
        保存する。"""
        return self.encoder.representation_shape()

    def to_stored_image(
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
        self.encoder.reset_memory_trace()
        state, next_memory = self._state(
            data.s_seq, data.a_seq, data.r_seq, data.scalar_obs_seq, data.rnn_state, -1, False
        )
        action = self.policy_head.get_action(state)
        value_report = self.value_head.value_report(self.value_head(state, action).output)
        if data.warmup:
            # バッファを貯めている間は一様乱数の行動を返す
            action = torch.rand_like(action) * 2.0 - 1.0
        return InferResult(
            action=action,
            value_report=value_report,
            rnn_state=next_memory,
            features=state,
        )

    def compute_loss(self, data: ReplayBufferData) -> LossResult:
        terms = self._loss_terms(data, False)
        return LossResult(loss=terms.total_loss, info=terms.info, td_error=terms.td_error)

    def infer_and_compute_loss(self, data: ReplayBufferData) -> InferLossResult:
        terms = self._loss_terms(data, True)
        # -Q(s,a) for the eligibility trace. Its gradient reaches the encoder too,
        # which learns from the critic as in the off-policy mode
        neg_value = -self.value_head.scalar_value(terms.state, terms.action_chunk.detach()).mean()
        infer_result = InferResult(
            action=terms.next_action,
            value_report=self.value_head.value_report(terms.next_critic_output),
            rnn_state=terms.next_memory,
            features=terms.next_state,
        )
        return InferLossResult(
            infer_result=infer_result,
            loss_result=LossResult(loss=terms.total_loss, info=terms.info, td_error=terms.td_error),
            et_info=EligibilityTraceInfo(
                actor_loss=terms.actor_loss,
                state_prediction_loss=terms.state_prediction_loss,
                neg_value=neg_value,
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
        memory: torch.Tensor,
        step: int,
        advance_trace: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """窓 (B, T, ...) の ``step`` 番目のステップの、保存した表現・前の行動・報酬・スカラーの
        観測と、それを読む前の再帰の状態 ``memory`` から、状態と次の再帰の状態を作る。"""
        return self.encoder(
            observations[:, step],
            actions[:, step],
            rewards[:, step],
            scalar_obs[:, step],
            memory,
            advance_trace,
        )

    def _prediction_loss(
        self,
        state: torch.Tensor,
        data: ReplayBufferData,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """将来の状態の予測の損失と、遷移ごとの内発報酬 (B,)。予測を切っているときは予測器が
        ないので、どちらも 0。"""
        if not self.use_state_predictor:
            return (
                torch.zeros((), device=state.device),
                {},
                torch.zeros(state.shape[0], device=state.device),
            )
        return self.prediction_head.compute_loss(
            state,
            data.actions[:, -self.horizon :],
            next_state,
            next_action,
        )

    def _loss_terms(self, data: ReplayBufferData, advance_trace: bool) -> LossTerms:
        """リプレイバッチの損失。現在の状態はチャンクの手前のステップから、次の状態は最後の
        ステップから作る。``advance_trace`` なら、現在の状態を作るときに記憶の層の RTRL の
        トレースを1ステップ進める。"""
        window = (data.observations, data.actions, data.rewards, data.scalar_obs)
        with torch.no_grad():
            next_state, next_memory = self._state(*window, data.rnn_state[:, -1], -1, False)
            # TD の目標は割引率ごとに、その割引率の方策の次の行動で作る。最後が主の割引率の
            # 方策の行動で、推論の結果と状態の予測にはそれを使う
            next_gamma_actions = self.policy_head.get_gamma_actions(next_state)
            next_action = next_gamma_actions[:, -1]
            next_critic_output = self.value_head(next_state, next_action).output
            next_q = self.value_head.gamma_values(next_state, next_gamma_actions)
        state, _ = self._state(
            *window, data.rnn_state[:, -self.horizon - 1], -self.horizon - 1, advance_trace
        )
        state_prediction_loss, state_prediction_info, intrinsic_reward = self._prediction_loss(
            state, data, next_state, next_action
        )
        # 内発報酬はチャンクの最初のステップの報酬に足す。TD の目標にだけ入り、観測の文には入らない
        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_rewards = torch.cat(
            [
                chunk_rewards[:, :1] + intrinsic_reward.view(-1, 1, 1),
                chunk_rewards[:, 1:],
            ],
            dim=1,
        )
        target_value = self.value_head.compute_target_value(
            next_q, chunk_rewards, data.dones[:, -self.horizon :]
        )
        action_chunk = data.actions[:, -self.horizon :]  # (B, horizon, action_dim)
        critic_loss, critic_info, td_error = self.value_head.compute_critic_loss(
            state, action_chunk, target_value, data.weights
        )
        actor_loss, actor_info = self.policy_head.compute_actor_loss(
            state, action_chunk, value_head=self.value_head
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
            next_memory=next_memory,
        )
