# SPDX-License-Identifier: MIT
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from omegaconf import DictConfig

from vla_streaming_rl.networks.interface import (
    EligibilityTraceInfo,
    InferInput,
    InferLossResult,
    InferResult,
    LossResult,
    NetworkInterface,
)
from vla_streaming_rl.networks.modules.backbone import SpatialTemporalEncoder
from vla_streaming_rl.networks.modules.image_processor import ImageProcessor
from vla_streaming_rl.networks.modules.value_head import DistributionalValueHead
from vla_streaming_rl.networks.modules.vlm_encoder import STATE_SCALAR_INDICES
from vla_streaming_rl.replay_buffer import ReplayBufferData


@dataclass
class LossTerms:
    """リプレイバッチ1つ分の損失と、それを計算する途中で得た次状態での推論結果。"""

    total_loss: torch.Tensor
    actor_loss: torch.Tensor
    info: dict
    state: torch.Tensor
    action_chunk: torch.Tensor
    next_state: torch.Tensor
    next_rnn_state: torch.Tensor
    next_action: torch.Tensor
    next_critic_output: torch.Tensor
    td_error: torch.Tensor


class ActorCriticWithActionValue(NetworkInterface):
    def __init__(
        self,
        *,
        image_shapes: list[tuple[int, ...]],
        action_space_shape: tuple[int],
        value_head_factory: Callable[[int, int], DistributionalValueHead],
        critic_loss_weight: float,
        prediction_head_factory,
        actor_critic_config: DictConfig,
        horizon: int,
        policy_head_factory,
    ) -> None:
        super().__init__()
        self.seq_len = actor_critic_config.seq_len
        self.critic_loss_weight = critic_loss_weight

        self.action_dim = action_space_shape[0]

        self.image_processor = ImageProcessor(image_shapes)
        token_dim = actor_critic_config.token_dim

        assert 0.0 <= actor_critic_config.token_dropout < 1.0, actor_critic_config.token_dropout
        self.token_dropout = actor_critic_config.token_dropout
        # 系列ごとに 0 からこの数までの一様なステップ数を引き、最新の方からその数のステップの
        # 観測（画像・報酬・スカラー）を落とす。観測が遅れて届く状況でも行動と価値を学ばせる
        assert actor_critic_config.observation_drop_steps_max >= 0, (
            actor_critic_config.observation_drop_steps_max
        )
        self.observation_drop_steps_max = actor_critic_config.observation_drop_steps_max
        # スカラーの観測は VLMEncoder と同じく、global_step を除いてそれぞれ固定の sin/cos の特徴で
        # トークンにする。周期を scalar_max_period から scalar_min_period まで対数等間隔にとり、
        # sin と cos でトークンの幅にする
        self.scalar_obs_dim = len(STATE_SCALAR_INDICES)
        assert token_dim % 2 == 0, token_dim
        periods = torch.logspace(
            math.log10(actor_critic_config.scalar_max_period),
            math.log10(actor_critic_config.scalar_min_period),
            token_dim // 2,
        )
        self.register_buffer("scalar_frequencies", 2.0 * math.pi / periods)

        self.encoder = SpatialTemporalEncoder(
            image_features_shapes=self.image_processor.output_shapes,
            image_latent_dim=token_dim,
            seq_len=self.seq_len,
            n_layer=actor_critic_config.encoder_block_num,
            action_dim=self.action_dim,
            scalar_obs_dim=self.scalar_obs_dim,
            temporal_model_type=actor_critic_config.temporal_model_type,
            layer_scale_init=actor_critic_config.layer_scale_init,
        )

        self.horizon = horizon
        self.policy_head = policy_head_factory(
            state_dim=self.encoder.output_dim,
            action_dim=self.action_dim,
            token_dim=token_dim,
        )

        self.value_head = value_head_factory(
            in_channels=self.encoder.output_dim,
            action_dim=self.action_dim,
            token_dim=token_dim,
        )
        self.disable_state_predictor = actor_critic_config.disable_state_predictor
        # 次状態の予測を切るときは予測器そのものを作らない。作ると使わない重みが
        # optimizer に入り、weight decay だけを受け続ける
        self.prediction_head = (
            None
            if self.disable_state_predictor
            else prediction_head_factory(
                image_latent_dim=token_dim,
                state_dim=self.encoder.output_dim,
                action_dim=self.action_dim,
            )
        )

        # 1ステップのトークンのうち観測にあたる位置。行動と register は含めない
        observation_token = torch.cat(
            [
                torch.ones(self.encoder.image_tokens_num, dtype=torch.bool),  # 画像
                torch.zeros(self.action_dim, dtype=torch.bool),  # 前の行動
                torch.ones(1 + self.scalar_obs_dim, dtype=torch.bool),  # 報酬とスカラー
                torch.zeros(1, dtype=torch.bool),  # register
            ]
        )
        self.register_buffer("observation_token", observation_token, persistent=False)

    def init_state(self) -> torch.Tensor:
        return self.encoder.init_state()

    def stored_image_shape(self) -> tuple[int, ...]:
        """The frozen encoder's output for the frames, not the frames: each image's
        (C, H', W') flattened and laid one after another."""
        return (sum(self.encoder.image_features_sizes),)

    def to_stored_image(self, images: list[torch.Tensor], instruction: str) -> torch.Tensor:
        # 指示の文は読まない
        del instruction
        return torch.cat(
            [self.image_processor.encode(image.unsqueeze(0)).flatten() for image in images]
        )

    def _token_keep(self, batch_size: int, steps: int, device: torch.device) -> torch.Tensor:
        """Which tokens of a batch's windows the encoder gets to read, the rest
        being the share ``token_dropout`` takes away: every token alike, whatever
        it carries, so no one kind of input can be leaned on alone. Only on the
        learning path."""
        shape = (batch_size, steps, self.encoder.space_len)
        return torch.rand(shape, device=device) >= self.token_dropout

    def _window(self, data: ReplayBufferData, start, stop, drop_steps_max: int) -> tuple:
        """The ``(image, action, reward, rnn_state, scalar_obs, token_keep)`` the
        encoder reads, sliced out of a replay batch
        over ``[start, stop)`` steps. 最新の方から 0 から ``drop_steps_max`` までの
        一様なステップ数の観測を落とす。"""
        observations = data.observations[:, start:stop]
        batch_size = observations.shape[0]
        token_keep = self._token_keep(batch_size, observations.shape[1], observations.device)
        steps = observations.shape[1]
        drop_steps = torch.randint(
            0, drop_steps_max + 1, (batch_size, 1), device=observations.device
        )
        dropped = torch.arange(steps, device=observations.device) >= steps - drop_steps
        token_keep &= ~(dropped.unsqueeze(-1) & self.observation_token)
        return (
            observations,
            data.actions[:, start:stop],
            data.rewards[:, start:stop],
            data.rnn_state[:, start],
            self._scalar_tokens(data.scalar_obs[:, start:stop]),
            token_keep,
        )

    def _scalar_tokens(self, scalar_obs: torch.Tensor) -> torch.Tensor:
        """スカラーの観測 (..., SCALAR_OBS_DIM) を、global_step を除いてそれぞれ sin/cos の特徴の
        トークン (..., scalar_obs_dim, hidden_image_dim) にする。"""
        angles = scalar_obs[..., STATE_SCALAR_INDICES].unsqueeze(-1) * self.scalar_frequencies
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    @torch.no_grad()
    def infer(self, data: InferInput) -> InferResult:
        assert data.s_seq.shape[0] == 1, "Batch size must be 1 for inference"

        x, rnn_state = self.encoder(
            data.s_seq,
            data.a_seq,
            data.r_seq,
            data.rnn_state,
            self._scalar_tokens(data.scalar_obs_seq),
            torch.ones(
                (1, data.s_seq.shape[1], self.encoder.space_len),
                dtype=torch.bool,
                device=data.s_seq.device,
            ),
        )  # (B, state_dim)

        # Get action chunk from policy_head
        action = self.policy_head.get_action(x)  # (B, horizon, action_dim)

        # Get action-value from value_head
        q_out = self.value_head(x, action)
        value_report = self.value_head.value_report(q_out.output)

        if data.warmup:
            # バッファを貯めている間は一様乱数の行動を返す。再帰状態はエピソードを
            # 追い続けるので、ネットワーク自体は通しておく
            action = torch.rand_like(action) * 2.0 - 1.0

        return InferResult(
            action=action,
            value_report=value_report,
            rnn_state=rnn_state,
            features=x,
        )

    def _prediction_loss(
        self,
        state: torch.Tensor,
        data: ReplayBufferData,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """将来の状態の予測の損失。予測を切っているときは予測器がないので 0。"""
        if self.disable_state_predictor:
            return torch.zeros((), device=state.device), {"seq_loss": 0.0}
        # ブートストラップは次の状態から行うので、1ステップ先の観測と次の状態が揃う
        # チャンク1ステップのときだけ使える
        assert self.horizon == 1, f"the successor prediction needs horizon 1; got {self.horizon}"
        with torch.no_grad():
            next_image_tokens = self.encoder.image_tokens(data.observations[:, -self.horizon])
        return self.prediction_head.compute_loss(
            state,
            data.actions[:, -self.horizon],
            next_image_tokens,
            data.rewards[:, -self.horizon],
            next_state,
            next_action[:, 0],
            data.dones[:, -self.horizon],
        )

    def _state_grad_info(
        self, state: torch.Tensor, critic_loss: torch.Tensor, seq_loss: torch.Tensor
    ) -> dict:
        """エンコーダの出力 ``state`` に、critic の損失と将来の状態の予測の損失がそれぞれ
        どれだけの大きさの勾配を流すか。ヘッドの分だけ逆伝播するので、エンコーダの中は
        通らない。予測を切っているときは critic の分だけ。"""
        critic_grad = torch.autograd.grad(
            self.critic_loss_weight * critic_loss, state, retain_graph=True
        )[0]
        info = {"grad/state_critic": critic_grad.norm().item()}
        if self.disable_state_predictor:
            return info
        seq_grad = torch.autograd.grad(seq_loss, state, retain_graph=True)[0]
        info["grad/state_seq"] = seq_grad.norm().item()
        info["grad/state_seq_over_critic"] = info["grad/state_seq"] / max(
            info["grad/state_critic"], 1e-12
        )
        return info

    def compute_loss(self, data: ReplayBufferData) -> LossResult:
        terms = self._loss_terms(data)
        return LossResult(loss=terms.total_loss, info=terms.info, td_error=terms.td_error)

    def infer_and_compute_loss(self, data: ReplayBufferData) -> InferLossResult:
        """Combined inference and loss computation."""
        terms = self._loss_terms(data)
        # -Q(s,a) for eligibility trace backward (detached from encoder)
        neg_value_detached = -self.value_head.scalar_value(
            terms.state.detach(), terms.action_chunk.detach()
        ).mean()
        infer_result = InferResult(
            action=terms.next_action,
            value_report=self.value_head.value_report(terms.next_critic_output),
            rnn_state=terms.next_rnn_state,
            features=terms.next_state,
        )
        return InferLossResult(
            infer_result=infer_result,
            loss_result=LossResult(loss=terms.total_loss, info=terms.info, td_error=terms.td_error),
            et_info=EligibilityTraceInfo(
                # Actor-only loss (no critic component)
                actor_entropy_loss=terms.actor_loss,
                neg_value=neg_value_detached,
                delta=terms.info["losses/delta"],
            ),
        )

    def _loss_terms(self, data: ReplayBufferData) -> LossTerms:
        """リプレイバッチの損失。現在の状態はチャンクの手前までの窓から、次の状態は
        チャンクの分だけ進めた窓から作る。"""
        # Bootstrap value: Q(s', μ(s')) on the next-state window, no grad.
        # ターゲットは観測を落とさない次状態から作る
        with torch.no_grad():
            next_state, next_rnn_state = self.encoder(*self._window(data, self.horizon, None, 0))
            next_action = self.policy_head.get_action(next_state)
            next_critic_output = self.value_head(next_state, next_action).output
        target_value = self.value_head.compute_target_value(
            next_critic_output, data.rewards[:, -self.horizon :], data.dones[:, -self.horizon :]
        )

        state, _ = self.encoder(
            *self._window(data, 0, -self.horizon, self.observation_drop_steps_max)
        )
        action_chunk = data.actions[:, -self.horizon :]  # (B, horizon, action_dim)
        critic_loss, critic_info, td_error = self.value_head.compute_critic_loss(
            state, action_chunk, target_value, data.weights
        )
        actor_loss, actor_info = self.policy_head.compute_actor_loss(
            state, action_chunk, value_head=self.value_head
        )
        seq_loss, seq_info = self._prediction_loss(state, data, next_state, next_action)
        info = {
            f"losses/{key}": value
            for key, value in {**critic_info, **actor_info, **seq_info}.items()
        }
        info.update(self._state_grad_info(state, critic_loss, seq_loss))
        return LossTerms(
            total_loss=self.critic_loss_weight * critic_loss + actor_loss + seq_loss,
            # 将来の状態の予測の損失は actor 側に入れる
            actor_loss=actor_loss + seq_loss,
            info=info,
            state=state,
            action_chunk=action_chunk,
            next_state=next_state,
            next_rnn_state=next_rnn_state,
            next_action=next_action,
            next_critic_output=next_critic_output,
            td_error=td_error,
        )
