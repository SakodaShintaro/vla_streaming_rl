# SPDX-License-Identifier: MIT
from collections.abc import Callable
from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch import nn

from ..replay_buffer import ReplayBufferData
from .interface import (
    EligibilityTraceInfo,
    InferInput,
    InferLossResult,
    InferResult,
    LossResult,
    NetworkInterface,
)
from .modules.temporal_block import GatedDeltaNetStream
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
    next_rnn_state: torch.Tensor
    next_action: torch.Tensor
    next_critic_output: torch.Tensor
    td_error: torch.Tensor


class VLAActorCriticWithActionValue(NetworkInterface):
    """画像と指示の文から行動を出し、エピソードの記憶を再帰の状態に持つ VLA。

    毎ステップ、観測の画像と指示の文をエンコーダ（凍結した VLM）に読ませてトークンにし、
    その後ろに学習するクエリのトークンを ``num_queries`` 個並べる。これをステップの順に
    つないだ1本の因果的な系列を、GatedDeltaNet の幹が読む::

        [ステップ1 の観測][クエリ] [ステップ2 の観測][クエリ] ...

    幹の状態は大きさが一定で、エピソードの間は途中で初期化しない。クエリの位置の出力は、
    それまでの全ステップと今のステップの観測を読んだ後の値で、方策ヘッドと行動価値ヘッドは
    このクエリの位置の出力だけを読む。観測のトークンは状態に書き込み、クエリは状態から行動に
    要る情報を読み出す。

    学習では、バッファに保存した窓の先頭の状態から窓を通して時間を遡って逆伝播する。推論では
    保存した今のステップの状態から、今のステップだけを進める。状態に過去の情報を残させるため、
    学習では観測をステップごと・入力の種類ごとに落とし、クエリの出力から数ステップ先の画像の
    変化を予測させる。

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
        horizon: int,
        critic_loss_weight: float,
        reset_on_episode_end: bool,
        vla_config: DictConfig,
    ) -> None:
        super().__init__()
        # バッファから学習に読む窓の長さ。推論は保存した状態から今のステップだけを進める
        self.seq_len = vla_config.seq_len
        self.infer_len = 1
        self.horizon = horizon
        self.critic_loss_weight = critic_loss_weight
        # エピソードの境目で状態を初期化するか。するなら、窓の途中でエピソードが始まる系列は
        # そこから 0 の状態で読み直す
        self.reset_on_episode_end = reset_on_episode_end
        self.observation_dropout = vla_config.observation_dropout
        self.modality_dropout = vla_config.modality_dropout
        self.forecast_steps = list(vla_config.forecast_steps)
        assert all(0 < step < self.seq_len + horizon for step in self.forecast_steps), (
            self.forecast_steps
        )
        self.forecast_pool = vla_config.forecast_pool
        self.forecast_loss_weight = vla_config.forecast_loss_weight
        device = "cuda"
        action_dim = action_space_shape[0]
        self.encoder = VLMEncoder(
            image_shapes=image_shapes, action_dim=action_dim, vla_config=vla_config, device=device
        )
        token_dim = self.encoder.token_dim
        self.num_queries = vla_config.num_queries
        self.queries = nn.Parameter(torch.randn(self.num_queries, token_dim, device=device) * 0.02)
        self.trunk = GatedDeltaNetStream(
            token_dim, vla_config.trunk_heads, vla_config.trunk_layers
        ).to(device)
        # 残差の大きさはクエリごとに違うので、ヘッドに渡す前に揃える
        self.output_norm = nn.LayerNorm(token_dim).to(device)
        state_dim = self.num_queries * token_dim
        self.policy_head = policy_head_factory(
            state_dim=state_dim, action_dim=action_dim, token_dim=token_dim
        ).to(device)
        self.value_head = value_head_factory(
            in_channels=state_dim, action_dim=action_dim, token_dim=token_dim
        ).to(device)
        # 最後のクエリの出力から、forecast_steps 先の画像の、今の画像からの変化を予測する。
        # 予測する先の数と縮めた格子の位置ごとに、クエリの出力に埋め込みを足して読む
        forecast_tokens_num = self.encoder.images_num * self.forecast_pool**2
        self.forecast_position = nn.Parameter(
            torch.randn(len(self.forecast_steps), forecast_tokens_num, token_dim, device=device)
            * 0.02
        )
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, vla_config.forecast_hidden_dim),
            nn.GELU(),
            nn.Linear(vla_config.forecast_hidden_dim, self.encoder.vlm_hidden_size),
        ).to(device)

    def init_state(self) -> torch.Tensor:
        return torch.zeros(1, *self.trunk.state_shape)

    def stored_image_shape(self) -> tuple[int, ...]:
        """VLM の表現の、画像のトークンの格子と末尾の位置と指示。射影は学習するので、
        その手前を保存する。"""
        return self.encoder.representation_shape()

    def to_stored_image(self, images: list[torch.Tensor], instruction: str) -> torch.Tensor:
        return self.encoder.represent(images, instruction)

    @torch.inference_mode()
    def infer(self, data: InferInput) -> InferResult:
        assert data.s_seq.shape[0] == 1, "Batch size must be 1 for inference"
        episode_rows = torch.ones(data.s_seq.shape[:2], dtype=torch.bool, device=data.s_seq.device)
        queries, rnn_state = self._stream(
            data.s_seq,
            data.a_seq,
            data.r_seq,
            data.scalar_obs_seq,
            data.rnn_state,
            episode_rows,
            0.0,
            0.0,
        )
        state = queries[:, -1].flatten(start_dim=1)
        action = self.policy_head.get_action(state)
        value_report = self.value_head.value_report(self.value_head(state, action).output)
        if data.warmup:
            # バッファを貯めている間は一様乱数の行動を返す。再帰状態はエピソードを
            # 追い続けるので、ネットワーク自体は通しておく
            action = torch.rand_like(action) * 2.0 - 1.0
        return InferResult(
            action=action,
            value_report=value_report,
            rnn_state=rnn_state,
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
            rnn_state=terms.next_rnn_state,
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

    def _stream(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        scalar_obs: torch.Tensor,
        rnn_state: torch.Tensor,
        episode_rows: torch.Tensor,
        observation_dropout: float,
        modality_dropout: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """窓 (B, T, ...) の各ステップのトークンの後ろにクエリを並べ、1本の系列にして状態
        ``rnn_state`` (B, trunk_layers, cache_size) から幹に通す。各ステップのクエリの位置の
        出力 (B, T, num_queries, token_dim) と、窓を読んだ後の状態を返す。

        ``episode_rows`` (B, T) は窓の最後のステップと同じエピソードのステップ。そうでない
        ステップは窓の先頭に並ぶので、その後ろから状態を 0 に戻して読み直す。"""
        tokens = self.encoder(
            observations, actions, rewards, scalar_obs, observation_dropout, modality_dropout
        )
        B, T = tokens.shape[:2]
        stream = torch.cat([tokens, self.queries.expand(B, T, -1, -1)], dim=2)
        step_len = stream.shape[2]
        start = (~episode_rows).sum(dim=1) * step_len
        output, rnn_state = self.trunk(stream.flatten(1, 2), rnn_state, start)
        queries = output.reshape(B, T, step_len, -1)[:, :, -self.num_queries :]
        return self.output_norm(queries), rnn_state

    def _episode_rows(self, dones: torch.Tensor) -> torch.Tensor:
        """窓の終端 (B, T, 1) から、窓の最後のステップと同じエピソードのステップ (B, T)。
        ステップ t の終端は t がエピソードの最後の観測であることを表すので、t から
        最後の1つ手前までに終端がなければ同じエピソード。エピソードの境目で状態を
        初期化しないなら、すべてのステップを読む。"""
        later = dones[:, :-1, 0].flip(1).cumsum(dim=1).flip(1)
        later = torch.cat([later, torch.zeros_like(later[:, :1])], dim=1)
        return (later == 0) | (not self.reset_on_episode_end)

    def _forecast_loss(
        self, last_queries: torch.Tensor, data: ReplayBufferData, episode_rows: torch.Tensor
    ) -> torch.Tensor:
        """窓の先頭からのステップ t の最後のクエリの出力 (B, T, token_dim) から、
        ``forecast_steps`` 先の画像の、ステップ t の画像からの変化を予測する損失。予測する先が
        窓の外にあるか、別のエピソードにあるものは数えない。"""
        targets = self.encoder.image_targets(data.observations, self.forecast_pool)
        window_len = targets.shape[1]
        device = targets.device
        rows = torch.arange(last_queries.shape[1], device=device)
        future = rows[:, None] + torch.tensor(self.forecast_steps, device=device)  # (T, K)
        inside = future < window_len
        future = future.clamp(max=window_len - 1)
        # ステップ t から t + k - 1 までの終端の数
        cumulative_dones = torch.cat(
            [torch.zeros_like(data.dones[:, :1, 0]), data.dones[:, :, 0].cumsum(dim=1)], dim=1
        )
        crosses = (cumulative_dones[:, future] - cumulative_dones[:, rows, None]) > 0
        valid = inside & ~crosses & episode_rows[:, :, None]  # (B, T, K)
        delta = targets[:, future] - targets[:, rows, None]  # (B, T, K, tokens, hidden)
        prediction = self.forecast_head(last_queries[:, :, None, None] + self.forecast_position)
        error = (prediction - delta).pow(2).mean(dim=(-2, -1))
        return (error * valid).sum() / valid.sum().clamp(min=1)

    def _loss_terms(self, data: ReplayBufferData) -> LossTerms:
        """リプレイバッチの損失。現在の状態はチャンクの手前までの窓から、次の状態は
        チャンクの分だけ進めた窓から、それぞれ窓の先頭に保存した状態を始めとして作る。"""

        def window(start: int, stop: int | None) -> tuple:
            return (
                data.observations[:, start:stop],
                data.actions[:, start:stop],
                data.rewards[:, start:stop],
                data.scalar_obs[:, start:stop],
            )

        # ターゲットは観測を落とさない次状態から作る
        with torch.no_grad():
            next_queries, next_rnn_state = self._stream(
                *window(self.horizon, None),
                data.rnn_state[:, self.horizon],
                self._episode_rows(data.dones[:, self.horizon :]),
                0.0,
                0.0,
            )
            next_state = next_queries[:, -1].flatten(start_dim=1)
            next_action = self.policy_head.get_action(next_state)
            next_critic_output = self.value_head(next_state, next_action).output
        target_value = self.value_head.compute_target_value(
            next_critic_output, data.rewards[:, -self.horizon :], data.dones[:, -self.horizon :]
        )

        episode_rows = self._episode_rows(data.dones[:, : -self.horizon])
        queries, _ = self._stream(
            *window(0, -self.horizon),
            data.rnn_state[:, 0],
            episode_rows,
            self.observation_dropout,
            self.modality_dropout,
        )
        state = queries[:, -1].flatten(start_dim=1)
        action_chunk = data.actions[:, -self.horizon :]  # (B, horizon, action_dim)
        critic_loss, critic_info, td_error = self.value_head.compute_critic_loss(
            state, action_chunk, target_value, data.weights
        )
        actor_loss, actor_info = self.policy_head.compute_actor_loss(
            state, action_chunk, value_head=self.value_head
        )
        forecast_loss = self._forecast_loss(queries[:, :, -1], data, episode_rows)
        info = {
            f"losses/{key}": value
            for key, value in {
                **critic_info,
                **actor_info,
                "forecast_loss": forecast_loss.item(),
            }.items()
        }
        # 将来の画像の予測の損失は actor 側に入れる
        actor_side_loss = actor_loss + self.forecast_loss_weight * forecast_loss
        return LossTerms(
            total_loss=self.critic_loss_weight * critic_loss + actor_side_loss,
            actor_loss=actor_side_loss,
            info=info,
            state=state,
            action_chunk=action_chunk,
            next_state=next_state,
            next_rnn_state=next_rnn_state,
            next_action=next_action,
            next_critic_output=next_critic_output,
            td_error=td_error,
        )
