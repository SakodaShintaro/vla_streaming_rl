# SPDX-License-Identifier: MIT
from collections.abc import Callable
from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch import nn

from ..replay_buffer import ReplayBufferData
from .modules.prediction_head import LatentTDPredictionHead
from .modules.siglip_encoder import SiglipEncoder
from .modules.value_head import DistributionalValueHead


@dataclass
class SiglipInference:
    """``infer`` の出力。"""

    action: torch.Tensor  # (1, action_dim)
    value_report: dict[str, float]  # value head diagnostics (incl. "value")
    state: torch.Tensor  # (1, state_dim) ヘッドが読んだ状態


@dataclass
class SiglipLoss:
    """``compute_replay_loss`` と ``compute_replay_and_streaming_loss`` の出力。``info`` は ``名前 -> 値`` の記録で、そのまま wandb に
    出す。"""

    loss: torch.Tensor
    info: dict  # scalar name -> float
    td_error: torch.Tensor  # (B,) 窓ごとの TD 誤差。リプレイの優先度の更新に使う
    # GDN の層があるとき、リプレイの窓の 2 行目以降を読む前の再帰の状態を今の重みで計算し直した
    # もの (B, window_len - 1, gdn_state_size)。バッファの保存した状態をこれで置き換える。
    # GDN の層がなければ None
    recurrent_states: torch.Tensor | None


class SiglipActorCriticWithActionValue(nn.Module):
    """凍結した SigLIP2 で観測のフレームを1枚ずつ読み、そのパッチの特徴から状態を作る
    アクタークリティック。

    毎ティック、エンコーダが現在のフレームを読んだ特徴を作り（``represent``）、バッファに
    保存する。推論も学習も保存した表現を読み戻し、SigLIP2 を通さない。
    状態はステップごとに ``to_state`` で作り、時間方向には何もしない。リプレイの窓は 2 ステップで、
    最初のステップの状態が現在の状態、最後のステップの状態が次の状態。

    ``siglip.num_gdn_layers`` が 0 より大きければ、ステップのトークンを潜在のトークンにまとめて
    GDN の層（``SiglipEncoder``）に1本の流れとして読ませ、その出力の記憶のトークンをステップの
    トークンに付け足してから混ぜる。毎ティック ``stream_step`` が流れを1ステップ進め、バッファに
    特徴と読む前の再帰の状態をつないだ1行を保存する。推論はその出力から行動を出す。リプレイの窓は
    ``gdn_burn_in_steps + gdn_bptt_steps + 1`` ステップで、最初の行に保存した再帰の状態から始め、
    burn-in のステップを勾配なしで読んで状態を今の重みに馴染ませてから、続くステップを BPTT で
    読む。その最後のステップが現在の状態、窓の最後のステップが次の状態。GDN の層の元の重みは
    この BPTT で、LoRA は ``compute_replay_and_streaming_loss`` の最新のステップの分で RTRL で学ぶ。
    ``reset_on_episode_end`` なら、窓の中のエピソードの境目で再帰の状態を空にする。
    保存した再帰の状態は書き込んだときの重みで計算したもので、重みが動くと古くなるので、窓を
    読むたびに 2 行目以降の分を今の重みで計算し直したものを返し、エージェントがバッファに書き戻す
    （``SiglipLoss.recurrent_states``）。
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
        siglip_config: DictConfig,
        reset_on_episode_end: bool,
    ) -> None:
        super().__init__()
        self.use_gdn = siglip_config.num_gdn_layers > 0
        self.burn_in_steps = siglip_config.gdn_burn_in_steps
        # 表現はバッファに保存するので、GDN の層がなければバッファから読む窓は今のステップだけで
        # よい。あれば burn-in と BPTT のステップ
        self.seq_len = (
            siglip_config.gdn_burn_in_steps + siglip_config.gdn_bptt_steps if self.use_gdn else 1
        )
        self.reset_on_episode_end = reset_on_episode_end
        self.critic_loss_weight = critic_loss_weight
        device = "cuda"
        action_dim = action_space_shape[0]
        self.encoder = SiglipEncoder(
            image_shapes=image_shapes,
            action_dim=action_dim,
            siglip_config=siglip_config,
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
        """バッファに保存する1行の形。GDN の層があれば、特徴を平らにしたものと読む前の再帰の状態を
        つないだもの。"""
        if self.use_gdn:
            positions, feature_dim = self.encoder.representation_shape()
            return (positions * feature_dim + self.encoder.gdn_state_size(),)
        return self.encoder.representation_shape()

    def represent(self, images: list[torch.Tensor]) -> torch.Tensor:
        """このティックの画像 (3, H, W) の並びを SigLIP2 で読んだ、バッファに保存する表現。"""
        return self.encoder.represent(images)

    def stream_step(
        self,
        images: list[torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        scalar_obs: torch.Tensor,
        reset_state: bool,
        use_trace: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GDN の層の流れをこのティックの分だけ進める。前の行動 (action_dim,)、正規化した報酬
        (1,)、スカラーの観測 (SCALAR_OBS_DIM,)。バッファに保存する1行と、記憶のトークンを
        付け足した混ぜる層の入力 (tokens_num + memory_tokens, token_dim) を返す。``use_trace`` なら出力は LoRA までの計算グラフ付きで、
        でなければ勾配なしで読む。"""
        features = self.encoder.represent(images)
        with torch.set_grad_enabled(use_trace):
            state_before, output = self.encoder.stream_step(
                features, action, reward, scalar_obs, reset_state, use_trace
            )
        return torch.cat([features.flatten(), state_before.detach()]), output

    @torch.inference_mode()
    def infer(self, window: ReplayBufferData, warmup: bool) -> SiglipInference:
        """バッファの最新の窓 (1, seq_len) の最後のステップの状態から行動を出す。``warmup``
        （バッファを貯めている間）なら一様乱数の行動を返す。"""
        assert window.observations.shape[0] == 1, "Batch size must be 1 for inference"
        return self._infer_state(self._state(window, -1), warmup)

    @torch.inference_mode()
    def infer_from_stream(self, output: torch.Tensor, warmup: bool) -> SiglipInference:
        """``stream_step`` が返した混ぜる層の入力から行動を出す。"""
        return self._infer_state(self.encoder.mix(output[None]), warmup)

    def _infer_state(self, state: torch.Tensor, warmup: bool) -> SiglipInference:
        action = self.policy_head.get_action(state)
        value_report = self.value_head.value_report(self.value_head(state, action).output)
        if warmup:
            action = torch.rand_like(action) * 2.0 - 1.0
        return SiglipInference(action=action, value_report=value_report, state=state)

    def compute_replay_loss(self, data: ReplayBufferData) -> SiglipLoss:
        """リプレイの窓の損失。"""
        state, next_state, recurrent_states = self._replay_states(data)
        result = self._head_terms(
            state,
            next_state,
            data.actions[:, -1],
            data.rewards[:, -1],
            data.dones[:, -1],
            data.weights,
        )
        result.recurrent_states = recurrent_states
        return result

    def compute_replay_and_streaming_loss(
        self,
        data: ReplayBufferData,
        previous_output: torch.Tensor,
        latest_output: torch.Tensor,
        latest: ReplayBufferData,
    ) -> SiglipLoss:
        """リプレイの窓 ``data`` と最新の遷移を1つのバッチにつないだ損失。最新の遷移の現在の状態は、
        前のティックの ``stream_step`` の出力 ``previous_output``（LoRA までの計算グラフ付き）から、
        次の状態はこのティックの出力 ``latest_output`` から作る。遷移の行動・報酬・終端は最新の
        窓 ``latest`` (1, 2) の最後の行。TD 誤差はリプレイの窓の分だけ返す。"""
        replay_state, replay_next_state, recurrent_states = self._replay_states(data)
        streaming_state = self.encoder.mix(previous_output[None])
        with torch.no_grad():
            streaming_next_state = self.encoder.mix(latest_output.detach()[None])
        result = self._head_terms(
            torch.cat([replay_state, streaming_state]),
            torch.cat([replay_next_state, streaming_next_state]),
            torch.cat([data.actions[:, -1], latest.actions[:, -1]]),
            torch.cat([data.rewards[:, -1], latest.rewards[:, -1]]),
            torch.cat([data.dones[:, -1], latest.dones[:, -1]]),
            torch.cat([data.weights, latest.weights]),
        )
        result.td_error = result.td_error[: data.weights.shape[0]]
        result.recurrent_states = recurrent_states
        return result

    def _replay_states(
        self, data: ReplayBufferData
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """リプレイの窓の現在の状態（最後から2番目のステップ）と次の状態（最後のステップ、
        勾配なし）。GDN の層があれば、2 行目以降を読む前の再帰の状態
        (B, window_len - 1, gdn_state_size) も返す（``SiglipLoss.recurrent_states``）。"""
        if not self.use_gdn:
            with torch.no_grad():
                next_state = self._state(data, -1)
            return self._state(data, -2), next_state, None
        batch, window_len = data.observations.shape[:2]
        positions, feature_dim = self.encoder.representation_shape()
        features = data.observations[..., : positions * feature_dim].view(
            batch, window_len, positions, feature_dim
        )
        recurrent = data.observations[:, 0, positions * feature_dim :]
        recomputed = []
        for step in range(window_len):
            if step > 0:
                if self.reset_on_episode_end:
                    # 前の行が終端なら、この行から新しいエピソードの流れ
                    recurrent = recurrent * (1.0 - data.dones[:, step - 1])
                recomputed.append(recurrent.detach())
            # burn-in と、次の状態の行は勾配なしで読む
            with torch.set_grad_enabled(self.burn_in_steps <= step < window_len - 1):
                tokens = self.encoder.step_tokens(
                    features[:, step],
                    data.actions[:, step],
                    data.rewards[:, step],
                    data.scalar_obs[:, step],
                )
                output, recurrent = self.encoder.read_step(tokens, recurrent)
                if step == window_len - 2:
                    state = self.encoder.mix(output)
        with torch.no_grad():
            next_state = self.encoder.mix(output)
        return state, next_state, torch.stack(recomputed, dim=1)

    def _head_terms(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        weights: torch.Tensor,
    ) -> SiglipLoss:
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
        # 内発報酬は遷移の報酬に足す。TD の目標にだけ入る
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
        return SiglipLoss(
            loss=self.critic_loss_weight * critic_loss + actor_loss + state_prediction_loss,
            info=info,
            td_error=td_error,
            recurrent_states=None,
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
