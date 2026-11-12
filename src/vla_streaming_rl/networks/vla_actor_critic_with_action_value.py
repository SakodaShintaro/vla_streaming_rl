# SPDX-License-Identifier: MIT
from collections.abc import Callable

import numpy as np
import torch
from omegaconf import DictConfig

from ..replay_buffer import ReplayBufferData
from .interface import InferInput, InferResult, LossResult, NetworkInterface
from .modules.prediction_head import LatentTDPredictionHead
from .modules.value_head import DistributionalValueHead
from .modules.vlm_encoder import VLMEncoder


class VLAActorCriticWithActionValue(NetworkInterface):
    """1枚の画像と指示の文から行動を出す VLA。

    毎ステップ、観測の画像とスカラーの観測をエンコーダ（凍結した VLM）に読ませて状態を作る。
    VLM は指示の文から始まるエピソードの流れを読み続けるので、過去の観測は VLM が自分の
    キャッシュに持ち越した分として反映される。方策ヘッドと行動価値ヘッドはその状態を読む。

    VLM は凍結して使うので、表現は観測を受け取ったときに一度だけ計算してバッファに保存する。
    推論も学習もそれを読み戻し、VLM を通さない。

    ``use_memory`` なら、エンコーダの LoRA 付きの GDN の層を通した画像のトークンを保存する。LoRA 付きの GDN の層の
    再帰の状態と RTRL のトレースはエンコーダが持ち、バッファには入れない。LoRA 付きの GDN の層の LoRA は
    ``compute_streaming_loss`` が最新のステップで RTRL で学習し、リプレイの ``compute_replay_loss`` は
    保存した表現から射影とヘッドを学習する。
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
        use_memory: bool,
    ) -> None:
        super().__init__()
        # 表現はバッファに保存するので、バッファから読む窓は今のステップだけでよい
        self.seq_len = 1
        self.horizon = horizon
        self.critic_loss_weight = critic_loss_weight
        device = "cuda"
        action_dim = action_space_shape[0]
        self.encoder = VLMEncoder(
            image_shapes=image_shapes,
            action_dim=action_dim,
            vla_config=vla_config,
            use_memory=use_memory,
            device=device,
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
        # LoRA 付きの GDN の層の状態は推論のたびに1ステップ進むので、行動のチャンクは1ステップに限る
        assert not use_memory or horizon == 1, horizon

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

    def reset_memory(self) -> None:
        self.encoder.reset_memory()

    def step_gdn_layers(self, step_tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder.step_gdn_layers(step_tokens)

    @torch.inference_mode()
    def infer(self, data: InferInput) -> InferResult:
        assert data.s_seq.shape[0] == 1, "Batch size must be 1 for inference"
        state = self._state(data.s_seq, data.a_seq, data.r_seq, data.scalar_obs_seq, -1)
        action = self.policy_head.get_action(state)
        value_report = self.value_head.value_report(self.value_head(state, action).output)
        if data.warmup:
            # バッファを貯めている間は一様乱数の行動を返す
            action = torch.rand_like(action) * 2.0 - 1.0
        return InferResult(action=action, value_report=value_report, features=state)

    def compute_replay_loss(self, data: ReplayBufferData) -> LossResult:
        """リプレイの窓の損失。現在の状態はチャンクの手前のステップから、次の状態は最後の
        ステップから、どちらも保存した表現で作る。"""
        window = (data.observations, data.actions, data.rewards, data.scalar_obs)
        with torch.no_grad():
            next_state = self._state(*window, -1)
        state = self._state(*window, -self.horizon - 1)
        return self._head_terms(data, state, next_state)

    def compute_streaming_loss(self, image: torch.Tensor, data: ReplayBufferData) -> LossResult:
        """最新の窓 ``data`` (B=1) の損失。現在の状態は、前のティックで ``step_gdn_layers`` が返した
        画像のトークン ``image`` (num_positions, hidden_size)（計算グラフ付き）から作り、勾配は
        LoRA 付きの GDN の層の LoRA まで届く。次の状態は保存した表現で作る。"""
        assert self.horizon == 1, self.horizon
        state = self.encoder(
            image.unsqueeze(0), data.actions[:, -2], data.rewards[:, -2], data.scalar_obs[:, -2]
        )
        with torch.no_grad():
            next_state = self._state(
                data.observations, data.actions, data.rewards, data.scalar_obs, -1
            )
        return self._head_terms(data, state, next_state)

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

    def _head_terms(
        self,
        data: ReplayBufferData,
        state: torch.Tensor,
        next_state: torch.Tensor,
    ) -> LossResult:
        """現在の状態 ``state``（勾配あり）と次の状態 ``next_state`` から、ヘッドの損失。
        行動・報酬・終端は窓 ``data`` の最後の horizon ステップのもの。"""
        with torch.no_grad():
            # TD の目標は割引率ごとに、その割引率の方策の次の行動で作る。最後が主の割引率の
            # 方策の行動で、状態の予測にはそれを使う
            next_gamma_actions = self.policy_head.get_gamma_actions(next_state)
            next_action = next_gamma_actions[:, -1]
            next_q = self.value_head.gamma_values(next_state, next_gamma_actions)
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
        return LossResult(
            loss=self.critic_loss_weight * critic_loss + actor_loss + state_prediction_loss,
            info=info,
            td_error=td_error,
        )
