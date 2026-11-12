# SPDX-License-Identifier: MIT
import math

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import SimbaBlock
from .head_output import HeadOutput
from .self_attention import SpatialTransformerBlock, get_fourier_embeds_from_coordinates
from .sparse_utils import apply_one_shot_pruning


def _euler_denoise(
    noise: torch.Tensor,
    denoising_time: float,
    denoising_steps: int,
    predict_fn,
) -> torch.Tensor:
    """Euler denoising via flow matching (forward: 0 -> denoising_time).

    Args:
        predict_fn: callable(x_t, t) -> clean data x_1 (x-prediction).
            Velocity is derived as v = (x_1_pred - x_t) / (1 - t).

    Returns:
        x_T を [-1, 1] に切ったもの, same shape as noise. 予測が有界なので tanh は掛けない
        （掛けると予測の範囲がさらに縮み、端で勾配が消える）。ノイズの分だけはみ出す行動を
        切り、critic が学習したことのない範囲の行動を評価しないようにする
    """
    x_t = noise
    dt = denoising_time / denoising_steps
    time_val = 0.0
    for _ in range(denoising_steps):
        t = torch.full((noise.shape[0],), time_val, device=noise.device)
        pred = predict_fn(x_t, t)
        denom = max(1e-4, 1.0 - time_val)
        v = (pred - x_t) / denom
        x_t = x_t + dt * v
        time_val += dt
    return x_t.clamp(-1.0, 1.0)


# 割引率の特徴の幅（状態をトークンとして読まない方策で、入力につなぐ分）
GAMMA_EMBEDDING_SIZE = 64


def _gamma_features(embed_dim: int, gamma: float, multi_gammas: list[float]) -> torch.Tensor:
    """割引率ごとの方策を条件づける特徴 (num_gammas, embed_dim)（AMAGO の multi-gamma）。
    補助の割引率を先に、主の割引率を最後に並べる（価値ヘッドと同じ並び）。実効の地平
    1/(1-γ) の log を sin/cos の特徴にしたもの。"""
    gammas = torch.tensor(list(multi_gammas) + [gamma], dtype=torch.float32)
    horizon_feature = torch.log1p(1.0 / (1.0 - gammas))
    return get_fourier_embeds_from_coordinates(embed_dim, horizon_feature.view(-1, 1))[:, 0, 0]


def _repeat_over_gammas(
    x: torch.Tensor, gamma_features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` (B, ...) を割引率の数だけ並べた (B * num_gammas, ...) と、各行の割引率の特徴
    (B * num_gammas, ...)。行は ``x`` の行ごとに割引率の並びで続く。"""
    num_gammas = gamma_features.size(0)
    feature_repeats = (x.size(0),) + (1,) * (gamma_features.dim() - 1)
    return x.repeat_interleave(num_gammas, dim=0), gamma_features.repeat(*feature_repeats)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiffusionPolicy(nn.Module):
    """方策は割引率ごとに持つ（AMAGO の multi-gamma）。割引率の特徴を入力につないで条件づけ、
    1つのネットワークで割引率ごとの行動を出す。割引率 g の方策は割引率 g の価値を最大にする
    ように学習し、環境で実行するのは主の割引率の方策の行動だけ。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        token_dim: int,
        hidden_dim: int,
        block_num: int,
        denoising_time: float,
        sparsity: float,
        denoising_steps: int,
        dacer_loss_weight: float,
        detach_state: bool,
        gamma: float,
        multi_gammas: list[float],
    ) -> None:
        super().__init__()
        # 状態をトークンとしては読まないので、トークンの幅は使わない
        del token_dim
        self.detach_state = detach_state
        time_embedding_size = 256
        total_action_dim = action_dim
        self.fc_in = nn.Linear(
            state_dim + total_action_dim + time_embedding_size + GAMMA_EMBEDDING_SIZE, hidden_dim
        )
        self.fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc_out = nn.Linear(hidden_dim, total_action_dim)
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
        self.action_dim = action_dim
        self.denoising_steps = denoising_steps
        self.denoising_time = denoising_time
        self.dacer_loss_weight = dacer_loss_weight
        self.t_embedder = TimestepEmbedder(time_embedding_size)
        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )
        self.register_buffer(
            "gamma_features", _gamma_features(GAMMA_EMBEDDING_SIZE, gamma, multi_gammas)
        )
        self.num_gammas = self.gamma_features.size(0)

    def forward(
        self, a: torch.Tensor, t: torch.Tensor, state: torch.Tensor, gamma_feature: torch.Tensor
    ) -> HeadOutput:
        """
        Args:
            a: flattened action (B, action_dim)
            t: timestep (B,)
            state: state embedding (B, state_dim)
            gamma_feature: その行の方策の割引率の特徴 (B, GAMMA_EMBEDDING_SIZE)
        """
        t = self.t_embedder(t)
        x = torch.cat([a, t, state, gamma_feature], 1)
        x = self.fc_in(x)

        x = self.fc_mid(x)
        x = self.norm(x)

        activation = x

        x = self.fc_out(x)
        x = torch.tanh(x)
        return HeadOutput(output=x, activation=activation)

    def _denoise(self, x: torch.Tensor, gamma_feature: torch.Tensor) -> torch.Tensor:
        """状態 (B, state_dim) と割引率の特徴 (B, GAMMA_EMBEDDING_SIZE) の行ごとの行動
        (B, action_dim)。"""
        bs = x.size(0)
        noise = torch.randn(bs, self.action_dim, device=x.device)

        def predict_fn(x_t, t):
            x_flat = x_t.view(bs, -1)
            return self.forward(x_flat, t, x, gamma_feature).output.view(bs, self.action_dim)

        return _euler_denoise(noise, self.denoising_time, self.denoising_steps, predict_fn)

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """主の割引率の方策の行動 (B, action_dim)。環境で実行するのはこれ。"""
        return self._denoise(x, self.gamma_features[-1:].expand(x.size(0), -1))

    def get_gamma_actions(self, x: torch.Tensor) -> torch.Tensor:
        """割引率ごとの方策の行動 (B, num_gammas, action_dim)。最後が主の割引率。"""
        return self._denoise(*_repeat_over_gammas(x, self.gamma_features)).view(
            x.size(0), self.num_gammas, self.action_dim
        )

    def compute_actor_loss(
        self,
        state: torch.Tensor,
        taken_action: torch.Tensor,
        *,
        value_head,
    ) -> tuple[torch.Tensor, dict]:
        """Advantage-maximizing actor loss + DACER2 score matching
        (https://arxiv.org/abs/2505.23426).

        割引率ごとの方策の行動を、それぞれの割引率の価値で評価して最大にする。
        """
        del taken_action
        if self.detach_state:
            state = state.detach()
        action = self.get_gamma_actions(state)
        B, G, action_dim = action.shape
        device = action.device

        for param in value_head.parameters():
            param.requires_grad_(False)
        policy_q = value_head.gamma_values(state, action)  # (B, G)
        actor_loss = -policy_q.mean()
        for param in value_head.parameters():
            param.requires_grad_(True)

        # DACER2 score matching: regress the policy's x-prediction onto a
        # Q-gradient-derived velocity target, with logarithmic time weighting.
        # 行は状態ごとに割引率の並びで続く (B * G, ...)
        eps = 1e-4
        t = torch.rand((B * G, 1), device=device) * (1 - eps) + eps
        w_t = torch.exp(0.4 * t + (-1.8))

        actions = action.view(B, G, -1).clone().detach().requires_grad_(True)
        q_values = value_head.gamma_values(state, actions.view(B, G, action_dim))
        q_grad = torch.autograd.grad(q_values.sum(), actions, create_graph=True)[0]
        q_grad = q_grad.view(B * G, -1)
        flat_actions = actions.detach().view(B * G, -1)

        noise = torch.randn_like(flat_actions).clamp(-3.0, 3.0)
        a_t = (1.0 - t) * noise + t * flat_actions
        with torch.no_grad():
            v_target = (1 - t) / t * q_grad + 1 / t * flat_actions
            v_target = v_target / (v_target.norm(dim=1, keepdim=True) + 1e-8)
            v_target = w_t * v_target
            # Convert to x-space (policy outputs the clean action prediction).
            target = a_t + (1 - t) * v_target

        pred_dict = self.forward(
            a_t, t.squeeze(1), *_repeat_over_gammas(state, self.gamma_features)
        )
        pred = pred_dict.output
        dacer_loss = F.mse_loss(pred, target)
        total_loss = actor_loss + dacer_loss * self.dacer_loss_weight

        info_dict = {
            "actor_loss": actor_loss.item(),
            "dacer_loss": dacer_loss.item(),
            "policy_q": policy_q.mean().item(),
        }
        return total_loss, info_dict


class QF3Policy(nn.Module):
    """QF3 (Fast Flow RL with Filtered Q-Gradients, arXiv:2610.08789) の flow の方策。

    ネットワークは速度場 v(s, x_τ, τ) を出し、行動はノイズ x_0 ~ N(0, I) から τ = 0 → 1 へ
    Euler で積分して作る。最後の層は 0 で初期化するので、学習の始めの flow は x_1 = x_0。

    actor の損失はリプレイの行動 a_buf への flow matching と、a_buf にノイズを乗せた x_τ から
    1 回の Euler で作った行動の Q の最大化の和。Q を読む行動は、速度の a_buf との差を成分ごとに
    [-α, α] に切って作る（式 (3)）ので、critic が学習した a_buf の近くでしか評価せず、差が α を
    超えた成分には Q の勾配が届かない（式 (5)）。flow matching は切らない速度に掛かる。

    方策は割引率ごとに持つ（AMAGO の multi-gamma）。割引率の特徴を入力につないで条件づけ、
    割引率 g の方策は割引率 g の価値を最大にするように学習し、環境で実行するのは主の割引率の
    方策の行動だけ。flow matching の目標はどの割引率でも同じリプレイの行動。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        token_dim: int,
        hidden_dim: int,
        block_num: int,
        sparsity: float,
        denoising_steps: int,
        clip_alpha: float,
        cfm_loss_weight: float,
        detach_state: bool,
        gamma: float,
        multi_gammas: list[float],
    ) -> None:
        super().__init__()
        # 状態をトークンとしては読まないので、トークンの幅は使わない
        del token_dim
        self.detach_state = detach_state
        time_embedding_size = 256
        total_action_dim = action_dim
        self.fc_in = nn.Linear(
            state_dim + total_action_dim + time_embedding_size + GAMMA_EMBEDDING_SIZE, hidden_dim
        )
        self.fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc_out = nn.Linear(hidden_dim, total_action_dim)
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
        self.action_dim = action_dim
        self.denoising_steps = denoising_steps
        self.clip_alpha = clip_alpha
        self.cfm_loss_weight = cfm_loss_weight
        self.t_embedder = TimestepEmbedder(time_embedding_size)
        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )
        self.register_buffer(
            "gamma_features", _gamma_features(GAMMA_EMBEDDING_SIZE, gamma, multi_gammas)
        )
        self.num_gammas = self.gamma_features.size(0)

    def forward(
        self, a: torch.Tensor, t: torch.Tensor, state: torch.Tensor, gamma_feature: torch.Tensor
    ) -> HeadOutput:
        """
        Args:
            a: flattened noisy action x_τ (B, action_dim)
            t: flow time τ (B,)
            state: state embedding (B, state_dim)
            gamma_feature: その行の方策の割引率の特徴 (B, GAMMA_EMBEDDING_SIZE)

        Returns:
            速度 v (B, action_dim)。有界にはしない
        """
        x = torch.cat([a, self.t_embedder(t), state, gamma_feature], 1)
        x = self.norm(self.fc_mid(self.fc_in(x)))
        activation = x
        return HeadOutput(output=self.fc_out(x), activation=activation)

    def _denoise(self, x: torch.Tensor, gamma_feature: torch.Tensor) -> torch.Tensor:
        """状態 (B, state_dim) と割引率の特徴 (B, GAMMA_EMBEDDING_SIZE) の行ごとの行動
        (B, action_dim)。τ = 0 → 1 を ``denoising_steps`` 回の Euler で積分する。"""
        bs = x.size(0)
        noise = torch.randn(bs, self.action_dim, device=x.device)

        def predict_fn(x_t, t):
            # 速度を x-prediction に直して渡す（_euler_denoise はそこから速度を戻す）
            velocity = self.forward(x_t.view(bs, -1), t, x, gamma_feature).output
            return x_t + (1.0 - t).view(bs, 1) * velocity

        return _euler_denoise(noise, 1.0, self.denoising_steps, predict_fn)

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """主の割引率の方策の行動 (B, action_dim)。環境で実行するのはこれ。"""
        return self._denoise(x, self.gamma_features[-1:].expand(x.size(0), -1))

    def get_gamma_actions(self, x: torch.Tensor) -> torch.Tensor:
        """割引率ごとの方策の行動 (B, num_gammas, action_dim)。最後が主の割引率。"""
        return self._denoise(*_repeat_over_gammas(x, self.gamma_features)).view(
            x.size(0), self.num_gammas, self.action_dim
        )

    def compute_actor_loss(
        self,
        state: torch.Tensor,
        taken_action: torch.Tensor,
        *,
        value_head,
    ) -> tuple[torch.Tensor, dict]:
        """QF3 の actor の損失 -Q(s, x̂_1,clip) + λ_cfm L_cfm（式 (4)）。

        割引率ごとの方策の行動を、それぞれの割引率の価値で評価して最大にする。
        """
        if self.detach_state:
            state = state.detach()
        B, G = state.size(0), self.num_gammas
        device = state.device

        # 行は状態ごとに割引率の並びで続く (B * G, ...)
        state_rep, gamma_feature = _repeat_over_gammas(state, self.gamma_features)
        replay_action = taken_action.reshape(B, -1).repeat_interleave(G, dim=0)
        t = torch.rand((B * G, 1), device=device)
        noise = torch.randn_like(replay_action)
        x_t = (1.0 - t) * noise + t * replay_action
        replay_velocity = replay_action - noise

        velocity = self.forward(x_t, t.squeeze(1), state_rep, gamma_feature).output
        cfm_loss = F.mse_loss(velocity, replay_velocity)

        # 速度のリプレイとの差を [-α, α] に切った 1 回の Euler の行動（式 (3)）
        gap = velocity - replay_velocity
        clipped_action = replay_action + (1.0 - t) * gap.clamp(-self.clip_alpha, self.clip_alpha)

        for param in value_head.parameters():
            param.requires_grad_(False)
        policy_q = value_head.gamma_values(
            state, clipped_action.view(B, G, self.action_dim)
        )  # (B, G)
        for param in value_head.parameters():
            param.requires_grad_(True)
        actor_loss = -policy_q.mean()
        total_loss = actor_loss + self.cfm_loss_weight * cfm_loss

        info_dict = {
            "actor_loss": actor_loss.item(),
            "cfm_loss": cfm_loss.item(),
            "policy_q": policy_q.mean().item(),
            # Q の勾配が届く成分の割合（論文 4.1 節）
            "qf3_inside_clip": (gap.detach().abs() < self.clip_alpha).float().mean().item(),
        }
        return total_loss, info_dict


class TokenDiffusionPolicy(nn.Module):
    """状態のトークンに、ノイズの乗った行動のトークンと拡散の時刻のトークンを並べ、Transformer の
    層で混ぜてから MLP で読む拡散の方策。混ぜる前後以外は :class:`DiffusionPolicy` と同じ。

    状態 (B, state_dim) は ``token_dim`` 幅のトークンを平らに並べたものとして読み戻す。行動の
    チャンクは成分ごとに sin/cos の特徴で1トークンにする（前の行動のトークンと同じ作り方）。
    混ぜる層は :class:`SpatialTransformerBlock` で、位置の埋め込みは 0、残差は layer scale で
    小さく始めるので、学習の始めは「トークンを並べて MLP で読む」のとほぼ同じ。

    方策は割引率ごとに持つ（AMAGO の multi-gamma）。割引率のトークンを並べて条件づけ、1つの
    ネットワークで割引率ごとの行動を出す。割引率 g の方策は割引率 g の価値を最大にするように
    学習し、環境で実行するのは主の割引率の方策の行動だけ。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        token_dim: int,
        mixer_layers: int,
        mixer_heads: int,
        mixer_layer_scale_init: float,
        hidden_dim: int,
        block_num: int,
        denoising_time: float,
        sparsity: float,
        denoising_steps: int,
        dacer_loss_weight: float,
        detach_state: bool,
        gamma: float,
        multi_gammas: list[float],
    ) -> None:
        super().__init__()
        assert state_dim % token_dim == 0, (state_dim, token_dim)
        self.detach_state = detach_state
        time_embedding_size = 256
        total_action_dim = action_dim
        # 状態のトークン、行動の成分ごとのトークン、時刻のトークン、割引率のトークン
        tokens_num = state_dim // token_dim + total_action_dim + 2
        mixed_dim = tokens_num * token_dim
        self.fc_in = nn.Linear(mixed_dim + total_action_dim + time_embedding_size, hidden_dim)
        self.fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc_out = nn.Linear(hidden_dim, total_action_dim)
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
        self.action_dim = action_dim
        self.denoising_steps = denoising_steps
        self.denoising_time = denoising_time
        self.dacer_loss_weight = dacer_loss_weight
        self.t_embedder = TimestepEmbedder(time_embedding_size)
        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )

        self.token_dim = token_dim
        self.t_token_embedder = TimestepEmbedder(token_dim)
        # 割引率のトークン (num_gammas, 1, token_dim)
        self.register_buffer(
            "gamma_tokens", _gamma_features(token_dim, gamma, multi_gammas).unsqueeze(1)
        )
        self.num_gammas = self.gamma_tokens.size(0)
        self.token_position = nn.Parameter(torch.zeros(1, tokens_num, token_dim))
        self.token_mixer = nn.Sequential(
            *[
                SpatialTransformerBlock(token_dim, mixer_heads, tokens_num, mixer_layer_scale_init)
                for _ in range(mixer_layers)
            ]
        )

    def forward(
        self, a: torch.Tensor, t: torch.Tensor, state: torch.Tensor, gamma_token: torch.Tensor
    ) -> HeadOutput:
        """
        Args:
            a: flattened action (B, action_dim)
            t: timestep (B,)
            state: state embedding (B, state_dim), state_dim // token_dim tokens laid flat
            gamma_token: その行の方策の割引率のトークン (B, 1, token_dim)
        """
        bs = a.size(0)
        state_tokens = state.view(bs, -1, self.token_dim)
        action_tokens = get_fourier_embeds_from_coordinates(self.token_dim, a.unsqueeze(1))[:, 0]
        t_token = self.t_token_embedder(t).unsqueeze(1)
        tokens = (
            torch.cat([state_tokens, action_tokens, t_token, gamma_token], dim=1)
            + self.token_position
        )
        mixed = self.token_mixer(tokens).flatten(start_dim=1)

        x = torch.cat([a, self.t_embedder(t), mixed], 1)
        x = self.norm(self.fc_mid(self.fc_in(x)))
        activation = x
        x = torch.tanh(self.fc_out(x))
        return HeadOutput(output=x, activation=activation)

    def _denoise(self, x: torch.Tensor, gamma_token: torch.Tensor) -> torch.Tensor:
        """状態 (B, state_dim) と割引率のトークン (B, 1, token_dim) の行ごとの行動
        (B, action_dim)。"""
        bs = x.size(0)
        noise = torch.randn(bs, self.action_dim, device=x.device)

        def predict_fn(x_t, t):
            x_flat = x_t.view(bs, -1)
            return self.forward(x_flat, t, x, gamma_token).output.view(bs, self.action_dim)

        return _euler_denoise(noise, self.denoising_time, self.denoising_steps, predict_fn)

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """主の割引率の方策の行動 (B, action_dim)。環境で実行するのはこれ。"""
        return self._denoise(x, self.gamma_tokens[-1:].expand(x.size(0), -1, -1))

    def get_gamma_actions(self, x: torch.Tensor) -> torch.Tensor:
        """割引率ごとの方策の行動 (B, num_gammas, action_dim)。最後が主の割引率。"""
        return self._denoise(*_repeat_over_gammas(x, self.gamma_tokens)).view(
            x.size(0), self.num_gammas, self.action_dim
        )

    def compute_actor_loss(
        self,
        state: torch.Tensor,
        taken_action: torch.Tensor,
        *,
        value_head,
    ) -> tuple[torch.Tensor, dict]:
        """Advantage-maximizing actor loss + DACER2 score matching
        (https://arxiv.org/abs/2505.23426).

        割引率ごとの方策の行動を、それぞれの割引率の価値で評価して最大にする。
        """
        del taken_action
        if self.detach_state:
            state = state.detach()
        action = self.get_gamma_actions(state)
        B, G, action_dim = action.shape
        device = action.device

        for param in value_head.parameters():
            param.requires_grad_(False)
        policy_q = value_head.gamma_values(state, action)  # (B, G)
        actor_loss = -policy_q.mean()
        for param in value_head.parameters():
            param.requires_grad_(True)

        # DACER2 score matching: regress the policy's x-prediction onto a
        # Q-gradient-derived velocity target, with logarithmic time weighting.
        # 行は状態ごとに割引率の並びで続く (B * G, ...)
        eps = 1e-4
        t = torch.rand((B * G, 1), device=device) * (1 - eps) + eps
        w_t = torch.exp(0.4 * t + (-1.8))

        actions = action.view(B, G, -1).clone().detach().requires_grad_(True)
        q_values = value_head.gamma_values(state, actions.view(B, G, action_dim))
        q_grad = torch.autograd.grad(q_values.sum(), actions, create_graph=True)[0]
        q_grad = q_grad.view(B * G, -1)
        flat_actions = actions.detach().view(B * G, -1)

        noise = torch.randn_like(flat_actions).clamp(-3.0, 3.0)
        a_t = (1.0 - t) * noise + t * flat_actions
        with torch.no_grad():
            v_target = (1 - t) / t * q_grad + 1 / t * flat_actions
            v_target = v_target / (v_target.norm(dim=1, keepdim=True) + 1e-8)
            v_target = w_t * v_target
            # Convert to x-space (policy outputs the clean action prediction).
            target = a_t + (1 - t) * v_target

        pred_dict = self.forward(a_t, t.squeeze(1), *_repeat_over_gammas(state, self.gamma_tokens))
        pred = pred_dict.output
        dacer_loss = F.mse_loss(pred, target)
        total_loss = actor_loss + dacer_loss * self.dacer_loss_weight

        info_dict = {
            "actor_loss": actor_loss.item(),
            "dacer_loss": dacer_loss.item(),
            "policy_q": policy_q.mean().item(),
        }
        return total_loss, info_dict


class CFGDiffusionPolicy(nn.Module):
    """
    Diffusion Policy with Classifier Free Guidance (CFGRL/pistar06)

    Training: Drop condition I (positive/negative) with condition_drop_prob probability
    Inference: Adjust guidance strength with beta (cfgrl_beta)

    方策は割引率ごとに持つ（AMAGO の multi-gamma）。割引率の特徴を入力につないで条件づけ、
    割引率 g の方策は割引率 g の価値での advantage の符号で条件づけて学習する。環境で実行
    するのは主の割引率の方策の行動だけ。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        token_dim: int,
        hidden_dim: int,
        block_num: int,
        denoising_time: float,
        sparsity: float,
        cfgrl_beta: float,
        denoising_steps: int,
        condition_drop_prob: float,
        detach_state: bool,
        gamma: float,
        multi_gammas: list[float],
    ) -> None:
        super().__init__()
        # 状態をトークンとしては読まないので、トークンの幅は使わない
        del token_dim
        self.detach_state = detach_state
        self.cfgrl_beta = cfgrl_beta
        self.condition_drop_prob = condition_drop_prob
        time_embedding_size = 256
        condition_embedding_size = 64
        total_action_dim = action_dim
        # Condition I embedding: 0=negative, 1=positive, 2=unconditional(dropout)
        self.condition_embedding = nn.Embedding(3, condition_embedding_size)

        self.fc_in = nn.Linear(
            state_dim
            + total_action_dim
            + time_embedding_size
            + condition_embedding_size
            + GAMMA_EMBEDDING_SIZE,
            hidden_dim,
        )
        self.fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc_out = nn.Linear(hidden_dim, total_action_dim)
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
        self.action_dim = action_dim
        self.denoising_steps = denoising_steps
        self.denoising_time = denoising_time
        self.t_embedder = TimestepEmbedder(time_embedding_size)
        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )
        self.register_buffer(
            "gamma_features", _gamma_features(GAMMA_EMBEDDING_SIZE, gamma, multi_gammas)
        )
        self.num_gammas = self.gamma_features.size(0)

    def forward(
        self,
        a: torch.Tensor,
        t: torch.Tensor,
        state: torch.Tensor,
        condition: torch.Tensor,
        gamma_feature: torch.Tensor,
    ) -> HeadOutput:
        """
        Args:
            a: flattened action (B, action_dim)
            t: timestep (B,)
            state: state embedding (B, state_dim)
            condition: condition I (B,) - 0=negative, 1=positive, 2=unconditional
            gamma_feature: その行の方策の割引率の特徴 (B, GAMMA_EMBEDDING_SIZE)
        """
        t_emb = self.t_embedder(t)
        cond_emb = self.condition_embedding(condition)  # (B, condition_embedding_size)
        x = torch.cat([a, t_emb, state, cond_emb, gamma_feature], 1)
        x = self.fc_in(x)

        x = self.fc_mid(x)
        x = self.norm(x)

        activation = x

        x = self.fc_out(x)
        x = torch.tanh(x)
        return HeadOutput(output=x, activation=activation)

    def _denoise(self, x: torch.Tensor, gamma_feature: torch.Tensor) -> torch.Tensor:
        """
        Inference using CFG, per row of state (B, state_dim) and gamma feature
        (B, GAMMA_EMBEDDING_SIZE).

        Guidance in output space: out = (1 - beta) * out_uncond + beta * out_positive
        Higher beta increases guidance towards positive advantage direction.
        """
        bs = x.size(0)
        device = x.device
        noise = torch.randn(bs, self.action_dim, device=device)

        # Condition labels: 1=positive, 2=unconditional
        cond_positive = torch.ones((bs,), dtype=torch.long, device=device)
        cond_uncond = torch.full((bs,), 2, dtype=torch.long, device=device)

        def predict_fn(x_t, t):
            x_flat = x_t.view(bs, -1)
            out_pos = self.forward(x_flat, t, x, cond_positive, gamma_feature).output
            out_unc = self.forward(x_flat, t, x, cond_uncond, gamma_feature).output
            out = (1 - self.cfgrl_beta) * out_unc + self.cfgrl_beta * out_pos
            return out.view(bs, self.action_dim)

        return _euler_denoise(noise, self.denoising_time, self.denoising_steps, predict_fn)

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """主の割引率の方策の行動 (B, action_dim)。環境で実行するのはこれ。"""
        return self._denoise(x, self.gamma_features[-1:].expand(x.size(0), -1))

    def get_gamma_actions(self, x: torch.Tensor) -> torch.Tensor:
        """割引率ごとの方策の行動 (B, num_gammas, action_dim)。最後が主の割引率。"""
        return self._denoise(*_repeat_over_gammas(x, self.gamma_features)).view(
            x.size(0), self.num_gammas, self.action_dim
        )

    def compute_actor_loss(
        self,
        state: torch.Tensor,
        taken_action: torch.Tensor,
        *,
        value_head: nn.Module,
    ) -> tuple[torch.Tensor, dict]:
        """CFGRL/pistar06: condition on advantage sign, drop with condition_drop_prob.

        データの行動の advantage は、割引率ごとに、その割引率の方策の行動の Q を基準に
        Q_g(s, a) - Q_g(s, π_g(s)) とする。中央値は割引率ごとに取る。"""
        if self.detach_state:
            state = state.detach()
        batch_size = state.shape[0]
        device = state.device
        G = self.num_gammas

        with torch.no_grad():
            data_q = value_head.to_values(value_head(state, taken_action).output)  # (B, G)
            policy_q = value_head.gamma_values(state, self.get_gamma_actions(state))  # (B, G)
            advantage = data_q - policy_q
            threshold = advantage.median(dim=0, keepdim=True).values
            # 行は状態ごとに割引率の並びで続く (B * G,)
            condition = (advantage >= threshold).long().view(-1)
            drop_mask = torch.rand(batch_size * G, device=device) < self.condition_drop_prob
            condition = torch.where(drop_mask, torch.full_like(condition, 2), condition)

        state_rep, gamma_feature = _repeat_over_gammas(state, self.gamma_features)
        action_flat = taken_action.view(batch_size, -1).repeat_interleave(G, dim=0)
        eps = 1e-4
        t = torch.rand((batch_size * G, 1), device=device) * (1 - eps) + eps
        noise = torch.randn_like(action_flat)
        noise = torch.clamp(noise, -3.0, 3.0)
        a_t = (1.0 - t) * noise + t * action_flat
        actor_output_dict = self.forward(a_t, t.squeeze(1), state_rep, condition, gamma_feature)
        pred = actor_output_dict.output
        actor_loss = F.mse_loss(pred, action_flat)

        positive_ratio = (condition == 1).float().mean().item()
        negative_ratio = (condition == 0).float().mean().item()
        uncond_ratio = (condition == 2).float().mean().item()
        info_dict = {
            "actor_loss": actor_loss.item(),
            "dacer_loss": 0.0,
            "advantage": advantage.mean().item(),
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "uncond_ratio": uncond_ratio,
        }
        return actor_loss, info_dict


class IMLEPolicy(nn.Module):
    """Single-step generator a = G(s, z), z ~ N(0, I), after cIMLE (IMLE-VLA).

    cIMLE draws ``sample_num`` candidates per state and trains only the one
    nearest the ground truth. Here the ground truth is replaced by the critic:
    the candidate with the highest Q is pushed up by Q-gradient ascent, while
    the others are left untouched, so different z can settle on different modes.

    方策は割引率ごとに持つ（AMAGO の multi-gamma）。割引率の特徴を入力につないで条件づけ、
    割引率 g の方策の候補は割引率 g の価値で選ぶ。環境で実行するのは主の割引率の方策の行動だけ。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        token_dim: int,
        hidden_dim: int,
        block_num: int,
        sparsity: float,
        sample_num: int,
        detach_state: bool,
        gamma: float,
        multi_gammas: list[float],
    ) -> None:
        super().__init__()
        # 状態をトークンとしては読まないので、トークンの幅は使わない
        del token_dim
        self.detach_state = detach_state
        self.action_dim = action_dim
        self.sample_num = sample_num
        total_action_dim = action_dim
        self.fc_in = nn.Linear(state_dim + total_action_dim + GAMMA_EMBEDDING_SIZE, hidden_dim)
        self.fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc_out = nn.Linear(hidden_dim, total_action_dim)
        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )
        self.register_buffer(
            "gamma_features", _gamma_features(GAMMA_EMBEDDING_SIZE, gamma, multi_gammas)
        )
        self.num_gammas = self.gamma_features.size(0)

    def forward(
        self, z: torch.Tensor, state: torch.Tensor, gamma_feature: torch.Tensor
    ) -> HeadOutput:
        """
        Args:
            z: latent noise (B, action_dim)
            state: state embedding (B, state_dim)
            gamma_feature: その行の方策の割引率の特徴 (B, GAMMA_EMBEDDING_SIZE)
        """
        x = torch.cat([z, state, gamma_feature], 1)
        x = self.fc_in(x)
        x = self.fc_mid(x)
        x = self.norm(x)
        activation = x
        x = self.fc_out(x)
        x = torch.tanh(x)
        return HeadOutput(output=x, activation=activation)

    def _generate(self, x: torch.Tensor, gamma_feature: torch.Tensor) -> torch.Tensor:
        """状態 (B, state_dim) と割引率の特徴 (B, GAMMA_EMBEDDING_SIZE) の行ごとの行動
        (B, action_dim)。"""
        bs = x.size(0)
        z = torch.randn(bs, self.action_dim, device=x.device)
        return self.forward(z, x, gamma_feature).output.view(bs, self.action_dim)

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """主の割引率の方策の行動 (B, action_dim)。環境で実行するのはこれ。"""
        return self._generate(x, self.gamma_features[-1:].expand(x.size(0), -1))

    def get_gamma_actions(self, x: torch.Tensor) -> torch.Tensor:
        """割引率ごとの方策の行動 (B, num_gammas, action_dim)。最後が主の割引率。"""
        return self._generate(*_repeat_over_gammas(x, self.gamma_features)).view(
            x.size(0), self.num_gammas, self.action_dim
        )

    def compute_actor_loss(
        self,
        state: torch.Tensor,
        taken_action: torch.Tensor,
        *,
        value_head: nn.Module,
    ) -> tuple[torch.Tensor, dict]:
        """Maximize Q of the best of ``sample_num`` candidates per state and gamma.

        割引率 g の方策の候補は割引率 g の価値で評価し、その中の最良を押し上げる。"""
        del taken_action
        if self.detach_state:
            state = state.detach()
        B = state.shape[0]
        m = self.sample_num
        G = self.num_gammas

        # 行は状態ごとに候補、候補ごとに割引率の並びで続く (B * m * G, ...)
        state_rep = state.repeat_interleave(m, dim=0)  # (B*m, state_dim)
        action = self._generate(*_repeat_over_gammas(state_rep, self.gamma_features)).view(
            B * m, G, self.action_dim
        )

        # The critic only scores the candidates: no gradient into its weights,
        # and none into the encoder through its state input.
        for param in value_head.parameters():
            param.requires_grad_(False)
        q = value_head.gamma_values(state_rep.detach(), action).view(B, m, G)
        for param in value_head.parameters():
            param.requires_grad_(True)

        # Only the winning candidate receives the ascent gradient.
        best = q.argmax(dim=1, keepdim=True)
        best_q = q.gather(1, best)
        actor_loss = -best_q.mean()

        # Mean pairwise L2 distance between the candidates of the same state and gamma.
        candidates = action.detach().view(B, m, G, -1).transpose(1, 2)  # (B, G, m, D)
        pairwise = torch.cdist(candidates, candidates)  # (B, G, m, m), zero diagonal
        action_gap = pairwise.sum(dim=(2, 3)) / (m * (m - 1))

        info_dict = {
            "actor_loss": actor_loss.item(),
            "dacer_loss": 0.0,
            "policy_q": best_q.mean().item(),
            "imle_q_gap": (q.max(dim=1).values - q.min(dim=1).values).mean().item(),
            "imle_action_gap": action_gap.mean().item(),
        }
        return actor_loss, info_dict


class TanhPolicy(nn.Module):
    """A squashed Gaussian over the action: a = tanh(u), u ~ N(mu(s), sigma(s)).

    The head every residual policy is built on (EXPO, arXiv:2506.16560): one
    distribution the entropy of which is a number the loss can hold to a target,
    rather than a sampler whose spread is whatever the denoiser happens to have.

    The temperature that balances that entropy against the critic is a dual
    variable, stepped here rather than left to the actor's optimizer: it is one
    scalar whose gradient is known in closed form, and at the actor's own
    learning rate it would take hundreds of thousands of steps to cross the
    range it has to cross, which leaves the loss maximizing entropy for as long
    as the critic is still near zero.

    方策は割引率ごとに持つ（AMAGO の multi-gamma）。割引率の特徴を入力につないで条件づけ、
    割引率 g の方策は割引率 g の価値を最大にするように学習する。温度は全割引率で1つ。環境で
    実行するのは主の割引率の方策の行動だけ。
    """

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0
    # Wide enough for any critic scale a run reaches, tight enough that a
    # transient cannot send the entropy term out of reach of the Q term.
    LOG_TEMPERATURE_MIN = -10.0
    LOG_TEMPERATURE_MAX = 2.0

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        token_dim: int,
        hidden_dim: int,
        block_num: int,
        sparsity: float,
        target_entropy_scale: float,
        init_temperature: float,
        temperature_lr: float,
        detach_state: bool,
        gamma: float,
        multi_gammas: list[float],
    ) -> None:
        super().__init__()
        # 状態をトークンとしては読まないので、トークンの幅は使わない
        del token_dim
        self.detach_state = detach_state
        assert init_temperature > 0.0, init_temperature
        assert temperature_lr > 0.0, temperature_lr
        self.action_dim = action_dim
        total_action_dim = action_dim
        self.fc_in = nn.Linear(state_dim + GAMMA_EMBEDDING_SIZE, hidden_dim)
        self.fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc_mean = nn.Linear(hidden_dim, total_action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, total_action_dim)
        # The mean starts at the middle of the action range, as the other heads'
        # output layers do; the spread starts at one, which tanh makes close to
        # uniform over the range.
        nn.init.zeros_(self.fc_mean.weight)
        nn.init.zeros_(self.fc_mean.bias)
        nn.init.zeros_(self.fc_log_std.bias)
        # Entropy of an action this wide that the temperature holds the policy to.
        self.target_entropy = -target_entropy_scale * total_action_dim
        self.temperature_lr = temperature_lr
        # A buffer, not a parameter: it is stepped by :meth:`compute_actor_loss`
        # and must stay out of the actor's optimizer, while still being saved.
        self.register_buffer("log_temperature", torch.tensor(math.log(init_temperature)))
        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )
        self.register_buffer(
            "gamma_features", _gamma_features(GAMMA_EMBEDDING_SIZE, gamma, multi_gammas)
        )
        self.num_gammas = self.gamma_features.size(0)

    def forward(self, state: torch.Tensor, gamma_feature: torch.Tensor) -> HeadOutput:
        """The distribution's parameters for ``state`` under the policy of the
        gamma ``gamma_feature`` (B, GAMMA_EMBEDDING_SIZE), ``output`` being the
        mean and the log standard deviation on one axis."""
        x = self.norm(self.fc_mid(self.fc_in(torch.cat([state, gamma_feature], dim=-1))))
        log_std = self.fc_log_std(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return HeadOutput(output=torch.cat([self.fc_mean(x), log_std], dim=-1), activation=x)

    def _sample(
        self, state: torch.Tensor, gamma_feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One reparameterized draw: the action and its log density under
        the squashed Gaussian, with the Gaussian's mean and log standard
        deviation it was drawn from.

        The correction for the squash is written as ``2 * (log 2 - u -
        softplus(-2u))`` rather than ``log(1 - tanh(u)^2)``, which underflows
        where the sample saturates.
        """
        head_out = self.forward(state, gamma_feature)
        mean, log_std = head_out.output.chunk(2, dim=-1)
        noise = torch.randn_like(mean)
        pre_tanh = mean + noise * log_std.exp()
        log_prob = (
            -0.5 * noise.pow(2)
            - log_std
            - 0.5 * math.log(2 * math.pi)
            - 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        ).sum(dim=-1)
        action = torch.tanh(pre_tanh).view(-1, self.action_dim)
        return action, log_prob, mean, log_std

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """主の割引率の方策の行動 (B, action_dim)。環境で実行するのはこれ。"""
        action, _, _, _ = self._sample(x, self.gamma_features[-1:].expand(x.size(0), -1))
        return action

    def get_gamma_actions(self, x: torch.Tensor) -> torch.Tensor:
        """割引率ごとの方策の行動 (B, num_gammas, action_dim)。最後が主の割引率。"""
        action, _, _, _ = self._sample(*_repeat_over_gammas(x, self.gamma_features))
        return action.view(x.size(0), self.num_gammas, self.action_dim)

    def compute_actor_loss(
        self,
        state: torch.Tensor,
        taken_action: torch.Tensor,
        *,
        value_head,
    ) -> tuple[torch.Tensor, dict]:
        """Soft actor-critic's actor loss, and the temperature's own step.

        The temperature is stepped first, in place: its loss has no parameters
        in common with the actor's, so nothing is lost by taking it outside the
        backward, and the step is gradient descent on ``-log_temperature *
        (log_prob + target_entropy)`` -- down while the policy is more random
        than the target asks, up while it is less.
        """
        del taken_action
        if self.detach_state:
            state = state.detach()
        # 行は状態ごとに割引率の並びで続く (B * G, ...)
        B, G = state.size(0), self.num_gammas
        action, log_prob, mean, log_std = self._sample(
            *_repeat_over_gammas(state, self.gamma_features)
        )

        with torch.no_grad():
            self.log_temperature += self.temperature_lr * (log_prob.mean() + self.target_entropy)
            self.log_temperature.clamp_(self.LOG_TEMPERATURE_MIN, self.LOG_TEMPERATURE_MAX)
        temperature = self.log_temperature.exp()

        for param in value_head.parameters():
            param.requires_grad_(False)
        policy_q = value_head.gamma_values(state, action.view(B, G, self.action_dim)).reshape(-1)
        for param in value_head.parameters():
            param.requires_grad_(True)

        actor_loss = (temperature * log_prob - policy_q).mean()
        info_dict = {
            "actor_loss": actor_loss.item(),
            "dacer_loss": 0.0,
            "policy_q": policy_q.mean().item(),
            "entropy": -log_prob.mean().item(),
            "temperature": temperature.item(),
            **self._distribution_report(mean.detach(), log_std.detach()),
        }
        return actor_loss, info_dict

    def _distribution_report(self, mean: torch.Tensor, log_std: torch.Tensor) -> dict:
        """方策の分布が行動空間のどこに寄っているか。行動の成分ごとの tanh(平均) のバッチ
        平均、標準偏差、tanh(平均) が端（絶対値 0.95 超）に張り付いている割合。"""
        squashed_mean = torch.tanh(mean).view(-1, self.action_dim)
        std = log_std.exp().view(-1, self.action_dim)
        return {
            **{
                f"policy_mean_{i}": squashed_mean[..., i].mean().item()
                for i in range(self.action_dim)
            },
            **{f"policy_std_{i}": std[..., i].mean().item() for i in range(self.action_dim)},
            "policy_saturation": (squashed_mean.abs() > 0.95).float().mean().item(),
        }
