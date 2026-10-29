# SPDX-License-Identifier: MIT
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.layers import GatedDeltaNet
from fla.models.utils import FLACache
from mamba_ssm.modules.mamba2 import Mamba2

from .self_attention import SelfAttention


class BaseTemporalBlock(nn.Module):
    """
    Generic block for temporal processing + MLP

    Args:
        hidden_dim: Hidden dimension
        temporal_layer: Layer for temporal processing
        layer_scale_init: LayerScale gamma initial value
    """

    def __init__(self, hidden_dim: int, temporal_layer: nn.Module, layer_scale_init: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Block 1: Norm + TemporalLayer + Residual
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.temporal = temporal_layer

        # Block 2: Norm + MLP + Residual
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim, bias=False),
            nn.Dropout(0.0),
        )
        self.temporal_gamma = nn.Parameter(torch.full((hidden_dim,), layer_scale_init))
        self.mlp_gamma = nn.Parameter(torch.full((hidden_dim,), layer_scale_init))

    def get_rnn_state_size(self) -> int:
        """Return size of rnn_state"""
        return self.temporal.get_rnn_state_size()

    def forward(
        self,
        x: torch.Tensor,  # [B, T, C]
        rnn_state: torch.Tensor,  # [1, B, C]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Block 1: Norm + TemporalLayer + Residual
        temporal_output, new_rnn_state = self.temporal(self.temporal_norm(x), rnn_state)
        x = x + self.temporal_gamma * temporal_output

        # Block 2: Norm + MLP + Residual
        x = x + self.mlp_gamma * self.mlp(self.mlp_norm(x))

        return x, new_rnn_state


class CausalAttentionLayer(nn.Module):
    """
    Temporal processing layer using CausalAttention

    Args:
        hidden_dim: Hidden dimension
        n_head: Number of attention heads
        max_position_embeddings: Maximum position embeddings
    """

    def __init__(self, hidden_dim: int, n_head: int, max_position_embeddings: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn = SelfAttention(hidden_dim, n_head, max_position_embeddings, use_rope=True)

        # Register causal mask
        matrix = torch.tril(torch.ones(max_position_embeddings, max_position_embeddings))
        causal_mask = torch.where(matrix == 0, float("-inf"), matrix)
        causal_mask = torch.where(matrix == 1, 0, causal_mask)
        self.register_buffer("causal_mask", causal_mask.contiguous())

    def get_rnn_state_size(self) -> int:
        """Return hidden_dim since state is not used"""
        return self.hidden_dim

    def forward(
        self,
        x: torch.Tensor,  # [B, T, C]
        rnn_state: torch.Tensor,  # [1, B, C]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.size()
        current_mask = self.causal_mask[:T, :T]
        output = self.attn(x, attn_mask=current_mask)
        return output, rnn_state


class MambaLayer(nn.Module):
    """
    Temporal processing layer using Mamba2

    Args:
        hidden_dim: Hidden dimension
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # headdim is the per-head SSM dimension and must divide d_inner
        # (= 2 * hidden_dim with Mamba2's default expand=2). 64 is Mamba2's
        # standard head size, giving nheads = 2 * hidden_dim / 64.
        self.mamba = Mamba2(d_model=hidden_dim, headdim=64)

    def get_rnn_state_size(self) -> int:
        """Return hidden_dim since state is not used"""
        return self.hidden_dim

    def forward(
        self,
        x: torch.Tensor,  # [B, T, C]
        rnn_state: torch.Tensor,  # [1, B, C]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.mamba(x)
        return output, rnn_state


class GRULayer(nn.Module):
    """
    Temporal processing layer using GRU

    Args:
        hidden_dim: Hidden dimension
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)

    def get_rnn_state_size(self) -> int:
        """Return GRU state size"""
        return self.hidden_dim

    def forward(
        self,
        x: torch.Tensor,  # [B, T, C]
        rnn_state: torch.Tensor,  # [1, B, C]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Make contiguous in case of non-contiguous tensor from repeat/reshape
        rnn_state = rnn_state.contiguous()
        output, new_rnn_state = self.gru(x, rnn_state)
        return output, new_rnn_state


class GatedDeltaNetLayer(nn.Module):
    """
    Temporal processing layer using Gated DeltaNet (single head), in plain PyTorch.

    fla の GatedDeltaNet と同じ計算を、torch.compile が丸ごと追える形で書いたもの。
    窓全体を1つのチャンクとして、デルタ則の再帰を下三角の連立方程式1回で解く。

    状態 S [K, V] の再帰（ステップ t）:
        S <- exp(g_t) * S
        u_t = beta_t * (v_t - S^T k_t)
        S <- S + k_t u_t^T
        o_t = S^T q_t

    Args:
        hidden_dim: Hidden dimension
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.key_dim = hidden_dim
        self.value_dim = 2 * hidden_dim
        qkv_dim = 2 * self.key_dim + self.value_dim
        self.conv_size = 4
        self.qkv_proj = nn.Linear(hidden_dim, qkv_dim, bias=False)
        # q, k, v ごとの因果的な depthwise 畳み込み
        self.qkv_conv = nn.Conv1d(qkv_dim, qkv_dim, self.conv_size, groups=qkv_dim, bias=False)
        self.a_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.b_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.g_proj = nn.Linear(hidden_dim, self.value_dim, bias=False)
        self.o_norm = nn.RMSNorm(self.value_dim, eps=1e-5)
        self.o_proj = nn.Linear(self.value_dim, hidden_dim, bias=False)
        # 減衰の初期化は fla に合わせる
        self.A_log = nn.Parameter(torch.log(torch.empty(1).uniform_(0, 16)))
        dt = torch.exp(torch.rand(1) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        dt = torch.clamp(dt, min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        # 状態: S [K, V] と、畳み込みが次の窓で読む直前の入力 [conv_size - 1, qkv_dim]
        self.recurrent_state_size = self.key_dim * self.value_dim
        self.conv_state_shape = (self.conv_size - 1, qkv_dim)
        self.cache_size = self.recurrent_state_size + math.prod(self.conv_state_shape)

    def get_rnn_state_size(self) -> int:
        """Return Gated DeltaNet state size"""
        return self.cache_size

    def forward(
        self,
        x: torch.Tensor,  # [B, T, C]
        rnn_state: torch.Tensor,  # [1, B, cache_size]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        state = rnn_state.squeeze(0)
        S0 = state[:, : self.recurrent_state_size].view(B, self.key_dim, self.value_dim)
        conv_state = state[:, self.recurrent_state_size :].view(B, *self.conv_state_shape)

        # 前の窓の末尾をつないで因果的に畳み込む
        qkv = torch.cat([conv_state, self.qkv_proj(x)], dim=1)  # [B, T + conv_size - 1, D]
        mixed = F.silu(self.qkv_conv(qkv.transpose(1, 2)).transpose(1, 2))
        q, k, v = mixed.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = F.normalize(q, dim=-1, eps=1e-6) * self.key_dim**-0.5
        k = F.normalize(k, dim=-1, eps=1e-6)
        beta = torch.sigmoid(self.b_proj(x))  # [B, T, 1]
        # 対数の減衰 g_t とその累積 gamma_t
        g = -self.A_log.exp() * F.softplus(self.a_proj(x).squeeze(-1) + self.dt_bias)
        gamma = g.cumsum(dim=1)  # [B, T]

        # decay[t, i] = exp(gamma_t - gamma_i)（i <= t）。それ以外は 0
        ones = torch.ones(T, T, dtype=torch.bool, device=x.device)
        log_decay = (gamma.unsqueeze(2) - gamma.unsqueeze(1)).masked_fill(~ones.tril(), -torch.inf)
        decay = log_decay.exp()  # [B, T, T]

        # (I + diag(beta) A) U = diag(beta) (V - diag(Gamma) K S0)、A は狭義下三角
        start_decay = gamma.exp().unsqueeze(-1)  # [B, T, 1]
        A = decay.masked_fill(~ones.tril(-1), 0.0) * (k @ k.transpose(1, 2))
        rhs = beta * (v - start_decay * (k @ S0))
        L = torch.eye(T, device=x.device) + beta * A
        u = torch.linalg.solve_triangular(L, rhs, upper=False, unitriangular=True)

        o = start_decay * (q @ S0) + (decay * (q @ k.transpose(1, 2))) @ u
        S = gamma[:, -1:].exp().unsqueeze(-1) * S0 + k.transpose(1, 2) @ (
            (gamma[:, -1:] - gamma).exp().unsqueeze(-1) * u
        )
        output = self.o_proj(self.o_norm(o) * F.silu(self.g_proj(x)))

        new_state = torch.cat(
            [S.reshape(B, -1), qkv[:, -(self.conv_size - 1) :].reshape(B, -1)], dim=-1
        )
        return output, new_state.unsqueeze(0)


class ChunkGatedDeltaNetLayer(nn.Module):
    """
    Temporal processing layer using fla's GatedDeltaNet (chunked kernels).

    状態（再帰の行列と短い畳み込みの直前の入力）は平らなテンソルで出し入れする。長い系列も
    チャンクに分けて解くので、トークンを何千も並べた系列を通せる。

    系列ごとに ``start`` の位置から状態を 0 に戻して読み直す。各系列を ``start`` の前と後の
    2本に分けて詰めて通すので、形はいつも同じになる。詰めた長さが変わると Triton の
    カーネルが作り直されて、1回に数十秒かかる。

    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of heads. The head dimension is hidden_dim // num_heads
    """

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, (hidden_dim, num_heads)
        head_dim = hidden_dim // num_heads
        head_v_dim = 2 * head_dim
        # layer_idx を渡さないと、fla はキャッシュを読み書きしない
        self.gdn = GatedDeltaNet(
            hidden_size=hidden_dim,
            expand_v=2,
            head_dim=head_dim,
            num_heads=num_heads,
            mode="chunk",
            layer_idx=0,
        )

        # recurrent_state: [B, num_heads, head_v_dim, head_dim]（fla は v を先に置く）
        self.recurrent_state_shape = (num_heads, head_v_dim, head_dim)
        self.recurrent_state_size = math.prod(self.recurrent_state_shape)
        # conv_state: q, k, v の ShortConvolution の状態 [B, dim, conv_size]
        conv_size = self.gdn.conv_size
        self.conv_state_shapes = [
            (num_heads * head_dim, conv_size),
            (num_heads * head_dim, conv_size),
            (num_heads * head_v_dim, conv_size),
        ]
        self.conv_state_sizes = [math.prod(shape) for shape in self.conv_state_shapes]
        self.cache_size = self.recurrent_state_size + sum(self.conv_state_sizes)

    def get_rnn_state_size(self) -> int:
        """Return GatedDeltaNet cache size"""
        return self.cache_size

    def _flatten_cache(self, cache: FLACache) -> torch.Tensor:
        """FLACache を [B, cache_size] にまとめる"""
        state = cache[0]
        batch_size = state["recurrent_state"].shape[0]
        parts = [state["recurrent_state"], *state["conv_state"]]
        return torch.cat([part.reshape(batch_size, -1).float() for part in parts], dim=-1)

    def _unflatten_cache(self, cache_flat: torch.Tensor, dtype: torch.dtype) -> FLACache:
        """[B, cache_size] を FLACache に戻す。0 の状態はキャッシュのない状態と同じ計算になる"""
        batch_size = cache_flat.shape[0]
        sizes = [self.recurrent_state_size, *self.conv_state_sizes]
        shapes = [self.recurrent_state_shape, *self.conv_state_shapes]
        recurrent_state, *conv_state = [
            part.reshape(batch_size, *shape).to(dtype).contiguous()
            for part, shape in zip(cache_flat.split(sizes, dim=-1), shapes, strict=True)
        ]
        cache = FLACache()
        cache.update(recurrent_state=recurrent_state, conv_state=tuple(conv_state), layer_idx=0)
        return cache

    def forward(
        self,
        x: torch.Tensor,  # [B, T, C]
        rnn_state: torch.Tensor,  # [B, cache_size]
        start: torch.Tensor,  # [B] long
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        # 前の1本は渡した状態から、後の1本は start が 0 なら渡した状態から、そうでなければ
        # 0 から始める
        lengths = torch.stack([start, T - start], dim=1).flatten()
        cu_seqlens = F.pad(lengths.cumsum(dim=0), (1, 0)).to(torch.int32)
        restart = (start == 0).unsqueeze(-1).to(rnn_state.dtype)
        initial_state = torch.stack([rnn_state, rnn_state * restart], dim=1).flatten(0, 1)
        cache = self._unflatten_cache(initial_state, x.dtype)
        output, _, cache = self.gdn(
            x.reshape(1, B * T, C), past_key_values=cache, use_cache=True, cu_seqlens=cu_seqlens
        )
        final_state = self._flatten_cache(cache).view(B, 2, -1)[:, 1]
        return output.view(B, T, -1), final_state


class GatedDeltaNetStream(nn.Module):
    """
    :class:`ChunkGatedDeltaNetLayer` を ``n_layer`` 段重ねた、1本の因果的な系列を読む幹。
    系列ごとに ``start`` の位置から状態を 0 に戻して読み直す。

    各段は Norm -> GatedDeltaNet -> 残差、Norm -> MLP -> 残差。状態は段ごとの平らな状態を
    並べた [B, n_layer, cache_size]。

    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of heads of each GatedDeltaNet
        n_layer: Number of layers
    """

    def __init__(self, hidden_dim: int, num_heads: int, n_layer: int) -> None:
        super().__init__()
        self.temporal_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layer)])
        self.temporals = nn.ModuleList(
            [ChunkGatedDeltaNetLayer(hidden_dim, num_heads) for _ in range(n_layer)]
        )
        self.mlp_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layer)])
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, 4 * hidden_dim, bias=False),
                    nn.GELU(),
                    nn.Linear(4 * hidden_dim, hidden_dim, bias=False),
                )
                for _ in range(n_layer)
            ]
        )
        self.state_shape = (n_layer, self.temporals[0].get_rnn_state_size())

    def forward(
        self,
        x: torch.Tensor,  # [B, T, C]
        rnn_state: torch.Tensor,  # [B, n_layer, cache_size]
        start: torch.Tensor,  # [B] long
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_states = []
        for i, temporal in enumerate(self.temporals):
            temporal_output, new_state = temporal(self.temporal_norms[i](x), rnn_state[:, i], start)
            x = x + temporal_output
            x = x + self.mlps[i](self.mlp_norms[i](x))
            new_states.append(new_state)
        return x, torch.stack(new_states, dim=1)


class IdentityLayer(nn.Module):
    """
    Identity temporal processing layer (for comparison)

    Args:
        hidden_dim: Hidden dimension
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def get_rnn_state_size(self) -> int:
        """Return hidden_dim since state is not used"""
        return self.hidden_dim

    def forward(
        self,
        x: torch.Tensor,  # [B, T, C]
        rnn_state: torch.Tensor,  # [1, B, C]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x, rnn_state


# Aliases for compatibility
def CausalTransformerBlock(
    hidden_dim: int, n_head: int, max_position_embeddings: int, layer_scale_init: float
) -> BaseTemporalBlock:
    """Causal Transformer block (with RoPE, applies causal mask internally)"""
    return BaseTemporalBlock(
        hidden_dim,
        CausalAttentionLayer(hidden_dim, n_head, max_position_embeddings),
        layer_scale_init,
    )


def MambaBlock(hidden_dim: int, layer_scale_init: float) -> BaseTemporalBlock:
    """Mamba state space model block (same structure as Transformer block)"""
    return BaseTemporalBlock(hidden_dim, MambaLayer(hidden_dim), layer_scale_init)


def GRUBlock(hidden_dim: int, layer_scale_init: float) -> BaseTemporalBlock:
    """GRU block (same structure as Transformer block)"""
    return BaseTemporalBlock(hidden_dim, GRULayer(hidden_dim), layer_scale_init)


def GdnBlock(hidden_dim: int, layer_scale_init: float) -> BaseTemporalBlock:
    """GatedDeltaNet block (same structure as Transformer block)"""
    return BaseTemporalBlock(hidden_dim, GatedDeltaNetLayer(hidden_dim), layer_scale_init)


def IdentityBlock(hidden_dim: int, layer_scale_init: float) -> BaseTemporalBlock:
    """Identity block (for comparison)"""
    return BaseTemporalBlock(hidden_dim, IdentityLayer(hidden_dim), layer_scale_init)
