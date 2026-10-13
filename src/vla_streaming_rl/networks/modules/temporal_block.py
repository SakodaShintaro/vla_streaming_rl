# SPDX-License-Identifier: MIT
import math

import torch
import torch.nn as nn
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
    Temporal processing layer using GatedDeltaNet

    Args:
        hidden_dim: Hidden dimension
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # 幅が小さいので、ヘッドは1つにして幅をそのままヘッドの次元にする。
        # カーネルはヘッドの次元が小さすぎると効率が悪い
        head_dim = hidden_dim
        head_v_dim = 2 * head_dim
        # layer_idx を渡さないと、fla はキャッシュを読み書きしない
        self.gdn = GatedDeltaNet(
            hidden_size=hidden_dim,
            expand_v=2,
            head_dim=head_dim,
            num_heads=1,
            mode="chunk",
            layer_idx=0,
        )

        # recurrent_state: [B, num_heads, head_v_dim, head_dim]（fla は v を先に置く）
        self.recurrent_state_shape = (1, head_v_dim, head_dim)
        self.recurrent_state_size = math.prod(self.recurrent_state_shape)
        # conv_state: q, k, v の ShortConvolution の状態 [B, dim, conv_size]
        conv_size = 4
        self.conv_state_shapes = [
            (head_dim, conv_size),
            (head_dim, conv_size),
            (head_v_dim, conv_size),
        ]
        self.conv_state_sizes = [math.prod(shape) for shape in self.conv_state_shapes]
        self.cache_size = self.recurrent_state_size + sum(self.conv_state_sizes)

    def get_rnn_state_size(self) -> int:
        """Return GatedDeltaNet cache size"""
        return self.cache_size

    def _flatten_cache(self, cache: FLACache) -> torch.Tensor:
        """FLACache を [1, B, cache_size] にまとめる"""
        state = cache[0]
        batch_size = state["recurrent_state"].shape[0]
        parts = [state["recurrent_state"], *state["conv_state"]]
        flat = torch.cat([part.reshape(batch_size, -1).float() for part in parts], dim=-1)
        return flat.unsqueeze(0)

    def _unflatten_cache(self, cache_flat: torch.Tensor, dtype: torch.dtype) -> FLACache:
        """[1, B, cache_size] を FLACache に戻す。0 の状態はキャッシュのない状態と同じ計算になる"""
        cache_flat = cache_flat.squeeze(0)
        batch_size = cache_flat.shape[0]
        sizes = [self.recurrent_state_size, *self.conv_state_sizes]
        shapes = [self.recurrent_state_shape, *self.conv_state_shapes]
        recurrent_state, *conv_state = [
            part.reshape(batch_size, *shape).to(dtype)
            for part, shape in zip(cache_flat.split(sizes, dim=-1), shapes)
        ]
        cache = FLACache()
        cache.update(recurrent_state=recurrent_state, conv_state=tuple(conv_state), layer_idx=0)
        return cache

    def forward(
        self,
        x: torch.Tensor,  # [B, T, C]
        rnn_state: torch.Tensor,  # [1, B, cache_size]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self._unflatten_cache(rnn_state, x.dtype)
        output, _, cache = self.gdn(x, past_key_values=cache, use_cache=True)
        return output, self._flatten_cache(cache)


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


if __name__ == "__main__":
    # Verification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    B, T, C = 2, 10, 32
    x = torch.randn(B, T, C, device=device)

    hidden_dim = C
    n_head = 4
    max_position_embeddings = 512
    layer_scale_init = 1e-4

    # CausalTransformerBlock
    print("\n=== CausalTransformerBlock ===")
    block = CausalTransformerBlock(
        hidden_dim, n_head, max_position_embeddings, layer_scale_init
    ).to(device)
    rnn_state = torch.zeros(1, B, block.get_rnn_state_size(), device=device)
    out, new_rnn_state = block(x, rnn_state)
    print(f"output shape: {out.shape}")
    print(f"new_rnn_state shape: {new_rnn_state.shape}")
    print(f"rnn_state_size: {block.get_rnn_state_size()}")

    # GRUBlock
    print("\n=== GRUBlock ===")
    gru_block = GRUBlock(hidden_dim, layer_scale_init).to(device)
    rnn_state = torch.zeros(1, B, gru_block.get_rnn_state_size(), device=device)
    out, new_rnn_state = gru_block(x, rnn_state)
    print(f"output shape: {out.shape}")
    print(f"new_rnn_state shape: {new_rnn_state.shape}")
    print(f"rnn_state_size: {gru_block.get_rnn_state_size()}")

    # MambaBlock
    print("\n=== MambaBlock ===")
    mamba_block = MambaBlock(hidden_dim, layer_scale_init).to(device)
    rnn_state = torch.zeros(1, B, mamba_block.get_rnn_state_size(), device=device)
    out, new_rnn_state = mamba_block(x, rnn_state)
    print(f"output shape: {out.shape}")
    print(f"new_rnn_state shape: {new_rnn_state.shape}")
    print(f"rnn_state_size: {mamba_block.get_rnn_state_size()}")

    # GdnBlock
    print("\n=== GdnBlock ===")
    gdn_block = GdnBlock(hidden_dim, layer_scale_init).to(device)
    rnn_state = torch.zeros(1, B, gdn_block.get_rnn_state_size(), device=device)
    out, new_rnn_state = gdn_block(x, rnn_state)
    print(f"output shape: {out.shape}")
    print(f"new_rnn_state shape: {new_rnn_state.shape}")
    print(f"rnn_state_size: {gdn_block.get_rnn_state_size()}")

    # IdentityBlock
    print("\n=== IdentityBlock ===")
    identity_block = IdentityBlock(hidden_dim, layer_scale_init).to(device)
    rnn_state = torch.zeros(1, B, identity_block.get_rnn_state_size(), device=device)
    out, new_rnn_state = identity_block(x, rnn_state)
    print(f"output shape: {out.shape}")
    print(f"new_rnn_state shape: {new_rnn_state.shape}")
    print(f"rnn_state_size: {identity_block.get_rnn_state_size()}")

    print("\nAll tests passed!")
