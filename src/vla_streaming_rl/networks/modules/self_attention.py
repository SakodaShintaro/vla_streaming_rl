# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb


def get_fourier_embeds_from_coordinates(embed_dim: int, coords: torch.Tensor) -> torch.Tensor:
    """
    Convert continuous coordinates to Fourier positional embeddings

    Converts continuous values like actions to high-dimensional vectors for Transformer processing.
    Creates rich representations by combining sin/cos functions of different frequencies.

    Args:
        embed_dim: Embedding dimension (must be even)
        coords: Coordinate tensor [B, T, coord_dim] or [B, T]

    Returns:
        torch.Tensor: Fourier embedding of shape [B, T, coord_dim, embed_dim]
    """
    device = coords.device
    dtype = coords.dtype

    # Expand 2D tensor to 3D [B, T] -> [B, T, 1]
    if coords.dim() == 2:
        coords = coords.unsqueeze(-1)

    batch_size, seq_len, coord_dim = coords.shape

    # Generate different frequencies (same principle as Transformer positional embedding)
    half_dim = embed_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype, device=device) * -emb)

    # Expand dimensions for broadcasting [half_dim] -> [1, 1, 1, half_dim]
    emb = emb.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    coords = coords.unsqueeze(-1)  # [B, T, coord_dim] -> [B, T, coord_dim, 1]

    # Multiply each coordinate value by each frequency [B, T, coord_dim, half_dim]
    emb = coords * emb

    # Create embedding vector by combining sin/cos
    # [sin(coord*freq1), cos(coord*freq1), sin(coord*freq2), cos(coord*freq2), ...]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    emb = emb.view(batch_size, seq_len, coord_dim, embed_dim)

    return emb


class SelfAttention(nn.Module):
    """
    Self-Attention module

    Args:
        hidden_dim: Hidden dimension
        n_head: Number of attention heads
        max_position_embeddings: Maximum position embeddings
        use_rope: Whether to use RoPE
    """

    def __init__(
        self, hidden_dim: int, n_head: int, max_position_embeddings: int, use_rope: bool
    ) -> None:
        super().__init__()
        assert hidden_dim % n_head == 0
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.res_drop_prob = nn.Dropout(0.0)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.n_head = n_head
        self.head_dim = hidden_dim // n_head
        self.qk_norm = True
        self.use_rope = use_rope

        if self.qk_norm:
            self.q_norm = nn.LayerNorm(hidden_dim)
            self.k_norm = nn.LayerNorm(hidden_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

        # Initialize LlamaRotaryEmbedding only when using RoPE
        if self.use_rope:
            rope_config = LlamaConfig(
                hidden_size=hidden_dim,
                num_attention_heads=n_head,
                max_position_embeddings=max_position_embeddings,
                head_dim=self.head_dim,
            )
            self.rotary_emb = LlamaRotaryEmbedding(rope_config)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        B, T, C = x.size()

        k = self.key(x)
        q = self.query(x)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE only when use_rope is True
        if self.use_rope:
            position_ids = torch.arange(T, device=x.device).unsqueeze(0)
            cos, sin = self.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if attn_mask is not None:
            attn_mask = attn_mask.to(q.dtype)
        y = (
            F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        y = self.res_drop_prob(self.proj(y))
        return y


class SpatialTransformerBlock(nn.Module):
    """
    Spatial Transformer block (no RoPE, no mask)

    Args:
        hidden_dim: Hidden dimension
        n_head: Number of attention heads
        max_position_embeddings: Maximum position embeddings
    """

    def __init__(self, hidden_dim: int, n_head: int, max_position_embeddings: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.attn = SelfAttention(
            hidden_dim,
            n_head,
            max_position_embeddings,
            use_rope=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim, bias=False),
            nn.Dropout(0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=None)
        x = x + self.mlp(self.ln2(x))
        return x
