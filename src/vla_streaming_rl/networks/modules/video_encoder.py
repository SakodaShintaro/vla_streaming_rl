# SPDX-License-Identifier: MIT
import math

import torch
from torch import nn
from torch.nn import functional as F


def _temporal_sinusoidal_pos_emb(
    num_frames: int, dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Fixed sinusoidal positional embedding for temporal positions. Returns (num_frames, dim)."""
    half = dim // 2
    positions = torch.arange(num_frames, device=device, dtype=torch.float32)
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32) * -(math.log(10000.0) / half)
    )
    angles = positions[:, None] * freqs[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).to(dtype)


def _temporal_causal_attention(
    attn_module: nn.Module,
    hidden_states: torch.Tensor,
    batch_size: int,
    seq_len: int,
    num_patches: int,
    temporal_pos_emb: torch.Tensor,
) -> torch.Tensor:
    """Compute causal temporal attention across frames for same-patch positions.

    Uses the same QKV weights as the spatial attention (no new parameters).
    Each batch element is processed independently (no cross-batch attention).

    Args:
        attn_module: Qwen3_5VisionAttention block (provides qkv, proj, scaling)
        hidden_states: (B * T * num_patches, hidden_dim)
        batch_size: B
        seq_len: T (frames per batch element)
        num_patches: n (spatial patches per frame)
        temporal_pos_emb: (T, hidden_dim) sinusoidal

    Returns:
        output: (B * T * num_patches, hidden_dim)
    """
    hidden_dim = hidden_states.shape[-1]
    num_heads = attn_module.num_heads
    head_dim = hidden_dim // num_heads

    # (B*T*n, d) -> (B, T, n, d) -> (B, n, T, d)
    x = hidden_states.view(batch_size, seq_len, num_patches, hidden_dim)
    x = x.transpose(1, 2)  # (B, n, T, d)

    # Add temporal position embedding: (T, d) broadcast to (B, n, T, d)
    x = x + temporal_pos_emb[None, None, :, :]

    # QKV projection
    qkv = attn_module.qkv(x.reshape(-1, hidden_dim))  # (B*n*T, 3*d)
    qkv = qkv.view(batch_size, num_patches, seq_len, 3, num_heads, head_dim)
    # -> (3, B*n, heads, T, head_dim)
    qkv = qkv.permute(3, 0, 1, 4, 2, 5).reshape(
        3, batch_size * num_patches, num_heads, seq_len, head_dim
    )
    q, k, v = qkv.unbind(0)

    # Causal mask: frame t attends to frames 0..t
    causal_mask = torch.triu(
        torch.full(
            (seq_len, seq_len),
            float("-inf"),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ),
        diagonal=1,
    )

    attn_out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=causal_mask, scale=attn_module.scaling
    )  # (B*n, heads, T, head_dim)

    # Output projection
    attn_out = attn_out.permute(0, 2, 1, 3).reshape(batch_size * num_patches * seq_len, hidden_dim)
    attn_out = attn_module.proj(attn_out)

    # (B*n*T, d) -> (B, n, T, d) -> (B, T, n, d) -> (B*T*n, d)
    attn_out = attn_out.view(batch_size, num_patches, seq_len, hidden_dim)
    attn_out = attn_out.transpose(1, 2).reshape(batch_size * seq_len * num_patches, hidden_dim)
    return attn_out


class VideoEncoder(nn.Module):
    """MEM-style video encoder that reuses Qwen3.5's ViT weights.

    c.f. https://www.pi.website/research/memory
    - Runs patch_embed + pos_embed on all frames
    - Loops through ViT blocks; every 4th layer adds causal temporal attention
    - After all blocks, drops past frames and keeps only last-frame patches
    - Runs PatchMerger on last-frame patches

    No new learnable parameters. K=1 produces identical output to the original ViT.
    """

    def forward(
        self,
        visual: nn.Module,
        all_pixel_values: torch.Tensor,
        all_image_grid_thw: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Run MEM-style video encoding and return last-frame embeddings.

        Args:
            visual: The Qwen3.5 VisionModel (model.model.visual)
            all_pixel_values: (B*T * num_patches_raw, patch_dim) - all frames
            all_image_grid_thw: (B*T, 3) - grid info for each frame
            batch_size: B
            seq_len: T (number of frames per batch element)

        Returns:
            image_embeds: (B * merged_tokens_per_image, llm_hidden_dim)
        """
        all_pixel_values = all_pixel_values.type(visual.dtype)
        num_images = batch_size * seq_len

        # --- Patch embed + position embed (same as original) ---
        hidden_states = visual.patch_embed(all_pixel_values)
        pos_embeds = visual.fast_pos_embed_interpolate(all_image_grid_thw)
        hidden_states = hidden_states + pos_embeds

        # --- Rotary position embedding for spatial attention ---
        rotary_pos_emb = visual.rot_pos_emb(all_image_grid_thw)
        total_tokens, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(total_tokens, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(total_tokens, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # --- cu_seqlens for spatial attention (each frame is independent) ---
        patches_per_image = (all_image_grid_thw[:, 1] * all_image_grid_thw[:, 2]).tolist()
        cu_seqlens = torch.zeros(num_images + 1, dtype=torch.int32, device=hidden_states.device)
        for i, n in enumerate(patches_per_image):
            cu_seqlens[i + 1] = cu_seqlens[i] + n

        # --- Temporal position embedding (fixed sinusoidal, no learnable params) ---
        hidden_dim = hidden_states.shape[-1]
        temporal_pos_emb = _temporal_sinusoidal_pos_emb(
            seq_len, hidden_dim, hidden_states.device, hidden_states.dtype
        )

        # Assume all images have the same spatial resolution
        num_patches = patches_per_image[0]

        # --- ViT blocks with temporal attention every 4th layer ---
        temporal_interval = 4
        for layer_idx, blk in enumerate(visual.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

            # Every 4th layer (0-indexed: 3, 7, 11, ...): add causal temporal attention
            use_temporal = (layer_idx + 1) % temporal_interval == 0 and seq_len > 1
            if use_temporal:
                temporal_out = _temporal_causal_attention(
                    blk.attn,
                    hidden_states,
                    batch_size,
                    seq_len,
                    num_patches,
                    temporal_pos_emb,
                )
                hidden_states = hidden_states + temporal_out

        # --- Extract last frame per batch element ---
        all_frame_hidden = hidden_states.view(num_images, num_patches, hidden_dim)
        last_frame_indices = [b * seq_len + (seq_len - 1) for b in range(batch_size)]
        last_frame_hidden = all_frame_hidden[last_frame_indices]  # (B, num_patches, hidden_dim)
        last_frame_hidden = last_frame_hidden.reshape(batch_size * num_patches, hidden_dim)

        # --- PatchMerger ---
        return visual.merger(last_frame_hidden)
