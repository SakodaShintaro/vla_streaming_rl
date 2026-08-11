# SPDX-License-Identifier: MIT
import torch
from torch import nn

from .temporal_block import (
    CausalAttentionLayer,
    GatedDeltaNetLayer,
    GRULayer,
    IdentityLayer,
    MambaLayer,
)

# Same cadence as VideoEncoder: temporal mixing every 4th ViT block.
TEMPORAL_INTERVAL = 4


def _build_temporal_layer(
    hidden_dim: int, n_head: int, tempo_len: int, temporal_model_type: str
) -> nn.Module:
    if temporal_model_type == "gru":
        return GRULayer(hidden_dim)
    if temporal_model_type == "mamba":
        return MambaLayer(hidden_dim)
    if temporal_model_type == "gdn":
        return GatedDeltaNetLayer(hidden_dim)
    if temporal_model_type == "transformer":
        return CausalAttentionLayer(hidden_dim, n_head, tempo_len)
    if temporal_model_type == "identity":
        return IdentityLayer(hidden_dim)
    raise ValueError(f"Unknown temporal_model_type: {temporal_model_type}")


class TemporalAdapter(nn.Module):
    """Norm -> temporal layer -> zero-initialized projection, added as a residual.

    ``out_proj`` starts at zero, so at initialization the adapter contributes
    nothing and the ViT behaves exactly like the stock per-frame encoder.
    """

    def __init__(
        self, hidden_dim: int, n_head: int, tempo_len: int, temporal_model_type: str
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.temporal = _build_temporal_layer(hidden_dim, n_head, tempo_len, temporal_model_type)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def get_rnn_state_size(self) -> int:
        return self.temporal.get_rnn_state_size()

    def forward(
        self, x: torch.Tensor, rnn_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (N, T, C) float32, rnn_state: (1, N, state_size)."""
        temporal_out, new_rnn_state = self.temporal(self.norm(x), rnn_state)
        return self.out_proj(temporal_out), new_rnn_state


class RecurrentVideoEncoder(nn.Module):
    """MEM-style video encoder whose temporal mixing is recurrent instead of attentive.

    Same skeleton as :class:`VideoEncoder` — the VLM's own ViT weights process every
    frame, every 4th block adds a temporal residual across frames at matching patch
    positions, and only the last frame survives into the LLM context. The difference
    is what carries the past: instead of causal attention over the frames inside the
    window, a recurrent layer (GRU / Mamba / GatedDeltaNet) carries ``rnn_state``
    across steps, so history is not bounded by the window.

    Only the temporal path adds parameters; the spatial path is untouched.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_head: int,
        num_patches: int,
        depth: int,
        seq_len: int,
        temporal_model_type: str,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.temporal_layer_indices = [i for i in range(depth) if (i + 1) % TEMPORAL_INTERVAL == 0]
        self.adapters = nn.ModuleList(
            [
                TemporalAdapter(hidden_dim, n_head, seq_len, temporal_model_type)
                for _ in self.temporal_layer_indices
            ]
        )

    def init_state(self) -> torch.Tensor:
        """(1, num_patches, state_size, num_adapters) recurrent state for batch size 1."""
        state_size = self.adapters[0].get_rnn_state_size()
        return torch.zeros(1, self.num_patches, state_size, len(self.adapters))

    def _apply_temporal(
        self,
        adapter: TemporalAdapter,
        hidden_states: torch.Tensor,
        batch_size: int,
        seq_len: int,
        rnn_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden_states: (B*T*num_patches, C); rnn_state: (1, B*num_patches, state_size)."""
        dtype = hidden_states.dtype
        # (B*T*n, C) -> (B, T, n, C) -> (B*n, T, C): one sequence per patch position.
        x = hidden_states.view(batch_size, seq_len, self.num_patches, self.hidden_dim)
        x = x.transpose(1, 2).reshape(batch_size * self.num_patches, seq_len, self.hidden_dim)
        temporal_out, rnn_state = adapter(x.to(torch.float32), rnn_state)
        # back to (B*T*n, C)
        temporal_out = temporal_out.view(
            batch_size, self.num_patches, seq_len, self.hidden_dim
        ).transpose(1, 2)
        temporal_out = temporal_out.reshape(-1, self.hidden_dim).to(dtype)
        return hidden_states + temporal_out, rnn_state

    def forward(
        self,
        visual: nn.Module,
        all_pixel_values: torch.Tensor,
        all_image_grid_thw: torch.Tensor,
        batch_size: int,
        seq_len: int,
        rnn_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode all frames and return last-frame embeddings + updated state.

        Args:
            visual: the VLM vision model (``model.model.visual``)
            all_pixel_values: (B*T * num_patches, patch_dim)
            all_image_grid_thw: (B*T, 3)
            batch_size: B
            seq_len: T
            rnn_state: (B, num_patches, state_size, num_adapters)

        Returns:
            image_embeds: (B * merged_tokens_per_image, llm_hidden_dim)
            rnn_state: (B, num_patches, state_size, num_adapters)
        """
        all_pixel_values = all_pixel_values.type(visual.dtype)
        num_images = batch_size * seq_len

        hidden_states = visual.patch_embed(all_pixel_values)
        hidden_states = hidden_states + visual.fast_pos_embed_interpolate(all_image_grid_thw)

        rotary_pos_emb = visual.rot_pos_emb(all_image_grid_thw)
        total_tokens, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(total_tokens, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(total_tokens, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # cu_seqlens for spatial attention (each frame is independent)
        patches_per_image = (all_image_grid_thw[:, 1] * all_image_grid_thw[:, 2]).tolist()
        cu_seqlens = torch.zeros(num_images + 1, dtype=torch.int32, device=hidden_states.device)
        for i, n in enumerate(patches_per_image):
            cu_seqlens[i + 1] = cu_seqlens[i] + n

        # External (B, num_patches, state_size, num_adapters)
        # -> per-adapter internal (1, B*num_patches, state_size)
        rnn_state_internal = rnn_state.reshape(
            1, batch_size * self.num_patches, -1, len(self.adapters)
        )

        new_layer_states = []
        for layer_idx, blk in enumerate(visual.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_idx not in self.temporal_layer_indices:
                continue
            adapter_idx = self.temporal_layer_indices.index(layer_idx)
            hidden_states, new_state = self._apply_temporal(
                self.adapters[adapter_idx],
                hidden_states,
                batch_size,
                seq_len,
                rnn_state_internal[:, :, :, adapter_idx],
            )
            new_layer_states.append(new_state)

        rnn_state = torch.stack(new_layer_states, dim=-1).reshape(
            batch_size, self.num_patches, -1, len(self.adapters)
        )

        # Keep only the last frame per batch element, then merge patches.
        all_frame_hidden = hidden_states.view(num_images, self.num_patches, self.hidden_dim)
        last_frame_indices = [b * seq_len + (seq_len - 1) for b in range(batch_size)]
        last_frame_hidden = all_frame_hidden[last_frame_indices]
        last_frame_hidden = last_frame_hidden.reshape(
            batch_size * self.num_patches, self.hidden_dim
        )

        return visual.merger(last_frame_hidden), rnn_state
