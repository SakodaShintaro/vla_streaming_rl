# ref. https://github.com/Kevin-thu/Epona/blob/main/models/stt.py

import torch
import torch.nn as nn
from einops import rearrange

from .self_attention import SpatialTransformerBlock
from .temporal_block import CausalTransformerBlock, GdnBlock, GRUBlock, IdentityBlock, MambaBlock


class SpatialTemporalBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_head: int,
        temporal_model_type: str,
        tempo_len: int,
        space_len: int,
    ) -> None:
        super().__init__()

        if temporal_model_type == "mamba":
            self.tempo_block = MambaBlock(hidden_dim)
        elif temporal_model_type == "transformer":
            self.tempo_block = CausalTransformerBlock(hidden_dim, n_head, tempo_len)
        elif temporal_model_type == "gru":
            self.tempo_block = GRUBlock(hidden_dim)
        elif temporal_model_type == "gdn":
            self.tempo_block = GdnBlock(hidden_dim)
        elif temporal_model_type == "identity":
            self.tempo_block = IdentityBlock(hidden_dim)
        else:
            raise ValueError(f"Unknown temporal_model_type: {temporal_model_type}")
        self.space_block = SpatialTransformerBlock(hidden_dim, n_head, space_len)

    def forward(
        self, x: torch.Tensor, rnn_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, S, C]
            rnn_state: [1, B*S, C] temporal block state

        Returns:
            x: [B, T, S, C]
            rnn_state: [1, B*S, C] updated temporal block state
        """
        b, t, s, c = x.shape
        x = rearrange(x, "b t s c -> (b s) t c")
        x, rnn_state = self.tempo_block(x, rnn_state)
        x = rearrange(x, "(b s) t c -> (b t) s c", b=b, s=s, t=t)
        x = self.space_block(x)
        x = rearrange(x, "(b t) s c -> b t s c", b=b, t=t)
        return x, rnn_state


class SpatialTemporalTransformer(nn.Module):
    def __init__(
        self,
        n_layer: int,
        space_len: int,
        tempo_len: int,
        hidden_dim: int,
        n_head: int,
        temporal_model_type: str,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.space_len = space_len
        self.temporal_model_type = temporal_model_type

        # Removed tempo_emb, keep only space_emb
        self.space_emb = nn.Parameter(torch.zeros(1, 1, space_len, self.hidden_dim))
        nn.init.normal_(self.space_emb.data, mean=0, std=0.02)

        self.spatial_temporal_blocks = nn.ModuleList(
            [
                SpatialTemporalBlock(
                    hidden_dim,
                    n_head,
                    temporal_model_type,
                    tempo_len,
                    space_len,
                )
                for _ in range(self.n_layer)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self, x: torch.Tensor, rnn_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the SpatialTemporalTransformer.

        Args:
            x: [B, T, S, C]
            rnn_state: [1, B*S, state_size, n_layer]
                state_size depends on temporal_model_type (C for GRU/Transformer/Mamba, cache_size for GDN)

        Returns:
            out: [B, T, S, C]
            rnn_state: [1, B*S, state_size, n_layer]
        """
        B, T, S, C = x.shape

        space_emb = torch.repeat_interleave(self.space_emb, T, dim=1)

        out = x + space_emb

        new_layer_states = []
        for i in range(self.n_layer):
            out, new_state = self.spatial_temporal_blocks[i](out, rnn_state[:, :, :, i])
            new_layer_states.append(new_state)

        # Combine each layer's state: n_layer of [1, B*S, state_size] -> [1, B*S, state_size, n_layer]
        rnn_state = torch.stack(new_layer_states, dim=-1)

        return out, rnn_state
