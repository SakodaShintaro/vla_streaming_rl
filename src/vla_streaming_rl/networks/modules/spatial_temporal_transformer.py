# ref. https://github.com/Kevin-thu/Epona/blob/main/models/stt.py

import torch
import torch.nn as nn
from einops import rearrange

from .self_attention import SpatialTransformerBlock
from .temporal_block import CausalTransformerBlock, GdnBlock, GRUBlock, IdentityBlock, MambaBlock


def build_temporal_block(
    temporal_model_type: str, hidden_dim: int, n_head: int, tempo_len: int, layer_scale_init: float
) -> nn.Module:
    """register にかける時間方向のブロック。"""
    assert temporal_model_type in ("mamba", "transformer", "gru", "gdn", "identity"), (
        temporal_model_type
    )
    if temporal_model_type == "mamba":
        return MambaBlock(hidden_dim, layer_scale_init)
    if temporal_model_type == "transformer":
        return CausalTransformerBlock(hidden_dim, n_head, tempo_len, layer_scale_init)
    if temporal_model_type == "gru":
        return GRUBlock(hidden_dim, layer_scale_init)
    if temporal_model_type == "gdn":
        return GdnBlock(hidden_dim, layer_scale_init)
    return IdentityBlock(hidden_dim, layer_scale_init)


class SpatialTemporalTransformer(nn.Module):
    """空間方向のブロックを全部通して各ステップの全トークンを混ぜ切ってから、空間の軸の
    末尾の register_num 個のトークン（register）だけに時間方向のブロックを重ねる。"""

    def __init__(
        self,
        spatial_block_num: int,
        temporal_block_num: int,
        space_len: int,
        tempo_len: int,
        hidden_dim: int,
        n_head: int,
        temporal_model_type: str,
        register_num: int,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temporal_block_num = temporal_block_num
        self.space_len = space_len
        self.register_num = register_num
        self.temporal_model_type = temporal_model_type

        # Removed tempo_emb, keep only space_emb
        self.space_emb = nn.Parameter(torch.zeros(1, 1, space_len, self.hidden_dim))
        nn.init.normal_(self.space_emb.data, mean=0, std=0.02)

        self.spatial_blocks = nn.ModuleList(
            [
                SpatialTransformerBlock(hidden_dim, n_head, space_len, layer_scale_init)
                for _ in range(spatial_block_num)
            ]
        )
        self.temporal_blocks = nn.ModuleList(
            [
                build_temporal_block(
                    temporal_model_type, hidden_dim, n_head, tempo_len, layer_scale_init
                )
                for _ in range(temporal_block_num)
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
            rnn_state: [1, B*R, state_size, temporal_block_num], R being the registers
                state_size depends on temporal_model_type (C for GRU/Transformer/Mamba, cache_size for GDN)

        Returns:
            out: [B, T, S, C]
            rnn_state: [1, B*R, state_size, temporal_block_num]
        """
        B, T, S, C = x.shape
        R = self.register_num

        space_emb = torch.repeat_interleave(self.space_emb, T, dim=1)
        out = rearrange(x + space_emb, "b t s c -> (b t) s c")
        for block in self.spatial_blocks:
            out = block(out)
        out = rearrange(out, "(b t) s c -> b t s c", b=B, t=T)

        # 混ぜ切った register だけを時間方向に流す
        registers = rearrange(out[:, :, S - R :], "b t r c -> (b r) t c")
        new_layer_states = []
        for i, block in enumerate(self.temporal_blocks):
            registers, new_state = block(registers, rnn_state[:, :, :, i])
            new_layer_states.append(new_state)
        registers = rearrange(registers, "(b r) t c -> b t r c", b=B, r=R)
        out = torch.cat([out[:, :, : S - R], registers], dim=2)

        # Combine each block's state: temporal_block_num of [1, B*R, state_size]
        # -> [1, B*R, state_size, temporal_block_num]
        rnn_state = torch.stack(new_layer_states, dim=-1)

        return out, rnn_state
