import torch
from torch import Tensor, nn

from .layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)


class FluxDiT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        vec_in_dim: int,
        context_in_dim: int,
        hidden_size: int,
        mlp_ratio: float,
        num_heads: int,
        depth_double_blocks: int,
        depth_single_blocks: int,
        axes_dim: tuple[int],
        theta: int,
        qkv_bias: bool,
        use_r_time: bool,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_r_time = use_r_time
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        self.noise_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        # Extra embedder used only by MeanFlow: encodes the time gap (r - t).
        # At r=t the embedding is r_in(timestep_embedding(0)), giving a clean
        # FM boundary so the same network can act as both FM and MeanFlow.
        self.r_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if use_r_time else None
        self.vector_in = MLPEmbedder(vec_in_dim, self.hidden_size)
        self.cond_in = nn.Linear(context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth_double_blocks)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        noise: Tensor,
        timesteps: Tensor,
        cond: Tensor,
        y: Tensor,
        r_timesteps: Tensor | None,
    ) -> dict[str, Tensor]:
        B, noise_seq_len, C = noise.shape
        _, cond_seq_len, _ = cond.shape

        # Generate positional encoding
        noise_pos_id = torch.arange(noise_seq_len, device=noise.device)
        noise_ids = noise_pos_id.unsqueeze(0).expand(B, -1).unsqueeze(-1).float()

        cond_pos_id = torch.arange(cond_seq_len, device=cond.device)
        cond_ids = cond_pos_id.unsqueeze(0).expand(B, -1).unsqueeze(-1).float()

        # running on sequences img
        noise = self.noise_in(noise)
        timesteps = timesteps.flatten()
        vec = self.time_in(timestep_embedding(timesteps, 256)) + self.vector_in(y)
        if self.use_r_time:
            # Encode the gap (r - t); embedding is r_in(0) when r=t (FM boundary).
            gap = r_timesteps.flatten() - timesteps
            vec = vec + self.r_in(timestep_embedding(gap, 256))
        cond = self.cond_in(cond)

        ids = torch.cat((cond_ids, noise_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            noise, cond = block(img=noise, cond=cond, vec=vec, pe=pe)

        noise = torch.cat((cond, noise), 1)
        for block in self.single_blocks:
            noise = block(noise, vec=vec, pe=pe)
        noise = noise[:, cond.shape[1] :, ...]

        output = self.final_layer(noise, vec)  # (B, noise_seq_len, out_channels)

        # dummy
        activation = torch.zeros((B, 1))

        return {"output": output, "activation": activation}
