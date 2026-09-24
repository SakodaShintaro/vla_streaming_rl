# SPDX-License-Identifier: MIT
import torch
from torch import nn


class RewardProcessor(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Expand scalar by repeating it to the required number of dimensions
        # x: (B, T, 1) -> embedded: (B, T, 1, embed_dim)
        embedded = x.unsqueeze(-1).expand(*x.shape, self.embed_dim)
        return embedded
