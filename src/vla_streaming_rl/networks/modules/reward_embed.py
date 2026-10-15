# SPDX-License-Identifier: MIT
import torch


def embed_reward(reward: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """報酬のスカラーを、そのまま ``embed_dim`` 個並べたトークンにする。
    (..., 1) -> (..., 1, embed_dim)"""
    return reward.unsqueeze(-1).expand(*reward.shape, embed_dim)
