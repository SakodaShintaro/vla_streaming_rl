# SPDX-License-Identifier: MIT
import numpy as np
import torch
from torch import nn


class RunningNormalizer(nn.Module):
    """Per-component running mean/std normalizer for a fixed-width vector.

    統計はネットワークの buffer として持つので、チェックポイントに保存され、評価や
    再開のときに学習時と同じ正規化が戻る。更新は gymnasium の RunningMeanStd と同じ
    並列の式で、1件ずつ足す。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float64))
        # RunningMeanStd と同じく、最初の1件で割り算が壊れないよう小さな件数から始める
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))
        self.epsilon = 1e-8

    @torch.no_grad()
    def update(self, x: np.ndarray) -> None:
        sample = torch.as_tensor(x, dtype=torch.float64, device=self.mean.device)
        delta = sample - self.mean
        total = self.count + 1.0
        self.mean += delta / total
        self.var.copy_((self.var * self.count + delta.pow(2) * self.count / total) / total)
        self.count.copy_(total)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device, x.dtype)
        std = torch.sqrt(self.var.to(x.device, x.dtype) + self.epsilon)
        return (x - mean) / std
