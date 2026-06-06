# SPDX-License-Identifier: MIT
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormedLinear(nn.Module):
    """Linear layer with weight projected onto the unit hypersphere in forward.

    Weights are L2-normalized along the input dimension at each forward call,
    so the stored parameters are free to be updated by any optimizer while the
    effective weights always have unit norm per output vector.  No bias is used.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.weight)
        with torch.no_grad():
            self.weight.copy_(F.normalize(self.weight, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, F.normalize(self.weight, dim=1))


class RSNorm(nn.Module):
    """Running-statistics normalization (SimbaV2, Eq. 3-4 of arXiv:2502.15280).

    Tracks a per-dimension running mean / variance over every input seen during
    training and standardizes inputs to zero mean, unit variance. The stats are
    buffers (no gradient, saved with the module) and update only in training
    mode, so evaluation is deterministic. The batched update is the standard
    parallel (Chan/Welford) variance merge, equivalent to the paper's per-step
    ``1/t`` recursion when fed one sample at a time.
    """

    _EPS = 1e-5

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("count", torch.zeros(()))
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))

    @torch.no_grad()
    def _update(self, x: torch.Tensor) -> None:
        batch_count = x.shape[0]
        if batch_count == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        new_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * (batch_count / new_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * (self.count * batch_count / new_count)
        self.var = m2 / new_count
        self.count = new_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update(x)
        return (x - self.mean) / torch.sqrt(self.var + self._EPS)


class HypersphericalEmbedding(nn.Module):
    """SimbaV2 input embedding (Eq. 9-10 of arXiv:2502.15280).

    RSNorm-standardize the input, append the constant shift coordinate
    ``c_shift`` and ℓ2-normalize so the result lands on the unit hypersphere
    while its original magnitude is retained in the extra coordinate (Eq. 9);
    then a NormedLinear + learnable scaler + ℓ2-normalization projects it onto
    the ``hidden_dim`` hypersphere (Eq. 10).
    """

    def __init__(self, in_dim: int, hidden_dim: int, c_shift: float) -> None:
        super().__init__()
        self.rsnorm = RSNorm(in_dim)
        self.register_buffer("c_shift", torch.full((1,), c_shift))
        self.linear = NormedLinear(in_dim + 1, hidden_dim)
        s_init = math.sqrt(2.0 / hidden_dim)
        self.scaler = nn.Parameter(torch.full((hidden_dim,), s_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rsnorm(x)
        shift = self.c_shift.expand(x.shape[0], 1)
        x = F.normalize(torch.cat([x, shift], dim=-1), dim=-1)  # õ  (Eq. 9)
        x = self.scaler * self.linear(x)  # s0 ⊙ (W0 õ)
        return F.normalize(x, dim=-1)  # h0  (Eq. 10)


class SimbaV2Block(nn.Module):
    """https://arxiv.org/abs/2502.15280

    Inverted-bottleneck MLP with:
      - L2 normalization instead of LayerNorm
      - NormedLinear (weight on unit hypersphere, no bias)
      - Learnable scaler vector
      - LERP residual connection
    All weight projection is handled inside forward, so no special optimizer
    or external hook is required.

    Args:
        channels: hidden dimension (dh)
    """

    _EXPANSION = 4
    _ALPHA_INIT = 0.1

    def __init__(self, channels: int) -> None:
        super().__init__()
        inner = channels * self._EXPANSION
        self.linear1 = NormedLinear(channels, inner)
        self.linear2 = NormedLinear(inner, channels)

        s_init = math.sqrt(2.0 / channels)
        self.scaler = nn.Parameter(torch.full((inner,), s_init))

        alpha_scale = 1.0 / math.sqrt(channels)
        self.alpha = nn.Parameter(torch.full((channels,), alpha_scale))
        self._alpha_ratio = self._ALPHA_INIT / alpha_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MLP + L2 Norm  (Eq.11)
        h = self.linear1(x)
        h = F.relu(h)
        h = h * self.scaler
        h = self.linear2(h)
        h = F.normalize(h, dim=-1)

        # LERP + L2 Norm  (Eq.12)
        alpha = self.alpha * self._alpha_ratio
        out = (1.0 - alpha) * x + alpha * h
        return F.normalize(out, dim=-1)


class SimbaBlock(nn.Module):
    """https://arxiv.org/abs/2410.09754"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(channels, elementwise_affine=False),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class BroBlock(nn.Module):
    """https://arxiv.org/abs/2405.16158"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
