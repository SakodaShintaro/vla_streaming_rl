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


class Scaler(nn.Module):
    """Learnable per-dimension scaler with decoupled init / learning scale
    (SimbaV2's ``Scaler``, Eq. 21 of arXiv:2502.15280).

    The stored parameter is initialized to ``scale`` and multiplied by the
    constant ``init / scale`` in forward, so the effective initial scaler is
    ``init`` while its gradient scale is controlled independently by ``scale``.
    Matches the reference ``Scaler`` exactly.
    """

    def __init__(self, dim: int, init: float, scale: float) -> None:
        super().__init__()
        self.scaler = nn.Parameter(torch.full((dim,), float(scale)))
        self._forward_scaler = init / scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaler * self._forward_scaler * x


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

    def __init__(
        self, in_dim: int, hidden_dim: int, scaler_init: float, scaler_scale: float
    ) -> None:
        super().__init__()
        # The reference normalizes observations with an external running
        # normalizer (``normalize_observation``); we fold that in here as RSNorm
        # since our critic input (pooled VLM features + action) is un-normalized.
        self.rsnorm = RSNorm(in_dim)
        self.register_buffer("c_shift", torch.full((1,), 3.0))
        self.linear = NormedLinear(in_dim + 1, hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rsnorm(x)
        shift = self.c_shift.expand(x.shape[0], 1)
        x = F.normalize(torch.cat([x, shift], dim=-1), dim=-1)  # õ  (Eq. 9)
        x = self.scaler(self.linear(x))  # s0 ⊙ (W0 õ)
        return F.normalize(x, dim=-1)  # h0  (Eq. 10)


class HyperMLP(nn.Module):
    """Inverted-bottleneck MLP on the hypersphere (SimbaV2 ``HyperMLP``, Eq. 11).

    NormedLinear (unit-norm weight, no bias) → Scaler → ReLU (+eps to avoid a
    zero vector) → NormedLinear → ℓ2-normalize. Matches the reference exactly.
    """

    _EPS = 1e-8

    def __init__(
        self, in_dim: int, inner_dim: int, out_dim: int, scaler_init: float, scaler_scale: float
    ) -> None:
        super().__init__()
        self.w1 = NormedLinear(in_dim, inner_dim)
        self.scaler = Scaler(inner_dim, scaler_init, scaler_scale)
        self.w2 = NormedLinear(inner_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.scaler(self.w1(x))
        x = F.relu(x) + self._EPS
        x = self.w2(x)
        return F.normalize(x, dim=-1)


class HyperLERPBlock(nn.Module):
    """SimbaV2 residual block (``HyperLERPBlock``, Eq. 12 of arXiv:2502.15280).

    A learnable linear interpolation (LERP) between the input and its HyperMLP
    transform, ℓ2-normalized back onto the hypersphere:
    ``x ← ℓ2norm(x + alpha ⊙ (mlp(x) − x))``. ``alpha`` is a Scaler with the
    reference's decoupled init/scale (``alpha_init = 1/(L+1)``,
    ``alpha_scale = 1/sqrt(hidden)``).
    """

    _EXPANSION = 4

    def __init__(
        self,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
    ) -> None:
        super().__init__()
        inner = hidden_dim * self._EXPANSION
        self.mlp = HyperMLP(
            hidden_dim,
            inner,
            hidden_dim,
            scaler_init / math.sqrt(self._EXPANSION),
            scaler_scale / math.sqrt(self._EXPANSION),
        )
        self.alpha = Scaler(hidden_dim, alpha_init, alpha_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.mlp(x)
        x = residual + self.alpha(x - residual)
        return F.normalize(x, dim=-1)


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
