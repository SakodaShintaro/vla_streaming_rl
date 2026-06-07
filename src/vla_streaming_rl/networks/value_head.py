# SPDX-License-Identifier: MIT
import math

import torch
from hl_gauss_pytorch import HLGaussLoss
from torch import nn

from .blocks import HyperLERPBlock, HypersphericalEmbedding, NormedLinear, Scaler, SimbaBlock
from .sparse_utils import apply_one_shot_pruning


def maybe_update_hl_gauss_range(
    module: nn.Module,
    target_value: torch.Tensor,
) -> None:
    observed_max = target_value.abs().max().item()
    if observed_max <= module.value_range:
        return
    module.value_range = observed_max
    device = module.hl_gauss_loss.support.device
    module.hl_gauss_loss = HLGaussLoss(
        min_value=-module.value_range,
        max_value=+module.value_range,
        num_bins=module.num_bins,
        clamp_to_range=True,
    ).to(device)


def weights_init_(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        # nn.init.orthogonal_(m.weight.data)
        nn.init.constant_(m.bias, 0)


class StateValueHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        block_num: int,
        num_bins: int,
        sparsity: float,
    ) -> None:
        super().__init__()
        self.fc_in = nn.Linear(in_channels, hidden_dim)
        self.fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc_out = nn.Linear(hidden_dim, num_bins)
        self.apply(weights_init_)

        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        result_dict = {}

        x = self.fc_in(x)
        x = self.fc_mid(x)
        x = self.norm(x)
        result_dict["activation"] = x

        output = self.fc_out(x)
        result_dict["output"] = output

        return result_dict


class ActionValueHead(nn.Module):
    """Dueling Architecture: Q(s,a) = V(s) + A(s,a)"""

    def __init__(
        self,
        in_channels: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int,
        block_num: int,
        num_bins: int,
        sparsity: float,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        total_action_dim = action_dim * horizon
        mid_dim = in_channels + total_action_dim

        # Value stream: V(s) - depends only on state
        self.v_fc_in = nn.Linear(in_channels, hidden_dim)
        self.v_fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.v_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.v_fc_out = nn.Linear(hidden_dim, num_bins)

        # Advantage stream: A(s,a) - depends on state and action
        self.a_fc_in = nn.Linear(mid_dim, hidden_dim)
        self.a_fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.a_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.a_fc_out = nn.Linear(hidden_dim, num_bins)

        self.apply(weights_init_)

        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: state embedding (B, state_dim)
            a: action chunk (B, horizon, action_dim)
        """
        result_dict = {}
        bs = a.size(0)
        a_flat = a.view(bs, -1)  # (B, horizon * action_dim)

        # Value stream: V(s)
        v = self.v_fc_in(x)
        v = self.v_fc_mid(v)
        v = self.v_norm(v)
        v_out = self.v_fc_out(v)  # (B, num_bins)

        # Advantage stream: A(s,a)
        xa = torch.cat([x, a_flat], dim=1)
        adv = self.a_fc_in(xa)
        adv = self.a_fc_mid(adv)
        adv = self.a_norm(adv)
        adv_out = self.a_fc_out(adv)  # (B, num_bins)

        result_dict["activation"] = torch.cat([v, adv], dim=1)

        # Q(s,a) = V(s) + A(s,a) in logit space
        output = v_out + adv_out
        result_dict["output"] = output

        return result_dict

    def get_advantage(self, x: torch.Tensor, a: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: state embedding (B, state_dim)
            a: action chunk (B, horizon, action_dim)
        """
        result_dict = {}
        bs = a.size(0)
        a_flat = a.view(bs, -1)  # (B, horizon * action_dim)

        xa = torch.cat([x, a_flat], dim=1)
        adv = self.a_fc_in(xa)
        adv = self.a_fc_mid(adv)
        adv = self.a_norm(adv)
        result_dict["activation"] = adv
        adv_out = self.a_fc_out(adv)  # (B, num_bins)
        result_dict["output"] = adv_out

        return result_dict


class HypersphericalActionValueHead(nn.Module):
    """SimbaV2 critic (arXiv:2502.15280): Q(s, a) with hyperspherical
    normalization of features *and* weights throughout.

    The state ``x`` and flattened action ``a`` are concatenated, embedded onto
    the unit hypersphere (RSNorm + shift + NormedLinear, Eq. 9-10), refined by
    ``block_num`` SimbaV2 blocks (NormedLinear + learnable scaler + ℓ2-norm +
    LERP residual, Eq. 11-12), and read out by a NormedLinear + scaler
    (Eq. 14). Because every linear uses unit-norm effective weights and every
    block output is ℓ2-normalized, feature / weight norms cannot grow during
    training — directly addressing the TD-loss-driven norm explosion that makes
    a plain MLP critic destabilize and the policy collapse mid-training.

    When ``num_bins > 1`` this is SimbaV2's distributional value estimation: the
    head emits ``num_bins`` categorical logits and holds an :class:`HLGaussLoss`
    (the repo's Gaussian-histogram cross-entropy drop-in for the paper's
    categorical critic). ``forward`` returns the raw logits in ``"output"``;
    convert to a scalar Q with ``hl_gauss_loss(output)`` and train with
    ``hl_gauss_loss(output, target)``; the support auto-expands via
    :func:`maybe_update_hl_gauss_range`.

    The value head mirrors the reference ``HyperCategoricalValue`` exactly:
    ``NormedLinear → Scaler(init=1) → NormedLinear + bias``. The internal scaler
    at init 1.0 (vs the embedder/block ``sqrt(2/dh)``) and the bias are
    essential — with a tiny output scaler and no bias the categorical logits
    stay ≈0, the softmax is pinned uniform, and the critic cannot leave its
    initial value (the freeze observed earlier). Drop-in for ``ActionValueHead``.
    """

    def __init__(
        self,
        in_channels: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int,
        block_num: int,
        num_bins: int,
        c_shift: float,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_bins = num_bins
        in_dim = in_channels + action_dim * horizon

        # Encoder scalers/alphas follow the reference critic config:
        #   scaler_init = scaler_scale = sqrt(2/dh);  alpha_init = 1/(L+1);
        #   alpha_scale = 1/sqrt(dh).
        scaler = math.sqrt(2.0 / hidden_dim)
        alpha_init = 1.0 / (block_num + 1)
        alpha_scale = 1.0 / math.sqrt(hidden_dim)
        self.embed = HypersphericalEmbedding(in_dim, hidden_dim, c_shift, scaler, scaler)
        self.blocks = nn.Sequential(
            *[
                HyperLERPBlock(hidden_dim, scaler, scaler, alpha_init, alpha_scale)
                for _ in range(block_num)
            ]
        )

        # Value head == reference HyperCategoricalValue: w1 → scaler(init=1) →
        # w2 + bias. The bias carries the (input-independent) marginal over
        # bins; the scaler at init 1.0 gives the logits enough magnitude for the
        # softmax to actually concentrate.
        self.value_w1 = NormedLinear(hidden_dim, hidden_dim)
        self.value_scaler = Scaler(hidden_dim, 1.0, 1.0)
        self.value_w2 = NormedLinear(hidden_dim, num_bins)
        self.value_bias = nn.Parameter(torch.zeros(num_bins))

        # Distributional (categorical) critic loss state. ``value_range`` starts
        # at 1.0 and is grown by ``maybe_update_hl_gauss_range`` as targets
        # exceed it, matching the other actor-critic networks in this repo.
        self.value_range = 1.0
        if num_bins > 1:
            self.hl_gauss_loss = HLGaussLoss(
                min_value=-self.value_range,
                max_value=+self.value_range,
                num_bins=num_bins,
                clamp_to_range=True,
            )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: state embedding (B, state_dim)
            a: action chunk (B, horizon, action_dim)
        """
        bs = a.size(0)
        h = torch.cat([x, a.view(bs, -1)], dim=1)
        h = self.embed(h)
        h = self.blocks(h)
        v = self.value_scaler(self.value_w1(h))
        logits = self.value_w2(v) + self.value_bias
        return {"output": logits, "activation": h}
