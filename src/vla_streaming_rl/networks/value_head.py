# SPDX-License-Identifier: MIT
import math

import torch
from hl_gauss_pytorch import HLGaussLoss
from torch import nn

from .blocks import HypersphericalEmbedding, NormedLinear, SimbaBlock, SimbaV2Block
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

    Drop-in for ``ActionValueHead``: same ``forward(x, a) -> {"output",
    "activation"}`` contract (scalar Q when ``num_bins == 1``).
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
        in_dim = in_channels + action_dim * horizon
        self.embed = HypersphericalEmbedding(in_dim, hidden_dim, c_shift)
        self.blocks = nn.Sequential(*[SimbaV2Block(hidden_dim) for _ in range(block_num)])
        self.out_linear = NormedLinear(hidden_dim, num_bins)
        s_init = math.sqrt(2.0 / hidden_dim)
        self.out_scaler = nn.Parameter(torch.full((num_bins,), s_init))

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
        out = self.out_scaler * self.out_linear(h)  # Eq. 14
        return {"output": out, "activation": h}
