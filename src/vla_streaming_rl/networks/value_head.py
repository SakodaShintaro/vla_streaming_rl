# SPDX-License-Identifier: MIT
import math

import torch
from hl_gauss_pytorch import HLGaussLoss
from torch import nn
from torch.nn import functional as F

from .blocks import HyperLERPBlock, HypersphericalEmbedding, NormedLinear, Scaler, SimbaBlock
from .head_output import HeadOutput
from .sparse_utils import apply_one_shot_pruning


class DistributionalValueHead(nn.Module):
    """Base for value heads: owns the HL-Gauss categorical-critic machinery so
    the value head — not the surrounding network — manages the distributional
    ⇄ scalar mapping.

    With ``num_bins > 1`` the head emits ``num_bins`` categorical logits and
    holds an :class:`HLGaussLoss` (Gaussian-histogram cross-entropy, the repo's
    drop-in for a categorical critic) whose support grows to track the targets.
    ``num_bins == 1`` is the plain scalar critic with no HL-Gauss; the methods
    degrade to identity / MSE so callers never branch on ``num_bins``.

    Subclasses call :meth:`_init_value_dist` once they know ``num_bins`` and
    return raw logits in ``forward``'s ``"output"``.
    """

    def _init_value_dist(self, num_bins: int) -> None:
        self.num_bins = num_bins
        # ``value_range`` starts at 1.0 and is grown by ``update_value_range``
        # as targets exceed it.
        self.value_range = 1.0
        if num_bins > 1:
            self.hl_gauss_loss = HLGaussLoss(
                min_value=-self.value_range,
                max_value=+self.value_range,
                num_bins=num_bins,
                clamp_to_range=True,
            )

    def to_value(self, logits: torch.Tensor) -> torch.Tensor:
        """Critic ``output`` → expected scalar value, with the bin dim reduced.

        ``num_bins > 1``: HL-Gauss expectation over the categorical logits.
        ``num_bins == 1``: the single output channel, squeezed.
        """
        if self.num_bins > 1:
            return self.hl_gauss_loss(logits)
        return logits.squeeze(-1)

    def value_report(self, logits: torch.Tensor) -> dict[str, float]:
        """Per-sample value diagnostics for inference / rollout logging.

        Always reports the expected scalar value under ``"value"``. For the
        distributional critic (``num_bins > 1``) it summarizes the categorical
        output with its ``"value_variance"`` (spread of the predicted return
        distribution) and the current ``"value_range"`` (HL-Gauss support
        half-width) — a compact alternative to logging every bin. Single-sample
        only (``B == 1``); every value is a Python float so callers can splat the
        dict straight into their metrics.
        """
        assert logits.shape[0] == 1, f"value_report expects B == 1, got B={logits.shape[0]}"
        report = {"value": self.to_value(logits).item()}
        if self.num_bins > 1:
            probs = logits.softmax(dim=-1).reshape(-1)
            centers = self.hl_gauss_loss.centers
            mean = (probs * centers).sum()
            variance = (probs * (centers - mean) ** 2).sum()
            report["value_variance"] = variance.item()
            report["value_range"] = self.value_range
        return report

    def update_value_range(self, target_value: torch.Tensor) -> None:
        """Grow the HL-Gauss support to cover ``target_value`` (no-op if scalar)."""
        if self.num_bins == 1:
            return
        observed_max = target_value.abs().max().item()
        if observed_max <= self.value_range:
            return
        self.value_range = observed_max
        device = self.hl_gauss_loss.support.device
        self.hl_gauss_loss = HLGaussLoss(
            min_value=-self.value_range,
            max_value=+self.value_range,
            num_bins=self.num_bins,
            clamp_to_range=True,
        ).to(device)

    def value_loss(self, logits: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
        """Regression/TD loss for the critic output.

        ``num_bins > 1``: HL-Gauss categorical cross-entropy against the
        support-projected target. ``num_bins == 1``: plain MSE on the scalar.
        Call :meth:`update_value_range` first when the target may exceed the
        current support.
        """
        if self.num_bins > 1:
            return self.hl_gauss_loss(logits, target_value)
        return F.mse_loss(self.to_value(logits), target_value)

    @torch.no_grad()
    def compute_target_value(
        self,
        next_output: torch.Tensor,
        chunk_rewards: torch.Tensor,
        chunk_dones: torch.Tensor,
    ) -> torch.Tensor:
        """n-step TD target for the action-chunk critic.

        Takes the next-state critic's raw ``output`` (logits) and reduces it to
        the bootstrap value internally (:meth:`to_value`), so callers never have
        to collapse the distribution themselves — the value head owns the
        distributional ⇄ scalar mapping end to end. Discounts ``chunk_rewards``
        over the ``horizon``-step chunk, stops the accumulation/bootstrap at the
        first ``done``, and bootstraps from that value. Uses ``self.horizon`` /
        ``self.gamma``, which the action-value subclasses set in ``__init__``.
        """
        next_q = self.to_value(next_output).view(-1)
        batch_size = chunk_rewards.size(0)
        device = chunk_rewards.device
        discounted_reward = torch.zeros(batch_size, device=device)
        gamma_power = 1.0
        continuing = torch.ones(batch_size, device=device)
        for i in range(self.horizon):
            discounted_reward += continuing * gamma_power * chunk_rewards[:, i].flatten()
            gamma_power *= self.gamma
            continuing *= 1 - chunk_dones[:, i].flatten()
        return discounted_reward + continuing * gamma_power * next_q

    def compute_critic_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        target_value: torch.Tensor,
        detach_critic: bool,
    ) -> tuple[torch.Tensor, dict]:
        """Distributional TD loss for Q(state, action_chunk) vs ``target_value``.

        Mirrors :meth:`PolicyHead.compute_actor_loss`: given the state, action
        chunk and TD target, the value head owns its own loss (forward →
        :meth:`update_value_range` → :meth:`value_loss`) and the scalar logging
        info. ``detach_critic`` stops the gradient into the encoder.
        """
        if detach_critic:
            state = state.detach()

        curr_critic_out = self(state, action_chunk)
        logits = curr_critic_out.output

        self.update_value_range(target_value)
        curr_critic_value = self.to_value(logits).view(-1)
        critic_loss = self.value_loss(logits, target_value)

        delta = target_value - curr_critic_value

        info = {
            "delta": delta.mean().item(),
            "critic_loss": critic_loss.item(),
            "curr_critic_value": curr_critic_value.mean().item(),
            "target_value": target_value.mean().item(),
            "value_range": self.value_range,
        }

        return critic_loss, info


def weights_init_(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        # nn.init.orthogonal_(m.weight.data)
        nn.init.constant_(m.bias, 0)


class ActionValueHead(DistributionalValueHead):
    """Dueling Architecture: Q(s,a) = V(s) + A(s,a)"""

    def __init__(
        self,
        in_channels: int,
        action_dim: int,
        horizon: int,
        gamma: float,
        hidden_dim: int,
        block_num: int,
        num_bins: int,
        sparsity: float,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.gamma = gamma
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
        self._init_value_dist(num_bins)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> HeadOutput:
        """
        Args:
            x: state embedding (B, state_dim)
            a: action chunk (B, horizon, action_dim)
        """
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

        # Q(s,a) = V(s) + A(s,a) in logit space
        output = v_out + adv_out

        return HeadOutput(output=output, activation=torch.cat([v, adv], dim=1))

    def get_advantage(self, x: torch.Tensor, a: torch.Tensor) -> HeadOutput:
        """
        Args:
            x: state embedding (B, state_dim)
            a: action chunk (B, horizon, action_dim)
        """
        bs = a.size(0)
        a_flat = a.view(bs, -1)  # (B, horizon * action_dim)

        xa = torch.cat([x, a_flat], dim=1)
        adv = self.a_fc_in(xa)
        adv = self.a_fc_mid(adv)
        adv = self.a_norm(adv)
        adv_out = self.a_fc_out(adv)  # (B, num_bins)

        return HeadOutput(output=adv_out, activation=adv)


class HypersphericalActionValueHead(DistributionalValueHead):
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
    convert to a scalar Q with :meth:`to_value` and train with
    :meth:`value_loss`; the support auto-expands via :meth:`update_value_range`.

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
        gamma: float,
        hidden_dim: int,
        block_num: int,
        num_bins: int,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.gamma = gamma
        self.action_dim = action_dim
        in_dim = in_channels + action_dim * horizon

        # Encoder scalers/alphas follow the reference critic config:
        #   scaler_init = scaler_scale = sqrt(2/dh);  alpha_init = 1/(L+1);
        #   alpha_scale = 1/sqrt(dh).
        scaler = math.sqrt(2.0 / hidden_dim)
        alpha_init = 1.0 / (block_num + 1)
        alpha_scale = 1.0 / math.sqrt(hidden_dim)
        self.embed = HypersphericalEmbedding(in_dim, hidden_dim, scaler, scaler)
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

        self._init_value_dist(num_bins)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> HeadOutput:
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
        return HeadOutput(output=logits, activation=h)

    def get_advantage(self, x: torch.Tensor, a: torch.Tensor) -> HeadOutput:
        """SimbaV2 has no separate advantage stream (pure Q(s, a)), so the
        advantage equals the Q output; maximizing it maximizes E[Q(s, π(s))]."""
        return self.forward(x, a)
