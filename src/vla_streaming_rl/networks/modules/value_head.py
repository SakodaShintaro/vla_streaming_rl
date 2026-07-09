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
    ⇄ scalar mapping, and the **multi-gamma** read-out.

    Multi-gamma (AMAGO, arXiv:2310.09971 §"multi-gamma"): the head predicts the
    action value for ``num_gammas`` discount factors at once from a *shared
    trunk* — the final layer emits ``num_gammas * num_bins`` logits, reshaped to
    ``(B, num_gammas, num_bins)``. The extra gammas are an auxiliary prediction
    task that enriches the representation; the actor / rollout maximize the mean
    over gammas (so :meth:`to_value` averages), while the critic is trained on a
    *per-gamma* TD target (:meth:`to_values`, :meth:`value_loss`). With
    ``num_gammas == 1`` everything collapses to the original single-gamma head:
    logits stay ``(B, num_bins)`` and every scalar caller is unchanged.

    With ``num_bins > 1`` the head emits categorical logits and holds an
    :class:`HLGaussLoss` (Gaussian-histogram cross-entropy, the repo's drop-in
    for a categorical critic) on a fixed ``[-1, 1]`` support. Because gammas
    closer to 1 have far larger returns, each gamma carries its own PopArt-like
    ``value_ranges[g]`` half-width: targets are divided by it before the
    categorical projection and predictions multiplied back, so every gamma uses
    the full bin support. ``num_bins == 1`` is the plain scalar critic (MSE, no
    HL-Gauss); ``value_ranges`` stay 1.0.

    Subclasses size their final layer to ``num_gammas * num_bins``, run logits
    through :meth:`_shape_logits` in ``forward``, and call :meth:`_init_value_dist`.
    """

    def _init_value_dist(self, num_bins: int, gammas: list[float]) -> None:
        self.num_bins = num_bins
        self.num_gammas = len(gammas)
        # ``gammas`` is the per-gamma discount list (auxiliary gammas first, the
        # primary/rollout discount last); the head owns it so it can build the
        # per-gamma TD target itself (:meth:`compute_target_value`).
        self.primary_gamma_index = self.num_gammas - 1
        self.register_buffer("gammas_tensor", torch.tensor(gammas, dtype=torch.float32))
        # Per-gamma support half-width (PopArt-like). Grown by
        # ``update_value_range`` as each gamma's targets exceed it; stays 1.0 for
        # the scalar (``num_bins == 1``) critic.
        self.register_buffer("value_ranges", torch.ones(self.num_gammas))
        if num_bins > 1:
            # Fixed canonical support; per-gamma scale handled by value_ranges.
            self.hl_gauss_loss = HLGaussLoss(
                min_value=-1.0,
                max_value=+1.0,
                num_bins=num_bins,
                clamp_to_range=True,
            )

    @property
    def value_range(self) -> float:
        """Primary-gamma support half-width (scalar, for logging — unchanged API)."""
        return float(self.value_ranges[self.primary_gamma_index])

    def _shape_logits(self, flat: torch.Tensor) -> torch.Tensor:
        """``(B, num_gammas * num_bins)`` → ``(B, num_bins)`` for a single gamma
        (legacy shape) else ``(B, num_gammas, num_bins)``."""
        if self.num_gammas == 1:
            return flat
        return flat.view(flat.shape[0], self.num_gammas, self.num_bins)

    def _normalized_value(self, logits: torch.Tensor) -> torch.Tensor:
        """Logits → value in the normalized ``[-1, 1]`` space, bin dim reduced.

        Shape ``(B,)`` for single-gamma logits, ``(B, num_gammas)`` otherwise.
        """
        if self.num_bins > 1:
            flat = logits.reshape(-1, self.num_bins)
            return self.hl_gauss_loss(flat).view(logits.shape[:-1])
        return logits.squeeze(-1)

    def to_values(self, logits: torch.Tensor) -> torch.Tensor:
        """Per-gamma denormalized values, shape ``(B, num_gammas)`` — the
        per-gamma TD-target read-out."""
        norm = self._normalized_value(logits)
        if self.num_gammas == 1:
            norm = norm.unsqueeze(-1)
        return norm * self.value_ranges

    def to_value(self, logits: torch.Tensor) -> torch.Tensor:
        """Scalar value per sample, **averaged over all gammas** — the actor /
        eligibility-trace / rollout objective. Shape ``(B,)``. For a single
        gamma this is just that gamma's value, so existing callers are unchanged.
        """
        return self.to_values(logits).mean(dim=1)

    def value_report(self, logits: torch.Tensor) -> dict[str, float]:
        """Per-sample value diagnostics for inference / rollout logging.

        Always reports the gamma-averaged scalar value under ``"value"``. With
        more than one gamma it adds each gamma's value under ``"value_g{γ}"``.
        For the distributional critic (``num_bins > 1``) it summarizes each
        gamma's categorical output with its ``"value_variance"`` (spread of the
        predicted return distribution) and ``"value_range"`` (PopArt support
        half-width) — a compact alternative to logging every bin; per-gamma runs
        suffix these with ``_g{γ}``. Single-sample only (``B == 1``); every value
        is a Python float so callers can splat the dict straight into metrics.
        """
        assert logits.shape[0] == 1, f"value_report expects B == 1, got B={logits.shape[0]}"
        per_gamma = self.to_values(logits).reshape(-1)  # (num_gammas,)
        multi = self.num_gammas > 1
        gammas = self.gammas_tensor.tolist()
        report = {"value": per_gamma.mean().item()}
        if multi:
            for g, v in zip(gammas, per_gamma):
                report[f"value_g{g:.4f}"] = v.item()
        if self.num_bins > 1:
            # ``centers`` live in the normalized [-1, 1] support; the real-unit
            # variance scales by each gamma's value_range². probs: (num_gammas, num_bins).
            probs = logits.softmax(dim=-1).reshape(self.num_gammas, self.num_bins)
            centers = self.hl_gauss_loss.centers
            mean = (probs * centers).sum(dim=-1, keepdim=True)
            variance = (probs * (centers - mean) ** 2).sum(dim=-1) * self.value_ranges**2
            for i, g in enumerate(gammas):
                sfx = f"_g{g:.4f}" if multi else ""
                report[f"value_variance{sfx}"] = variance[i].item()
                report[f"value_range{sfx}"] = self.value_ranges[i].item()
        return report

    def update_value_range(self, target_value: torch.Tensor) -> None:
        """Grow each gamma's HL-Gauss support to cover its targets (no-op if scalar)."""
        if self.num_bins == 1:
            return
        target = target_value.view(target_value.shape[0], self.num_gammas)
        observed = target.abs().amax(dim=0)
        with torch.no_grad():
            self.value_ranges.copy_(
                torch.maximum(self.value_ranges, observed.to(self.value_ranges))
            )

    def value_loss(self, logits: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
        """Per-gamma regression/TD loss, averaged over gammas (and batch).

        ``num_bins > 1``: HL-Gauss categorical cross-entropy against the
        per-gamma-normalized target. ``num_bins == 1``: plain MSE on the scalar.
        Call :meth:`update_value_range` first when targets may exceed the support.
        """
        target = target_value.view(target_value.shape[0], self.num_gammas)
        if self.num_bins > 1:
            target_norm = (target / self.value_ranges).clamp(-1.0, 1.0)
            return self.hl_gauss_loss(logits.reshape(-1, self.num_bins), target_norm.reshape(-1))
        return F.mse_loss(self.to_values(logits), target)

    @torch.no_grad()
    def compute_target_value(
        self,
        next_output: torch.Tensor,
        chunk_rewards: torch.Tensor,
        chunk_dones: torch.Tensor,
    ) -> torch.Tensor:
        next_q = self.to_values(next_output)  # (B, num_gammas)
        batch_size = chunk_rewards.size(0)
        gammas = self.gammas_tensor.to(chunk_rewards.device)  # (G,)
        discounted_reward = torch.zeros(batch_size, self.num_gammas, device=chunk_rewards.device)
        gamma_power = torch.ones(self.num_gammas, device=chunk_rewards.device)  # (G,)
        continuing = torch.ones(batch_size, 1, device=chunk_rewards.device)  # (B, 1)
        for i in range(self.horizon):
            reward_i = chunk_rewards[:, i].view(batch_size, 1)
            discounted_reward += continuing * gamma_power.unsqueeze(0) * reward_i
            gamma_power = gamma_power * gammas
            continuing = continuing * (1 - chunk_dones[:, i].view(batch_size, 1))
        return discounted_reward + continuing * gamma_power.unsqueeze(0) * next_q

    def compute_critic_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        target_value: torch.Tensor,
        detach_critic: bool,
    ) -> tuple[torch.Tensor, dict]:
        """Distributional TD loss for Q(state, action_chunk) vs ``target_value``.

        Mirrors :meth:`PolicyHead.compute_actor_loss`: given the state, action
        chunk and the per-gamma TD target ``(B, num_gammas)`` (built by
        :meth:`compute_target_value`), the value head owns its loss (forward →
        :meth:`update_value_range` → :meth:`value_loss`) and the scalar logging
        info. ``detach_critic`` stops the gradient into the encoder.
        """
        if detach_critic:
            state = state.detach()

        logits = self(state, action_chunk).output

        self.update_value_range(target_value)
        # ``to_value`` averages over gammas; the per-gamma loss is in ``value_loss``.
        curr_critic_value = self.to_value(logits).view(-1)
        critic_loss = self.value_loss(logits, target_value)

        delta = target_value.mean(dim=1) - curr_critic_value

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
        hidden_dim: int,
        block_num: int,
        num_bins: int,
        gammas: list[float],
        sparsity: float,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        total_action_dim = action_dim * horizon
        mid_dim = in_channels + total_action_dim
        out_dim = len(gammas) * num_bins

        # Value stream: V(s) - depends only on state
        self.v_fc_in = nn.Linear(in_channels, hidden_dim)
        self.v_fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.v_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.v_fc_out = nn.Linear(hidden_dim, out_dim)

        # Advantage stream: A(s,a) - depends on state and action
        self.a_fc_in = nn.Linear(mid_dim, hidden_dim)
        self.a_fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.a_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.a_fc_out = nn.Linear(hidden_dim, out_dim)

        self.apply(weights_init_)

        nn.init.zeros_(self.v_fc_out.weight)
        nn.init.zeros_(self.v_fc_out.bias)
        nn.init.zeros_(self.a_fc_out.weight)
        nn.init.zeros_(self.a_fc_out.bias)

        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )
        self._init_value_dist(num_bins, gammas)

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
        v_out = self.v_fc_out(v)  # (B, num_gammas * num_bins)

        # Advantage stream: A(s,a)
        xa = torch.cat([x, a_flat], dim=1)
        adv = self.a_fc_in(xa)
        adv = self.a_fc_mid(adv)
        adv = self.a_norm(adv)
        adv_out = self.a_fc_out(adv)  # (B, num_gammas * num_bins)

        # Q(s,a) = V(s) + A(s,a) in logit space
        output = self._shape_logits(v_out + adv_out)

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
        adv_out = self._shape_logits(self.a_fc_out(adv))  # (B, [G,] num_bins)

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
        hidden_dim: int,
        block_num: int,
        num_bins: int,
        gammas: list[float],
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        in_dim = in_channels + action_dim * horizon
        num_gammas = len(gammas)

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
        self.value_w2 = NormedLinear(hidden_dim, num_gammas * num_bins)
        self.value_bias = nn.Parameter(torch.zeros(num_gammas * num_bins))

        self._init_value_dist(num_bins, gammas)

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
        logits = self._shape_logits(self.value_w2(v) + self.value_bias)
        return HeadOutput(output=logits, activation=h)

    def get_advantage(self, x: torch.Tensor, a: torch.Tensor) -> HeadOutput:
        """SimbaV2 has no separate advantage stream (pure Q(s, a)), so the
        advantage equals the Q output; maximizing it maximizes E[Q(s, π(s))]."""
        return self.forward(x, a)
