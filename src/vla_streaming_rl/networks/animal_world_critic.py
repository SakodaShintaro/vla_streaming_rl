# SPDX-License-Identifier: MIT
"""The Animal-AI PPO network with WCM's world-critic head, ported from ``~/work/WCM``.

WCM (arXiv 2607.29613) trains a critic whose shared representation has to
support two objectives: an action-free value estimate over the history, and an
action-conditioned prediction of the next latent state. The second one is what
this file adds. The value estimate here is already action-free -- PPO's value
head reads the recurrent output, which never sees an action -- so what is missing is
the dynamics branch and the collapse guard that lets its target be the encoder's
own latent.

Three pieces come across from ``world_critic/model.py``:

- :class:`GatedDynamicsBlock`, the FiLM-modulated residual update, initialized so
  the gate starts near closed and the branch starts near identity,
- :class:`ActionConditionedDynamics`, predicting ``z_(t+1) = z_t + delta(h_t, a_t)``
  with the recurrent output as the history context, and
- :class:`SIGReg`, the LeJEPA sketched isotropic-Gaussian regularizer. The
  prediction target is the encoder's own detached latent, so nothing stops the
  encoder from collapsing to a constant except this term.

The one structural change is the action encoder: WCM's actions are continuous
robot commands through an MLP, while Animal-AI's are one of nine discrete
branch pairs, so an embedding table takes that slot.

The dynamics loss rides on the minibatch the PPO update already runs, over the
step pairs that stay inside one episode, so ``seq_len`` is what bounds it: a
window of ``n`` steps contributes ``n - 1`` pairs, and it has to be raised above
what plain PPO runs with.
"""

import torch
import torch.nn.functional as F
from torch import nn

from vla_streaming_rl.networks.animal_ppo import (
    ACTION_NUM,
    HIDDEN_NODES,
    TEMPORAL_UNITS,
    AnimalPPONetwork,
)


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regularizer, from ``world_critic/model.py``.

    The latent is projected onto random unit directions and each projection's
    empirical characteristic function is compared against a standard normal's,
    integrated over ``knots`` quadrature points with a Gaussian window.
    """

    def __init__(self, knots: int, num_projections: int) -> None:
        super().__init__()
        self.num_projections = num_projections
        points = torch.linspace(0, 3, knots, dtype=torch.float32)
        delta = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * delta, dtype=torch.float32)
        weights[[0, -1]] = delta
        window = torch.exp(-points.square() / 2)
        self.register_buffer("points", points)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, latent: torch.Tensor, projections: torch.Tensor) -> torch.Tensor:
        """``latent`` is ``(groups, samples, dim)``: the statistic is taken over
        the sample axis, one distribution per group."""
        assert projections.shape == (latent.size(-1), self.num_projections), (
            f"Expected projection matrix {(latent.size(-1), self.num_projections)}, "
            f"got {projections.shape}"
        )
        projected = (latent @ projections).unsqueeze(-1) * self.points
        error = (projected.cos().mean(-3) - self.phi).square() + projected.sin().mean(-3).square()
        statistic = (error @ self.weights) * latent.size(-2)
        return statistic.mean()

    def loss(self, latent: torch.Tensor) -> torch.Tensor:
        """The regularizer on a ``(..., latent_dim)`` batch of latents.

        WCM takes one statistic per time index, over a batch large enough for
        each to be a distribution of its own. The batches here are small -- one
        sequence in streaming mode -- so every step of every sequence is instead
        pooled into a single sample set.
        """
        samples = latent.reshape(-1, latent.shape[-1]).float().unsqueeze(0)
        projections = normalized_random_projections(
            samples.size(-1), self.num_projections, samples.device, torch.float32
        )
        return self(samples, projections)


def normalized_random_projections(
    latent_dim: int, num_projections: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return F.normalize(torch.randn(latent_dim, num_projections, device=device, dtype=dtype), dim=0)


class DynamicsMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class GatedDynamicsBlock(nn.Module):
    """A FiLM-modulated, gated residual update in the predictor's hidden space.

    The gate bias starts at -2 so the block begins close to an identity update
    without severing the action path.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.context_norm = nn.LayerNorm(latent_dim)
        self.condition = nn.Linear(latent_dim, latent_dim * 3)
        self.update = DynamicsMLP(latent_dim, hidden_dim, latent_dim, dropout)

        nn.init.normal_(self.condition.weight, std=0.02)
        nn.init.zeros_(self.condition.bias)
        with torch.no_grad():
            self.condition.bias[latent_dim * 2 :].fill_(-2.0)

    def forward(self, hidden: torch.Tensor, action_latent: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.condition(action_latent).chunk(3, dim=-1)
        conditioned = self.context_norm(hidden) * (1.0 + scale) + shift
        return hidden + torch.sigmoid(gate) * self.update(conditioned)


class ActionConditionedDynamics(nn.Module):
    """``z_(t+1) = z_t + delta(h_t, a_t)``: the visual latent is the residual
    anchor, the recurrent output supplies the history, and the action only ever
    modulates this module -- never the value head.

    ``action_encoder`` maps a step's action to one latent-sized vector, which is
    what decides whether the actions are discrete branch indexes (an embedding
    table) or continuous vectors (an MLP, as in WCM).
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
        action_encoder: nn.Module,
    ) -> None:
        super().__init__()
        assert depth >= 1, f"dynamics_depth must be at least 1, got {depth}"
        self.action_encoder = action_encoder
        self.blocks = nn.ModuleList(
            GatedDynamicsBlock(latent_dim, hidden_dim, dropout) for _ in range(depth)
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )
        nn.init.normal_(self.delta_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(
        self,
        current_state_latent: torch.Tensor,
        context: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        assert current_state_latent.shape == context.shape, (
            f"Current-state/context shapes differ: {current_state_latent.shape} vs {context.shape}"
        )
        action_latent = self.action_encoder(actions)
        assert action_latent.shape == context.shape, (
            f"Encoded actions must be {context.shape}, got {action_latent.shape}"
        )
        hidden = context
        for block in self.blocks:
            hidden = block(hidden, action_latent)
        return current_state_latent + self.delta_head(hidden)


def next_state_prediction_loss(
    predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Mean squared error over the ``valid`` step pairs of ``(B, T, latent_dim)``
    predictions; ``valid`` is ``(B, T, 1)`` and drops the pairs that cross an
    episode boundary."""
    squared = (predicted - target).square() * valid
    return squared.sum() / valid.sum().clamp_min(1.0) / predicted.shape[-1]


class AnimalWorldCriticNetwork(AnimalPPONetwork):
    """:class:`AnimalPPONetwork` plus the world-critic branch.

    ``forward`` is the parent's, so acting is unchanged and existing PPO
    checkpoints load into the body. Only ``forward_for_update`` differs: off one
    pass of the trunk it reads the two latents the auxiliary losses need -- the
    per-step visual latent the prediction target is taken from, and the recurrent
    context the prediction is conditioned on -- and returns their loss alongside
    the heads the PPO update already expects.
    """

    def __init__(
        self,
        observation_space_shape: tuple[int, ...],
        vels_size: int,
        image_encoder_type: str,
        image_encoder_output_dim: int,
        image_encode_mode: str,
        image_encoder_trainable: bool,
        temporal_model_type: str,
        latent_dim: int,
        dynamics_depth: int,
        dynamics_mlp_ratio: float,
        dynamics_dropout: float,
        next_state_coef: float,
        sigreg_coef: float,
        sigreg_knots: int,
        sigreg_projections: int,
    ) -> None:
        super().__init__(
            observation_space_shape,
            vels_size,
            image_encoder_type,
            image_encoder_output_dim,
            image_encode_mode,
            image_encoder_trainable,
            temporal_model_type,
        )
        self.latent_dim = latent_dim
        self.next_state_coef = next_state_coef
        self.sigreg_coef = sigreg_coef
        self.state_projection = nn.Sequential(
            nn.LayerNorm(HIDDEN_NODES),
            nn.Linear(HIDDEN_NODES, latent_dim),
        )
        self.context_projection = nn.Sequential(
            nn.LayerNorm(TEMPORAL_UNITS),
            nn.Linear(TEMPORAL_UNITS, latent_dim),
        )
        action_embedding = nn.Embedding(ACTION_NUM, latent_dim)
        nn.init.normal_(action_embedding.weight, std=0.02)
        self.dynamics = ActionConditionedDynamics(
            latent_dim=latent_dim,
            hidden_dim=int(latent_dim * dynamics_mlp_ratio),
            depth=dynamics_depth,
            dropout=dynamics_dropout,
            action_encoder=action_embedding,
        )
        self.sigreg = SIGReg(sigreg_knots, sigreg_projections)

    def forward_for_update(
        self,
        visual: torch.Tensor,
        vels: torch.Tensor,
        state: torch.Tensor,
        dones: torch.Tensor,
        actions: torch.Tensor,
        sequence_num: int,
    ) -> tuple:
        visual_latent, hidden = self.embed(visual, vels)
        temporal_out, _ = self.recurrent(hidden, state, dones, sequence_num)
        state_latent = self.state_projection(visual_latent).reshape(
            sequence_num, -1, self.latent_dim
        )
        context_latent = self.context_projection(temporal_out).reshape(
            sequence_num, -1, self.latent_dim
        )
        # the window is sequence-major, so a pair is the step and the one after
        # it; a step flagged fresh starts an episode and cuts the pair before it
        fresh = dones.reshape(sequence_num, -1)
        predicted = self.dynamics(
            state_latent[:, :-1], context_latent[:, :-1], actions.reshape(sequence_num, -1)[:, :-1]
        )
        target = state_latent[:, 1:].detach()
        valid = (fresh[:, 1:] < 0.5).to(predicted.dtype).unsqueeze(-1)

        next_state_loss = next_state_prediction_loss(predicted, target, valid)
        sigreg_loss = self.sigreg.loss(context_latent)
        auxiliary_loss = self.next_state_coef * next_state_loss + self.sigreg_coef * sigreg_loss
        reported = {
            "next_state": float(next_state_loss.item()),
            "sigreg": float(sigreg_loss.item()),
        }
        return (
            self.logits_head(temporal_out),
            self.value_head(temporal_out).squeeze(-1),
            auxiliary_loss,
            reported,
        )
