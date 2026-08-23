# SPDX-License-Identifier: MIT
"""The off-policy Animal-AI actor-critic with WCM's world-critic objective.

The same branch ``networks/animal_world_critic.py`` adds to the PPO network,
moved onto :class:`AnimalActorCriticWithActionValue` so it runs under
:class:`~vla_streaming_rl.agents.off_policy.OffPolicyAgent` and
:class:`~vla_streaming_rl.agents.streaming.StreamingAgent`. Nothing about the
objective needs on-policy data: both terms are self-supervised on the window
itself, so replayed windows train them exactly as fresh ones do.

Two differences from the PPO port:

- the actions are the continuous chunks the replay buffer stores, so the action
  encoder is WCM's own MLP rather than an embedding table, and
- the window comes from the replay buffer, which samples across episode
  boundaries, so ``dones`` masks the step pairs that cross one.
"""

import torch
from torch import nn

from vla_streaming_rl.networks.animal_actor_critic import (
    AnimalActorCriticWithActionValue,
)
from vla_streaming_rl.networks.animal_ppo import HIDDEN_NODES, TEMPORAL_UNITS
from vla_streaming_rl.networks.animal_world_critic import (
    ActionConditionedDynamics,
    DynamicsMLP,
    SIGReg,
    next_state_prediction_loss,
)
from vla_streaming_rl.replay_buffer import ReplayBufferData


class AnimalWorldCriticActorCritic(AnimalActorCriticWithActionValue):
    def __init__(
        self,
        *,
        observation_space_shape: tuple[int, ...],
        action_space_shape: tuple[int, ...],
        value_head_factory,
        horizon: int,
        policy_type: str,
        actor_hidden_dim: int,
        actor_block_num: int,
        denoising_time: float,
        denoising_steps: int,
        dacer_loss_weight: float,
        som_alpha: float,
        som_w: float,
        sparsity: float,
        critic_loss_weight: float,
        detach_actor: bool,
        detach_critic: bool,
        image_encoder_type: str,
        image_encoder_output_dim: int,
        image_encode_mode: str,
        image_encoder_trainable: bool,
        temporal_model_type: str,
        wcm_latent_dim: int,
        wcm_dynamics_depth: int,
        wcm_dynamics_mlp_ratio: float,
        wcm_dynamics_dropout: float,
        wcm_next_state_coef: float,
        wcm_sigreg_coef: float,
        wcm_sigreg_knots: int,
        wcm_sigreg_projections: int,
    ) -> None:
        super().__init__(
            observation_space_shape=observation_space_shape,
            action_space_shape=action_space_shape,
            value_head_factory=value_head_factory,
            horizon=horizon,
            policy_type=policy_type,
            actor_hidden_dim=actor_hidden_dim,
            actor_block_num=actor_block_num,
            denoising_time=denoising_time,
            denoising_steps=denoising_steps,
            dacer_loss_weight=dacer_loss_weight,
            som_alpha=som_alpha,
            som_w=som_w,
            sparsity=sparsity,
            critic_loss_weight=critic_loss_weight,
            detach_actor=detach_actor,
            detach_critic=detach_critic,
            image_encoder_type=image_encoder_type,
            image_encoder_output_dim=image_encoder_output_dim,
            image_encode_mode=image_encode_mode,
            image_encoder_trainable=image_encoder_trainable,
            temporal_model_type=temporal_model_type,
        )
        self.wcm_next_state_coef = wcm_next_state_coef
        self.wcm_sigreg_coef = wcm_sigreg_coef
        self.state_projection = nn.Sequential(
            nn.LayerNorm(HIDDEN_NODES),
            nn.Linear(HIDDEN_NODES, wcm_latent_dim),
        )
        self.context_projection = nn.Sequential(
            nn.LayerNorm(TEMPORAL_UNITS),
            nn.Linear(TEMPORAL_UNITS, wcm_latent_dim),
        )
        dynamics_hidden_dim = int(wcm_latent_dim * wcm_dynamics_mlp_ratio)
        self.dynamics = ActionConditionedDynamics(
            latent_dim=wcm_latent_dim,
            hidden_dim=dynamics_hidden_dim,
            depth=wcm_dynamics_depth,
            dropout=wcm_dynamics_dropout,
            action_encoder=DynamicsMLP(
                self.action_dim, dynamics_hidden_dim, wcm_latent_dim, wcm_dynamics_dropout
            ),
        )
        self.sigreg = SIGReg(wcm_sigreg_knots, wcm_sigreg_projections)

    def _encode_current_window(
        self, data: ReplayBufferData
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        temporal_out, visual_latent, _ = self.encoder.forward_steps(
            *self._window(data, 0, -self.horizon)
        )
        assert temporal_out.shape[1] >= 2, (
            f"The world-critic loss needs a window of at least 2 steps, got {temporal_out.shape[1]}: "
            "raise seq_len"
        )
        state_latent = self.state_projection(visual_latent)
        context_latent = self.context_projection(temporal_out)

        actions = data.actions[:, 0 : -self.horizon]
        predicted = self.dynamics(state_latent[:, :-1], context_latent[:, :-1], actions[:, :-1])
        target = state_latent[:, 1:].detach()
        dones = data.dones[:, 0 : -self.horizon].reshape(temporal_out.shape[0], -1)
        valid = (dones[:, :-1] < 0.5).to(predicted.dtype).unsqueeze(-1)

        next_state_loss = next_state_prediction_loss(predicted, target, valid)
        sigreg_loss = self.sigreg.loss(context_latent)
        auxiliary_loss = (
            self.wcm_next_state_coef * next_state_loss + self.wcm_sigreg_coef * sigreg_loss
        )
        auxiliary_info = {
            "wcm_next_state": next_state_loss.item(),
            "wcm_sigreg": sigreg_loss.item(),
        }
        return temporal_out[:, -1], auxiliary_loss, auxiliary_info
