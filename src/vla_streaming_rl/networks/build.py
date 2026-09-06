# SPDX-License-Identifier: MIT
import functools
from collections.abc import Callable

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from vla_streaming_rl.networks.modules.value_head import (
    ActionValueHead,
    DistributionalValueHead,
    HypersphericalActionValueHead,
)


def _build_value_head(
    in_channels: int,
    action_dim: int,
    *,
    critic_arch: str,
    horizon: int,
    gamma: float,
    multi_gammas: list[float],
    hidden_dim: int,
    block_num: int,
    num_bins: int,
    sparsity: float,
) -> DistributionalValueHead:
    """Build the action-value head for a state of width ``in_channels``.

    Networks receive this (via ``functools.partial`` fixing every config knob but
    ``in_channels`` and ``action_dim``) and call it with their own state width and
    action dim — the two shape values only the network knows. All value-related
    construction — critic architecture, discount, bins, hidden sizes — lives here,
    so a value change touches only this builder and ``value_head``, never the
    networks.

    Multi-gamma (AMAGO): the head predicts the action value for several discounts
    at once. ``gamma`` is the primary/rollout discount and is kept last; the
    auxiliary ``multi_gammas`` come first. The head owns this list and builds the
    per-gamma TD target / loss; empty ``multi_gammas`` == single-gamma (original).
    """
    gammas = list(multi_gammas) + [gamma]
    if critic_arch == "simbav2":
        return HypersphericalActionValueHead(
            in_channels=in_channels,
            action_dim=action_dim,
            horizon=horizon,
            gammas=gammas,
            hidden_dim=hidden_dim,
            block_num=block_num,
            num_bins=num_bins,
        )
    if critic_arch == "dueling":
        return ActionValueHead(
            in_channels=in_channels,
            action_dim=action_dim,
            horizon=horizon,
            gammas=gammas,
            hidden_dim=hidden_dim,
            block_num=block_num,
            num_bins=num_bins,
            sparsity=sparsity,
        )
    raise ValueError(f"Unknown critic_arch: {critic_arch!r} (expected 'simbav2'/'dueling')")


def build_network(
    args: DictConfig,
    observation_space_shape: tuple[int, ...],
    action_space_shape: tuple[int, ...],
    parse_action_text: Callable[[str], tuple[np.ndarray, bool]] | None,
    prompt_builder,
    device: torch.device,
) -> nn.Module:
    # The PPO network has its own value head baked in and reads no critic
    # config, so it is settled before the shared value-head factory is built.
    if args.network_class == "animal_ppo":
        from vla_streaming_rl.networks.animal_ppo import AnimalPPONetwork

        # scaled velocity (vx, vy, vz) plus the health that stands in for the clock
        return AnimalPPONetwork(
            observation_space_shape=observation_space_shape,
            vels_size=4,
            image_encoder_type=args.image_encoder_type,
            image_encoder_output_dim=args.image_encoder_output_dim,
            image_encode_mode=args.image_encode_mode,
            image_encoder_trainable=args.image_encoder_trainable,
            temporal_model_type=args.temporal_model_type,
        ).to(device)

    if args.network_class == "animal_world_critic":
        from vla_streaming_rl.networks.animal_world_critic import AnimalWorldCriticNetwork

        # a window of n steps carries n - 1 next-state pairs, so 1 trains nothing
        assert args.seq_len >= 2, (
            f"seq_len {args.seq_len} leaves no next-state pair for the world-critic loss"
        )

        return AnimalWorldCriticNetwork(
            observation_space_shape=observation_space_shape,
            vels_size=4,
            image_encoder_type=args.image_encoder_type,
            image_encoder_output_dim=args.image_encoder_output_dim,
            image_encode_mode=args.image_encode_mode,
            image_encoder_trainable=args.image_encoder_trainable,
            temporal_model_type=args.temporal_model_type,
            latent_dim=args.wcm_latent_dim,
            dynamics_depth=args.wcm_dynamics_depth,
            dynamics_mlp_ratio=args.wcm_dynamics_mlp_ratio,
            dynamics_dropout=args.wcm_dynamics_dropout,
            next_state_coef=args.wcm_next_state_coef,
            sigreg_coef=args.wcm_sigreg_coef,
            sigreg_knots=args.wcm_sigreg_knots,
            sigreg_projections=args.wcm_sigreg_projections,
        ).to(device)

    # One factory for every network: all critic config comes from ``args`` and is
    # bound here, leaving ``in_channels`` and ``action_dim`` for the network to
    # supply at call time (see ``_build_value_head``).
    value_head_factory = functools.partial(
        _build_value_head,
        critic_arch=args.critic_arch,
        horizon=args.horizon,
        gamma=args.gamma,
        multi_gammas=list(args.multi_gammas),
        hidden_dim=args.critic_hidden_dim,
        block_num=args.critic_block_num,
        num_bins=args.num_bins,
        sparsity=args.sparsity,
    )

    if args.network_class == "actor_critic_with_action_value":
        from vla_streaming_rl.networks.actor_critic_with_action_value import (
            ActorCriticWithActionValue,
        )

        network = ActorCriticWithActionValue(
            observation_space_shape=observation_space_shape,
            action_space_shape=action_space_shape,
            value_head_factory=value_head_factory,
            sparsity=args.sparsity,
            seq_len=args.seq_len,
            dacer_loss_weight=args.dacer_loss_weight,
            critic_loss_weight=args.critic_loss_weight,
            predictor_step_num=args.predictor_step_num,
            encoder_block_num=args.encoder_block_num,
            layer_scale_init=args.layer_scale_init,
            temporal_model_type=args.temporal_model_type,
            horizon=args.horizon,
            policy_type=args.policy_type,
            actor_hidden_dim=args.actor_hidden_dim,
            actor_block_num=args.actor_block_num,
            denoising_time=args.denoising_time,
            denoising_steps=args.denoising_steps,
            som_alpha=args.som_alpha,
            som_w=args.som_w,
            predictor_hidden_dim=args.predictor_hidden_dim,
            predictor_block_num=args.predictor_block_num,
            detach_actor=args.detach_actor,
            detach_critic=args.detach_critic,
            detach_predictor=args.detach_predictor,
            disable_state_predictor=args.disable_state_predictor,
            predictor_type=args.predictor_type,
            image_encoder_type=args.image_encoder_type,
            image_encoder_output_dim=args.image_encoder_output_dim,
            image_encode_mode=args.image_encode_mode,
            image_encoder_trainable=args.image_encoder_trainable,
            vlm_model_id=args.vlm_model_id,
            cot_tokens_num=args.cot_tokens_num,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            cot_mode=args.cot_mode,
            cot_steps_per_chain=args.cot_steps_per_chain,
            cot_pool=args.cot_pool,
            cot_cuda_graph=args.cot_cuda_graph,
            prompt_builder=prompt_builder,
        ).to(device)

    elif args.network_class == "animal_actor_critic":
        from vla_streaming_rl.networks.animal_actor_critic import AnimalActorCriticWithActionValue

        network = AnimalActorCriticWithActionValue(
            observation_space_shape=observation_space_shape,
            action_space_shape=action_space_shape,
            value_head_factory=value_head_factory,
            horizon=args.horizon,
            policy_type=args.policy_type,
            actor_hidden_dim=args.actor_hidden_dim,
            actor_block_num=args.actor_block_num,
            denoising_time=args.denoising_time,
            denoising_steps=args.denoising_steps,
            dacer_loss_weight=args.dacer_loss_weight,
            som_alpha=args.som_alpha,
            som_w=args.som_w,
            sparsity=args.sparsity,
            critic_loss_weight=args.critic_loss_weight,
            detach_actor=args.detach_actor,
            detach_critic=args.detach_critic,
            image_encoder_type=args.image_encoder_type,
            image_encoder_output_dim=args.image_encoder_output_dim,
            image_encode_mode=args.image_encode_mode,
            image_encoder_trainable=args.image_encoder_trainable,
            temporal_model_type=args.temporal_model_type,
        ).to(device)

    elif args.network_class == "vlm_actor_critic_with_action_value":
        from vla_streaming_rl.networks.vlm_actor_critic_with_action_value import (
            VLMActorCriticWithActionValue,
        )

        network = VLMActorCriticWithActionValue(
            observation_space_shape=observation_space_shape,
            action_space_shape=action_space_shape,
            parse_action_text=parse_action_text,
            value_head_factory=value_head_factory,
            seq_len=args.seq_len,
            horizon=args.horizon,
            critic_loss_weight=args.critic_loss_weight,
            denoising_steps=args.denoising_steps,
            denoising_time=args.denoising_time,
            dacer_loss_weight=args.dacer_loss_weight,
            som_alpha=args.som_alpha,
            som_w=args.som_w,
            text_q_margin=args.text_q_margin,
            text_action_mode=args.text_action_mode,
            predictor_step_num=args.predictor_step_num,
            disable_state_predictor=args.disable_state_predictor,
            detach_actor=args.detach_actor,
            detach_critic=args.detach_critic,
            detach_predictor=args.detach_predictor,
            use_lora=args.use_lora,
            vlm_model_id=args.vlm_model_id,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            pad_token_id=args.pad_token_id,
            num_state_queries=args.num_state_queries,
            state_out_dim=args.state_out_dim,
            actor_hidden_dim=args.actor_hidden_dim,
            actor_block_num=args.actor_block_num,
            predictor_hidden_dim=args.predictor_hidden_dim,
            predictor_block_num=args.predictor_block_num,
            sparsity=args.sparsity,
            image_mode=args.image_mode,
            predictor_type=args.predictor_type,
            policy_type=args.policy_type,
            image_encoder_type=args.image_encoder_type,
            image_encoder_output_dim=args.image_encoder_output_dim,
            image_encode_mode=args.image_encode_mode,
            image_encoder_trainable=args.image_encoder_trainable,
        ).to(device)

    else:
        raise ValueError(f"Unknown network class: {args.network_class}")

    return network
