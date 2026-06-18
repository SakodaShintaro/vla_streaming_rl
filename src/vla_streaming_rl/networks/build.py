# SPDX-License-Identifier: MIT
import functools
from collections.abc import Callable

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from vla_streaming_rl.networks.value_head import (
    ActionValueHead,
    DistributionalValueHead,
    HypersphericalActionValueHead,
)


def _build_value_head(
    in_channels: int,
    *,
    critic_arch: str,
    action_dim: int,
    horizon: int,
    gamma: float,
    multi_gammas: list[float],
    hidden_dim: int,
    block_num: int,
    num_bins: int,
    sparsity: float,
) -> DistributionalValueHead:
    """Build the action-value head for a state of width ``in_channels``.

    Networks receive this (via ``functools.partial`` fixing everything but
    ``in_channels``) and call it with their own state dim. All value-related
    construction — critic architecture, discount, bins, hidden sizes — lives
    here, so a value change touches only this builder and ``value_head``, never
    the networks.

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
    device: torch.device,
) -> nn.Module:
    if args.network_class == "actor_critic_with_action_value":
        from vla_streaming_rl.networks.actor_critic_with_action_value import (
            ActorCriticWithActionValue,
        )

        value_head_factory = functools.partial(
            _build_value_head,
            critic_arch=args.critic_arch,
            action_dim=action_space_shape[0],
            horizon=args.horizon,
            gamma=args.gamma,
            multi_gammas=list(args.multi_gammas),
            hidden_dim=args.critic_hidden_dim,
            block_num=args.critic_block_num,
            num_bins=args.num_bins,
            sparsity=args.sparsity,
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
        ).to(device)
        network = torch.compile(network)
    elif args.network_class == "vlm_actor_critic_with_action_value":
        from vla_streaming_rl.networks.vlm_actor_critic_with_action_value import (
            VLMActorCriticWithActionValue,
        )

        value_head_factory = functools.partial(
            _build_value_head,
            critic_arch=args.critic_arch,
            action_dim=action_space_shape[0],
            horizon=args.horizon,
            gamma=args.gamma,
            multi_gammas=list(args.multi_gammas),
            hidden_dim=args.critic_hidden_dim,
            block_num=args.critic_block_num,
            num_bins=args.num_bins,
            sparsity=args.sparsity,
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
        ).to(device)
    elif args.network_class == "simlingo":
        from vla_streaming_rl.networks.simlingo_network import ACTION_DIM, SimLingoNetwork

        # SimLingo's dueling critic is always scalar (num_bins == 1); SimbaV2
        # honors the configured bins. The critic acts on a single (horizon == 1)
        # 60-D waypoint action.
        critic_num_bins = args.num_bins if args.critic_arch == "simbav2" else 1
        value_head_factory = functools.partial(
            _build_value_head,
            critic_arch=args.critic_arch,
            action_dim=ACTION_DIM,
            horizon=1,
            gamma=args.gamma,
            multi_gammas=list(args.multi_gammas),
            hidden_dim=args.critic_hidden_dim,
            block_num=args.critic_block_num,
            num_bins=critic_num_bins,
            sparsity=0.0,
        )

        network = SimLingoNetwork(
            value_head_factory=value_head_factory,
            device=device,
            actor_loss_type=args.actor_loss_type,
            awr_num_samples=args.awr_num_samples,
            awr_temperature=args.awr_temperature,
            awr_sample_noise=args.awr_sample_noise,
        ).to(device)
    elif args.network_class == "libero_pi05":
        from vla_streaming_rl.networks.libero_pi05_network import LiberoPi05Network

        # pi0.5 loads its own pretrained backbone + processors from the HF
        # checkpoint; the env observation/action shapes are unused.
        network = LiberoPi05Network(
            checkpoint_repo=args.pi05_checkpoint_repo,
            device=device,
        )
    else:
        raise ValueError(f"Unknown network class: {args.network_class}")

    return network
