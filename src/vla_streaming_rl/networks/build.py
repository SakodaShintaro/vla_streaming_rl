# SPDX-License-Identifier: MIT
from collections.abc import Callable

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn


def build_network(
    args: DictConfig,
    observation_space_shape: tuple[int, ...],
    action_space_shape: tuple[int, ...],
    parse_action_text: Callable[[str], tuple[np.ndarray, bool]] | None,
    device: torch.device,
    compile: bool,
) -> nn.Module:
    if args.network_class == "actor_critic_with_state_value":
        from vla_streaming_rl.networks.actor_critic_with_state_value import (
            ActorCriticWithStateValue,
        )

        network = ActorCriticWithStateValue(
            observation_space_shape=observation_space_shape,
            action_space_shape=action_space_shape,
            gamma=args.gamma,
            clip_param_policy=args.clip_param_policy,
            clip_param_value=args.clip_param_value,
            num_bins=args.num_bins,
            predictor_step_num=args.predictor_step_num,
            critic_loss_weight=args.critic_loss_weight,
            encoder=args.encoder,
            seq_len=args.seq_len,
            encoder_block_num=args.encoder_block_num,
            temporal_model_type=args.temporal_model_type,
            horizon=args.horizon,
            critic_block_num=args.critic_block_num,
            policy_type=args.policy_type,
            predictor_hidden_dim=args.predictor_hidden_dim,
            predictor_block_num=args.predictor_block_num,
            disable_state_predictor=args.disable_state_predictor,
            predictor_type=args.predictor_type,
        )
    elif args.network_class == "actor_critic_with_action_value":
        from vla_streaming_rl.networks.actor_critic_with_action_value import (
            ActorCriticWithActionValue,
        )

        network = ActorCriticWithActionValue(
            observation_space_shape=observation_space_shape,
            action_space_shape=action_space_shape,
            gamma=args.gamma,
            num_bins=args.num_bins,
            sparsity=args.sparsity,
            seq_len=args.seq_len,
            dacer_loss_weight=args.dacer_loss_weight,
            critic_loss_weight=args.critic_loss_weight,
            predictor_step_num=args.predictor_step_num,
            encoder=args.encoder,
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
            critic_hidden_dim=args.critic_hidden_dim,
            critic_block_num=args.critic_block_num,
            predictor_hidden_dim=args.predictor_hidden_dim,
            predictor_block_num=args.predictor_block_num,
            detach_actor=args.detach_actor,
            detach_critic=args.detach_critic,
            detach_predictor=args.detach_predictor,
            disable_state_predictor=args.disable_state_predictor,
            predictor_type=args.predictor_type,
        )
    elif args.network_class == "vlm_actor_critic_with_action_value":
        from vla_streaming_rl.networks.vlm_actor_critic_with_action_value import (
            VLMActorCriticWithActionValue,
        )

        network = VLMActorCriticWithActionValue(
            observation_space_shape=observation_space_shape,
            action_space_shape=action_space_shape,
            parse_action_text=parse_action_text,
            gamma=args.gamma,
            num_bins=args.num_bins,
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
            critic_hidden_dim=args.critic_hidden_dim,
            critic_block_num=args.critic_block_num,
            predictor_hidden_dim=args.predictor_hidden_dim,
            predictor_block_num=args.predictor_block_num,
            sparsity=args.sparsity,
            image_mode=args.image_mode,
            predictor_type=args.predictor_type,
            policy_type=args.policy_type,
        )
    elif args.network_class == "simlingo":
        from vla_streaming_rl.networks.simlingo_network import SimLingoNetwork

        network = SimLingoNetwork(
            critic_arch=args.critic_arch,
            critic_hidden_dim=args.critic_hidden_dim,
            critic_block_num=args.critic_block_num,
            critic_c_shift=args.critic_c_shift,
            num_bins=args.num_bins,
            device=device,
        )
    else:
        raise ValueError(f"Unknown network class: {args.network_class}")

    network = network.to(device)
    if compile:
        network = torch.compile(network)
    return network
