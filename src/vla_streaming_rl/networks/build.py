# SPDX-License-Identifier: MIT

import hydra
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

    Networks receive this (via the config's ``value_head`` node fixing every knob but
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
            temporal_model_type=args.temporal_model_type,
        ).to(device)

    # 全ネットワーク共通のヘッド工場。critic の設定は config の value_head
    # ノードが単一ソースで束ね、ネットワークは自分しか知らない in_channels と
    # action_dim を呼び出し時に渡す。
    value_head_factory = hydra.utils.instantiate(args.value_head)

    policy_head_factory = hydra.utils.instantiate(args.policy_head)

    if args.network_class == "actor_critic_with_action_value":
        from vla_streaming_rl.networks.actor_critic_with_action_value import (
            ActorCriticWithActionValue,
        )

        network = ActorCriticWithActionValue(
            observation_space_shape=observation_space_shape,
            action_space_shape=action_space_shape,
            value_head_factory=value_head_factory,
            seq_len=args.seq_len,
            critic_loss_weight=args.critic_loss_weight,
            predictor_step_num=args.predictor_step_num,
            encoder_block_num=args.encoder_block_num,
            layer_scale_init=args.layer_scale_init,
            temporal_model_type=args.temporal_model_type,
            horizon=args.horizon,
            policy_head_factory=policy_head_factory,
            predictor_hidden_dim=args.predictor_hidden_dim,
            predictor_block_num=args.predictor_block_num,
            detach_actor=args.detach_actor,
            detach_critic=args.detach_critic,
            detach_predictor=args.detach_predictor,
            disable_state_predictor=args.disable_state_predictor,
            predictor_type=args.predictor_type,
            image_encoder_type=args.image_encoder_type,
            image_encoder_output_dim=args.image_encoder_output_dim,
            vlm_model_id=args.vlm_model_id,
            vlm_load_in_4bit=args.vlm_load_in_4bit,
            cot_tokens_num=args.cot_tokens_num,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            cot_mode=args.cot_mode,
            cot_steps_per_chain=args.cot_steps_per_chain,
            cot_dropout=args.cot_dropout,
            token_dropout=args.token_dropout,
            bc_loss_weight=args.bc_loss_weight,
            cot_pool=args.cot_pool,
            cot_cuda_graph=args.cot_cuda_graph,
            chain_generator_factory=hydra.utils.instantiate(args.chain_generator),
            prompt_builder=prompt_builder,
        ).to(device)

    elif args.network_class == "vlm_actor_critic_with_action_value":
        from vla_streaming_rl.networks.vlm_actor_critic_with_action_value import (
            VLMActorCriticWithActionValue,
        )

        network = VLMActorCriticWithActionValue(
            observation_space_shape=observation_space_shape,
            action_space_shape=action_space_shape,
            value_head_factory=value_head_factory,
            seq_len=args.seq_len,
            horizon=args.horizon,
            critic_loss_weight=args.critic_loss_weight,
            policy_head_factory=policy_head_factory,
            reasoning_loss_weight=args.reasoning_loss_weight,
            reasoning_max_tokens=args.reasoning_max_tokens,
            reasoning_temperature=args.reasoning_temperature,
            predictor_step_num=args.predictor_step_num,
            disable_state_predictor=args.disable_state_predictor,
            detach_actor=args.detach_actor,
            detach_critic=args.detach_critic,
            detach_predictor=args.detach_predictor,
            use_lora=args.use_lora,
            vlm_model_id=args.vlm_model_id,
            vlm_load_in_4bit=args.vlm_load_in_4bit,
            pad_token_id=args.pad_token_id,
            num_state_queries=args.num_state_queries,
            state_out_dim=args.state_out_dim,
            predictor_hidden_dim=args.predictor_hidden_dim,
            predictor_block_num=args.predictor_block_num,
            cot_steps_per_chain=args.cot_steps_per_chain,
            predictor_type=args.predictor_type,
            image_encoder_type=args.image_encoder_type,
            image_encoder_output_dim=args.image_encoder_output_dim,
        ).to(device)

    else:
        raise ValueError(f"Unknown network class: {args.network_class}")

    return network
