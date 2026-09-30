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
    detach_state: bool,
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
            detach_state=detach_state,
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
            detach_state=detach_state,
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
            temporal_model_type=args.actor_critic.temporal_model_type,
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
            critic_loss_weight=args.critic_loss_weight,
            prediction_head_factory=hydra.utils.instantiate(args.prediction_head),
            actor_critic_config=args.actor_critic,
            horizon=args.horizon,
            policy_head_factory=policy_head_factory,
            high_level_config=args.high_level,
            prompt_builder=prompt_builder,
        ).to(device)

    elif args.network_class == "vlm_actor_critic_with_action_value":
        # このネットワークは会話をリプレイバッファの seq_len 行から組み直すので、
        # 高レベル方策の窓も同じ長さでなければならない
        assert args.high_level.seq_len == args.vla.seq_len, (
            f"high_level.seq_len {args.high_level.seq_len} must equal vla.seq_len {args.vla.seq_len}"
        )
        from vla_streaming_rl.networks.vlm_actor_critic_with_action_value import (
            VLMActorCriticWithActionValue,
        )

        network = VLMActorCriticWithActionValue(
            observation_space_shape=observation_space_shape,
            action_space_shape=action_space_shape,
            value_head_factory=value_head_factory,
            horizon=args.horizon,
            critic_loss_weight=args.critic_loss_weight,
            policy_head_factory=policy_head_factory,
            vla_config=args.vla,
            pad_token_id=args.replay_buffer.pad_token_id,
            cot_steps_per_chain=args.high_level.cot_steps_per_chain,
        ).to(device)

    else:
        raise ValueError(f"Unknown network class: {args.network_class}")

    return network
