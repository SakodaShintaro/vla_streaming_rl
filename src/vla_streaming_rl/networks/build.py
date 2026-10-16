# SPDX-License-Identifier: MIT

from collections.abc import Callable

import hydra
import numpy as np
import torch
from gymnasium import Env
from omegaconf import DictConfig
from torch import nn

from vla_streaming_rl.image_obs import image_obs_names


def build_network(
    args: DictConfig,
    env: Env,
    parse_action_text: Callable[[str], tuple[np.ndarray, bool]],
    text_action_names: list[str],
    device: torch.device,
) -> nn.Module:
    """``parse_action_text`` と ``text_action_names`` は返答に書かせる行動の決まりで、
    環境ごとのプロンプトの builder が持つ。返答を書くネットワークだけが使う。"""
    # 観測の画像ごとの (3, H, W)、IMAGE_OBS_NAMES の並び
    image_shapes = [
        env.observation_space[name].shape for name in image_obs_names(env.observation_space)
    ]
    action_space_shape = env.action_space.shape
    # 全ネットワーク共通のヘッド工場。どのクラスを使うかと設定は config の policy_head と
    # value_head（configs/policy_head, configs/value_head から選ぶ）が決め、ネットワークは
    # 自分しか知らない状態の幅と action_dim を呼び出し時に渡す
    value_head_factory = hydra.utils.instantiate(args.value_head)

    policy_head_factory = hydra.utils.instantiate(args.policy_head)

    if args.network_class == "actor_critic_with_action_value":
        from vla_streaming_rl.networks.actor_critic_with_action_value import (
            ActorCriticWithActionValue,
        )

        network = ActorCriticWithActionValue(
            image_shapes=image_shapes,
            action_space_shape=action_space_shape,
            value_head_factory=value_head_factory,
            critic_loss_weight=args.critic_loss_weight,
            prediction_head_factory=hydra.utils.instantiate(args.prediction_head),
            actor_critic_config=args.actor_critic,
            horizon=args.horizon,
            policy_head_factory=policy_head_factory,
        ).to(device)

    elif args.network_class == "vlm_actor_critic_with_action_value":
        from vla_streaming_rl.networks.vlm_actor_critic_with_action_value import (
            VLMActorCriticWithActionValue,
        )

        network = VLMActorCriticWithActionValue(
            action_space_shape=action_space_shape,
            value_head_factory=value_head_factory,
            horizon=args.horizon,
            critic_loss_weight=args.critic_loss_weight,
            policy_head_factory=policy_head_factory,
            vla_config=args.vla,
            parse_action_text=parse_action_text,
            text_action_names=text_action_names,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            decision_fps=env.metadata["decision_fps"],
        ).to(device)

    else:
        raise ValueError(f"Unknown network class: {args.network_class}")

    return network
