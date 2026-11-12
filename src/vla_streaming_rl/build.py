# SPDX-License-Identifier: MIT
import hydra
from gymnasium import Env
from omegaconf import DictConfig

from vla_streaming_rl.agents.instruction import ENV_INSTRUCTIONS, TOPDOWN_INSTRUCTION
from vla_streaming_rl.agents.off_policy import OffPolicyAgent
from vla_streaming_rl.image_obs import image_obs_names
from vla_streaming_rl.networks.vla_actor_critic_with_action_value import (
    VLAActorCriticWithActionValue,
)
from vla_streaming_rl.reward_scaler import build_reward_scaler


def build_all(env: Env, args: DictConfig) -> tuple[VLAActorCriticWithActionValue, OffPolicyAgent]:
    """config が表すネットワークとエージェントを、どの入口のスクリプトでも同じようにつないだもの。"""
    image_names = image_obs_names(env.observation_space)
    # ヘッドのクラスと設定は config の policy_head / value_head / prediction_head が決め、
    # ネットワークは自分しか知らない状態の幅と action_dim を呼び出し時に渡す。
    # 4 ビットに量子化した VLM を含みうるので .to(device) はせず、内部で cuda に置く
    network = VLAActorCriticWithActionValue(
        image_shapes=[env.observation_space[name].shape for name in image_names],
        action_space_shape=env.action_space.shape,
        value_head_factory=hydra.utils.instantiate(args.value_head),
        policy_head_factory=hydra.utils.instantiate(args.policy_head),
        prediction_head_factory=hydra.utils.instantiate(args.prediction_head),
        horizon=args.horizon,
        critic_loss_weight=args.critic_loss_weight,
        vla_config=args.vla,
        # LoRA 付きの GDN の層は streaming の更新の RTRL で学ぶので、そのときだけ持つ
        use_memory=args.streaming_update,
    )
    instruction = ENV_INSTRUCTIONS[args.env_id]
    if "topdown_image" in image_names:
        instruction = f"{instruction} {TOPDOWN_INSTRUCTION}"
    agent = OffPolicyAgent(
        action_space=env.action_space,
        image_names=image_names,
        network=network,
        normalizing_by_return=args.normalizing_by_return,
        reward_scaler=build_reward_scaler(args.reward_scale_mode, args.fixed_reward_scale),
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        max_grad_norm=args.max_grad_norm,
        use_done=args.use_done,
        seq_len=network.seq_len,
        horizon=args.horizon,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        weight_decay=args.weight_decay,
        buffer_size=args.replay_buffer.size,
        buffer_device=args.replay_buffer.device,
        priority_alpha=args.replay_buffer.priority_alpha,
        priority_beta=args.replay_buffer.priority_beta,
        priority_epsilon=args.replay_buffer.priority_epsilon,
        reset_on_episode_end=args.reset_on_episode_end,
        streaming_update=args.streaming_update,
        instruction=instruction,
    )
    return network, agent
