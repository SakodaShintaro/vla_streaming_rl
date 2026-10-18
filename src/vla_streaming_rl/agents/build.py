# SPDX-License-Identifier: MIT
import torch
from gymnasium import Env
from omegaconf import DictConfig

from vla_streaming_rl.agents.instruction import ENV_INSTRUCTIONS
from vla_streaming_rl.image_obs import image_obs_names
from vla_streaming_rl.reward_scaler import build_reward_scaler


def build_all(env: Env, args: DictConfig) -> tuple[torch.nn.Module, object]:
    """The network and agent a config describes, wired together the way every
    entry script needs them."""
    from vla_streaming_rl.networks.build import build_network

    network = build_network(args, env, torch.device("cuda"))
    return network, build_agent(env, network, args)


def build_agent(env: Env, network: torch.nn.Module, args: DictConfig):
    instruction = ENV_INSTRUCTIONS[args.env_id]
    if args.agent_type == "streaming":
        from vla_streaming_rl.agents.streaming import StreamingAgent

        return StreamingAgent(
            action_space=env.action_space,
            image_names=image_obs_names(env.observation_space),
            network=network,
            normalizing_by_return=args.normalizing_by_return,
            reward_scaler=build_reward_scaler(args.reward_scale_mode, args.fixed_reward_scale),
            max_grad_norm=args.max_grad_norm,
            use_done=args.use_done,
            seq_len=network.seq_len,
            horizon=args.horizon,
            use_eligibility_trace=args.use_eligibility_trace,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            weight_decay=args.weight_decay,
            gamma=args.gamma,
            et_lambda=args.et_lambda,
            buffer_device=args.replay_buffer.device,
            reset_on_episode_end=args.reset_on_episode_end,
            instruction=instruction,
        )

    assert args.agent_type == "off_policy", f"Unknown agent_type: {args.agent_type!r}"
    from vla_streaming_rl.agents.off_policy import OffPolicyAgent

    return OffPolicyAgent(
        action_space=env.action_space,
        image_names=image_obs_names(env.observation_space),
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
        instruction=instruction,
    )
