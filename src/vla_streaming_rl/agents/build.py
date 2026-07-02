# SPDX-License-Identifier: MIT
import torch
from gymnasium import Env
from omegaconf import DictConfig


def build_agent(env: Env, network: torch.nn.Module, args: DictConfig):
    if args.agent_type == "standard":
        from vla_streaming_rl.agents.standard import StandardAgent

        agent = StandardAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            network=network,
            learning_mode=args.learning_mode,
            normalizing_by_return=args.normalizing_by_return,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            max_grad_norm=args.max_grad_norm,
            use_done=args.use_done,
            seq_len=args.seq_len,
            horizon=args.horizon,
            use_eligibility_trace=args.use_eligibility_trace,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            weight_decay=args.weight_decay,
            gamma=args.gamma,
            et_lambda=args.et_lambda,
            buffer_size=args.buffer_size,
            buffer_device=args.buffer_device,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            pad_token_id=args.pad_token_id,
        )

    elif args.agent_type == "simlingo":
        from vla_streaming_rl.agents.simlingo import SimLingoAgent

        agent = SimLingoAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            network=network,
            gamma=args.gamma,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            exploration_noise=args.exploration_noise,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            learning_mode=args.learning_mode,
            et_lambda=args.et_lambda,
        )

    elif args.agent_type == "libero_pi05":
        from vla_streaming_rl.agents.libero_pi05 import LiberoPi05Agent

        relabeler = None
        if args.vlac.enabled:
            from vla_streaming_rl.networks.vlac import VlacRewardRelabeler

            relabeler = VlacRewardRelabeler(
                milestone_interval=args.vlac.milestone_interval,
                check_interval=args.vlac.check_interval,
                gamma=args.gamma,
            )

        agent = LiberoPi05Agent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            network=network,
            learning_mode=args.learning_mode,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            et_lambda=args.et_lambda,
            gamma=args.gamma,
            relabeler=relabeler,
            vlac_ref_num=args.vlac.ref_num,
            reach_reward_scale=args.reach_reward_scale,
        )

    else:
        raise ValueError(f"Unknown agent type: {args.agent_type}")

    return agent
