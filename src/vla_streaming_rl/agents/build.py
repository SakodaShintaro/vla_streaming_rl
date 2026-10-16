# SPDX-License-Identifier: MIT
import torch
from gymnasium import Env
from omegaconf import DictConfig

from vla_streaming_rl.agents.prompt import PromptBuilder, build_prompt_builder
from vla_streaming_rl.image_obs import image_obs_names


def build_all(env: Env, args: DictConfig) -> tuple[torch.nn.Module, object]:
    """The prompt builder, network and agent a config describes, wired
    together the way every entry script needs them."""
    from vla_streaming_rl.networks.build import build_network

    prompt_builder = build_prompt_builder(env, args)
    network = build_network(
        args,
        env,
        prompt_builder.parse_action_text,
        prompt_builder.text_action_names(),
        torch.device("cuda"),
    )
    return network, build_agent(env, network, prompt_builder, args)


def build_agent(
    env: Env, network: torch.nn.Module, prompt_builder: PromptBuilder, args: DictConfig
):
    if args.agent_type == "streaming":
        from vla_streaming_rl.agents.streaming import StreamingAgent

        return StreamingAgent(
            action_space=env.action_space,
            image_names=image_obs_names(env.observation_space),
            network=network,
            normalizing_by_return=args.normalizing_by_return,
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
            prompt_builder=prompt_builder,
        )

    assert args.agent_type == "off_policy", f"Unknown agent_type: {args.agent_type!r}"
    from vla_streaming_rl.agents.off_policy import OffPolicyAgent

    return OffPolicyAgent(
        action_space=env.action_space,
        image_names=image_obs_names(env.observation_space),
        network=network,
        normalizing_by_return=args.normalizing_by_return,
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
        use_reflection=args.vla.use_reflection,
        reset_on_episode_end=args.reset_on_episode_end,
        prompt_builder=prompt_builder,
    )
