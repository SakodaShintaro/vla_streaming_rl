# SPDX-License-Identifier: MIT
import torch
from gymnasium import Env
from omegaconf import DictConfig

from vla_streaming_rl.agents.prompt import PromptBuilder


def build_agent(
    env: Env, network: torch.nn.Module, prompt_builder: PromptBuilder, args: DictConfig
):
    if args.agent_type == "zeroshot_vlm":
        from vla_streaming_rl.agents.vlm_backends import build_vlm_backend
        from vla_streaming_rl.agents.zeroshot_vlm import ZeroShotVLMAgent

        # This baseline only ever acts by writing the action out, so it is the
        # one agent whose regime is fixed rather than configured.
        assert args.text_action_mode == "text_action", (
            f"zeroshot_vlm writes its action as text, so text_action_mode has to be "
            f'"text_action", not {args.text_action_mode!r}'
        )

        return ZeroShotVLMAgent(
            action_space=env.action_space,
            parse_action_text=env.unwrapped.parse_action_text,
            action_spec=env.unwrapped.action_spec,
            backend=build_vlm_backend(args, env.unwrapped.action_choices),
            seq_len=args.seq_len,
            image_side=args.image_side,
            reset_on_episode_end=args.reset_on_episode_end,
            prompt_builder=prompt_builder,
        )

    if args.agent_type == "animal_ppo":
        from vla_streaming_rl.agents.animal_ppo import AnimalPPOAgent

        return AnimalPPOAgent(
            action_space=env.action_space,
            network=network,
            horizon=args.horizon,
            steps_num=args.steps_num,
            minibatch_size=args.minibatch_size,
            mini_epochs=args.mini_epochs,
            seq_len=args.seq_len,
            gamma=args.gamma,
            lam=args.lam,
            learning_rate=args.learning_rate,
            e_clip=args.e_clip,
            entropy_coef=args.entropy_coef,
            critic_coef=args.critic_coef,
            clip_value=args.clip_value,
            normalize_advantage=args.normalize_advantage,
            max_grad_norm=args.max_grad_norm,
            velocity_scale=list(args.velocity_scale),
            health_scale=args.health_scale,
            reset_on_episode_end=args.reset_on_episode_end,
            prompt_builder=prompt_builder,
        )

    if args.agent_type == "streaming":
        from vla_streaming_rl.agents.streaming import StreamingAgent

        return StreamingAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            network=network,
            normalizing_by_return=args.normalizing_by_return,
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
            buffer_device=args.buffer_device,
            max_prompt_tokens=args.max_prompt_tokens,
            pad_token_id=args.pad_token_id,
            reset_on_episode_end=args.reset_on_episode_end,
            prompt_builder=prompt_builder,
        )

    assert args.agent_type == "off_policy", f"Unknown agent_type: {args.agent_type!r}"
    from vla_streaming_rl.agents.off_policy import OffPolicyAgent

    return OffPolicyAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        network=network,
        normalizing_by_return=args.normalizing_by_return,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        max_grad_norm=args.max_grad_norm,
        use_done=args.use_done,
        seq_len=args.seq_len,
        horizon=args.horizon,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        weight_decay=args.weight_decay,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        max_prompt_tokens=args.max_prompt_tokens,
        pad_token_id=args.pad_token_id,
        reset_on_episode_end=args.reset_on_episode_end,
        prompt_builder=prompt_builder,
    )
