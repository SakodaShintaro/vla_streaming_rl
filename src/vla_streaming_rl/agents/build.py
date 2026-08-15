# SPDX-License-Identifier: MIT
import torch
from gymnasium import Env
from omegaconf import DictConfig


def build_agent(env: Env, network: torch.nn.Module, args: DictConfig):
    if args.agent_type == "zeroshot_vlm":
        from vla_streaming_rl.agents.zeroshot_vlm import ZeroShotVLMAgent

        return ZeroShotVLMAgent(
            action_space=env.action_space,
            parse_action_text=env.unwrapped.parse_action_text,
            action_spec=env.unwrapped.action_spec,
            model_id=args.openrouter_model_id,
            seq_len=args.seq_len,
            max_new_tokens=args.max_new_tokens,
            reasoning_max_tokens=args.reasoning_max_tokens,
            image_side=args.image_side,
            temperature=args.temperature,
        )

    from vla_streaming_rl.agents.standard import StandardAgent

    return StandardAgent(
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
        max_prompt_tokens=args.max_prompt_tokens,
        pad_token_id=args.pad_token_id,
    )
