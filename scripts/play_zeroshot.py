# SPDX-License-Identifier: MIT
import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch

from vla_streaming_rl.agents.zeroshot_vlm import ZeroShotVLMAgent
from vla_streaming_rl.utils import concat_images, convert_to_uint8
from vla_streaming_rl.wrappers import make_env

os.environ["TOKENIZERS_PARALLELISM"] = "false"


CAR_RACING_ACTION_SPEC = "steer=<value>, accel=<value> where each <value> is a float in [-1, 1]"
PANEL_ACTION_SPEC = (
    "dx=<value>, dy=<value>, button=<value> where each <value> is a float in [-1, 1]"
)

CONTINUOUS_ACTION_SPECS = {
    "CarRacing-v3": CAR_RACING_ACTION_SPEC,
    "STL10Panel-v0": PANEL_ACTION_SPEC,
    "TrackingSquare-v0": PANEL_ACTION_SPEC,
}


def wrap_format_hint(action_spec: str) -> str:
    return (
        "Use the following structured response format. "
        "First, describe what you see in the current image inside <perception>...</perception>. "
        "Then, lay out your strategy step by step inside <reasoning>...</reasoning>, "
        "justifying the action you intend to take based on the current image and the "
        "previous reward (if shown). "
        f"Finally, output the action inside <answer>...</answer> using the format: {action_spec}. "
        "The text inside <answer> must contain ONLY the action — no commentary, no labels."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_id",
        type=str,
        default="CarRacing-v3",
        choices=["CarRacing-v3", "STL10Panel-v0", "TrackingSquare-v0"],
    )
    parser.add_argument("--agent_type", type=str, choices=["random", "vlm"], required=True)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--render", type=int, default=1, choices=[0, 1])
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--vlm_model_id", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--seq_len",
        type=int,
        default=0,
        help="Number of past (image, response) turns to feed back as in-context history (FIFO).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)

    return parser.parse_args()


class RandomAgent:
    def __init__(self, action_space: gym.spaces.Box) -> None:
        self.action_space = action_space

    def reset(self, task_prompt: str) -> None:
        del task_prompt

    def select_action(self, obs: np.ndarray, prev_reward: float) -> tuple[np.ndarray, dict]:
        del obs, prev_reward
        return self.action_space.sample(), {}


def run_episode(env, agent, render: bool):
    obs, reset_info = env.reset()
    agent.reset(reset_info.get("task_prompt", ""))

    total_reward = 0.0
    step_count = 0
    bgr_image_list = []

    obs_for_render = convert_to_uint8(obs.copy().transpose(1, 2, 0))
    bgr_image_list.append(concat_images([env.render(), obs_for_render]))
    prev_reward = 0.0

    while True:
        action, agent_info = agent.select_action(obs, prev_reward)
        obs, reward, termination, truncation, env_info = env.step(action)

        prev_reward = reward
        total_reward += reward
        step_count += 1

        text = agent_info.get("text")
        if text is not None:
            parse_used = agent_info.get("parse_used", "strict")
            print(f"  [step {step_count:4d}] reward={reward:+.3f} parse={parse_used} text={text!r}")

        obs_for_render = convert_to_uint8(obs.copy().transpose(1, 2, 0))
        bgr_image = concat_images([env.render(), obs_for_render])
        bgr_image_list.append(bgr_image)

        if render:
            cv2.imshow(env.spec.id if env.spec is not None else "play", bgr_image)
            cv2.waitKey(1)

        if termination or truncation:
            break

    episode_length = env_info.get("episode", {}).get("l", step_count)
    episode_reward = env_info.get("episode", {}).get("r", total_reward)

    return episode_reward, episode_length, bgr_image_list, step_count


if __name__ == "__main__":
    args = parse_args()

    exp_name = f"ZEROSHOT_{args.agent_type.upper()}"

    # seeding
    seed = args.seed if args.seed != -1 else np.random.randint(0, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(__file__).resolve().parent / "results" / f"{datetime_str}_{exp_name}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # save seed to file
    with open(result_dir / "seed.txt", "w") as f:
        f.write(str(seed))

    video_dir = result_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    # env setup
    env = make_env(args.env_id, None)
    env.action_space.seed(seed)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    # Prime the env once so we can construct the agent with the initial task prompt.
    obs, reset_info = env.reset(seed=seed)
    initial_task_prompt = reset_info.get("task_prompt", "")

    # agent setup
    action_dim = int(np.prod(env.action_space.shape))
    if args.agent_type == "random":
        agent = RandomAgent(env.action_space)
    elif args.agent_type == "vlm":
        parse_action_text = getattr(env.unwrapped, "parse_action_text", None)
        assert parse_action_text is not None, (
            f"VLM agent requires env.unwrapped.parse_action_text; not defined for {args.env_id}"
        )
        action_spec = CONTINUOUS_ACTION_SPECS[args.env_id]
        format_hint = wrap_format_hint(action_spec)
        agent = ZeroShotVLMAgent(
            model_id=args.vlm_model_id,
            parse_action_text=parse_action_text,
            action_dim=action_dim,
            format_hint=format_hint,
            seq_len=args.seq_len,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        raise ValueError(f"Unknown agent_type: {args.agent_type}")

    print(f"Running {args.num_episodes} episodes with {args.agent_type} agent...")
    if args.agent_type == "vlm":
        print(f"  model_id={args.vlm_model_id}")
        print(f"  seq_len={args.seq_len}  max_new_tokens={args.max_new_tokens}")
        print(f"  task_prompt={initial_task_prompt!r}")

    episode_rewards = []
    episode_lengths = []
    best_reward = -float("inf")
    best_video = None
    global_step = 0

    start_time = time.time()

    for episode in range(args.num_episodes):
        episode_reward, episode_length, bgr_image_list, step_count = run_episode(
            env, agent, render=args.render
        )

        global_step += step_count
        elapsed_time = time.time() - start_time
        sps = global_step / elapsed_time if elapsed_time > 0 else 0

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Save best episode video
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_video = bgr_image_list

        print(
            f"Episode {episode + 1:2d}: Reward = {episode_reward:7.2f}, Length = {episode_length:3d}, "
            f"Global Step = {global_step:5d}, SPS = {sps:6.1f}"
        )

        # Save episode video
        video_path = video_dir / f"ep_{episode + 1:03d}.mp4"
        rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in bgr_image_list]
        imageio.mimsave(str(video_path), rgb_images, fps=10, macro_block_size=1)

    # Save best episode video
    if best_video:
        video_path = video_dir / "best_episode.mp4"
        rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in best_video]
        imageio.mimsave(str(video_path), rgb_images, fps=10, macro_block_size=1)

    # Calculate and display statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    elapsed_time = time.time() - start_time
    final_sps = global_step / elapsed_time if elapsed_time > 0 else 0

    print(f"\n=== Results after {args.num_episodes} episodes ===")
    print(f"Agent type: {args.agent_type}")
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Average episode length: {avg_length:.1f}")
    print(f"Total steps: {global_step}")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Steps per second: {final_sps:.1f}")

    # Save results to file
    with open(result_dir / "results.txt", "w") as f:
        f.write(f"Agent type: {args.agent_type}\n")
        if args.agent_type == "vlm":
            f.write(f"VLM model: {args.vlm_model_id}\n")
            f.write(f"seq_len: {args.seq_len}\n")
            f.write(f"max_new_tokens: {args.max_new_tokens}\n")
        f.write(f"Number of episodes: {args.num_episodes}\n")
        f.write(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}\n")
        f.write(f"Best reward: {best_reward:.2f}\n")
        f.write(f"Average episode length: {avg_length:.1f}\n")
        f.write(f"Total steps: {global_step}\n")
        f.write(f"Total time: {elapsed_time:.1f} seconds\n")
        f.write(f"Steps per second: {final_sps:.1f}\n")
        f.write(f"Episode rewards: {episode_rewards}\n")

    env.close()
    cv2.destroyAllWindows()
