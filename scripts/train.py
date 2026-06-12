# SPDX-License-Identifier: MIT
# This script was initially inspired by CleanRL https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
import logging
import os
import warnings

os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS.*")
logging.getLogger("httpx").setLevel(logging.WARNING)

import csv
import random
import time
from pathlib import Path

import cv2
import hydra
import imageio
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from vla_streaming_rl.agents.off_policy import OffPolicyAgent
from vla_streaming_rl.agents.streaming import StreamingAgent
from vla_streaming_rl.networks.build import build_network
from vla_streaming_rl.self_forcing.goal_predictor import WorldModelGoalPredictor
from vla_streaming_rl.utils import concat_labeled_images
from vla_streaming_rl.wrappers import make_env

torch.set_float32_matmul_precision("high")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _viz_resize(image: np.ndarray, scale: float) -> np.ndarray:
    """Downscale (or upscale) the observation render panel.

    This is applied only to the copy that goes into ``concat_labeled_images``
    / ``cv2.imshow``, never to data used for any computation.
    """
    if scale == 1.0:
        return image
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def save_episode_data(
    video_dir: Path,
    image_dir: Path,
    obs_dir: Path,
    name: str,
    bgr_image_list: list[np.ndarray],
    action_list: list[np.ndarray],
    reward_list: list[float],
    obs_list: list[np.ndarray],
) -> None:
    """Save episode video, images, actions and rewards"""
    if not bgr_image_list:
        return

    # Stable-panel contract: an agent emits the same set of equally-shaped
    # panels every step, so every render frame has the same size. Fail loudly
    # if that is violated rather than letting the video encoder error out.
    frame_sizes = {img.shape for img in bgr_image_list}
    if len(frame_sizes) != 1:
        raise ValueError(
            f"Episode '{name}' produced frames of differing sizes {frame_sizes}; "
            "an agent's panel set / shapes must stay constant across the run."
        )

    # libx264 (yuv420p) requires even width/height; the concatenated panel strip
    # can be an odd size. Pad one black row/column when needed.
    h, w = bgr_image_list[0].shape[:2]
    if h % 2 or w % 2:
        bgr_image_list = [
            np.pad(img, ((0, h % 2), (0, w % 2), (0, 0)), mode="constant")
            for img in bgr_image_list
        ]

    video_path = video_dir / f"{name}.mp4"
    rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in bgr_image_list]
    imageio.mimsave(str(video_path), rgb_images, fps=10, macro_block_size=1)

    # Save as images
    curr_image_dir = image_dir / f"{name}"
    curr_image_dir.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(bgr_image_list):
        image_path = curr_image_dir / f"{idx:08d}.png"
        cv2.imwrite(str(image_path), img)

    # Save raw observations as images
    obs_image_dir = obs_dir / f"{name}"
    obs_image_dir.mkdir(parents=True, exist_ok=True)
    for idx, obs in enumerate(obs_list):
        obs_hwc = (obs.transpose(1, 2, 0) * 255).astype(np.uint8)
        obs_bgr = cv2.cvtColor(obs_hwc, cv2.COLOR_RGB2BGR)
        obs_path = obs_image_dir / f"{idx:08d}.png"
        cv2.imwrite(str(obs_path), obs_bgr)

    # Save actions and rewards to TSV file
    tsv_path = curr_image_dir / "log.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("step\t")
        for i in range(len(action_list[0])):
            f.write(f"action{i}\t")
        f.write("reward\n")
        for step, (action, reward) in enumerate(zip(action_list, reward_list)):
            f.write(f"{step}\t")
            for a in action:
                f.write(f"{float(a):.6f}\t")
            f.write(f"{float(reward):.6f}\n")


def main(args: DictConfig, exp_name: str, seed: int, result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=f"vla_streaming_rl_{args.env_id}",
        config=OmegaConf.to_container(args, resolve=True),
        name=exp_name,
        group=args.wandb_group,
        save_code=True,
        settings=wandb.Settings(quiet=True),
        dir=str(result_dir),
    )

    # seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)

    # Create result directories and save files only if not in debug mode
    if not args.debug:
        # save seed to file
        with open(result_dir / "seed.txt", "w") as f:
            f.write(str(seed))

        # Save git show -s and git diff results
        with open(result_dir / "git_info.txt", "w") as f:
            f.write(f"git show -s:\n{os.popen('git show -s').read()}\n")
            f.write(f"git diff:\n{os.popen('git diff').read()}\n")

        video_dir = result_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)

        image_dir = result_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        obs_dir = result_dir / "obs"
        obs_dir.mkdir(parents=True, exist_ok=True)
    else:
        result_dir = None
        video_dir = None
        image_dir = None
        obs_dir = None
    log_episode_path = result_dir / "log_episode.tsv" if result_dir is not None else None
    log_episode_file = None
    log_episode_writer = None

    # env setup
    env = make_env(args.env_id, args.env_factory, result_dir=result_dir)
    env.unwrapped.max_step_count = args.max_step_count
    env.action_space.seed(seed)

    eval_range = env.unwrapped.eval_range

    start_time = time.time()

    # start the game
    global_step = 0
    score_list = []
    score_sum_all = 0.0
    episode_count = 0
    best_score = -float("inf")
    # Don't reset the env here — the first iteration of the episode loop
    # does that with seed=seed (was: double-reset wasted route 0 in the
    # bench2drive sequential cursor and confused the eval writer).
    step_limit = args.step_limit
    episode_limit = args.episode_limit
    checkpoint_interval = max(1, step_limit // 10)

    network = build_network(
        args,
        observation_space_shape=env.observation_space.shape,
        action_space_shape=env.action_space.shape,
        parse_action_text=getattr(env.unwrapped, "parse_action_text", None),
        device=torch.device("cuda"),
    )

    goal_predictor = WorldModelGoalPredictor(
        enabled=bool(args.self_forcing.enabled),
        config_path=args.self_forcing.config_path,
        checkpoint_path=args.self_forcing.checkpoint_path,
        device=torch.device("cuda"),
        num_context_blocks=int(args.self_forcing.num_context_blocks),
    )

    if args.agent_type == "off_policy":
        agent = OffPolicyAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            network=network,
            normalizing_by_return=args.normalizing_by_return,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            max_grad_norm=args.max_grad_norm,
            use_done=args.use_done,
            accumulation_steps=args.accumulation_steps,
            seq_len=args.seq_len,
            horizon=args.horizon,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            buffer_device=args.buffer_device,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            pad_token_id=args.pad_token_id,
            goal_predictor=goal_predictor,
        )
    elif args.agent_type == "streaming":
        agent = StreamingAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            network=network,
            normalizing_by_return=args.normalizing_by_return,
            max_grad_norm=args.max_grad_norm,
            use_done=args.use_done,
            accumulation_steps=args.accumulation_steps,
            seq_len=args.seq_len,
            horizon=args.horizon,
            use_eligibility_trace=args.use_eligibility_trace,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            et_lambda=args.et_lambda,
            buffer_device=args.buffer_device,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            pad_token_id=args.pad_token_id,
            goal_predictor=goal_predictor,
        )
    elif args.agent_type == "simlingo":
        from vla_streaming_rl.agents.simlingo import SimLingoAgent

        agent = SimLingoAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            env=env,
            network=network,
            scratch_dir=(result_dir / "simlingo_scratch")
            if result_dir is not None
            else Path("/tmp/simlingo_scratch"),
            gamma=float(args.gamma),
            buffer_size=int(args.buffer_size),
            batch_size=int(args.batch_size),
            learning_starts=int(args.learning_starts),
            exploration_noise=float(args.exploration_noise),
            actor_lr=float(args.actor_lr),
            critic_lr=float(args.critic_lr),
            max_grad_norm=float(args.max_grad_norm),
            learning_mode=str(args.learning_mode),
            et_lambda=float(args.et_lambda),
            awr_num_samples=int(args.awr_num_samples),
            awr_temperature=float(args.awr_temperature),
            awr_sample_noise=float(args.awr_sample_noise),
        )
    else:
        raise ValueError(f"Unknown agent type: {args.agent_type}")

    parameter_count = sum(p.numel() for p in agent.network.parameters())
    print(f"Parameter count: {parameter_count:,}")

    episode_id = 0
    while True:
        # Stop when the env has dispensed every scenario in a fixed playlist
        # (e.g. Bench2Drive220 sequential mode). For envs without a finite
        # playlist this is always False.
        runtime = getattr(env.unwrapped, "runtime", None)
        if runtime is not None and getattr(runtime, "is_exhausted", False):
            break

        # Stop once we've run the configured number of episodes.
        if episode_id >= episode_limit:
            break

        # initialize episode (seed only the first call so the gym RNG is
        # set once; subsequent resets keep advancing it).
        obs, reset_info = env.reset(seed=seed) if episode_id == 0 else env.reset()
        task_prompt = reset_info["task_prompt"] if args.use_prompt else ""

        # initial action
        result = agent.select_action(global_step, obs, 0.0, False, False, task_prompt)
        action = result.action

        # initial render. The trainer only owns the environment / observation
        # panels; prediction, reward, goal, bev, ... arrive via result.panels.
        obs_for_render = obs.copy().transpose(1, 2, 0)
        obs_viz = _viz_resize(obs_for_render, args.render_scale)
        panels = {
            "environment": env.render(),
            "observation": obs_viz,
            **result.panels,
        }
        initial_rgb_image = concat_labeled_images(panels)
        bgr_image_list = [cv2.cvtColor(initial_rgb_image, cv2.COLOR_RGB2BGR)]

        # action and reward history for this episode
        action_list = []
        reward_list = []
        obs_list = [obs.copy()]

        while True:
            global_step += 1

            # step
            env_step_start = time.time()
            obs, reward, terminated, truncated, env_info = env.step(action)
            env_step_time_msec = (time.time() - env_step_start) * 1000
            task_prompt = env_info["task_prompt"] if args.use_prompt else ""

            # save action, reward, and observation
            action_list.append(action.copy())
            reward_list.append(reward)
            obs_list.append(obs.copy())

            agent_step_start = time.time()
            result = agent.step(global_step, obs, reward, terminated, truncated, task_prompt)
            action = result.action
            agent_step_time_msec = (time.time() - agent_step_start) * 1000

            # render
            obs_for_render = obs.copy().transpose(1, 2, 0)

            # log: metrics are already scalar telemetry (images live in panels)
            elapsed_time_sec = time.time() - start_time
            elapsed_time_min = elapsed_time_sec / 60
            data_dict = {
                "global_step": global_step,
                "elapsed_time_min": elapsed_time_min,
                "SPS": global_step / elapsed_time_sec,
                "reward": reward,
                "env_step_msec": env_step_time_msec,
                "agent_step_msec": agent_step_time_msec,
                **result.metrics,
            }
            wandb.log(data_dict)

            obs_viz = _viz_resize(obs_for_render, args.render_scale)
            panels = {
                "environment": env.render(),
                "observation": obs_viz,
                **result.panels,
            }
            rgb_image = concat_labeled_images(panels)
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            bgr_image_list.append(bgr_image)
            if args.render:
                cv2.imshow(args.env_id, bgr_image)
                cv2.waitKey(1)

            if global_step % checkpoint_interval == 0 and result_dir is not None:
                module = network._orig_mod if hasattr(network, "_orig_mod") else network
                trainable_state = {
                    name: param.detach().cpu()
                    for name, param in module.named_parameters()
                    if param.requires_grad
                }
                torch.save(trainable_state, result_dir / "checkpoint.pt")

            if terminated or truncated:
                break

            if global_step >= step_limit:
                break

        if global_step >= step_limit:
            break

        score = env_info["episode"]["r"]
        score_list.append(score)
        score_list = score_list[-eval_range:]
        recent_average_score = np.mean(score_list)
        score_sum_all += float(score)
        episode_count += 1
        total_average_score = score_sum_all / episode_count

        if args.normalizing_by_return:
            agent.reward_processor.update(score)

        elapsed_time_sec = time.time() - start_time
        elapsed_time_hour = elapsed_time_sec / 3600
        data_dict = {
            "global_step": global_step,
            "episode_id": episode_id,
            "episodic_return": env_info["episode"]["r"],
            "episodic_length": env_info["episode"]["l"],
            "SPS": global_step / elapsed_time_sec,
            "elapsed_time_hour": elapsed_time_hour,
            "total_average_score": total_average_score,
        }
        # Bench2Drive sequential-mode bookkeeping: surface which scenario
        # just finished so a wandb sweep across the 220 routes is plottable.
        if "scenario_index" in env_info:
            data_dict["scenario_index"] = env_info["scenario_index"]
            data_dict["scenarios_done"] = int(env_info["scenario_index"]) + 1
            data_dict["scenarios_total"] = env_info["scenarios_total"]
        # The env auto-flushes its Bench2Drive eval artifacts on the
        # terminating step and exposes the summary scores here.
        for k, v in env_info.get("eval_summary", {}).items():
            if isinstance(v, (int, float)):
                data_dict[f"eval/{k}"] = v
        # for animalai_env
        if "pass_mark" in env_info:
            success = float(score >= env_info["pass_mark"])
            data_dict["success"] = success
            data_dict["pass_mark"] = env_info["pass_mark"]
            arena_name = env_info.get("arena_name", "")
            if arena_name:
                data_dict[f"success/{arena_name}"] = success
                data_dict[f"episodic_return/{arena_name}"] = score
            if "cleared_count" in env_info:
                data_dict["cleared_count"] = env_info["cleared_count"]
                data_dict["is_revisit"] = float(env_info.get("is_revisit", False))
                data_dict["advanced"] = float(env_info.get("advanced", False))
        if len(score_list) >= eval_range:
            data_dict["recent_average_score"] = recent_average_score
        wandb.log(data_dict)

        if result_dir is not None:
            if log_episode_writer is None:
                log_episode_file = open(log_episode_path, "w", newline="")
                fieldnames = list(data_dict.keys()) + ["recent_average_score"]
                log_episode_writer = csv.DictWriter(
                    log_episode_file, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
                )
                log_episode_writer.writeheader()
            log_episode_writer.writerow(data_dict)
            log_episode_file.flush()

        if episode_id % args.print_interval == 0:
            print(
                f"Ep: {episode_id}\tStep: {global_step}\tLast score: {score:.2f}\tAverage score: {recent_average_score:.2f}\tLength: {env_info['episode']['l']:.2f}\tElapsed time: {elapsed_time_hour:.2f}h"
            )

        feedback_text = input("Feedback: ") if args.use_feedback else ""
        episode_end_info = agent.on_episode_end(score, feedback_text)
        wandb.log(episode_end_info)

        is_best = score > best_score

        if is_best and result_dir is not None:
            with open(result_dir / "best_score.txt", "w") as f:
                f.write(f"{episode_id + 1}\t{score:.2f}")
            best_score = score
            save_episode_data(
                video_dir,
                image_dir,
                obs_dir,
                "best_episode",
                bgr_image_list,
                action_list,
                reward_list,
                obs_list,
            )

        if (
            episode_id == 0 or (episode_id + 1) % args.image_save_interval == 0
        ) and result_dir is not None:
            save_episode_data(
                video_dir,
                image_dir,
                obs_dir,
                f"ep_{episode_id + 1:08d}",
                bgr_image_list,
                action_list,
                reward_list,
                obs_list,
            )

        episode_id += 1

    env.close()
    # env.close() auto-merges the Bench2Drive eval sweep and stashes the
    # Driving Score / Success Rate / Efficiency / Comfort summary on the
    # env. Push it into wandb.summary so the run is self-describing.
    final_eval_summary = getattr(env.unwrapped, "final_eval_summary", None)
    if final_eval_summary is not None:
        wandb.summary.update({f"final/{k}": v for k, v in final_eval_summary.items()})
        print("=== Bench2Drive220 final ===")
        for k, v in final_eval_summary.items():
            print(f"  {k}: {v}")

    if log_episode_file is not None:
        log_episode_file.close()
    wandb.finish()


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def hydra_main(cfg: DictConfig) -> None:
    # Hydra's output dir is our result dir (configured via hydra.run.dir)
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    # Restore cwd so relative paths in the code work correctly
    os.chdir(hydra.utils.get_original_cwd())

    if cfg.debug:
        cfg.off_wandb = True
        cfg.learning_starts = max(10, cfg.seq_len + cfg.horizon + 5)
        cfg.render = 0
        cfg.step_limit = 100
        cfg.buffer_size = int(2e4)

    if cfg.off_wandb:
        os.environ["WANDB_MODE"] = "offline"

    if not os.environ.get("DISPLAY"):
        print("Because a headless environment is detected, rendering is automatically disabled.")
        cfg.render = 0

    exp_name = f"{cfg.agent_type.upper()}_{cfg.exp_name}"
    seed = cfg.seed if cfg.seed != -1 else np.random.randint(0, 10000)

    for i in range(cfg.trial_num):
        suffix = f"_{i:02d}" if cfg.trial_num > 1 else ""
        trial_dir = hydra_output_dir / f"trial{suffix}" if cfg.trial_num > 1 else hydra_output_dir
        main(cfg, exp_name + suffix, seed + i, trial_dir)


if __name__ == "__main__":
    hydra_main()
