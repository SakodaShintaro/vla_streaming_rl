# SPDX-License-Identifier: MIT
# This script was initially inspired by CleanRL https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
from vla_streaming_rl.script_setup import setup_runtime

setup_runtime()

import csv
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import hydra
import imageio
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from vla_streaming_rl.agents.build import build_all
from vla_streaming_rl.checkpoint import load_checkpoint_weights, save_checkpoint
from vla_streaming_rl.envs.animalai_env import training_levels
from vla_streaming_rl.script_setup import disable_render_if_headless, resolve_seed, seed_everything
from vla_streaming_rl.utils import render_frame
from vla_streaming_rl.wrappers import make_env


@dataclass
class TrainState:
    """The trainer's running counters: initialized fresh, persisted to
    train_state.json every episode, and restored by ``load_resume_state``."""

    global_step: int
    episode_id: int
    episode_count: int
    score_sum_all: float
    success_sum_all: float
    success_episode_count: int
    best_score: float
    score_list: list[float]
    success_list: list[float]

    @classmethod
    def fresh(cls) -> "TrainState":
        return cls(
            global_step=0,
            episode_id=0,
            episode_count=0,
            score_sum_all=0.0,
            success_sum_all=0.0,
            success_episode_count=0,
            best_score=-float("inf"),
            score_list=[],
            success_list=[],
        )


@dataclass
class EpisodeRecord:
    """Everything one episode leaves behind for ``save_episode_data``."""

    bgr_images: list[np.ndarray]
    actions: list[np.ndarray]
    rewards: list[float]
    observations: list[np.ndarray]
    texts: list[dict[str, str]]
    xyzs: list[tuple[float, float, float]]

    @classmethod
    def fresh(cls) -> "EpisodeRecord":
        return cls(bgr_images=[], actions=[], rewards=[], observations=[], texts=[], xyzs=[])


def save_episode_texts(episode_log_dir: Path, text_list: list[dict[str, str]]) -> None:
    """The free-form text agents emitted, one row per rendered frame.

    Text a network draws into a panel is legible in the video but not
    searchable; this is the same content as characters, so a run's chain of
    thought can be read back and grepped. Tabs and newlines are escaped to keep
    one step on one line.
    """
    keys = list(text_list[0].keys())
    if not keys:
        return

    def escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")

    tsv_path = episode_log_dir / "texts.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("step\t" + "\t".join(keys) + "\n")
        for step, texts in enumerate(text_list):
            f.write(f"{step}\t" + "\t".join(escape(texts[key]) for key in keys) + "\n")


def save_episode_data(episode_dir: Path, name: str, record: EpisodeRecord) -> None:
    """Save episode videos, actions, rewards and positions"""
    if not record.bgr_images:
        return

    # Stable-panel contract: an agent emits the same set of equally-shaped
    # panels every step, so every render frame has the same size. Fail loudly
    # if that is violated rather than letting the video encoder error out.
    frame_sizes = {img.shape for img in record.bgr_images}
    assert len(frame_sizes) == 1, (
        f"Episode '{name}' produced frames of differing sizes {frame_sizes}; "
        "an agent's panel set / shapes must stay constant across the run."
    )

    # libx264 (yuv420p) requires even width/height; the concatenated panel strip
    # can be an odd size. Pad one black row/column when needed.
    bgr_image_list = record.bgr_images
    h, w = bgr_image_list[0].shape[:2]
    if h % 2 or w % 2:
        bgr_image_list = [
            np.pad(img, ((0, h % 2), (0, w % 2), (0, 0)), mode="constant") for img in bgr_image_list
        ]

    episode_log_dir = episode_dir / f"{name}"
    episode_log_dir.mkdir(parents=True, exist_ok=True)

    video_path = episode_log_dir / "render.mp4"
    rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in bgr_image_list]
    imageio.mimsave(str(video_path), rgb_images, fps=10, macro_block_size=1)

    obs_rgb_images = [
        (obs.transpose(1, 2, 0) * 255).astype(np.uint8) for obs in record.observations
    ]
    obs_sizes = {img.shape for img in obs_rgb_images}
    assert len(obs_sizes) == 1, (
        f"Episode '{name}' produced observations of differing sizes {obs_sizes}"
    )
    obs_h, obs_w = obs_rgb_images[0].shape[:2]
    if obs_h % 2 or obs_w % 2:
        obs_rgb_images = [
            np.pad(img, ((0, obs_h % 2), (0, obs_w % 2), (0, 0)), mode="constant")
            for img in obs_rgb_images
        ]
    obs_video_path = episode_log_dir / "obs.mp4"
    imageio.mimsave(str(obs_video_path), obs_rgb_images, fps=10, macro_block_size=1)

    save_episode_texts(episode_log_dir, record.texts)

    # One row per step: the action taken, the reward it drew, and where the
    # agent stood after it, so an episode's trajectory can be drawn from the
    # run rather than re-simulated. ``xyzs`` is empty for an env that reports
    # no position, and the x/y/z columns are then left out rather than filled
    # with a stand-in.
    columns = [f"action{i}" for i in range(len(record.actions[0]))] + ["reward"]
    rows = [
        [f"{float(a):.6f}" for a in action] + [f"{float(reward):.6f}"]
        for action, reward in zip(record.actions, record.rewards)
    ]
    if record.xyzs:
        columns = columns + ["x", "y", "z"]
        rows = [row + [f"{float(v):.4f}" for v in xyz] for row, xyz in zip(rows, record.xyzs)]

    with open(episode_log_dir / "log.tsv", "w", encoding="utf-8") as f:
        f.write("step\t" + "\t".join(columns) + "\n")
        for step, row in enumerate(rows):
            f.write(f"{step}\t" + "\t".join(row) + "\n")


def write_git_info(result_dir: Path) -> None:
    """Save branch name, git show -s and git diff results"""
    branch_name = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True
    ).stdout
    git_show = subprocess.run(["git", "show", "-s"], capture_output=True, text=True).stdout
    git_diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True).stdout
    with open(result_dir / "git_info.txt", "w") as f:
        f.write(f"branch:\n{branch_name}\n")
        f.write(f"git show -s:\n{git_show}\n")
        f.write(f"git diff:\n{git_diff}\n")


def write_arena_stats(path: Path, curriculum: dict, best_score_per_arena: dict) -> None:
    """Per-arena record (attempts / successes / success rate / best score).

    `cleared` is the arena selector's own verdict, which depends on the mode:
    one pass is enough for most, while "success" wants a repeatable pass rate.

    Human-readable, and also read back by ``load_resume_state`` on resume."""
    attempts = curriculum["arena_attempts"]
    successes = curriculum["arena_successes"]
    cleared = curriculum["arena_cleared"]
    with open(path, "w") as f:
        f.write("arena\tattempts\tsuccesses\tsuccess_rate\tbest_score\tcleared\n")
        for arena in sorted(attempts):
            success_rate = successes[arena] / attempts[arena]
            f.write(
                f"{arena}\t{attempts[arena]}\t{successes[arena]}\t{success_rate:.4f}"
                f"\t{best_score_per_arena[arena]:.6f}\t{int(cleared[arena])}\n"
            )


def write_success_rate(path: Path, curriculum: dict) -> None:
    """Cleared-arena rate per Olympics level ("category") plus the total.

    One row per level the run has attempted so far, keyed by the "XX" prefix of
    the arena label, and an "all" row over every attempted arena."""
    cleared = curriculum["arena_cleared"]
    arenas_by_level: dict[str, list[str]] = {}
    for arena in sorted(cleared):
        arenas_by_level.setdefault(arena.split("-")[0], []).append(arena)
    with open(path, "w") as f:
        f.write("category,arenas,successes,success_rate\n")
        for level, arenas in sorted(arenas_by_level.items()):
            successes = sum(int(cleared[arena]) for arena in arenas)
            f.write(f"{level},{len(arenas)},{successes},{successes / len(arenas):.4f}\n")
        successes = sum(int(cleared[arena]) for arena in cleared)
        f.write(f"all,{len(cleared)},{successes},{successes / len(cleared):.4f}\n")


def load_resume_state(resume_dir: Path, network, agent, env) -> tuple[TrainState, dict[str, float]]:
    """Restore weights / optimizer / counters / curriculum from a previous run dir.

    checkpoint.pt is required; the other files are optional so directories
    written before this resume mechanism existed still load (their missing
    parts keep the fresh defaults)."""
    load_checkpoint_weights(resume_dir / "checkpoint.pt", network)

    optimizer_path = resume_dir / "optimizer.pt"
    if optimizer_path.exists():
        agent.load_optimizer_state_dict(torch.load(optimizer_path, map_location="cuda"))
        print(f"Resume: loaded optimizer states from {optimizer_path}")
    else:
        print(f"Resume: {optimizer_path} not found, optimizers start fresh")

    state = TrainState.fresh()
    curriculum_progress: dict = {}
    train_state_path = resume_dir / "train_state.json"
    if train_state_path.exists():
        data = json.loads(train_state_path.read_text())
        if "curriculum_progress" in data:
            curriculum_progress = data["curriculum_progress"]
        for state_field in fields(TrainState):
            if state_field.name in data:
                setattr(state, state_field.name, data[state_field.name])
        print(f"Resume: loaded counters from {train_state_path}")
    else:
        print(f"Resume: {train_state_path} not found, counters start from zero")

    best_score_per_arena: dict[str, float] = {}
    arena_stats_path = resume_dir / "arena_stats.tsv"
    set_curriculum = getattr(env.unwrapped, "set_curriculum_state", None)
    if arena_stats_path.exists() and set_curriculum is not None:
        attempts = {}
        successes = {}
        cleared_count = 0
        for row in arena_stats_path.read_text().splitlines()[1:]:
            arena, n_attempt, n_success, _rate, best, cleared = row.split("\t")
            attempts[arena] = int(n_attempt)
            successes[arena] = int(n_success)
            best_score_per_arena[arena] = float(best)
            cleared_count += int(cleared)
        set_curriculum(attempts, successes, curriculum_progress)
        print(
            f"Resume: loaded {len(attempts)} arena records from {arena_stats_path} "
            f"(cleared={cleared_count})"
        )
    return state, best_score_per_arena


def save_train_state(
    result_dir: Path, state: TrainState, env, best_score_per_arena: dict[str, float]
) -> None:
    """Persist the light resume state (the heavy weights / optimizer files
    follow the checkpoint_interval cadence instead)."""
    train_state = asdict(state)
    train_state["score_list"] = [float(s) for s in state.score_list]
    train_state["success_list"] = [float(s) for s in state.success_list]
    train_state["score_sum_all"] = float(state.score_sum_all)
    train_state["success_sum_all"] = float(state.success_sum_all)
    train_state["best_score"] = float(state.best_score)
    get_curriculum = getattr(env.unwrapped, "get_curriculum_state", None)
    if get_curriculum is not None:
        curriculum = get_curriculum()
        train_state["curriculum_progress"] = curriculum["progress"]
        write_arena_stats(result_dir / "arena_stats.tsv", curriculum, best_score_per_arena)
        write_success_rate(result_dir / "success_rate.csv", curriculum)
    (result_dir / "train_state.json").write_text(json.dumps(train_state, indent=2))


def build_episode_log(
    state: TrainState, env_info: dict, eval_range: int, elapsed_time_sec: float
) -> dict:
    """The per-episode wandb row, updating the running counters in ``state``."""
    score = env_info["episode"]["r"]
    state.score_list.append(score)
    state.score_list = state.score_list[-eval_range:]
    state.score_sum_all += float(score)
    state.episode_count += 1

    data_dict = {
        "global_step": state.global_step,
        "episode_id": state.episode_id,
        "episodic_return": env_info["episode"]["r"],
        "episodic_length": env_info["episode"]["l"],
        "SPS": state.global_step / elapsed_time_sec,
        "elapsed_time_hour": elapsed_time_sec / 3600,
        "total_average_score": state.score_sum_all / state.episode_count,
    }
    # Bench2Drive sequential-mode bookkeeping: surface which scenario
    # just finished so a wandb sweep across the 220 routes is plottable.
    if "scenario_index" in env_info:
        data_dict["scenario_index"] = env_info["scenario_index"]
        data_dict["scenarios_done"] = int(env_info["scenario_index"]) + 1
        data_dict["scenarios_total"] = env_info["scenarios_total"]
    # The env auto-flushes its Bench2Drive eval artifacts on the
    # terminating step and exposes the summary scores here.
    if "eval_summary" in env_info:
        for k, v in env_info["eval_summary"].items():
            if isinstance(v, (int, float)):
                data_dict[f"eval/{k}"] = v
    # for animalai_env
    if "pass_mark" in env_info:
        success = float(score >= env_info["pass_mark"])
        data_dict["success"] = success
        state.success_sum_all += success
        state.success_episode_count += 1
        state.success_list.append(success)
        state.success_list = state.success_list[-eval_range:]
        data_dict["success_episode_count"] = state.success_episode_count
        data_dict["total_success_rate"] = state.success_sum_all / state.success_episode_count
        if len(state.success_list) >= eval_range:
            data_dict["recent_success_rate"] = float(np.mean(state.success_list))
        arena_name = env_info["arena_name"] if "arena_name" in env_info else ""
        if arena_name:
            data_dict[f"success/{arena_name}"] = success
            data_dict[f"episodic_return/{arena_name}"] = score
        if "cleared_count" in env_info:
            data_dict["cleared_count"] = env_info["cleared_count"]
            data_dict["stage"] = env_info["stage"]
            data_dict["round_index"] = env_info["round_index"]
            data_dict["round_success_rate"] = env_info["round_success_rate"]
            data_dict["last_round_success_rate"] = env_info["last_round_success_rate"]
            for level, rate in env_info["last_round_level_success_rate"].items():
                data_dict[f"last_round_success_rate/{level}"] = rate
            data_dict["advanced"] = float(env_info["advanced"])
    if len(state.score_list) >= eval_range:
        data_dict["recent_average_score"] = float(np.mean(state.score_list))
    return data_dict


def final_evaluation(
    args: DictConfig, agent, network, env, seed: int, global_step: int, result_dir: Path
) -> None:
    """Post-training evaluation: the Animal-AI Testbed sweep for AnimalAI runs,
    and the Bench2Drive final summary the closed env stashed on itself."""
    if args.env_id == "AnimalAI-v0" and not args.debug:
        from test_trained_agent import run_testbed

        eval_factory = OmegaConf.merge(args.env_factory, {"mode": "eval"})
        eval_env = make_env(args.env_id, eval_factory, result_dir=None)
        eval_env.action_space.seed(seed)
        # A fresh env starts its counter at 0, but the network reads the global
        # step as an observation and was trained at this run's values.
        eval_env.unwrapped.set_global_step(global_step)
        if network is not None:
            network.eval()
        testbed_metrics = run_testbed(
            agent,
            eval_env,
            seed,
            args.render,
            args.env_id,
            global_step,
            result_dir / "eval" / "final",
            args.env_factory.train_variant,
            training_levels(
                args.env_factory.mode,
                args.env_factory.train_variant,
                list(args.env_factory.train_levels),
            ),
        )
        wandb.summary.update({f"testbed/{k}": v for k, v in testbed_metrics.items()})
        eval_env.close()
    # env.close() auto-merges the Bench2Drive eval sweep and stashes the
    # Driving Score / Success Rate / Efficiency / Comfort summary on the
    # env. Push it into wandb.summary so the run is self-describing.
    final_eval_summary = getattr(env.unwrapped, "final_eval_summary", None)
    if final_eval_summary is not None:
        wandb.summary.update({f"final/{k}": v for k, v in final_eval_summary.items()})
        print("=== Bench2Drive220 final ===")
        for k, v in final_eval_summary.items():
            print(f"  {k}: {v}")


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

    seed_everything(seed)
    with open(result_dir / "seed.txt", "w") as f:
        f.write(str(seed))
    write_git_info(result_dir)

    episode_log_dir = result_dir / "episode_log"
    episode_log_dir.mkdir(parents=True, exist_ok=True)

    log_episode_path = result_dir / "log_episode.tsv"
    log_episode_file = None
    log_episode_writer = None

    # env setup
    env = make_env(args.env_id, args.env_factory, result_dir=result_dir)
    env.unwrapped.max_step_count = args.max_step_count
    env.action_space.seed(seed)

    eval_range = env.unwrapped.eval_range

    start_time = time.time()

    state = TrainState.fresh()
    best_score_per_arena: dict[str, float] = {}
    step_limit = args.step_limit
    episode_limit = args.episode_limit
    time_limit_sec = args.time_limit_hour * 3600
    checkpoint_interval = max(1, step_limit // 10)

    def out_of_budget() -> bool:
        return state.global_step >= step_limit or time.time() - start_time >= time_limit_sec

    # Every agent composes its own language input; the env only publishes state.
    # The chain of thought writes into the same conversation, so the builder is
    # made before the network that carries the chain.
    network, agent = build_all(env, args)

    parameter_count = (
        sum(p.numel() for p in agent.network.parameters()) if network is not None else 0
    )
    print(f"Parameter count: {parameter_count:,}")

    if args.resume_dir is not None:
        state, best_score_per_arena = load_resume_state(Path(args.resume_dir), network, agent, env)
        print(
            f"Resumed from {args.resume_dir}: "
            f"global_step={state.global_step} episode_id={state.episode_id}"
        )
        set_global_step = getattr(env.unwrapped, "set_global_step", None)
        if set_global_step is not None:
            set_global_step(state.global_step)

    while True:
        # Stop when the env has dispensed every scenario in a fixed playlist:
        # Animal-AI's "sequential" and "eval" arena orders, Bench2Drive220's
        # sequential runtime. An env with no finite playlist does not publish
        # the flag and is left to episode_limit / step_limit.
        if getattr(env.unwrapped, "is_exhausted", False):
            break

        # Stop once we've run the configured number of episodes.
        if state.episode_id >= episode_limit:
            break

        # initialize episode (seed only the first call so the gym RNG is
        # set once; subsequent resets keep advancing it).
        obs, reset_info = env.reset(seed=seed) if state.episode_id == 0 else env.reset()

        # initial action
        result = agent.select_action(state.global_step, obs, 0.0, False, False, reset_info)
        action = result.action

        # The trainer only owns the environment / observation panels; goal,
        # bev, ... arrive via result.panels. The initial render leads.
        record = EpisodeRecord.fresh()
        rgb_image = render_frame(env, obs, result, args.render_scale)
        record.bgr_images.append(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        record.observations.append(obs["image"].copy())
        record.texts.append(result.texts)
        # Animal-AI reports the agent's arena position every step; an env with no
        # arena reports none and record.xyzs stays empty.
        log_position = "agent_xyz" in reset_info

        while True:
            state.global_step += 1

            # step
            env_step_start = time.time()
            obs, reward, terminated, truncated, env_info = env.step(action)
            env_step_time_msec = (time.time() - env_step_start) * 1000

            record.actions.append(action.copy())
            record.rewards.append(reward)
            record.observations.append(obs["image"].copy())
            if log_position:
                record.xyzs.append(env_info["agent_xyz"])

            agent_step_start = time.time()
            result = agent.step(state.global_step, obs, reward, terminated, truncated, env_info)
            action = result.action
            agent_step_time_msec = (time.time() - agent_step_start) * 1000

            # log: metrics are already scalar telemetry (images live in panels)
            elapsed_time_sec = time.time() - start_time
            wandb.log(
                {
                    "global_step": state.global_step,
                    "elapsed_time_min": elapsed_time_sec / 60,
                    "SPS": state.global_step / elapsed_time_sec,
                    "reward": reward,
                    "env_step_msec": env_step_time_msec,
                    "agent_step_msec": agent_step_time_msec,
                    **result.metrics,
                }
            )

            rgb_image = render_frame(env, obs, result, args.render_scale)
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            record.bgr_images.append(bgr_image)
            record.texts.append(result.texts)
            if args.render:
                cv2.imshow(args.env_id, bgr_image)
                cv2.waitKey(1)

            if state.global_step % checkpoint_interval == 0:
                save_checkpoint(result_dir, network, agent)

            if terminated or truncated:
                break

            if out_of_budget():
                break

        if out_of_budget():
            break

        score = env_info["episode"]["r"]
        if args.normalizing_by_return:
            agent.reward_processor.update(score)

        elapsed_time_sec = time.time() - start_time
        data_dict = build_episode_log(state, env_info, eval_range, elapsed_time_sec)
        wandb.log(data_dict)

        if log_episode_writer is None:
            log_episode_file = open(log_episode_path, "w", newline="")
            fieldnames = list(data_dict.keys()) + [
                "recent_average_score",
                "recent_success_rate",
            ]
            log_episode_writer = csv.DictWriter(
                log_episode_file, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
            )
            log_episode_writer.writeheader()
        log_episode_writer.writerow(data_dict)
        log_episode_file.flush()

        if state.episode_id % args.print_interval == 0:
            recent_average_score = float(np.mean(state.score_list))
            print(
                f"Ep: {state.episode_id}\tStep: {state.global_step}\tLast score: {score:.2f}\tAverage score: {recent_average_score:.2f}\tLength: {env_info['episode']['l']:.2f}\tElapsed time: {elapsed_time_sec / 3600:.2f}h"
            )

        episode_end_info = agent.on_episode_end(score)
        wandb.log(episode_end_info)

        arena_name = env_info["arena_name"] if "arena_name" in env_info else ""

        if arena_name:
            arena_is_best = (
                arena_name not in best_score_per_arena or score > best_score_per_arena[arena_name]
            )
            if arena_is_best:
                best_score_per_arena[arena_name] = score
                save_episode_data(episode_log_dir, f"best_{arena_name}", record)
        else:
            is_best = score > state.best_score
            if is_best:
                with open(result_dir / "best_score.txt", "w") as f:
                    f.write(f"{state.episode_id + 1}\t{score:.2f}")
                state.best_score = score
                save_episode_data(episode_log_dir, "best_episode", record)

        if state.episode_id == 0 or (state.episode_id + 1) % args.image_save_interval == 0:
            save_episode_data(episode_log_dir, f"ep_{state.episode_id + 1:08d}", record)

        state.episode_id += 1
        save_train_state(result_dir, state, env, best_score_per_arena)

    save_checkpoint(result_dir, network, agent)

    env.close()

    final_evaluation(args, agent, network, env, seed, state.global_step, result_dir)

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
        cfg.render = False
        cfg.step_limit = 100
        cfg.buffer_size = int(2e4)

    if cfg.off_wandb:
        os.environ["WANDB_MODE"] = "offline"

    cfg.render = disable_render_if_headless(cfg.render)

    exp_name = f"{cfg.agent_type.upper()}_{cfg.exp_name}"
    seed = resolve_seed(cfg.seed)

    for i in range(cfg.trial_num):
        suffix = f"_{i:02d}" if cfg.trial_num > 1 else ""
        trial_dir = hydra_output_dir / f"trial{suffix}" if cfg.trial_num > 1 else hydra_output_dir
        main(cfg, exp_name + suffix, seed + i, trial_dir)


if __name__ == "__main__":
    hydra_main()
