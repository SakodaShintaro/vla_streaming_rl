# SPDX-License-Identifier: MIT
# This script was initially inspired by CleanRL https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
from vla_streaming_rl.script_setup import setup_runtime

setup_runtime()

import csv
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from vla_streaming_rl.agents.build import build_all
from vla_streaming_rl.checkpoint import load_checkpoint_weights, save_checkpoint
from vla_streaming_rl.envs.animalai.curriculum import training_levels
from vla_streaming_rl.episode_log import EpisodeRecord, save_episode_data
from vla_streaming_rl.render import render_frame
from vla_streaming_rl.script_setup import disable_render_if_headless, resolve_seed, seed_everything
from vla_streaming_rl.testbed import run_testbed
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

    `cleared` is the arena selector's own verdict: passed at least once.

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
    """Restore weights / optimizer / counters / curriculum from a previous run dir,
    as ``save_checkpoint`` and ``save_train_state`` wrote them."""
    load_checkpoint_weights(resume_dir / "checkpoint.pt", network)
    agent.load_optimizer_state_dict(torch.load(resume_dir / "optimizer.pt", map_location="cuda"))

    data = json.loads((resume_dir / "train_state.json").read_text())
    best_score_per_arena: dict[str, float] = {}
    set_curriculum = getattr(env.unwrapped, "set_curriculum_state", None)
    if set_curriculum is not None:
        # カリキュラムのある環境では、進み具合とアリーナごとの記録も一緒に書いてある
        curriculum_progress = data.pop("curriculum_progress")
        attempts = {}
        successes = {}
        arena_stats_path = resume_dir / "arena_stats.tsv"
        for row in arena_stats_path.read_text().splitlines()[1:]:
            arena, n_attempt, n_success, _rate, best, _cleared = row.split("\t")
            attempts[arena] = int(n_attempt)
            successes[arena] = int(n_success)
            best_score_per_arena[arena] = float(best)
        set_curriculum(attempts, successes, curriculum_progress)
        print(f"Resume: loaded {len(attempts)} arena records from {arena_stats_path}")
    return TrainState(**data), best_score_per_arena


def save_train_state(
    result_dir: Path, state: TrainState, env, best_score_per_arena: dict[str, float]
) -> None:
    """Persist the light resume state (the heavy weights / optimizer files
    are saved at every tenth of the run's progress instead)."""
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
    state: TrainState, env_info: dict, mode: str, eval_range: int, elapsed_time_sec: float
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
        if mode in ("sequential", "random"):
            data_dict["cleared_count"] = env_info["cleared_count"]
            data_dict["round_index"] = env_info["round_index"]
            data_dict["round_success_rate"] = env_info["round_success_rate"]
            # 周が閉じたエピソードでだけ記録する（それ以外は前周の値の繰り返しになる）
            if env_info["round_closed"]:
                data_dict["last_round_success_rate"] = env_info["last_round_success_rate"]
                for level, rate in env_info["last_round_level_success_rate"].items():
                    data_dict[f"last_round_success_rate/{level}"] = rate
    if len(state.score_list) >= eval_range:
        data_dict["recent_average_score"] = float(np.mean(state.score_list))
    return data_dict


class EpisodeTable:
    """log_episode.tsv: one row per episode, the same scalars the episode logs
    to wandb. The columns are fixed by the first episode's row (plus the
    running averages, which only appear once enough episodes have ended)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = None
        self._writer = None

    def write(self, data_dict: dict) -> None:
        if self._writer is None:
            self._file = open(self.path, "w", newline="")
            fieldnames = list(data_dict.keys()) + [
                "recent_average_score",
                "recent_success_rate",
            ]
            self._writer = csv.DictWriter(
                self._file, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
            )
            self._writer.writeheader()
        self._writer.writerow(data_dict)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def save_episode_videos(
    result_dir: Path,
    state: TrainState,
    arena_name: str,
    score: float,
    record: EpisodeRecord,
    best_score_per_arena: dict[str, float],
    args: DictConfig,
) -> None:
    """終わったエピソードの記録を残す。アリーナのある環境ではアリーナごとの最高点の
    エピソードを、ない環境では全体の最高点のエピソードを残し、それとは別に
    ``save_interval`` エピソードごとに1つ残す。"""
    episode_log_dir = result_dir / "episode_log"
    if arena_name:
        arena_is_best = (
            arena_name not in best_score_per_arena or score > best_score_per_arena[arena_name]
        )
        if arena_is_best:
            best_score_per_arena[arena_name] = score
            save_episode_data(episode_log_dir, f"best_{arena_name}", record)
    elif score > state.best_score:
        with open(result_dir / "best_score.txt", "w") as f:
            f.write(f"{state.episode_id + 1}\t{score:.2f}")
        state.best_score = score
        save_episode_data(episode_log_dir, "best_episode", record)

    if state.episode_id == 0 or (state.episode_id + 1) % args.save_interval == 0:
        save_episode_data(episode_log_dir, f"ep_{state.episode_id + 1:08d}", record)


def final_evaluation(
    args: DictConfig, agent, network, env, seed: int, global_step: int, result_dir: Path
) -> None:
    """Post-training evaluation: the Animal-AI Testbed sweep for AnimalAI runs,
    and the Bench2Drive final summary the closed env stashed on itself."""
    if args.env_id == "AnimalAI-v0" and args.final_evaluation and not args.debug:
        eval_factory = OmegaConf.merge(args.env_factory, {"mode": "eval"})
        eval_env = make_env(args.env_id, eval_factory, result_dir=None, seed=seed)
        eval_env.action_space.seed(seed)
        # A fresh env starts its counter at 0, but the network reads the global
        # step as an observation and was trained at this run's values.
        eval_env.unwrapped.set_global_step(global_step)
        network.eval()
        testbed_metrics = run_testbed(
            agent,
            eval_env,
            seed,
            args.render,
            args.render_scale,
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

    (result_dir / "episode_log").mkdir(parents=True, exist_ok=True)

    episode_table = EpisodeTable(result_dir / "log_episode.tsv")

    # env setup
    env = make_env(args.env_id, args.env_factory, result_dir=result_dir, seed=seed)
    env.action_space.seed(seed)

    eval_range = env.unwrapped.eval_range
    # The curriculum mode only exists for AnimalAI; other envs have no mode-specific logs.
    mode = args.env_factory.mode if args.env_id == "AnimalAI-v0" else ""

    start_time = time.time()

    state = TrainState.fresh()
    best_score_per_arena: dict[str, float] = {}
    step_limit = args.step_limit
    episode_limit = args.episode_limit
    time_limit_sec = args.time_limit_hour * 3600

    def out_of_budget() -> bool:
        return state.global_step >= step_limit or time.time() - start_time >= time_limit_sec

    def progress_tenths() -> int:
        """学習がどこまで進んだかを10段階で。ステップ数・エピソード数・時間のうち、上限に
        いちばん近いもので測る。環境によってどれで終わるかが違うため"""
        fraction = max(
            state.global_step / step_limit,
            state.episode_id / episode_limit,
            (time.time() - start_time) / time_limit_sec,
        )
        return int(fraction * 10)

    # Every agent composes its own language input; the env only publishes state.
    # The chain of thought writes into the same conversation, so the builder is
    # made before the network that carries the chain.
    network, agent = build_all(env, args)

    parameter_count = sum(p.numel() for p in agent.network.parameters())
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

    # 最後に重みを保存した時点の進み具合
    checkpoint_tenths = progress_tenths()

    # ネットワークが行動として読む、前のステップでエージェントが返した行動。エージェントは
    # エピソードをまたいで持ち越すので、ここでも持ち越す
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    while True:
        # Stop when the env has dispensed every scenario in a fixed playlist:
        # Animal-AI's "eval" arena order, Bench2Drive220's
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

        # The trainer only owns the environment / observation panels; goal,
        # bev, ... arrive via result.panels. The initial render leads.
        record = EpisodeRecord.start(
            render_frame(env, obs, reset_info, action, result, args.render_scale),
            obs,
            result.texts,
            reset_info,
        )
        action = result.action

        while True:
            state.global_step += 1

            # step
            env_step_start = time.time()
            obs, reward, terminated, truncated, env_info = env.step(action)
            env_step_time_msec = (time.time() - env_step_start) * 1000

            agent_step_start = time.time()
            result = agent.step(state.global_step, obs, reward, terminated, truncated, env_info)
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

            frame = render_frame(env, obs, env_info, action, result, args.render_scale)
            record.add_step(action, reward, obs, env_info, frame, result.texts)
            action = result.action
            if args.render:
                cv2.imshow(args.env_id, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            if terminated or truncated:
                break

            if out_of_budget():
                break

        if out_of_budget():
            break

        score = env_info["episode"]["r"]
        elapsed_time_sec = time.time() - start_time
        data_dict = build_episode_log(state, env_info, mode, eval_range, elapsed_time_sec)
        wandb.log(data_dict)
        episode_table.write(data_dict)

        if state.episode_id % args.print_interval == 0:
            recent_average_score = float(np.mean(state.score_list))
            print(
                f"Ep: {state.episode_id}\tStep: {state.global_step}\tLast score: {score:.2f}\tAverage score: {recent_average_score:.2f}\tLength: {env_info['episode']['l']:.2f}\tElapsed time: {elapsed_time_sec / 3600:.2f}h"
            )

        agent.on_episode_end(score)

        arena_name = env_info["arena_name"] if "arena_name" in env_info else ""

        save_episode_videos(
            result_dir, state, arena_name, score, record, best_score_per_arena, args
        )

        state.episode_id += 1
        save_train_state(result_dir, state, env, best_score_per_arena)
        # 重みは進み具合の 10 分の 1 ごとに、カウンタと同じエピソードの境目で保存する。
        # 再開したときに重みとカウンタの時点がずれないように
        if progress_tenths() > checkpoint_tenths:
            checkpoint_tenths = progress_tenths()
            save_checkpoint(result_dir, network, agent)

    save_checkpoint(result_dir, network, agent)

    env.close()

    final_evaluation(args, agent, network, env, seed, state.global_step, result_dir)

    episode_table.close()
    wandb.finish()


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def hydra_main(cfg: DictConfig) -> None:
    # Hydra's output dir is our result dir (configured via hydra.run.dir)
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    # Restore cwd so relative paths in the code work correctly
    os.chdir(hydra.utils.get_original_cwd())

    if cfg.debug:
        cfg.off_wandb = True
        cfg.learning_starts = max(10, cfg.actor_critic.seq_len + cfg.horizon + 5)
        cfg.render = False
        cfg.step_limit = 100
        cfg.replay_buffer.size = int(2e4)

    if cfg.off_wandb:
        os.environ["WANDB_MODE"] = "offline"

    cfg.render = disable_render_if_headless(cfg.render)

    exp_name = f"{HydraConfig.get().runtime.choices['agent']}_{cfg.exp_name}"
    seed = resolve_seed(cfg.seed)

    for i in range(cfg.trial_num):
        suffix = f"_{i:02d}" if cfg.trial_num > 1 else ""
        trial_dir = hydra_output_dir / f"trial{suffix}" if cfg.trial_num > 1 else hydra_output_dir
        main(cfg, exp_name + suffix, seed + i, trial_dir)


if __name__ == "__main__":
    hydra_main()
