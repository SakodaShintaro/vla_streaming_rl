# SPDX-License-Identifier: MIT
"""Animal-AI Olympics の評価アリーナを、学習しない方策で1回ずつ回す。

学習の終わりの評価（scripts/train.py）と、チェックポイントからの評価
（scripts/test_trained_agent.py）の両方が使う。"""

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from vla_streaming_rl.agents.base import Agent
from vla_streaming_rl.envs.animalai_curriculum import seen_in_training
from vla_streaming_rl.episode_log import EpisodeRecord, save_episode_data
from vla_streaming_rl.render import render_frame


def run_arena(
    agent: Agent,
    env,
    seed: int,
    render: bool,
    render_scale: float,
    window_name: str,
    global_step: int,
    prev_action: np.ndarray,
) -> tuple[str, bool, float, EpisodeRecord, np.ndarray]:
    """Play the next arena the env serves; return its (name, passed, score),
    what the episode leaves behind, recorded the way training records it, and
    the action the agent returned last, which it reads at the next reset.

    `global_step` is the training step the checkpoint was written at. Below
    `learning_starts` OffPolicyAgent treats every step as warmup and lets the
    network act as it does while the buffer fills, which a trained checkpoint
    is always past.
    """
    obs, reset_info = env.reset(seed=seed, options=None)
    arena_name = reset_info["arena_name"]
    result = agent.select_action(global_step, obs, 0.0, False, False, reset_info)
    record = EpisodeRecord.start(
        render_frame(env, obs, reset_info, prev_action, result, render_scale),
        obs,
        result.texts,
        reset_info,
    )
    action = result.action

    while True:
        obs, reward, terminated, truncated, env_info = env.step(action)
        result = agent.select_action(global_step, obs, reward, terminated, truncated, env_info)
        frame = render_frame(env, obs, env_info, action, result, render_scale)
        record.add_step(action, reward, obs, env_info, frame, result.texts)
        action = result.action
        if render:
            cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if terminated or truncated:
            break

    score = env_info["episode"]["r"]
    success = bool(score >= env_info["pass_mark"])
    agent.on_episode_end(score)
    return arena_name, success, score, record, action


def run_testbed(
    agent: Agent,
    env,
    seed: int,
    render: bool,
    render_scale: float,
    window_name: str,
    global_step: int,
    result_dir: Path,
    train_variant: str,
    train_levels: list[str],
) -> dict[str, float]:
    """Sweep every arena the env's SequentialSelector serves and write results.

    Returns per-level pass rates plus the overall "cleared" count and
    "success_rate" so callers can push them into run summaries. Training draws
    one variant of every competition task, so part of the Testbed is arenas the
    run has trained on; `seen_in_training` splits the sweep into that part and
    the held-out one, and both rates are reported. Every episode is kept under
    ``episode_log/<arena>``.
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    selector = env.unwrapped.selector
    arena_count = len(selector.arenas)
    seen = seen_in_training(train_variant, train_levels)
    print(f"Running {arena_count} arenas, 1 episode each.")

    result_path = result_dir / "test_result.tsv"
    level_attempts: Counter[str] = Counter()
    level_successes: Counter[str] = Counter()
    split_attempts = {"seen": 0, "held_out": 0}
    split_successes = {"seen": 0, "held_out": 0}
    with open(result_path, "w") as f:
        f.write("arena\tsuccess\tscore\tseen_in_training\n")
        success_count = 0
        # エージェントは前の行動をエピソードをまたいで持ち越すので、表示もそれに合わせる
        prev_action = np.zeros(env.action_space.shape, dtype=np.float32)
        while not selector.is_exhausted:
            arena_name, success, score, record, prev_action = run_arena(
                agent, env, seed, render, render_scale, window_name, global_step, prev_action
            )
            save_episode_data(result_dir / "episode_log", arena_name, record)
            success_count += int(success)
            # Competition arenas are named XX-YY-ZZ (level-task-variant).
            level = arena_name.split("-")[0]
            level_attempts[level] += 1
            level_successes[level] += int(success)
            split = "seen" if arena_name in seen else "held_out"
            split_attempts[split] += 1
            split_successes[split] += int(success)
            f.write(f"{arena_name}\t{int(success)}\t{score:.6f}\t{int(arena_name in seen)}\n")
            f.flush()
            done = sum(level_attempts.values())
            print(f"[{done}/{arena_count}] {arena_name}\tsuccess={int(success)}\tscore={score:.2f}")

    success_rate = success_count / arena_count
    print(f"Cleared {success_count}/{arena_count} arenas ({success_rate:.1%}).")
    summary_lines = [f"cleared: {success_count}/{arena_count} ({success_rate:.1%})"]
    metrics: dict[str, float] = {
        "cleared": success_count,
        "arena_count": arena_count,
        "success_rate": success_rate,
    }
    for split in ("seen", "held_out"):
        n_success = split_successes[split]
        n_attempt = split_attempts[split]
        if n_attempt == 0:
            continue
        summary_lines.append(f"{split}: {n_success}/{n_attempt} ({n_success / n_attempt:.1%})")
        metrics[f"success_rate_{split}"] = n_success / n_attempt
        metrics[f"arena_count_{split}"] = n_attempt
    for level in sorted(level_attempts):
        n_success = level_successes[level]
        n_attempt = level_attempts[level]
        summary_lines.append(
            f"level {level}: {n_success}/{n_attempt} ({n_success / n_attempt:.1%})"
        )
        metrics[f"level_{level}"] = n_success / n_attempt
    (result_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")
    return metrics
