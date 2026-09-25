# SPDX-License-Identifier: MIT
"""Run every Animal-AI Olympics arena exactly once with a frozen policy.

Loads a trained checkpoint (weights only), then sweeps every XX-YY-ZZ.yaml
arena under external/animal-ai/configs/competition/ in order, one episode
each, with the network in eval mode and no optimizer step ever taken. The
sweep order and its end come from the env's SequentialSelector.

Instead of hydra flags, this script takes the path to a checkpoint.pt file
(as saved by scripts/train.py) and reconstructs the training config from
the wandb run recorded alongside it (<run_dir>/wandb/*/files/config.yaml), so
running an eval never requires re-specifying agent/env/network overrides by
hand.
"""

from vla_streaming_rl.script_setup import setup_runtime

setup_runtime()

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import yaml
from omegaconf import DictConfig, OmegaConf

from vla_streaming_rl.agents.build import build_all
from vla_streaming_rl.checkpoint import load_checkpoint_weights
from vla_streaming_rl.envs.animalai_env import seen_in_training, training_levels
from vla_streaming_rl.script_setup import disable_render_if_headless, resolve_seed, seed_everything
from vla_streaming_rl.utils import render_frame
from vla_streaming_rl.wrappers import make_env


def _coerce_numeric_strings(value):
    """YAML 1.1 leaves exponent floats like '1e-05' as strings; recover them as floats."""
    if isinstance(value, dict):
        return {key: _coerce_numeric_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_numeric_strings(item) for item in value]
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def load_wandb_config(run_dir: Path) -> DictConfig:
    """Reconstruct the resolved training config from a wandb run's config.yaml."""
    config_paths = sorted(run_dir.glob("wandb/run-*/files/config.yaml"))
    assert config_paths, f"No wandb config.yaml found under {run_dir}/wandb/"
    raw = yaml.safe_load(config_paths[-1].read_text())
    raw.pop("_wandb", None)
    flat = {key: _coerce_numeric_strings(entry["value"]) for key, entry in raw.items()}
    return OmegaConf.create(flat)


def load_global_step(run_dir: Path) -> int:
    """The training step the checkpoint was written at, from train_state.json."""
    train_state_path = run_dir / "train_state.json"
    assert train_state_path.exists(), f"{train_state_path} not found"
    return int(json.loads(train_state_path.read_text())["global_step"])


def _show(env, obs, result, render: bool, window_name: str) -> None:
    if not render:
        return
    frame = render_frame(env, obs, result, 1.0)
    cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def run_arena(
    agent, env, seed: int, render: bool, window_name: str, global_step: int
) -> tuple[str, bool, float]:
    """Play the next arena the env serves; return its (name, passed, score).

    `global_step` is the training step the checkpoint was written at. It gates
    OffPolicyAgent's warmup (below `learning_starts` it returns
    uniform random actions instead of querying the network), which a trained
    checkpoint is always past.
    """
    obs, reset_info = env.reset(seed=seed, options=None)
    arena_name = reset_info["arena_name"]
    result = agent.select_action(global_step, obs, 0.0, False, False, reset_info)
    action = result.action
    _show(env, obs, result, render, window_name)

    while True:
        obs, reward, terminated, truncated, env_info = env.step(action)
        result = agent.select_action(global_step, obs, reward, terminated, truncated, env_info)
        action = result.action
        _show(env, obs, result, render, window_name)

        if terminated or truncated:
            break

    score = env_info["episode"]["r"]
    success = bool(score >= env_info["pass_mark"])
    return arena_name, success, score


def run_testbed(
    agent,
    env,
    seed: int,
    render: bool,
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
    the held-out one, and both rates are reported.
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
        while not selector.is_exhausted:
            arena_name, success, score = run_arena(
                agent, env, seed, render, window_name, global_step
            )
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


def main(
    args: DictConfig, checkpoint_path: Path, result_dir: Path, seed: int, render: bool
) -> None:
    seed_everything(seed)

    global_step = load_global_step(checkpoint_path.parent)

    env = make_env(args.env_id, args.env_factory, result_dir=None)
    env.action_space.seed(seed)
    # The network reads the global step as an observation, so a fresh env's 0
    # would be far outside anything the checkpoint was trained at.
    env.unwrapped.set_global_step(global_step)

    network, agent = build_all(env, args)
    load_checkpoint_weights(checkpoint_path, network)
    network.eval()

    run_testbed(
        agent,
        env,
        seed,
        render,
        args.env_id,
        global_step,
        result_dir,
        args.env_factory.train_variant,
        training_levels(
            args.env_factory.mode,
            args.env_factory.train_variant,
            list(args.env_factory.train_levels),
        ),
    )

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to a checkpoint.pt file saved by scripts/train.py",
    )
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    checkpoint_path = cli_args.checkpoint.resolve()
    run_dir = checkpoint_path.parent
    cfg = load_wandb_config(run_dir)
    # Always evaluate on the paper's Testbed sweep, regardless of which mode
    # trained this checkpoint (see SequentialSelector in animalai_env.py).
    cfg.env_factory.mode = "eval"

    seed = resolve_seed(cli_args.seed)
    eval_dir = run_dir / "eval" / checkpoint_path.stem
    render = disable_render_if_headless(cli_args.render)

    print(OmegaConf.to_yaml(cfg))
    main(cfg, checkpoint_path, eval_dir, seed, render)
