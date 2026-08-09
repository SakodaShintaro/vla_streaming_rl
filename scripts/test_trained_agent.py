# SPDX-License-Identifier: MIT
"""Run every Animal-AI Olympics arena exactly once with a frozen policy.

Loads a trained checkpoint (weights only), then sweeps every XX-YY-ZZ.yaml
arena under external/animal-ai/configs/competition/ in order, one episode
each, with the network in eval mode and no optimizer step ever taken. The
sweep order and its end come from the env's SequentialSelector.

Instead of hydra flags, this script takes the path to a single checkpoint_*.pt
file (as saved by scripts/train.py) and reconstructs the training config from
the wandb run recorded alongside it (<run_dir>/wandb/*/files/config.yaml), so
running an eval never requires re-specifying agent/env/network overrides by
hand.
"""

import argparse
import logging
import os
import random
import warnings
from pathlib import Path

os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS.*")
logging.getLogger("httpx").setLevel(logging.WARNING)

import cv2
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from vla_streaming_rl.agents.build import build_agent
from vla_streaming_rl.networks.build import build_network
from vla_streaming_rl.utils import concat_labeled_images, overlay_caption
from vla_streaming_rl.wrappers import make_env

torch.set_float32_matmul_precision("high")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    if not config_paths:
        raise FileNotFoundError(f"No wandb config.yaml found under {run_dir}/wandb/")
    raw = yaml.safe_load(config_paths[-1].read_text())
    raw.pop("_wandb", None)
    flat = {key: _coerce_numeric_strings(entry["value"]) for key, entry in raw.items()}
    return OmegaConf.create(flat)


def load_checkpoint_weights(checkpoint_path: Path, network: torch.nn.Module) -> None:
    trainable_state = torch.load(checkpoint_path, map_location="cuda")
    module = network._orig_mod if hasattr(network, "_orig_mod") else network
    missing, unexpected = module.load_state_dict(trainable_state, strict=False)
    if unexpected:
        raise ValueError(f"checkpoint parameters not found in the network: {unexpected[:5]}")

    # `missing` includes two harmless cases: frozen params (never saved) and
    # names that alias a shared tensor already restored under a different key
    # (e.g. tied lm_head/embed_tokens, or image_processor reused by prediction_head).
    # Resolve every trainable missing name to its tensor and flag it only if
    # that exact tensor was never touched by any key actually in the checkpoint.
    loaded_ids = set()
    for name, param in module.named_parameters(remove_duplicate=False):
        if name in trainable_state:
            loaded_ids.add(id(param))
    name_to_param = dict(module.named_parameters(remove_duplicate=False))
    unaccounted_missing = [
        name
        for name in missing
        if name in name_to_param
        and name_to_param[name].requires_grad
        and id(name_to_param[name]) not in loaded_ids
    ]
    if unaccounted_missing:
        raise ValueError(
            f"trainable network parameters not found in the checkpoint: {unaccounted_missing[:5]}"
        )
    print(f"Loaded {len(trainable_state)} weight tensors from {checkpoint_path}")


def run_arena(
    agent, env, seed: int, render: bool, window_name: str, global_step: int
) -> tuple[str, bool, float]:
    """Play the next arena the env serves; return its (name, passed, score).

    `global_step` must exceed the agent's `learning_starts`: StandardAgent
    treats smaller steps as off-policy warmup and returns uniform random
    actions instead of querying the network.
    """
    obs, reset_info = env.reset(seed=seed, options=None)
    arena_name = reset_info["arena_name"]
    result = agent.select_action(global_step, obs, 0.0, False, False, reset_info)
    action = result.action

    if render:
        panels = {
            "environment": overlay_caption(env.render(), f"{obs['language']}  reward: {0.0:+.3f}"),
            "observation": obs["image"].copy().transpose(1, 2, 0),
            **result.panels,
        }
        cv2.imshow(window_name, cv2.cvtColor(concat_labeled_images(panels), cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    while True:
        obs, reward, terminated, truncated, env_info = env.step(action)
        result = agent.select_action(global_step, obs, reward, terminated, truncated, env_info)
        action = result.action

        if render:
            panels = {
                "environment": overlay_caption(
                    env.render(), f"{obs['language']}  reward: {reward:.3f}"
                ),
                "observation": obs["image"].copy().transpose(1, 2, 0),
                **result.panels,
            }
            cv2.imshow(window_name, cv2.cvtColor(concat_labeled_images(panels), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if terminated or truncated:
            break

    score = env_info["episode"]["r"]
    success = bool(score >= env_info["pass_mark"])
    return arena_name, success, score


def main(
    args: DictConfig, checkpoint_path: Path, result_dir: Path, seed: int, render: bool
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)

    env = make_env(args.env_id, args.env_factory, result_dir=None)
    env.action_space.seed(seed)

    network = build_network(
        args,
        observation_space_shape=env.observation_space["image"].shape,
        action_space_shape=env.action_space.shape,
        parse_action_text=getattr(env.unwrapped, "parse_action_text", None),
        device=torch.device("cuda"),
    )
    agent = build_agent(env, network, args)
    load_checkpoint_weights(checkpoint_path, network)
    network.eval()

    selector = env.unwrapped.selector
    arena_count = len(selector.arenas)
    print(f"Running {arena_count} arenas, 1 episode each.")

    result_path = result_dir / "test_result.tsv"
    level_attempts: dict[str, int] = {}
    level_successes: dict[str, int] = {}
    with open(result_path, "w") as f:
        f.write("arena\tsuccess\tscore\n")
        success_count = 0
        while not selector.is_exhausted:
            arena_name, success, score = run_arena(
                agent, env, seed, render, args.env_id, args.learning_starts + 1
            )
            success_count += int(success)
            # Competition arenas are named XX-YY-ZZ (level-task-variant).
            level = arena_name.split("-")[0]
            level_attempts[level] = level_attempts.get(level, 0) + 1
            level_successes[level] = level_successes.get(level, 0) + int(success)
            f.write(f"{arena_name}\t{int(success)}\t{score:.6f}\n")
            f.flush()
            done = sum(level_attempts.values())
            print(f"[{done}/{arena_count}] {arena_name}\tsuccess={int(success)}\tscore={score:.2f}")

    print(f"Cleared {success_count}/{arena_count} arenas.")
    summary_lines = [f"cleared: {success_count}/{arena_count}"]
    for level in sorted(level_attempts):
        n_success = level_successes[level]
        n_attempt = level_attempts[level]
        summary_lines.append(
            f"level {level}: {n_success}/{n_attempt} ({n_success / n_attempt:.1%})"
        )
    (result_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to a checkpoint_<step>.pt file saved by scripts/train.py",
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

    seed = cli_args.seed if cli_args.seed != -1 else np.random.randint(0, 10000)
    eval_dir = run_dir / "eval" / checkpoint_path.stem
    render = cli_args.render
    if render and not os.environ.get("DISPLAY"):
        print("Because a headless environment is detected, rendering is automatically disabled.")
        render = False

    print(OmegaConf.to_yaml(cfg))
    main(cfg, checkpoint_path, eval_dir, seed, render)
