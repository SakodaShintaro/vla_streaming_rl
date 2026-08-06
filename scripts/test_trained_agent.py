# SPDX-License-Identifier: MIT
"""Run every Animal-AI Olympics arena exactly once with a frozen policy.

Loads a trained checkpoint (weights only), then sweeps every XX-YY-ZZ.yaml
arena under external/animal-ai/configs/competition/ in order, one episode
each, with the network in eval mode and no optimizer step ever taken. The
sweep order and its end come from the env's SequentialSelector, so run this
with `env=animalai env_factory.mode=eval`.
"""

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

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from vla_streaming_rl.agents.build import build_agent
from vla_streaming_rl.networks.build import build_network
from vla_streaming_rl.wrappers import make_env

torch.set_float32_matmul_precision("high")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_checkpoint_weights(checkpoint_dir: Path, network: torch.nn.Module) -> None:
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    trainable_state = torch.load(checkpoint_path, map_location="cuda")
    module = network._orig_mod if hasattr(network, "_orig_mod") else network
    missing, unexpected = module.load_state_dict(trainable_state, strict=False)
    if unexpected:
        raise ValueError(f"checkpoint parameters not found in the network: {unexpected[:5]}")
    print(f"Loaded {len(trainable_state)} weight tensors from {checkpoint_path}")


def run_arena(agent, env, seed: int) -> tuple[str, bool, float]:
    """Play the next arena the env serves; return its (name, passed, score)."""
    obs, reset_info = env.reset(seed=seed, options=None)
    arena_name = reset_info["arena_name"]
    result = agent.select_action(0, obs, 0.0, False, False, reset_info)
    action = result.action

    while True:
        obs, reward, terminated, truncated, env_info = env.step(action)
        result = agent.select_action(0, obs, reward, terminated, truncated, env_info)
        action = result.action
        if terminated or truncated:
            break

    score = env_info["episode"]["r"]
    success = bool(score >= env_info["pass_mark"])
    return arena_name, success, score


def main(args: DictConfig, result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed != -1 else np.random.randint(0, 10000)
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
    load_checkpoint_weights(Path(args.resume_dir), network)
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
            arena_name, success, score = run_arena(agent, env, seed)
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


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def hydra_main(cfg: DictConfig) -> None:
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    os.chdir(hydra.utils.get_original_cwd())

    if not os.environ.get("DISPLAY"):
        print("Because a headless environment is detected, rendering is automatically disabled.")
        cfg.render = 0

    if cfg.resume_dir is None:
        raise ValueError("test.py requires resume_dir to point at a trained checkpoint directory.")

    print(OmegaConf.to_yaml(cfg))
    main(cfg, hydra_output_dir)


if __name__ == "__main__":
    hydra_main()
