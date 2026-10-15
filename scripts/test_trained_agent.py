# SPDX-License-Identifier: MIT
"""Run the Animal-AI Olympics evaluation arenas exactly once with a frozen policy.

Loads a trained checkpoint (weights only), then sweeps the XX-YY-ZZ.yaml arenas
under external/animal-ai/configs/competition/ that ``eval_levels`` and
``eval_variants`` in configs/env/animalai.yaml pick (all 900 for the paper's
Testbed), in order, one episode
each, with the network in eval mode and no optimizer step ever taken. The
sweep order and its end come from the env's SequentialSelector. Every
episode's videos, actions, rewards, positions and texts are kept under
``episode_log/<arena>``.

Instead of hydra flags, this script takes the path to a checkpoint.pt file
(as saved by scripts/train.py) and reconstructs the training config from
the wandb run recorded alongside it (<run_dir>/wandb/*/files/config.yaml), so
running an eval never requires re-specifying agent/env/network overrides by
hand.
"""

from vla_streaming_rl.script_setup import setup_runtime

setup_runtime()

import argparse
import re
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

from vla_streaming_rl.agents.build import build_all
from vla_streaming_rl.checkpoint import load_checkpoint_weights, load_global_step
from vla_streaming_rl.envs.animalai.curriculum import training_levels
from vla_streaming_rl.script_setup import disable_render_if_headless, resolve_seed, seed_everything
from vla_streaming_rl.testbed import run_testbed
from vla_streaming_rl.wrappers import make_env

EXPONENT_FLOAT_RE = re.compile(r"-?\d+(\.\d*)?[eE][-+]?\d+")


def _coerce_numeric_strings(value):
    """YAML 1.1 leaves exponent floats like '1e-05' as strings; recover them as floats."""
    if isinstance(value, dict):
        return {key: _coerce_numeric_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_numeric_strings(item) for item in value]
    # "01" のようなゼロ埋めのラベルは文字列のまま残す
    if isinstance(value, str) and EXPONENT_FLOAT_RE.fullmatch(value) is not None:
        return float(value)
    return value


def load_wandb_config(run_dir: Path) -> DictConfig:
    """Reconstruct the resolved training config from a wandb run's config.yaml."""
    config_paths = sorted(run_dir.glob("wandb/run-*/files/config.yaml"))
    assert config_paths, f"No wandb config.yaml found under {run_dir}/wandb/"
    raw = yaml.safe_load(config_paths[-1].read_text())
    raw.pop("_wandb", None)
    flat = {key: _coerce_numeric_strings(entry["value"]) for key, entry in raw.items()}
    return OmegaConf.create(flat)


def main(
    args: DictConfig, checkpoint_path: Path, result_dir: Path, seed: int, render: bool
) -> None:
    seed_everything(seed)

    global_step = load_global_step(checkpoint_path.parent)

    env = make_env(args.env_id, args.env_factory, result_dir=None, seed=seed)
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
        args.render_scale,
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
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def load_eval_config(run_dir: Path) -> DictConfig:
    """学習時の config に、評価の設定を重ねたもの。評価で回すアリーナは、学習時ではなく、
    いまの configs/env/animalai.yaml に従う。"""
    cfg = load_wandb_config(run_dir)
    # Always evaluate on the eval sweep, regardless of which mode trained this
    # checkpoint (see SequentialSelector in envs/animalai/curriculum.py).
    cfg.env_factory.mode = "eval"
    env_config = OmegaConf.load(Path(__file__).parents[1] / "configs" / "env" / "animalai.yaml")
    cfg.env_factory.eval_levels = env_config.env_factory.eval_levels
    cfg.env_factory.eval_variants = env_config.env_factory.eval_variants
    return cfg


if __name__ == "__main__":
    cli_args = parse_args()
    checkpoint_path = cli_args.checkpoint.resolve()
    run_dir = checkpoint_path.parent
    cfg = load_eval_config(run_dir)

    seed = resolve_seed(cli_args.seed)
    eval_dir = run_dir / "eval" / checkpoint_path.stem
    render = disable_render_if_headless(not cli_args.no_render)

    print(OmegaConf.to_yaml(cfg))
    main(cfg, checkpoint_path, eval_dir, seed, render)
