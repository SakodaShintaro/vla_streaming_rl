# SPDX-License-Identifier: MIT
"""Collect episodes with a frozen policy for the language-model check of RTRL.

Loads a trained checkpoint (weights only) and its training config from the run's
wandb record, then plays the life mode with no optimizer step until ``--games``
games have ended. Every tick the agent reads is kept, including each episode's
reset frame and terminal frame, in the order the agent read them, so the stream
can be read again across episode boundaries. Per episode, ``ep_XXXXXX.npz`` holds
what the encoder read on each tick: the images (uint8, lossless), the scalar
observation, the normalized previous action and the shaped reward. The game the
episode belongs to and its arena, score and success go to ``episodes.tsv``, and
the instruction to ``instruction.txt``.
"""

from vla_streaming_rl.script_setup import setup_runtime

setup_runtime()

import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from test_trained_agent import load_wandb_config

from vla_streaming_rl.agents.build import build_all
from vla_streaming_rl.checkpoint import load_checkpoint_weights, load_global_step
from vla_streaming_rl.scalar_obs import read_scalar_obs
from vla_streaming_rl.script_setup import seed_everything
from vla_streaming_rl.wrappers import make_env


def to_uint8(image: np.ndarray) -> np.ndarray:
    """The env's uint8 image that the wrapper divided by 255, back as uint8."""
    restored = np.rint(image * 255.0)
    assert np.allclose(restored, image * 255.0, atol=1e-3), "image is not a scaled uint8"
    return restored.astype(np.uint8)


def main(checkpoint_path: Path, games: int, seed: int) -> None:
    run_dir = checkpoint_path.parent
    cfg = load_wandb_config(run_dir)
    # 試行のあとに足した vla の項目は、いまの既定値で埋める
    common = OmegaConf.load(Path(__file__).parents[1] / "configs" / "agent" / "common.yaml")
    cfg.vla = OmegaConf.merge(common.vla, cfg.vla)
    assert cfg.env_factory.mode == "life", cfg.env_factory.mode
    out_dir = run_dir / "lm_data" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=False)

    seed_everything(seed)
    global_step = load_global_step(run_dir)
    env = make_env(cfg.env_id, cfg.env_factory, result_dir=None, seed=seed)
    env.action_space.seed(seed)
    # The network reads the global step as an observation
    env.unwrapped.set_global_step(global_step)

    network, agent = build_all(env, cfg)
    load_checkpoint_weights(checkpoint_path, network)
    network.eval()
    (out_dir / "instruction.txt").write_text(agent.instruction)

    table = open(out_dir / "episodes.tsv", "w")
    table.write("episode\tgame\tarena\tticks\tscore\tsuccess\n")
    episode = 0
    games_ended = 0
    while games_ended < games:
        obs, info = env.reset(seed=seed) if episode == 0 else env.reset()
        reward, terminated, truncated = 0.0, False, False
        ticks: dict[str, list] = {
            **{name: [] for name in agent.image_names},
            "scalar_obs": [],
            "prev_action": [],
            "reward": [],
        }
        while True:
            # What the encoder reads on this tick (see Agent._observe)
            for name in agent.image_names:
                ticks[name].append(to_uint8(obs[name]))
            ticks["scalar_obs"].append(read_scalar_obs(obs))
            ticks["prev_action"].append(
                ((agent.prev_action - agent.action_bias) / agent.action_scale).astype(np.float32)
            )
            ticks["reward"].append(np.float32(info["shaped_reward"]))
            result = agent.select_action(global_step, obs, reward, terminated, truncated, info)
            if terminated or truncated:
                break
            obs, reward, terminated, truncated, info = env.step(result.action)

        np.savez(out_dir / f"ep_{episode:06d}.npz", **{k: np.stack(v) for k, v in ticks.items()})
        score = info["episode"]["r"]
        success = int(score >= info["pass_mark"])
        table.write(
            f"{episode}\t{games_ended}\t{info['arena_name']}\t{len(ticks['reward'])}"
            f"\t{score:.6f}\t{success}\n"
        )
        table.flush()
        print(
            f"episode {episode}  game {games_ended}  {info['arena_name']}"
            f"  ticks={len(ticks['reward'])}  score={score:.2f}  success={success}",
            flush=True,
        )
        games_ended = info["games_ended"]
        episode += 1

    table.close()
    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="checkpoint.pt saved by scripts/train.py")
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args.checkpoint.resolve(), cli_args.games, cli_args.seed)
