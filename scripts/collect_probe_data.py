# SPDX-License-Identifier: MIT
"""Collect (representation, agent position) pairs for an offline linear probe.

Runs a frozen, trained policy through the same Animal-AI arena (``ARENA_STEM``)
``NUM_REPEATS`` times, recording at every step the representation the network
hands to its policy/value/prediction heads (``last_features``)
together with the agent's true arena position (``info["agent_xyz"]``). Also
renders the same panel video train.py writes, for visual sanity-checking.
Saves everything to ``probe_data.npz``; run scripts/visualize_linear_probe.py
on that file to fit the probe and render it.

``ARENA_STEM`` names a competition arena, so run this with
``env=animalai env_factory.mode=eval`` (the env resolves a pinned arena
against the arenas its selector serves).
"""

from vla_streaming_rl.script_setup import setup_runtime

setup_runtime()

import os
from pathlib import Path

import hydra
import imageio
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from vla_streaming_rl.agents.build import build_all
from vla_streaming_rl.checkpoint import load_checkpoint_weights
from vla_streaming_rl.script_setup import resolve_seed, seed_everything
from vla_streaming_rl.utils import render_frame
from vla_streaming_rl.wrappers import make_env

# TODO: swap for a hand-authored arena yaml once one exists.
ARENA_STEM = "01-30-03"
NUM_REPEATS = 10


def collect_arena(
    agent,
    env,
    seed: int,
    arena_stem: str,
    episode_id: int,
    feature_list: list,
    xyz_list: list,
    arena_list: list,
    episode_list: list,
    video_path: Path,
) -> None:
    obs, reset_info = env.reset(seed=seed, options={"arena_stem": arena_stem})
    result = agent.select_action(0, obs, 0.0, False, False, reset_info)
    feature_list.append(agent.last_features.squeeze(0).cpu().numpy())
    xyz_list.append(reset_info["agent_xyz"])
    arena_list.append(arena_stem)
    episode_list.append(episode_id)
    action = result.action

    frame_list = [render_frame(env, obs, result, 1.0)]

    while True:
        obs, reward, terminated, truncated, env_info = env.step(action)
        result = agent.select_action(0, obs, reward, terminated, truncated, env_info)
        feature_list.append(agent.last_features.squeeze(0).cpu().numpy())
        xyz_list.append(env_info["agent_xyz"])
        arena_list.append(arena_stem)
        episode_list.append(episode_id)
        action = result.action

        frame_list.append(render_frame(env, obs, result, 1.0))

        if terminated or truncated:
            break

    imageio.mimsave(str(video_path), frame_list, fps=10, macro_block_size=1)


def main(args: DictConfig, result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    video_dir = result_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    seed = resolve_seed(args.seed)
    seed_everything(seed)

    env = make_env(args.env_id, args.env_factory, result_dir=None)
    env.action_space.seed(seed)

    network, agent = build_all(env, args)
    load_checkpoint_weights(Path(args.resume_dir) / "checkpoint.pt", network)
    network.eval()

    print(f"Collecting representations over {NUM_REPEATS} repeats of arena {ARENA_STEM}.")

    feature_list: list = []
    xyz_list: list = []
    arena_list: list = []
    episode_list: list = []
    for i in range(NUM_REPEATS):
        video_path = video_dir / f"ep_{i + 1:04d}_{ARENA_STEM}.mp4"
        collect_arena(
            agent,
            env,
            seed + i,
            ARENA_STEM,
            i,
            feature_list,
            xyz_list,
            arena_list,
            episode_list,
            video_path,
        )
        print(f"[{i + 1}/{NUM_REPEATS}] {ARENA_STEM}\tsamples so far={len(feature_list)}")

    env.close()

    features = np.stack(feature_list).astype(np.float32)
    targets = np.array(xyz_list, dtype=np.float32)
    arenas = np.array(arena_list, dtype="U16")
    episodes = np.array(episode_list, dtype=np.int64)

    data_path = result_dir / "probe_data.npz"
    np.savez(data_path, features=features, xyz=targets, arena=arenas, episode=episodes)
    print(f"Saved {features.shape[0]} samples (feature dim {features.shape[1]}) to {data_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def hydra_main(cfg: DictConfig) -> None:
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    os.chdir(hydra.utils.get_original_cwd())

    assert cfg.resume_dir is not None, (
        "collect_probe_data.py requires resume_dir to point at a trained checkpoint directory."
    )

    print(OmegaConf.to_yaml(cfg))
    main(cfg, hydra_output_dir)


if __name__ == "__main__":
    hydra_main()
