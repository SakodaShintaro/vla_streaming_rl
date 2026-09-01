# SPDX-License-Identifier: MIT
"""Collect (representation, agent position) pairs for an offline linear probe.

Runs a frozen, trained policy through the same Animal-AI arena (``ARENA_STEM``)
``NUM_REPEATS`` times, recording at every step the representation the network
hands to its policy/value/prediction heads (``last_features``)
together with the agent's true arena position (``info["agent_xyz"]``). Also
renders the same panel video train.py writes, for visual sanity-checking.
Saves everything to ``probe_data.npz``; run scripts/compute_linear_probe.py
on that file to fit and evaluate the probe.

``ARENA_STEM`` names a competition arena, so run this with
``env=animalai env_factory.mode=eval`` (the env resolves a pinned arena
against the arenas its selector serves).
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
import imageio
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from vla_streaming_rl.agents.build import build_agent
from vla_streaming_rl.networks.build import build_network
from vla_streaming_rl.utils import concat_labeled_images, overlay_caption
from vla_streaming_rl.wrappers import make_env

torch.set_float32_matmul_precision("high")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# TODO: swap for a hand-authored arena yaml once one exists.
ARENA_STEM = "01-30-03"
NUM_REPEATS = 10


def load_checkpoint_weights(checkpoint_dir: Path, network: torch.nn.Module) -> None:
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    trainable_state = torch.load(checkpoint_path, map_location="cuda")
    module = network._orig_mod if hasattr(network, "_orig_mod") else network
    missing, unexpected = module.load_state_dict(trainable_state, strict=False)
    if unexpected:
        raise ValueError(f"checkpoint parameters not found in the network: {unexpected[:5]}")
    print(f"Loaded {len(trainable_state)} weight tensors from {checkpoint_path}")


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

    obs_viz = (obs["image"].copy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    panels = {
        "environment": overlay_caption(
            env.render(), f"{result.texts['prompt']}  reward: {0.0:+.3f}"
        ),
        "observation": obs_viz,
    }
    frame_list = [concat_labeled_images(panels)]

    while True:
        obs, reward, terminated, truncated, env_info = env.step(action)
        result = agent.select_action(0, obs, reward, terminated, truncated, env_info)
        feature_list.append(agent.last_features.squeeze(0).cpu().numpy())
        xyz_list.append(env_info["agent_xyz"])
        arena_list.append(arena_stem)
        episode_list.append(episode_id)
        action = result.action

        obs_viz = (obs["image"].copy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        panels = {
            "environment": overlay_caption(
                env.render(), f"{result.texts['prompt']}  reward: {reward:+.3f}"
            ),
            "observation": obs_viz,
        }
        frame_list.append(concat_labeled_images(panels))

        if terminated or truncated:
            break

    imageio.mimsave(str(video_path), frame_list, fps=10, macro_block_size=1)


def main(args: DictConfig, result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    video_dir = result_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

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

    if not os.environ.get("DISPLAY"):
        print("Because a headless environment is detected, rendering is automatically disabled.")
        cfg.render = 0

    if cfg.resume_dir is None:
        raise ValueError(
            "collect_probe_data.py requires resume_dir to point at a trained checkpoint directory."
        )

    print(OmegaConf.to_yaml(cfg))
    main(cfg, hydra_output_dir)


if __name__ == "__main__":
    hydra_main()
