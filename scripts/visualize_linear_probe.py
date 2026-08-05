# SPDX-License-Identifier: MIT
"""Animate the linear probe's predicted vs. true agent position over time.

Reads a ``probe_data.npz`` produced by scripts/collect_probe_data.py
(features, xyz, arena, episode arrays), fits the same closed-form ridge
linear probe as scripts/compute_linear_probe.py (split by episode: 9 train /
1 valid out of the default 10 collected episodes), and renders a single
video (one train episode, then one valid episode) with the actual gameplay
footage (from collect_probe_data.py's saved per-episode video) on the left
and the probe's predicted vs. true position on the right: a top-down (x, z)
trail (matching AnimalAIEnv's top-down render orientation) plus a
y-over-time line plot. The gameplay footage is what makes the plot
interpretable — the raw (x, z) trail alone doesn't show what's happening in
the arena.
"""

import argparse
from pathlib import Path

import cv2
import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRAIN_RATIO = 0.9
RIDGE_LAMBDA = 1.0
ARENA_SIZE_M = 40.0
TRAIL_LENGTH = 30
PANEL_HEIGHT_PX = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("seed", type=int)
    return parser.parse_args()


def split_episodes(episode_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique_episodes = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique_episodes)
    n_train = max(1, int(round(len(perm) * TRAIN_RATIO)))
    return perm[:n_train], perm[n_train:]


def fit_linear_probe(features: np.ndarray, targets: np.ndarray, train_mask: np.ndarray) -> tuple:
    mean = features[train_mask].mean(axis=0)
    std = features[train_mask].std(axis=0) + 1e-6
    x_train = (features[train_mask] - mean) / std
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=np.float32)], axis=1)

    gram = x_train.T @ x_train + RIDGE_LAMBDA * np.eye(x_train.shape[1], dtype=np.float32)
    weight = np.linalg.solve(gram, x_train.T @ targets[train_mask])
    return weight, mean, std


def predict(features: np.ndarray, weight: np.ndarray, mean: np.ndarray, std: np.ndarray):
    x = (features - mean) / std
    x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    return x @ weight


def read_video_frames(video_path: Path) -> list:
    reader = imageio.get_reader(str(video_path))
    frame_list = [frame for frame in reader]
    reader.close()
    return frame_list


def render_probe_panel(
    true_xyz: np.ndarray, pred_xyz: np.ndarray, split_name: str, episode_id: int, t: int
) -> np.ndarray:
    n = true_xyz.shape[0]
    lo = max(0, t - TRAIL_LENGTH)
    fig, (ax_top, ax_y) = plt.subplots(1, 2, figsize=(10, 5))

    ax_top.set_title(f"{split_name} ep {episode_id}  step {t}/{n - 1}  top-down (x, z)")
    ax_top.set_xlim(0.0, ARENA_SIZE_M)
    ax_top.set_ylim(0.0, ARENA_SIZE_M)
    ax_top.set_aspect("equal")
    ax_top.plot(true_xyz[lo : t + 1, 0], true_xyz[lo : t + 1, 2], color="tab:blue", alpha=0.5)
    ax_top.plot(pred_xyz[lo : t + 1, 0], pred_xyz[lo : t + 1, 2], color="tab:red", alpha=0.5)
    ax_top.scatter(*true_xyz[t, [0, 2]], color="tab:blue", s=80, label="true")
    ax_top.scatter(*pred_xyz[t, [0, 2]], color="tab:red", s=80, label="predicted")
    ax_top.legend(loc="upper right")

    ax_y.set_title("height (y) over time")
    ax_y.set_xlim(0, n - 1)
    ax_y.plot(true_xyz[: t + 1, 1], color="tab:blue", label="true")
    ax_y.plot(pred_xyz[: t + 1, 1], color="tab:red", label="predicted")
    ax_y.axvline(t, color="gray", linestyle="--", alpha=0.5)
    ax_y.legend(loc="upper right")

    fig.tight_layout()
    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return panel


def render_episode_frames(
    true_xyz: np.ndarray,
    pred_xyz: np.ndarray,
    gameplay_frames: list,
    split_name: str,
    episode_id: int,
) -> list:
    n = true_xyz.shape[0]
    frame_list = []
    for t in range(n):
        probe_panel = render_probe_panel(true_xyz, pred_xyz, split_name, episode_id, t)
        h, w = gameplay_frames[t].shape[:2]
        new_w = int(round(w * PANEL_HEIGHT_PX / h))
        gameplay_panel = cv2.resize(gameplay_frames[t], (new_w, PANEL_HEIGHT_PX))
        frame_list.append(np.hstack([gameplay_panel, probe_panel]))
    return frame_list


if __name__ == "__main__":
    args = parse_args()
    data_path = Path(args.data_path)
    data = np.load(data_path)
    features = data["features"]
    targets = data["xyz"]
    episodes = data["episode"]
    arenas = data["arena"]

    train_episodes, valid_episodes = split_episodes(episodes, args.seed)
    train_mask = np.isin(episodes, train_episodes)

    weight, mean, std = fit_linear_probe(features, targets, train_mask)
    predicted = predict(features, weight, mean, std)

    train_episode_id = int(np.sort(train_episodes)[0])
    valid_episode_id = int(np.sort(valid_episodes)[0])

    frame_list = []
    for split_name, episode_id in [("train", train_episode_id), ("valid", valid_episode_id)]:
        mask = episodes == episode_id
        arena_name = arenas[mask][0]
        video_in_path = data_path.parent / "video" / f"ep_{episode_id + 1:04d}_{arena_name}.mp4"
        gameplay_frames = read_video_frames(video_in_path)
        frame_list += render_episode_frames(
            targets[mask], predicted[mask], gameplay_frames, split_name, episode_id
        )

    video_dir = data_path.parent / "probe_viz"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"probe_train{train_episode_id}_valid{valid_episode_id}.mp4"
    imageio.mimsave(str(video_path), frame_list, fps=10, macro_block_size=1)
    print(f"Saved {video_path}")
