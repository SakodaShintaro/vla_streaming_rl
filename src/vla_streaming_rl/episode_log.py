# SPDX-License-Identifier: MIT
"""1エピソード分の記録（描画、観測、行動、報酬、位置、テキスト）と、その保存。
学習と評価の両方で、同じ形のログを残すために使う。"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio
import numpy as np


@dataclass
class EpisodeRecord:
    """Everything one episode leaves behind for ``save_episode_data``."""

    bgr_images: list[np.ndarray]
    actions: list[np.ndarray]
    rewards: list[float]
    observations: list[np.ndarray]
    texts: list[dict[str, str]]
    xyzs: list[tuple[float, float, float]]

    @classmethod
    def fresh(cls) -> "EpisodeRecord":
        return cls(bgr_images=[], actions=[], rewards=[], observations=[], texts=[], xyzs=[])


def save_episode_texts(episode_log_dir: Path, text_list: list[dict[str, str]]) -> None:
    """The free-form text agents emitted, one row per rendered frame.

    Text a network draws into a panel is legible in the video but not
    searchable; this is the same content as characters, so a run's chain of
    thought can be read back and grepped. Tabs and newlines are escaped to keep
    one step on one line.
    """
    keys = list(text_list[0].keys())
    if not keys:
        return

    def escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")

    tsv_path = episode_log_dir / "texts.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("step\t" + "\t".join(keys) + "\n")
        for step, texts in enumerate(text_list):
            f.write(f"{step}\t" + "\t".join(escape(texts[key]) for key in keys) + "\n")


def save_episode_data(episode_dir: Path, name: str, record: EpisodeRecord) -> None:
    """Save episode videos, actions, rewards and positions"""
    if not record.bgr_images:
        return

    # Stable-panel contract: an agent emits the same set of equally-shaped
    # panels every step, so every render frame has the same size. Fail loudly
    # if that is violated rather than letting the video encoder error out.
    frame_sizes = {img.shape for img in record.bgr_images}
    assert len(frame_sizes) == 1, (
        f"Episode '{name}' produced frames of differing sizes {frame_sizes}; "
        "an agent's panel set / shapes must stay constant across the run."
    )

    # libx264 (yuv420p) requires even width/height; the concatenated panel strip
    # can be an odd size. Pad one black row/column when needed.
    bgr_image_list = record.bgr_images
    h, w = bgr_image_list[0].shape[:2]
    if h % 2 or w % 2:
        bgr_image_list = [
            np.pad(img, ((0, h % 2), (0, w % 2), (0, 0)), mode="constant") for img in bgr_image_list
        ]

    episode_log_dir = episode_dir / f"{name}"
    episode_log_dir.mkdir(parents=True, exist_ok=True)

    video_path = episode_log_dir / "render.mp4"
    rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in bgr_image_list]
    imageio.mimsave(str(video_path), rgb_images, fps=10, macro_block_size=1)

    obs_rgb_images = [
        (obs.transpose(1, 2, 0) * 255).astype(np.uint8) for obs in record.observations
    ]
    obs_sizes = {img.shape for img in obs_rgb_images}
    assert len(obs_sizes) == 1, (
        f"Episode '{name}' produced observations of differing sizes {obs_sizes}"
    )
    obs_h, obs_w = obs_rgb_images[0].shape[:2]
    if obs_h % 2 or obs_w % 2:
        obs_rgb_images = [
            np.pad(img, ((0, obs_h % 2), (0, obs_w % 2), (0, 0)), mode="constant")
            for img in obs_rgb_images
        ]
    obs_video_path = episode_log_dir / "obs.mp4"
    imageio.mimsave(str(obs_video_path), obs_rgb_images, fps=10, macro_block_size=1)

    save_episode_texts(episode_log_dir, record.texts)

    # One row per step: the action taken, the reward it drew, and where the
    # agent stood after it, so an episode's trajectory can be drawn from the
    # run rather than re-simulated. ``xyzs`` is empty for an env that reports
    # no position, and the x/y/z columns are then left out rather than filled
    # with a stand-in.
    columns = [f"action{i}" for i in range(len(record.actions[0]))] + ["reward"]
    rows = [
        [f"{float(a):.6f}" for a in action] + [f"{float(reward):.6f}"]
        for action, reward in zip(record.actions, record.rewards)
    ]
    if record.xyzs:
        columns = columns + ["x", "y", "z"]
        rows = [row + [f"{float(v):.4f}" for v in xyz] for row, xyz in zip(rows, record.xyzs)]

    with open(episode_log_dir / "log.tsv", "w", encoding="utf-8") as f:
        f.write("step\t" + "\t".join(columns) + "\n")
        for step, row in enumerate(rows):
            f.write(f"{step}\t" + "\t".join(row) + "\n")
