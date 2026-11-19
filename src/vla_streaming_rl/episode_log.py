# SPDX-License-Identifier: MIT
"""1エピソード分の記録（描画、観測、行動、報酬、位置、テキスト）と、その保存。
学習と評価の両方で、同じ形のログを残すために使う。"""

from dataclasses import dataclass
from pathlib import Path

import imageio
import numpy as np

from vla_streaming_rl.scalar_obs import SCALAR_OBS_NAMES, read_scalar_obs


@dataclass
class EpisodeRecord:
    """エピソードの中でエージェントが行動を決めた観測ごとに、並びの順に1つずつ積む。最初はリセット
    直後の観測で、そのあとはステップごとに返った観測。どのリストも同じ位置が同じ観測の分で、
    リプレイバッファの1行と同じく、その観測で行動を決めるときにエージェントが読んだもの（観測、
    前の行動、観測と一緒に返った報酬）と、そのときの位置を持つ。"""

    # 描画したフレーム（RGB）
    frames: list[np.ndarray]
    # その観測が返る前に実行した行動（エージェントが前の行動として読むもの）。リセット直後の分は
    # 前のエピソードの最後の行動
    prev_actions: list[np.ndarray]
    # その観測と一緒に返った環境の報酬と、エージェントが読む整形した報酬。リセット直後の環境の
    # 報酬は 0
    rewards: list[float]
    shaped_rewards: list[float]
    observations: list[np.ndarray]
    texts: list[dict[str, str]]
    xyzs: list[tuple[float, float, float]]
    # 位置と一緒に記録する、エージェントの向き（yaw、度）
    yaws: list[float]
    # エージェントが画像と並べて読むスカラーの観測（SCALAR_OBS_NAMES の並び）
    scalar_obs: list[np.ndarray]
    # 位置と向きを記録するか。Animal-AI はアリーナの中の位置と向きを毎ステップ出すが、出さない
    # 環境では xyzs と yaws は空のまま
    log_position: bool

    @classmethod
    def start(
        cls,
        frame: np.ndarray,
        obs: dict,
        texts: dict[str, str],
        reset_info: dict,
        prev_action: np.ndarray,
    ) -> "EpisodeRecord":
        """リセット直後の描画と観測から、エピソードの記録を始める。``prev_action`` はリセット直後の
        観測でエージェントが前の行動として読んだもの（前のエピソードの最後の行動）。"""
        log_position = "agent_xyz" in reset_info
        return cls(
            frames=[frame],
            prev_actions=[prev_action.copy()],
            rewards=[0.0],
            shaped_rewards=[reset_info["shaped_reward"]],
            observations=[obs["image"].copy()],
            texts=[texts],
            xyzs=[reset_info["agent_xyz"]] if log_position else [],
            yaws=[reset_info["agent_rotation"][1]] if log_position else [],
            scalar_obs=[read_scalar_obs(obs)],
            log_position=log_position,
        )

    def add_step(
        self,
        action: np.ndarray,
        reward: float,
        obs: dict,
        info: dict,
        frame: np.ndarray,
        texts: dict[str, str],
    ) -> None:
        """1ステップ分を足す。``action`` を取って ``reward`` と ``obs`` が返り、その観測に
        エージェントが応えたあとの描画が ``frame``。"""
        self.prev_actions.append(action.copy())
        self.rewards.append(reward)
        self.shaped_rewards.append(info["shaped_reward"])
        self.observations.append(obs["image"].copy())
        if self.log_position:
            self.xyzs.append(info["agent_xyz"])
            # agent_rotation は (pitch, yaw, roll)
            self.yaws.append(info["agent_rotation"][1])
        self.scalar_obs.append(read_scalar_obs(obs))
        self.frames.append(frame)
        self.texts.append(texts)


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
    if not record.frames:
        return

    # Stable-panel contract: an agent emits the same set of equally-shaped
    # panels every step, so every render frame has the same size. Fail loudly
    # if that is violated rather than letting the video encoder error out.
    frame_sizes = {img.shape for img in record.frames}
    assert len(frame_sizes) == 1, (
        f"Episode '{name}' produced frames of differing sizes {frame_sizes}; "
        "an agent's panel set / shapes must stay constant across the run."
    )

    # libx264 (yuv420p) requires even width/height; the concatenated panel strip
    # can be an odd size. Pad one black row/column when needed.
    frame_list = record.frames
    h, w = frame_list[0].shape[:2]
    if h % 2 or w % 2:
        frame_list = [
            np.pad(img, ((0, h % 2), (0, w % 2), (0, 0)), mode="constant") for img in frame_list
        ]

    episode_log_dir = episode_dir / f"{name}"
    episode_log_dir.mkdir(parents=True, exist_ok=True)

    video_path = episode_log_dir / "render.mp4"
    imageio.mimsave(str(video_path), frame_list, fps=10, macro_block_size=1)

    obs_rgb_images = [
        (obs.transpose(1, 2, 0) * 255).astype(np.uint8) for obs in record.observations
    ]
    obs_sizes = {img.shape for img in obs_rgb_images}
    assert len(obs_sizes) == 1, (
        f"Episode '{name}' produced observations of differing sizes {obs_sizes}"
    )
    obs_video_path = episode_log_dir / "obs.mp4"
    imageio.mimsave(
        str(obs_video_path),
        obs_rgb_images,
        fps=10,
        macro_block_size=1,
        codec="libx264rgb",
        pixelformat="rgb24",
        output_params=["-crf", "0"],
    )

    save_episode_texts(episode_log_dir, record.texts)

    columns = [f"prev_action{i}" for i in range(len(record.prev_actions[0]))] + [
        "reward",
        "shaped_reward",
    ]
    rows = [
        [f"{float(a):.6f}" for a in prev_action] + [f"{float(reward):.6f}", f"{float(shaped):.6f}"]
        for prev_action, reward, shaped in zip(
            record.prev_actions, record.rewards, record.shaped_rewards, strict=True
        )
    ]
    if record.xyzs:
        columns = columns + ["x", "y", "z", "yaw"]
        rows = [
            row + [f"{float(v):.4f}" for v in xyz] + [f"{float(yaw):.2f}"]
            for row, xyz, yaw in zip(rows, record.xyzs, record.yaws, strict=True)
        ]
    # 観測のスカラー
    columns = columns + list(SCALAR_OBS_NAMES)
    rows = [
        row + [f"{float(v):.6f}" for v in scalar_obs]
        for row, scalar_obs in zip(rows, record.scalar_obs, strict=True)
    ]

    with open(episode_log_dir / "log.tsv", "w", encoding="utf-8") as f:
        f.write("step\t" + "\t".join(columns) + "\n")
        for step, row in enumerate(rows):
            f.write(f"{step}\t" + "\t".join(row) + "\n")
