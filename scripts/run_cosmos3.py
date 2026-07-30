# SPDX-License-Identifier: MIT
import argparse
import dataclasses
import gzip
import json
from datetime import datetime
from pathlib import Path

import cv2
import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from vla_streaming_rl.cosmos3.policy import CosmosEdgePolicy

AV_DOMAIN_NAME = "av"
AV_ACTION_DIM = 9
AV_LATERAL_IDX, AV_VERTICAL_IDX, AV_FORWARD_IDX = 0, 1, 2
CARLA_TO_AV_FRAME = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
ROAD_OPTION_TO_TEXT = {
    1: "turns left",
    2: "turns right",
    3: "goes straight",
    4: "follows the lane",
}


@dataclasses.dataclass
class EpisodeData:
    annos: list[dict]
    frames: list[Image.Image]
    road_options: list[int]


def load_episode(episode_dir: Path) -> EpisodeData:
    num_frames = len(list((episode_dir / "anno").glob("*.json.gz")))
    annos = [json.load(gzip.open(episode_dir / f"anno/{i:05d}.json.gz")) for i in range(num_frames)]
    frames = [
        Image.open(episode_dir / f"camera/rgb_front/{i:05d}.jpg").convert("RGB")
        for i in range(num_frames)
    ]
    return EpisodeData(
        annos=annos,
        frames=frames,
        road_options=[a["command_near"] for a in annos],
    )


def load_world2cam(anno: dict) -> np.ndarray:
    return np.array(anno["sensors"]["CAM_FRONT"]["world2cam"])


def relative_camera_pose(anno_from: dict, anno_to: dict) -> tuple[np.ndarray, np.ndarray]:
    world2cam_from = load_world2cam(anno_from)
    world2cam_to = load_world2cam(anno_to)
    to_in_from = world2cam_from @ np.linalg.inv(world2cam_to)
    return to_in_from[:3, :3], to_in_from[:3, 3]


def carla_to_av_frame(
    rotation: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return (
        CARLA_TO_AV_FRAME @ rotation @ CARLA_TO_AV_FRAME.T,
        CARLA_TO_AV_FRAME @ translation,
    )


def rotation_6d_from_matrix(rotation: np.ndarray) -> np.ndarray:
    return np.concatenate([rotation[:, 0], rotation[:, 1]])


def rotation_matrix_from_6d(rotation_6d: np.ndarray) -> np.ndarray:
    a1, a2 = rotation_6d[:3], rotation_6d[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2_ortho = a2 - np.dot(b1, a2) * b1
    b2 = a2_ortho / (np.linalg.norm(a2_ortho) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def av_pose_delta(anno_from: dict, anno_to: dict) -> np.ndarray:
    rotation, translation = carla_to_av_frame(*relative_camera_pose(anno_from, anno_to))
    return np.concatenate([translation, rotation_6d_from_matrix(rotation)])


def history_action_from_episode(
    episode: EpisodeData, history_start: int, current: int
) -> torch.Tensor:
    rows = [
        av_pose_delta(episode.annos[i], episode.annos[i + 1]) for i in range(history_start, current)
    ]
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def ground_truth_future_positions(
    episode: EpisodeData, center: int, future_chunk_size: int
) -> np.ndarray:
    anchor = episode.annos[center]
    positions = []
    for j in range(center, center + future_chunk_size + 1):
        _rotation, translation = carla_to_av_frame(*relative_camera_pose(anchor, episode.annos[j]))
        positions.append(translation)
    return np.array(positions)


def integrate_ego_path(action_chunk: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    points = [pose[:3, 3].copy()]
    for row in action_chunk:
        step = np.eye(4)
        step[:3, :3] = rotation_matrix_from_6d(row[3:9])
        step[:3, 3] = row[:3]
        pose = pose @ step
        points.append(pose[:3, 3].copy())
    return np.array(points)


def build_prompt(road_option: int) -> str:
    return f"The ego vehicle {ROAD_OPTION_TO_TEXT[road_option]}, closely following the curve of the road ahead."


@torch.no_grad()
def run_policy(
    policy: CosmosEdgePolicy,
    frames: list[Image.Image],
    prompt: str,
    history_action: torch.Tensor,
    last_history_frame_idx: int,
    num_inference_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    encoding = policy.encode(frames, prompt, history_action)
    full_chunk, vision_latents = policy.sample_action_chunk(encoding, num_inference_steps)
    future_chunk = full_chunk[last_history_frame_idx:].float().cpu().numpy()
    predicted_frame = policy.decode_vision_latents(vision_latents, -1)
    return future_chunk, predicted_frame


def render_topdown_plot(
    predicted_path: np.ndarray,
    ground_truth_path: np.ndarray,
    width_px: int,
    height_px: int,
    lateral_range: tuple[float, float],
    forward_range: tuple[float, float],
) -> np.ndarray:
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax.plot(
        ground_truth_path[:, AV_LATERAL_IDX],
        ground_truth_path[:, AV_FORWARD_IDX],
        "g-o",
        label="ground truth",
    )
    ax.plot(
        predicted_path[:, AV_LATERAL_IDX],
        predicted_path[:, AV_FORWARD_IDX],
        "r-o",
        label="predicted",
    )
    ax.plot([0], [0], "k^", markersize=12)
    ax.set_xlabel("lateral [m] (+ = right)")
    ax.set_ylabel("forward [m]")
    ax.set_title("top-down future path")
    ax.legend()
    ax.set_xlim(lateral_range)
    ax.set_ylim(forward_range)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return buf[:, :, :3].copy()


def compose_panel(
    current_frame_rgb: np.ndarray,
    predicted_frame_rgb: np.ndarray,
    predicted_path: np.ndarray,
    ground_truth_path: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    lateral_range: tuple[float, float],
    forward_range: tuple[float, float],
) -> np.ndarray:
    left_width = canvas_width // 2
    right_width = canvas_width - left_width
    top_height = canvas_height // 2
    bottom_height = canvas_height - top_height

    topdown = render_topdown_plot(
        predicted_path, ground_truth_path, left_width, canvas_height, lateral_range, forward_range
    )
    if topdown.shape[:2] != (canvas_height, left_width):
        topdown = cv2.resize(topdown, (left_width, canvas_height), interpolation=cv2.INTER_AREA)

    current_resized = cv2.resize(
        current_frame_rgb, (right_width, top_height), interpolation=cv2.INTER_AREA
    )
    predicted_resized = cv2.resize(
        predicted_frame_rgb, (right_width, bottom_height), interpolation=cv2.INTER_CUBIC
    )
    right_half = np.concatenate([current_resized, predicted_resized], axis=0)
    return np.concatenate([topdown, right_half], axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=Path, default=(Path.home() / "data/bench2drive"))
    parser.add_argument("--num_cond_latent_frames", type=int, default=2)
    parser.add_argument("--future_chunk_size", type=int, default=8)
    parser.add_argument("--frame_stride", type=int, default=4)
    parser.add_argument("--resolution_tier", type=int, default=480, choices=[256, 480, 704, 720])
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--action_fps", type=float, default=10.0)
    parser.add_argument("--min_sun_altitude", type=float, default=30.0)
    parser.add_argument("--max_fog_density", type=float, default=20.0)
    parser.add_argument("--max_precipitation", type=float, default=20.0)
    parser.add_argument("--canvas_width", type=int, default=1280)
    parser.add_argument("--canvas_height", type=int, default=720)
    parser.add_argument("--video_fps", type=float, default=10.0)
    parser.add_argument("--topdown_lateral_range", type=float, nargs=2, default=(-5.0, 5.0))
    parser.add_argument("--topdown_forward_range", type=float, nargs=2, default=(-1.0, 10.0))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert args.future_chunk_size % 4 == 0, (
        "future_chunk_size must be a multiple of 4 (Cosmos3 VAE temporal factor)"
    )
    last_history_frame_idx = (args.num_cond_latent_frames - 1) * 4
    model_chunk = last_history_frame_idx + args.future_chunk_size
    history_len = last_history_frame_idx + 1

    camera_dirs = sorted(args.root_dir.rglob("camera"))
    episode_dirs = [camera_dir.parent for camera_dir in camera_dirs]
    assert episode_dirs, f"no bench2drive episodes found under {args.root_dir}"
    episode_dir = episode_dirs[0]
    print(f"episode: {episode_dir.name}")

    episode = load_episode(episode_dir)
    num_frames = len(episode.frames)
    first_center = history_len - 1
    last_center = num_frames - 1 - args.future_chunk_size
    assert first_center <= last_center, (
        f"episode has only {num_frames} frames; need at least "
        f"{history_len + args.future_chunk_size} for one panel"
    )
    print(
        f"episode has {num_frames} frames; rendering panels for frames {first_center}..{last_center}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = CosmosEdgePolicy.from_pretrained(
        "nvidia/Cosmos3-Edge", torch_dtype=torch.bfloat16, enable_safety_checker=False
    ).to(device)
    policy.vae.to(torch.float32)
    policy.setup(
        chunk_size=model_chunk,
        domain_name=AV_DOMAIN_NAME,
        resolution_tier=args.resolution_tier,
        num_inference_steps=args.num_inference_steps,
        num_cond_latent_frames=args.num_cond_latent_frames,
        fps=args.action_fps,
    )

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).resolve().parent / "results" / f"{datetime_str}_COSMOS3_CLIP_PANELS"
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    panels_rgb = []
    for center in range(first_center, last_center + 1, args.frame_stride):
        history_start = center - history_len + 1
        clip_frames = episode.frames[history_start : center + 1]
        history_action = history_action_from_episode(episode, history_start, center).to(device)
        prompt = build_prompt(episode.road_options[center])

        action_chunk, predicted_frame = run_policy(
            policy,
            clip_frames,
            prompt,
            history_action,
            last_history_frame_idx,
            args.num_inference_steps,
        )
        torch.cuda.empty_cache()

        predicted_path = integrate_ego_path(action_chunk)
        ground_truth_path = ground_truth_future_positions(episode, center, args.future_chunk_size)

        current_frame_rgb = np.array(clip_frames[-1])
        panel_rgb = compose_panel(
            current_frame_rgb,
            predicted_frame,
            predicted_path,
            ground_truth_path,
            args.canvas_width,
            args.canvas_height,
            tuple(args.topdown_lateral_range),
            tuple(args.topdown_forward_range),
        )
        Image.fromarray(panel_rgb).save(panel_dir / f"panel_{center:05d}.png")
        panels_rgb.append(panel_rgb)
        print(f"frame {center}/{last_center} done, prompt={prompt!r}")

    video_path = output_dir / "rollout.mp4"
    imageio.mimsave(str(video_path), panels_rgb, fps=args.video_fps, macro_block_size=1)
    print(f"\nsaved {len(panels_rgb)} panels and video to {output_dir}")


if __name__ == "__main__":
    main()
