# SPDX-License-Identifier: MIT
import argparse
import dataclasses
import gzip
import json
import time
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
AV_LATERAL_IDX = 0
AV_VERTICAL_IDX = 1
AV_FORWARD_IDX = 2
CARLA_TO_AV_FRAME = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
ROAD_OPTION_TO_TEXT = {
    1: "turns left",
    2: "turns right",
    3: "goes straight",
    4: "follows the lane",
}


@dataclasses.dataclass
class EpisodeData:
    world2cams: list[np.ndarray]
    frames: list[Image.Image]
    road_options: list[int]
    intrinsic: np.ndarray
    camera_height: float


def load_episode(episode_dir: Path) -> EpisodeData:
    num_frames = len(list((episode_dir / "anno").glob("*.json.gz")))
    annos = [json.load(gzip.open(episode_dir / f"anno/{i:05d}.json.gz")) for i in range(num_frames)]
    frames = [
        Image.open(episode_dir / f"camera/rgb_front/{i:05d}.jpg").convert("RGB")
        for i in range(num_frames)
    ]
    return EpisodeData(
        world2cams=[np.array(a["sensors"]["CAM_FRONT"]["world2cam"]) for a in annos],
        frames=frames,
        road_options=[a["command_near"] for a in annos],
        intrinsic=np.array(annos[0]["sensors"]["CAM_FRONT"]["intrinsic"]),
        camera_height=float(annos[0]["sensors"]["CAM_FRONT"]["cam2ego"][2][3]),
    )


def relative_camera_pose(
    world2cam_from: np.ndarray, world2cam_to: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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


def av_pose_delta(world2cam_from: np.ndarray, world2cam_to: np.ndarray) -> np.ndarray:
    rotation, translation = carla_to_av_frame(*relative_camera_pose(world2cam_from, world2cam_to))
    return np.concatenate([translation, rotation_6d_from_matrix(rotation)])


def history_action_from_episode(
    episode: EpisodeData, history_start: int, current: int
) -> torch.Tensor:
    rows = [
        av_pose_delta(episode.world2cams[i], episode.world2cams[i + 1])
        for i in range(history_start, current)
    ]
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def ground_truth_future_positions(
    episode: EpisodeData, center: int, future_chunk_size: int
) -> np.ndarray:
    anchor = episode.world2cams[center]
    positions = []
    for j in range(center, center + future_chunk_size + 1):
        _rotation, translation = carla_to_av_frame(
            *relative_camera_pose(anchor, episode.world2cams[j])
        )
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
    future_chunk_size: int,
    num_inference_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    encode_start = time.perf_counter()
    encoding = policy.encode(frames, prompt, history_action)
    sample_start = time.perf_counter()
    full_chunk, vision_latents = policy.sample_action_chunk(encoding, num_inference_steps)
    future_chunk = full_chunk[last_history_frame_idx:].float().cpu().numpy()
    decode_start = time.perf_counter()
    predicted_frames = policy.decode_vision_latents(vision_latents, slice(-future_chunk_size, None))
    decode_end = time.perf_counter()
    print(
        f"encode: {int((sample_start - encode_start) * 1000)}msec, "
        f"sample_action_chunk: {int((decode_start - sample_start) * 1000)}msec, "
        f"decode_vision_latents: {int((decode_end - decode_start) * 1000)}msec"
    )
    return future_chunk, predicted_frames


def project_path_to_image(
    path: np.ndarray, intrinsic: np.ndarray, camera_height: float
) -> np.ndarray:
    ground_path = path.copy()
    ground_path[:, AV_VERTICAL_IDX] += camera_height / 4
    in_front = ground_path[:, AV_FORWARD_IDX] > 0.1
    camera_points = ground_path[in_front]
    projected = camera_points @ intrinsic.T
    return projected[:, :2] / projected[:, 2:3]


def draw_path(
    image_rgb: np.ndarray,
    path: np.ndarray,
    intrinsic: np.ndarray,
    camera_height: float,
    color: tuple[int, int, int],
) -> None:
    pixels = project_path_to_image(path, intrinsic, camera_height).astype(int)
    for p1, p2 in zip(pixels[:-1], pixels[1:]):
        cv2.line(image_rgb, tuple(p1), tuple(p2), color, 2)
    for p in pixels:
        cv2.circle(image_rgb, tuple(p), 4, color, -1)


def draw_legend(
    image_rgb: np.ndarray,
    ground_truth_color: tuple[int, int, int],
    predicted_color: tuple[int, int, int],
) -> None:
    cv2.putText(
        image_rgb,
        "ground truth",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        ground_truth_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_rgb,
        "predicted",
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        predicted_color,
        2,
        cv2.LINE_AA,
    )


def label_thumbnail(thumbnail_rgb: np.ndarray, text: str) -> None:
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    x = thumbnail_rgb.shape[1] - text_size[0] - 4
    cv2.putText(
        thumbnail_rgb, text, (x, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
    )


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
    predicted_frames_rgb: np.ndarray,
    predicted_path: np.ndarray,
    ground_truth_path: np.ndarray,
    intrinsic: np.ndarray,
    camera_height: float,
    lateral_range: tuple[float, float],
    forward_range: tuple[float, float],
) -> np.ndarray:
    ground_truth_color = (0, 255, 0)
    predicted_color = (255, 0, 0)
    overlaid = current_frame_rgb.copy()
    draw_path(overlaid, ground_truth_path, intrinsic, camera_height, ground_truth_color)
    draw_path(overlaid, predicted_path, intrinsic, camera_height, predicted_color)
    draw_legend(overlaid, ground_truth_color, predicted_color)

    height, width = overlaid.shape[:2]
    thumb_width = width // 4
    thumb_height = height // len(predicted_frames_rgb)
    thumbnails = []
    for step, frame in enumerate(predicted_frames_rgb, start=1):
        thumbnail = cv2.resize(frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        label_thumbnail(thumbnail, f"t+{step}")
        thumbnails.append(thumbnail)

    topdown_width = width // 2
    topdown = render_topdown_plot(
        predicted_path, ground_truth_path, topdown_width, height, lateral_range, forward_range
    )
    if topdown.shape[:2] != (height, topdown_width):
        topdown = cv2.resize(topdown, (topdown_width, height), interpolation=cv2.INTER_AREA)

    return np.concatenate([topdown, overlaid, np.concatenate(thumbnails, axis=0)], axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=Path, default=(Path.home() / "data/bench2drive"))
    parser.add_argument("--num_cond_latent_frames", type=int, default=2)
    parser.add_argument("--future_chunk_size", type=int, default=4)
    parser.add_argument("--frame_stride", type=int, default=4)
    parser.add_argument("--resolution_tier", type=int, default=480, choices=[256, 480, 704, 720])
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--action_fps", type=float, default=10.0)
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

        run_policy_start = time.perf_counter()
        action_chunk, predicted_frames = run_policy(
            policy,
            clip_frames,
            prompt,
            history_action,
            last_history_frame_idx,
            args.future_chunk_size,
            args.num_inference_steps,
        )
        run_policy_msec = int((time.perf_counter() - run_policy_start) * 1000)
        print(f"run_policy: {run_policy_msec}msec")

        predicted_path = integrate_ego_path(action_chunk)
        ground_truth_path = ground_truth_future_positions(episode, center, args.future_chunk_size)

        current_frame_rgb = np.array(clip_frames[-1])
        panel_rgb = compose_panel(
            current_frame_rgb,
            predicted_frames,
            predicted_path,
            ground_truth_path,
            episode.intrinsic,
            episode.camera_height,
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
