"""Streaming evaluation of the Self-Forcing world model on bench2drive valid.

For each valid episode we replay the front-camera jpgs one frame at a time
through ``WorldModelGoalPredictor.step(obs)`` and record the goal frame the
predictor returns. The lookahead is fixed by the predictor:
``delta = block_pix + 1 - predict_interval`` pixel frames into the future
(block_pix = fpb*4 = 12). We score the predicted goal at step ``t`` against
the real jpg ``delta`` frames later (PSNR).

Run:
  uv run python scripts/self_forcing/infer_valid.py \
    --config_path configs/self_forcing/b2d_finetune.yaml \
    --b2d_root /path/to/bench2drive \
  uv run python scripts/self_forcing/infer_valid.py \
    --config_path configs/self_forcing/b2d_finetune.yaml \
    --b2d_root /path/to/bench2drive \
    --checkpoint_path logs/.../checkpoint_model_001000/model.pt \
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.io import write_video
from tqdm import tqdm

from vla_streaming_rl.self_forcing.goal_predictor import WorldModelGoalPredictor
from vla_streaming_rl.self_forcing.utils.misc import set_seed

_WAN_FPS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional LoRA fine-tune ckpt. None = base self_forcing_dmd.pt only.",
    )
    parser.add_argument(
        "--b2d_root",
        type=str,
        required=True,
        help="Bench2Drive root (contains splits.json and per-episode rgb_front jpgs).",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=None,
        help="Output root. Required only when --checkpoint_path is omitted; "
        "with a checkpoint, results are saved next to it.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Limit to first N valid episodes (default: all).",
    )
    parser.add_argument(
        "--num_context_blocks",
        type=int,
        default=1,
        help="K: how many fpb-latent blocks of past pixels to use as context.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Cap each episode at this many pixel frames (default: all).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fps",
        type=int,
        default=_WAN_FPS,
        help="Mp4 playback fps (bench2drive native rate is 10).",
    )
    args = parser.parse_args()
    if args.checkpoint_path is None and args.out_root is None:
        parser.error("--out_root is required when --checkpoint_path is not given.")
    return args


def _load_real_frames(
    episode_dir: Path, max_frames: int | None, target_h: int, target_w: int
) -> torch.Tensor:
    """Load and resize episode jpgs. Returns (T, H, W, 3) uint8."""
    rgb_dir = episode_dir / "camera" / "rgb_front"
    paths = sorted(rgb_dir.glob("*.jpg"))
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise RuntimeError(f"no jpgs found in {rgb_dir}")
    resize = transforms.Resize((target_h, target_w))
    frames = []
    for p in paths:
        img = resize(Image.open(p).convert("RGB"))
        frames.append(torch.from_numpy(np.asarray(img)))  # (H, W, 3) uint8
    return torch.stack(frames, dim=0)


def _per_frame_psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    """pred, ref: (H, W, 3) uint8. Returns scalar PSNR in dB."""
    p = pred.astype(np.float32)
    r = ref.astype(np.float32)
    mse = float(((p - r) ** 2).mean())
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _annotate(frame: np.ndarray, label: str) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(
        out, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )
    return out


def main() -> None:
    args = parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.checkpoint_path:
        out_dir = Path(args.checkpoint_path).parent / f"{stamp}"
    else:
        out_dir = args.out_root / f"{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    set_seed(args.seed)
    torch.set_grad_enabled(False)

    print(f"output dir: {out_dir}")
    print("building world model")
    predictor = WorldModelGoalPredictor(
        enabled=True,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=device,
        num_context_blocks=args.num_context_blocks,
    )
    predict_interval = predictor.predict_interval
    block_pix = predictor.block_pix
    delta = block_pix + 1 - predict_interval  # constant lookahead the predictor delivers

    b2d_root = Path(args.b2d_root)
    splits = json.load(open(b2d_root / "splits.json"))
    episodes = splits["valid"]
    if args.num_episodes is not None:
        episodes = episodes[: args.num_episodes]
    print(f"valid episodes to infer: {len(episodes)}")

    summary = []
    ep_pbar = tqdm(episodes, desc="episodes", unit="ep")
    for i, ep in enumerate(ep_pbar):
        ep_pbar.set_postfix_str(ep, refresh=False)
        real_uint8 = _load_real_frames(
            b2d_root / ep, args.max_frames, predictor.target_h, predictor.target_w
        )  # (T, H, W, 3) uint8
        T = real_uint8.shape[0]
        if T <= delta:
            tqdm.write(f"[{i + 1}/{len(episodes)}] {ep}: SKIP (T={T} <= delta={delta})")
            continue

        predictor.reset()
        t0 = time.time()
        pred_frames: list[np.ndarray] = []
        for t in tqdm(range(T), desc=f"  steps ({ep[:30]})", unit="f", leave=False):
            obs = real_uint8[t].numpy().astype(np.float32).transpose(2, 0, 1) / 255.0
            goal = predictor.step(obs)  # (H, W, 3) uint8
            pred_frames.append(goal)

        # Pair pred[t] with real[t + delta]; the first few steps have no real
        # prediction yet so PSNR there is just black-vs-real.
        psnrs: list[float] = []
        compare_frames: list[np.ndarray] = []
        for t in range(T - delta):
            pred = pred_frames[t]
            ref = real_uint8[t + delta].numpy()
            psnr = _per_frame_psnr(pred, ref)
            psnrs.append(psnr)
            # cycle_step = how many env steps have passed since the most recent
            # inference fired. goal_frame_idx = which frame of the cached
            # block this is (block_pix - N + cycle_step). target_offset is
            # frames-into-future from the prediction time.
            cycle_step = t % predict_interval
            goal_frame_idx = (block_pix - predict_interval) + cycle_step
            target_offset = goal_frame_idx + 1
            label_real = f"real t+{delta}"
            label_pred = f"pred {cycle_step}f ago, +{target_offset}f from pred  PSNR {psnr:.1f}dB"
            row = np.concatenate([_annotate(ref, label_real), _annotate(pred, label_pred)], axis=1)
            compare_frames.append(row)

        mean_psnr = float(np.mean(psnrs)) if psnrs else float("nan")
        write_video(
            str(out_dir / f"{ep}_compare.mp4"),
            torch.from_numpy(np.stack(compare_frames, axis=0)),
            fps=args.fps,
        )

        dt = time.time() - t0
        running = [s["mean_psnr_db"] for s in summary] + [mean_psnr]
        running_mean = float(np.mean(running))
        ep_pbar.set_postfix(
            ep=ep[:30],
            psnr=f"{mean_psnr:.2f}dB",
            avg=f"{running_mean:.2f}dB",
            dt=f"{dt:.1f}s",
        )
        tqdm.write(
            f"[{i + 1}/{len(episodes)}] {ep}: T={T} delta={delta} "
            f"mean_PSNR={mean_psnr:.2f}dB dt={dt:.1f}s"
        )
        summary.append(
            {
                "episode": ep,
                "T": T,
                "delta_frames": delta,
                "mean_psnr_db": round(mean_psnr, 3),
                "per_frame_psnr_db": [round(p, 3) for p in psnrs],
                "dt_seconds": round(dt, 2),
            }
        )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "checkpoint_path": args.checkpoint_path,
                "num_context_blocks": args.num_context_blocks,
                "predict_interval": predict_interval,
                "delta_frames": delta,
                "fps": args.fps,
                "seed": args.seed,
                "episodes": summary,
            },
            f,
            indent=2,
        )
    print(f"done. wrote {len(summary)} videos to {out_dir}")


if __name__ == "__main__":
    main()
