# SPDX-License-Identifier: MIT
"""Benchmark end-to-end VLM inference (ViT + LLM) with varying sequence lengths.

Compares:
  - main branch: K images as separate <image> tokens → K*64 LLM tokens
  - feat/mem branch: VideoEncoder → 64 LLM tokens regardless of K

Usage:
    uv run python scripts/bench_video_encoder.py \
        --model_id Qwen/Qwen3.5-0.8B \
        --image_dir local/image/ep_00000001 \
        --num_warmup 3 --num_runs 10
"""

import argparse
import glob
import time

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--num_warmup", type=int, required=True)
    parser.add_argument("--num_runs", type=int, required=True)
    return parser.parse_args()


def load_frames(image_dir: str, max_frames: int) -> list:
    paths = sorted(glob.glob(f"{image_dir}/*.png"))[:max_frames]
    return [Image.open(p).convert("RGB").resize((84, 84)) for p in paths]


def bench(fn, num_warmup: int, num_runs: int) -> tuple:
    """Run warmup + timed runs, return (mean_ms, std_ms)."""
    with torch.no_grad():
        for _ in range(num_warmup):
            fn()
            torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    return np.mean(times) * 1000, np.std(times) * 1000


def build_standard_inputs(processor, frames, is_qwen35: bool, device: str):
    """Build VLM inputs with all K frames as separate <image> tokens (main branch style)."""
    content = []
    for f in frames:
        content.append({"type": "image", "image": f})
        content.append({"type": "text", "text": "reward 0.00"})
    messages = [[{"role": "user", "content": content}]]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if is_qwen35:
        proc_images, _ = process_vision_info(messages)
    else:
        proc_images, _ = process_vision_info(messages, image_patch_size=16)

    inputs = processor(text=text, images=proc_images, return_tensors="pt", padding=True)
    inputs.pop("token_type_ids", None)
    return {
        k: v.to(device).to(torch.bfloat16) if v.dtype.is_floating_point else v.to(device)
        for k, v in inputs.items()
    }


def build_mem_inputs(processor, frames, is_qwen35: bool, device: str):
    """Build VLM inputs with only last frame as <image> + all frames for VideoEncoder."""
    # Prompt: past rewards as text, last frame as image
    content = []
    for _ in frames[:-1]:
        content.append({"type": "text", "text": "reward 0.00"})
    content.append({"type": "image", "image": frames[-1]})
    content.append({"type": "text", "text": "reward 0.00"})
    messages = [[{"role": "user", "content": content}]]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if is_qwen35:
        last_proc, _ = process_vision_info(messages)
    else:
        last_proc, _ = process_vision_info(messages, image_patch_size=16)
    inputs = processor(text=text, images=last_proc, return_tensors="pt", padding=True)
    inputs.pop("token_type_ids", None)

    # All frames for VideoEncoder
    all_messages = [[{"role": "user", "content": [{"type": "image", "image": f}]}] for f in frames]
    if is_qwen35:
        all_proc, _ = process_vision_info(all_messages)
    else:
        all_proc, _ = process_vision_info(all_messages, image_patch_size=16)
    all_inputs = processor(
        text=["x"] * len(frames), images=all_proc, return_tensors="pt", padding=True
    )

    inputs["all_pixel_values"] = all_inputs["pixel_values"]
    inputs["all_image_grid_thw"] = all_inputs["image_grid_thw"]
    inputs["seq_len"] = len(frames)

    return {
        k: v.to(device).to(torch.bfloat16)
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
        else v.to(device)
        if isinstance(v, torch.Tensor)
        else v
        for k, v in inputs.items()
    }


def run_standard_forward(vlm_model, inputs):
    """Standard VLM forward: pixel_values → ViT → LLM (all images as tokens)."""
    return vlm_model.forward(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )


def run_mem_forward(vlm_model, visual, video_encoder, inputs):
    """MEM-style forward: VideoEncoder → last-frame embed → masked_scatter → LLM."""
    vlm_inner = vlm_model.model
    inputs_embeds = vlm_inner.get_input_embeddings()(inputs["input_ids"])

    batch_size = inputs["input_ids"].shape[0]
    seq_len = inputs["seq_len"]

    last_frame_embeds = video_encoder(
        visual, inputs["all_pixel_values"], inputs["all_image_grid_thw"], batch_size, seq_len
    )
    last_frame_embeds = last_frame_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

    image_token_id = vlm_inner.config.image_token_id
    image_mask = (inputs["input_ids"] == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, last_frame_embeds)

    position_ids = vlm_inner.compute_3d_position_ids(
        input_ids=inputs["input_ids"],
        image_grid_thw=inputs["image_grid_thw"],
        video_grid_thw=None,
        inputs_embeds=inputs_embeds,
        attention_mask=inputs["attention_mask"],
        past_key_values=None,
    )

    return vlm_model.forward(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        attention_mask=inputs["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )


def main():
    args = parse_args()
    device = "cuda"
    is_qwen35 = "Qwen3.5" in args.model_id

    print(f"Loading model: {args.model_id}")
    vlm_model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa" if is_qwen35 else "flash_attention_2",
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    visual = vlm_model.model.visual

    # Try importing VideoEncoder
    video_encoder = None
    try:
        from vla_streaming_rl.networks.modules.video_encoder import VideoEncoder

        video_encoder = VideoEncoder().to(device)
        print("VideoEncoder found (feat/mem branch)")
    except ImportError:
        print("VideoEncoder not found (main branch)")

    # Load frames
    max_seq = 16
    frames = load_frames(args.image_dir, max_seq)
    print(f"Loaded {len(frames)} frames from {args.image_dir}")
    print(f"Warmup: {args.num_warmup}, Timed: {args.num_runs}")
    print()

    seq_lens = [1, 2, 4, 8, 16]

    # --- Standard: all images as <image> tokens ---
    print("=" * 60)
    print("Standard (K images → K*64 LLM tokens)")
    print("=" * 60)
    for K in seq_lens:
        available = min(K, len(frames))
        inputs = build_standard_inputs(processor, frames[:available], is_qwen35, device)
        num_tokens = inputs["input_ids"].shape[1]

        mean_ms, std_ms = bench(
            lambda inp=inputs: run_standard_forward(vlm_model, inp),
            args.num_warmup,
            args.num_runs,
        )
        print(f"  K={available:2d}: {mean_ms:6.1f} ± {std_ms:.1f} ms  (LLM tokens: {num_tokens})")

    # --- MEM: VideoEncoder → 64 LLM tokens ---
    if video_encoder is not None:
        print()
        print("=" * 60)
        print("MEM (K frames → VideoEncoder → 64 LLM tokens)")
        print("=" * 60)
        for K in seq_lens:
            available = min(K, len(frames))
            inputs = build_mem_inputs(processor, frames[:available], is_qwen35, device)
            num_tokens = inputs["input_ids"].shape[1]

            mean_ms, std_ms = bench(
                lambda inp=inputs: run_mem_forward(vlm_model, visual, video_encoder, inp),
                args.num_warmup,
                args.num_runs,
            )
            print(
                f"  K={available:2d}: {mean_ms:6.1f} ± {std_ms:.1f} ms  (LLM tokens: {num_tokens})"
            )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
