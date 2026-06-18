# SPDX-License-Identifier: MIT
import argparse
import time
from pathlib import Path
from typing import Callable

import torch
from qwen_vl_utils import process_vision_info

from vla_streaming_rl.networks.modules.vlm_backbone import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument(
        "--model_id",
        type=str,
        choices=["Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3.5-0.8B"],
        required=True,
    )
    parser.add_argument("--seq_len", type=int, default=1)
    parser.add_argument("--tokens_per_step", type=int, default=10)
    parser.add_argument("--max_context_words", type=int, default=50)
    parser.add_argument("--prompt", type=str, default="Describe the scene.")
    parser.add_argument("--mode", choices=["image", "video"], default="image")
    parser.add_argument("--max_images", type=int, default=10)
    return parser.parse_args()


def prepare_inputs_qwen35(processor, text, messages):
    images, videos = process_vision_info(messages)
    return processor(
        text=text,
        images=images,
        videos=videos,
        return_tensors="pt",
    )


def prepare_inputs_qwenvl(processor, text, messages):
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    return processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )


PREPARE_INPUTS_FN = {
    "qwen35": prepare_inputs_qwen35,
    "qwenvl": prepare_inputs_qwenvl,
}


def build_messages(
    prompt: str,
    mode: str,
    image_paths: list[Path],
    previous_text: str,
) -> list[dict]:
    user_content: list[dict[str, object]] = []
    user_content.append({"type": "text", "text": prompt})

    if mode == "video":
        video_sources = [f"file://{path.resolve()}" for path in image_paths]
        user_content.append({"type": "video", "video": video_sources})
    else:
        for path in image_paths:
            user_content.append({"type": "image", "image": f"file://{path.resolve()}"})

    messages = [{"role": "user", "content": user_content}]

    if previous_text:
        messages.append({"role": "assistant", "content": previous_text})

    return messages


def truncate_context(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[-max_words:])


def collect_image_paths(images_dir: Path) -> list[Path]:
    all_paths = sorted(images_dir.glob("*"))
    return [p for p in all_paths if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]


def compute_steps(num_images: int, seq_len: int) -> tuple[int, int]:
    effective_seq_len = min(seq_len, num_images)
    num_steps = max(1, num_images - effective_seq_len + 1)
    return num_steps, effective_seq_len


def generate_and_decode(
    model,
    processor,
    messages: list[dict],
    previous_text: str,
    tokens_per_step: int,
    prepare_inputs_fn: Callable,
) -> tuple[str, float]:
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not previous_text,
        continue_final_message=bool(previous_text),
    )

    inputs = prepare_inputs_fn(processor, text, messages).to(model.device)

    start = time.time()
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=tokens_per_step,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    elapsed_msec = (time.time() - start) * 1000

    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text, elapsed_msec


def sliding_window_generate(
    model,
    processor,
    all_image_paths: list[Path],
    prompt: str,
    mode: str,
    seq_len: int,
    tokens_per_step: int,
    max_context_words: int,
    prepare_inputs_fn: Callable,
) -> None:
    num_steps, effective_seq_len = compute_steps(len(all_image_paths), seq_len)

    print(f"Found {len(all_image_paths)} images")
    print(f"Sliding window: seq_len={effective_seq_len}, tokens_per_step={tokens_per_step}")
    print(f"Will process {num_steps} steps")

    accumulated_text = ""
    total_time = 0.0

    with torch.inference_mode():
        for step in range(num_steps):
            window_start = step
            window_end = step + effective_seq_len
            current_images = all_image_paths[window_start:window_end]

            context = truncate_context(accumulated_text, max_context_words)
            messages = build_messages(prompt, mode, current_images, context)

            new_text, elapsed_msec = generate_and_decode(
                model, processor, messages, context, tokens_per_step, prepare_inputs_fn
            )
            total_time += elapsed_msec

            accumulated_text = (accumulated_text + " " + new_text).strip()

            print(f"\n--- Step {step}/{num_steps - 1} (images {window_start}-{window_end - 1}) ---")
            print(f"Generated: {new_text}")
            print(f"Time: {elapsed_msec:.2f} ms")

    print("\n" + "=" * 60)
    print("FINAL OUTPUT:")
    print("=" * 60)
    print(accumulated_text)
    print(f"\nTotal time: {total_time:.2f} ms")
    print(f"Average time per step: {total_time / num_steps:.2f} ms")


MODEL_TYPE_MAP = {
    "Qwen/Qwen3.5-0.8B": "qwen35",
    "Qwen/Qwen3-VL-2B-Instruct": "qwenvl",
}


def main() -> None:
    args = parse_args()

    all_image_paths = collect_image_paths(args.images_dir)[: args.max_images]
    assert all_image_paths, f"No images found in {args.images_dir}"

    model_type = MODEL_TYPE_MAP[args.model_id]
    model, processor = load_model(args.model_id, use_lora=False, device="cuda")

    sliding_window_generate(
        model,
        processor,
        all_image_paths,
        args.prompt,
        args.mode,
        args.seq_len,
        args.tokens_per_step,
        args.max_context_words,
        PREPARE_INPUTS_FN[model_type],
    )


if __name__ == "__main__":
    main()
