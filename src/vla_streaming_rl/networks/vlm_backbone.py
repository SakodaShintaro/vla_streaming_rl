# SPDX-License-Identifier: MIT
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch import nn
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)


def load_model(
    model_id: str, use_lora: bool, device: torch.device
) -> tuple[nn.Module, AutoProcessor]:
    """Load Qwen3.5 model and processor."""

    # quantization has a negative effect on performance, so we disable it by default for now
    # True:4.30 steps/sec, False 5.40 steps/sec
    use_quantization = False

    bnb_config = None
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device,
    )
    if use_lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=[
                # Language model
                "down_proj",
                "o_proj",
                "k_proj",
                "q_proj",
                "gate_proj",
                "up_proj",
                "v_proj",
                # Vision encoder (attn.proj only, not patch_embed.proj)
                "qkv",
                r"attn\.proj",
                "linear_fc1",
                "linear_fc2",
            ],
            use_dora=True,
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(model_id, device_map=device)

    return model, processor


def _images_to_pil(images: torch.Tensor) -> list[Image.Image]:
    """Convert (N, C, H, W) float tensor in [0,1] to list of PIL Images."""
    result = []
    for i in range(images.shape[0]):
        img_np = (images[i].to(torch.float32).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        result.append(Image.fromarray(img_np))
    return result


def prepare_vlm_inputs(
    processor: AutoProcessor,
    images: torch.Tensor,
    task_prompts: list[str],
    image_mode: str,
) -> dict[str, torch.Tensor]:
    """Build VLM messages and prepare model inputs.

    With image_mode="mem", only the last frame is placed as <image> in the prompt and
    a separate video encoder is expected to consume the full frame stack externally
    (via "all_pixel_values") and inject the last-frame embedding back at the
    <image_pad> positions. With image_mode="sequence", every frame is placed as its
    own <image> in the prompt and is encoded independently by the VLM's vision
    encoder.

    Args:
        processor: AutoProcessor instance
        images: (B, T, C, H, W) tensor
        task_prompts: List of task prompt strings, one per batch element
        image_mode: "mem" or "sequence"

    Returns:
        Dictionary of model inputs. Extra keys:
            all_pixel_values: pixel_values for all B*T frames
            all_image_grid_thw: grid_thw for all B*T frames
            seq_len: number of frames T
    """
    device = images.device
    batch_size, seq_len = images.shape[:2]

    # --- Build prompt: last frame only (mem) or all T frames (sequence) ---
    messages = []
    for b in range(batch_size):
        content: list[dict] = []
        prompt = task_prompts[b]
        if prompt:
            content.append({"type": "text", "text": prompt})
        if image_mode == "mem":
            frame_indices = [seq_len - 1]
        elif image_mode == "sequence":
            frame_indices = list(range(seq_len))
        else:
            raise ValueError(f"Unknown image_mode: {image_mode}")
        for t in frame_indices:
            img = images[b, t].to(torch.float32)
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            content.append({"type": "image", "image": Image.fromarray(img_np)})
        messages.append([{"role": "user", "content": content}])

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    prompt_proc_images, _ = process_vision_info(messages)

    inputs = processor(text=text, images=prompt_proc_images, return_tensors="pt", padding=True)
    inputs.pop("token_type_ids", None)

    # --- Process ALL frames to get pixel_values for video encoder ---
    all_pil = _images_to_pil(images.reshape(-1, *images.shape[2:]))  # B*T images
    all_messages = [
        [{"role": "user", "content": [{"type": "image", "image": img}]}] for img in all_pil
    ]
    all_proc_images, _ = process_vision_info(all_messages)

    all_inputs = processor(
        text=["x"] * len(all_pil), images=all_proc_images, return_tensors="pt", padding=True
    )

    inputs["all_pixel_values"] = all_inputs["pixel_values"]
    inputs["all_image_grid_thw"] = all_inputs["image_grid_thw"]
    inputs["seq_len"] = seq_len

    inputs = {
        k: v.to(device).to(torch.bfloat16)
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
        else v.to(device)
        if isinstance(v, torch.Tensor)
        else v
        for k, v in inputs.items()
    }
    return inputs
