# SPDX-License-Identifier: MIT
"""Build the VLM's model inputs out of a window of frames and a task prompt.

The frames go through the backbone's native video channel: the processor pairs
adjacent frames into temporal patches and prefixes each patch with the plaintext
``<x.y seconds>`` timestamp it was pretrained on. SimpleMemVLA (arXiv 2609.05533)
finds that format to be what makes a history usable as memory -- shuffling frame
order or dropping the timestamps breaks the temporal reading -- and that packing
the same frames as independent images instead asks the model to relearn temporal
structure from scratch. It is also far cheaper: at B=2, T=8 the video channel
builds its inputs in 1.6 ms against 12.0 ms for the multi-image packing, and
occupies 72 context tokens against 1024.

This used to sit behind a caching layer that pre-rendered the chat template and
pinned the image grid, on the grounds that ``AutoProcessor.__call__`` was too
costly per step. On transformers 5.15 that is no longer true, so the processor
is called directly.
"""

import torch
from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata


def build_vlm_inputs(
    processor: AutoProcessor,
    images: torch.Tensor,
    task_prompts: list[str],
    history_fps: float,
) -> dict:
    """Build VLM inputs.

    Args:
        images: (B, T, C, H, W) float tensor in [0, 1].
        task_prompts: one prompt per batch element.
        history_fps: the environment's decision rate, which is what the plaintext
            timestamps labelling each temporal patch are written from.

    Returns the prompt (``input_ids``, ``attention_mask``, ``mm_token_type_ids``),
    the grid M-RoPE reads (``video_grid_thw``), what the vision tower is fed
    (``vision_pixel_values``, ``vision_grid_thw``) and the window length.
    """
    assert images.ndim == 5, f"expected (B, T, C, H, W); got {tuple(images.shape)}"
    assert history_fps > 0.0, history_fps
    B, T = images.shape[:2]
    assert len(task_prompts) == B, f"task_prompts length {len(task_prompts)} != batch size {B}"
    temporal_patch_size = processor.video_processor.temporal_patch_size
    assert T % temporal_patch_size == 0, (
        f"the window must pack into temporal patches: T={T} is not divisible by "
        f"temporal_patch_size={temporal_patch_size}"
    )
    device = images.device

    # frames_indices and fps are what the processor turns into the plaintext
    # timestamps; giving it the window's own rate makes them window-relative,
    # the convention SimpleMemVLA deploys.
    metadata = VideoMetadata(
        total_num_frames=T,
        fps=history_fps,
        duration=T / history_fps,
        frames_indices=list(range(T)),
        video_backend="tensor",
    )
    messages = [
        [{"role": "user", "content": [{"type": "text", "text": p}, {"type": "video"}]}]
        for p in task_prompts
    ]
    texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(
        text=texts,
        videos=[images[b] for b in range(B)],
        video_metadata=[metadata] * B,
        do_sample_frames=False,
        do_rescale=False,
        padding=True,
        return_tensors="pt",
    )
    grid = encoded["video_grid_thw"].to(device)
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "mm_token_type_ids": encoded["mm_token_type_ids"].to(device),
        "video_grid_thw": grid,
        "vision_pixel_values": encoded["pixel_values_videos"].to(device).to(torch.bfloat16),
        "vision_grid_thw": grid,
        "seq_len": T,
    }
