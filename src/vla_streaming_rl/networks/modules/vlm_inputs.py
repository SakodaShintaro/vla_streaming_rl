# SPDX-License-Identifier: MIT
"""How the VLMs in this repo are handed a prompt builder's conversation: the
chat template rendered once, with the frames pulled out beside it."""

import torch
from torchvision.transforms.v2 import functional as TF
from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata


def render_conversation(
    processor: AutoProcessor, conversation: list[dict], enable_thinking: bool
) -> tuple[str, list[torch.Tensor]]:
    """The conversation as the processor takes it: the chat template's text
    with a placeholder where each frame goes, and the frames in that order.

    ``enable_thinking`` is what the template renders at the generation prompt:
    off closes the model's own think block before it starts.
    """
    images = [
        part["image"]
        for turn in conversation
        for part in turn["content"]
        if part["type"] == "image"
    ]
    messages = [
        {
            "role": turn["role"],
            "content": [
                {key: value for key, value in part.items() if key in ("type", "text")}
                for part in turn["content"]
            ],
        }
        for turn in conversation
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    return text, images


def _to_frame(frame) -> torch.Tensor:
    """A frame as the processor is handed it, a (C, H, W) float tensor in
    [0, 1], out of whatever the agent gave its builder: that, or an 8-bit
    picture."""
    return TF.to_dtype(TF.to_image(frame), torch.float32, scale=True)


def encode_conversation(
    processor: AutoProcessor, conversation: list[dict], enable_thinking: bool, device: torch.device
) -> dict:
    """What the model's forward takes for one conversation, its frames and its
    videos included.

    A video part carries its frames, the rate they were taken at and the tick
    each was taken on, which is what the timestamps the processor writes between
    them count in seconds. The model folds a video's frames in pairs, so a video
    holds an even number of them. Its token goes in bare: the processor's own
    replacement opens and closes the vision section around every pair, and the
    chat template's wrapper around it would nest a second one.
    """
    text, images = render_conversation(processor, conversation, enable_thinking)
    videos = [part for turn in conversation for part in turn["content"] if part["type"] == "video"]
    assert all(len(video["video"]) % 2 == 0 for video in videos), (
        "a video holds an even number of frames"
    )
    wrapped_video = (
        f"{processor.vision_start_token}{processor.video_token}{processor.vision_end_token}"
    )
    return processor(
        text=[text.replace(wrapped_video, processor.video_token)],
        images=[_to_frame(image) for image in images] or None,
        videos=[torch.stack([_to_frame(frame) for frame in video["video"]]) for video in videos]
        or None,
        video_metadata=[
            VideoMetadata(
                total_num_frames=len(video["video"]),
                fps=video["fps"],
                frames_indices=video["frames_indices"],
            )
            for video in videos
        ]
        or None,
        do_sample_frames=False,
        do_rescale=False,
        return_tensors="pt",
    ).to(device)


def build_vlm_inputs(
    processor: AutoProcessor,
    texts: list[str],
    images: list[list[torch.Tensor]],
    device: torch.device,
) -> dict:
    """Build VLM inputs for a batch of rendered conversations.

    Args:
        texts: one rendered conversation per batch element (see
            :func:`render_conversation`).
        images: per batch element, the (C, H, W) float frames in [0, 1] its
            placeholders stand for, in order. The image processor does its own
            resizing, so the frames are handed over untouched.

    Returns the prompt (``input_ids``, ``attention_mask``, ``mm_token_type_ids``),
    the grids M-RoPE reads (``image_grid_thw``; ``video_grid_thw`` is None), what
    the vision tower is fed (``vision_pixel_values``, ``vision_grid_thw``) and the
    placeholder the tower's tokens are scattered into (``vision_token_id``).
    """
    assert len(texts) == len(images), f"{len(texts)} texts for {len(images)} image lists"
    encoded = processor(
        text=texts,
        images=images,
        do_rescale=False,
        padding=True,
        return_tensors="pt",
    ).to(device)
    grid = encoded["image_grid_thw"]
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "mm_token_type_ids": encoded["mm_token_type_ids"],
        "image_grid_thw": grid,
        "video_grid_thw": None,
        "vision_pixel_values": encoded["pixel_values"].to(torch.bfloat16),
        "vision_grid_thw": grid,
        "vision_token_id": processor.image_token_id,
    }
