# SPDX-License-Identifier: MIT
"""How every VLM in this repo is handed its input: the frames the agent was shown
as one video, under the prompt builder's two turns.

The video is read off a window of the last ``seq_len`` ticks at an interval of
``frame_stride`` ticks, counted back from the newest frame so that one is always
in it: the window is how far back the model sees, the stride how often.
"""

import torch
import torch.nn.functional as F
from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata

FRAME_SIZE = 256


class FrameWindow:
    """The last ``length`` frames an agent was shown, the window a video is
    read off.

    Fewer than ``length`` frames -- the start of an episode -- are padded by
    repeating the earliest, so the window is always full and its oldest slot is
    what the agent saw first.
    """

    def __init__(self, length: int) -> None:
        assert length >= 1, length
        self.length = length
        self._frames = []

    def reset(self) -> None:
        self._frames = []

    def push(self, frame: torch.Tensor) -> None:
        """``frame``: (C, H, W) float in [0, 1]."""
        self._frames = (self._frames + [frame])[-self.length :]

    def frames(self) -> torch.Tensor:
        """(length, C, H, W)."""
        assert self._frames, "no frame has been pushed since the reset"
        padding = [self._frames[0]] * (self.length - len(self._frames))
        return torch.stack(padding + self._frames)


def strided_frames(images: torch.Tensor, frame_stride: int) -> torch.Tensor:
    """Every ``frame_stride``-th frame of the window, counted back from its last
    so the newest frame is always kept. ``images`` is (..., T, C, H, W); the
    result has ceil(T / frame_stride) frames in their original order."""
    assert frame_stride >= 1, frame_stride
    dim = images.ndim - 4
    T = images.shape[dim]
    index = torch.arange(T - 1, -1, -frame_stride, device=images.device).flip(0)
    return images.index_select(dim, index)


def chat_messages(system_text: str, turn_text: str, vision_type: str) -> list[dict]:
    """The two turns every VLM here reads: the standing task as the system, and
    the frames followed by this tick's text as the user. ``vision_type`` is the
    placeholder the frames take, ``"video"`` or ``"image"``."""
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {
            "role": "user",
            "content": [{"type": vision_type}, {"type": "text", "text": turn_text}],
        },
    ]


def build_vlm_inputs(
    processor: AutoProcessor,
    images: torch.Tensor,
    system_texts: list[str],
    turn_texts: list[str],
    decision_fps: float,
    frame_stride: int,
    enable_thinking: bool,
) -> dict:
    """Build VLM inputs.

    Args:
        images: (B, T, C, H, W) float tensor in [0, 1], the whole window; the
            video is every ``frame_stride``-th frame of it counted back from
            the last (see :func:`strided_frames`).
        system_texts, turn_texts: the prompt builder's two turns, one pair per
            batch element.
        decision_fps: the environment's decision rate; divided by the stride, it
            is what the plaintext timestamps labelling each temporal patch are
            written from.
        enable_thinking: what the chat template renders at the generation prompt:
            off closes the model's own think block before it starts.

    Returns the prompt (``input_ids``, ``attention_mask``, ``mm_token_type_ids``),
    the grids M-RoPE reads (``image_grid_thw``, ``video_grid_thw``; one of the two
    is None), what the vision tower is fed (``vision_pixel_values``,
    ``vision_grid_thw``), the placeholder the tower's tokens are scattered into
    (``vision_token_id``), the window length, and under ``encoded`` the
    processor's own output as ``generate`` takes it.
    """
    assert images.ndim == 5, f"expected (B, T, C, H, W); got {tuple(images.shape)}"
    assert decision_fps > 0.0, decision_fps
    images = strided_frames(images, frame_stride)
    video_fps = decision_fps / frame_stride
    B, T = images.shape[:2]
    assert len(system_texts) == B, f"system_texts length {len(system_texts)} != batch size {B}"
    assert len(turn_texts) == B, f"turn_texts length {len(turn_texts)} != batch size {B}"
    device = images.device

    if T == 1:
        return _single_frame_inputs(
            processor, images[:, 0], system_texts, turn_texts, enable_thinking, device
        )

    temporal_patch_size = processor.video_processor.temporal_patch_size
    assert T % temporal_patch_size == 0, (
        f"the video must pack into temporal patches: {T} frames (seq_len / frame_stride) "
        f"is not divisible by temporal_patch_size={temporal_patch_size}"
    )

    # Bicubic to match the resample the processor would have used itself.
    frames = F.interpolate(
        images.reshape(B * T, *images.shape[2:]),
        size=(FRAME_SIZE, FRAME_SIZE),
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 1.0)
    frames = frames.reshape(B, T, *frames.shape[1:])

    # frames_indices and fps are what the processor turns into the plaintext
    # timestamps; giving it the window's own rate makes them window-relative,
    # the convention SimpleMemVLA deploys.
    metadata = VideoMetadata(
        total_num_frames=T,
        fps=video_fps,
        duration=T / video_fps,
        frames_indices=list(range(T)),
        video_backend="tensor",
    )
    messages = [
        chat_messages(system_text, turn_text, "video")
        for system_text, turn_text in zip(system_texts, turn_texts, strict=True)
    ]
    texts = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    encoded = processor(
        text=texts,
        videos=[frames[b] for b in range(B)],
        video_metadata=[metadata] * B,
        do_sample_frames=False,
        do_rescale=False,
        padding=True,
        return_tensors="pt",
    ).to(device)
    grid = encoded["video_grid_thw"]
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "mm_token_type_ids": encoded["mm_token_type_ids"],
        "image_grid_thw": None,
        "video_grid_thw": grid,
        "vision_pixel_values": encoded["pixel_values_videos"].to(torch.bfloat16),
        "vision_grid_thw": grid,
        "vision_token_id": processor.video_token_id,
        "seq_len": T,
        "encoded": encoded,
    }


def _single_frame_inputs(
    processor: AutoProcessor,
    frame: torch.Tensor,
    system_texts: list[str],
    turn_texts: list[str],
    enable_thinking: bool,
    device: torch.device,
) -> dict:
    """The one-frame window, packed as a plain image.

    A window of one has no pair to form a temporal patch with, and the video
    channel has nothing to say about it: no motion for the Conv3d to read across
    the temporal kernel and one timestamp carrying no relation. Sending it as an
    image is both what the format is for and what this repo did before the video
    channel existed, so a `seq_len: 1` run stays comparable with those. The
    image processor does its own upscaling here -- its minimum-pixel floor is
    the FRAME_SIZE grid already -- so the frames are handed over untouched.
    """
    B = frame.shape[0]
    messages = [
        chat_messages(system_text, turn_text, "image")
        for system_text, turn_text in zip(system_texts, turn_texts, strict=True)
    ]
    texts = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    encoded = processor(
        text=texts,
        images=[frame[b] for b in range(B)],
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
        "seq_len": 1,
        "encoded": encoded,
    }
