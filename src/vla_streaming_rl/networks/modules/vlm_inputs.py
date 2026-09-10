# SPDX-License-Identifier: MIT
import torch
import torch.nn.functional as F
from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata

FRAME_SIZE = 256


def build_video_media(processor: AutoProcessor, frames: list[dict], decision_fps: float) -> dict:
    """The frames as the processor takes them: one clip, with the timestamps read
    off the episode steps they were seen on.

    Shared so that every reader of a ``PromptBuilder`` conversation sends the
    same pixels and the same timestamps. A probe that packs them differently from
    the run it is probing measures a different prompt.

    The processor pairs adjacent frames into temporal patches, so an odd count
    has no packing; the oldest frame is dropped rather than the newest, which is
    the one being asked about. A lone first frame has no oldest to drop, so it is
    repeated to fill the patch, which is what the image channel does to a single
    image anyway.
    """
    assert frames, "no frame to build a clip from"
    assert decision_fps > 0.0, decision_fps
    temporal_patch_size = processor.video_processor.temporal_patch_size
    clip = frames[len(frames) % temporal_patch_size :]
    if not clip:
        clip = frames + frames[-1:] * (temporal_patch_size - len(frames))
    pixels = F.interpolate(
        torch.stack([frame["image"].to(torch.float32) for frame in clip]),
        size=(FRAME_SIZE, FRAME_SIZE),
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 1.0)
    metadata = VideoMetadata(
        total_num_frames=len(clip),
        fps=decision_fps,
        duration=len(clip) / decision_fps,
        frames_indices=[frame["step"] for frame in clip],
        video_backend="tensor",
    )
    return {"videos": [pixels], "video_metadata": [metadata], "do_sample_frames": False}


def build_vlm_inputs(
    processor: AutoProcessor,
    images: torch.Tensor,
    task_prompts: list[str],
    decision_fps: float,
) -> dict:
    """Build VLM inputs.

    Args:
        images: (B, T, C, H, W) float tensor in [0, 1].
        task_prompts: one prompt per batch element.
        decision_fps: the environment's decision rate, which is what the plaintext
            timestamps labelling each temporal patch are written from.

    Returns the prompt (``input_ids``, ``attention_mask``, ``mm_token_type_ids``),
    the grids M-RoPE reads (``image_grid_thw``, ``video_grid_thw``; one of the two
    is None), what the vision tower is fed (``vision_pixel_values``,
    ``vision_grid_thw``), the placeholder the tower's tokens are scattered into
    (``vision_token_id``) and the window length.
    """
    assert images.ndim == 5, f"expected (B, T, C, H, W); got {tuple(images.shape)}"
    assert decision_fps > 0.0, decision_fps
    B, T = images.shape[:2]
    assert len(task_prompts) == B, f"task_prompts length {len(task_prompts)} != batch size {B}"
    device = images.device

    if T == 1:
        return _single_frame_inputs(processor, images[:, 0], task_prompts, device)

    temporal_patch_size = processor.video_processor.temporal_patch_size
    assert T % temporal_patch_size == 0, (
        f"the window must pack into temporal patches: T={T} is not divisible by "
        f"temporal_patch_size={temporal_patch_size}"
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
        fps=decision_fps,
        duration=T / decision_fps,
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
        videos=[frames[b] for b in range(B)],
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
        "image_grid_thw": None,
        "video_grid_thw": grid,
        "vision_pixel_values": encoded["pixel_values_videos"].to(device).to(torch.bfloat16),
        "vision_grid_thw": grid,
        "vision_token_id": processor.video_token_id,
        "seq_len": T,
    }


def _single_frame_inputs(
    processor: AutoProcessor,
    frame: torch.Tensor,
    task_prompts: list[str],
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
        [{"role": "user", "content": [{"type": "text", "text": p}, {"type": "image"}]}]
        for p in task_prompts
    ]
    texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(
        text=texts,
        images=[frame[b] for b in range(B)],
        do_rescale=False,
        padding=True,
        return_tensors="pt",
    )
    grid = encoded["image_grid_thw"].to(device)
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "mm_token_type_ids": encoded["mm_token_type_ids"].to(device),
        "image_grid_thw": grid,
        "video_grid_thw": None,
        "vision_pixel_values": encoded["pixel_values"].to(device).to(torch.bfloat16),
        "vision_grid_thw": grid,
        "vision_token_id": processor.image_token_id,
        "seq_len": 1,
    }
