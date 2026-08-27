# SPDX-License-Identifier: MIT
"""Build VLM model inputs while keeping the per-step text path light.

The HF AutoProcessor.__call__ rebuilds chat-template metadata + BatchFeature
every call, allocating thousands of small Python objects per training step.
In long Bench2Drive routes the heap state churned by scenario runtime
amplifies that allocation cost (observed: 2 ms -> 17 ms over a single run on
the same input shape).

We split the cost:
    - Image side is cached aggressively: input dimensions are fixed, so
      ``image_grid_thw`` is constant. Tensors go straight into the
      image_processor (the original ``prepare_vlm_inputs`` did
      tensor->CPU->numpy->PIL->tensor just so the processor could do
      PIL->tensor again — pointless work).
    - Text side runs the tokenizer fresh every step. The chat template is
      cached at init with a sentinel for the prompt so per-step we only do
      ``str.replace`` + tokenizer, skipping ``apply_chat_template`` and the
      processor's image-placeholder expansion. This keeps Chain-of-Thought-
      style varying prompts cheap.
"""

from collections.abc import Sequence

import torch
from PIL import Image
from transformers import AutoProcessor

_PROMPT_SENTINEL = "<<<<TASK_PROMPT_SENTINEL>>>>"


class VLMInputCache:
    """Caches image grid + chat template; tokenizes text per call."""

    def __init__(
        self,
        processor: AutoProcessor,
        observation_shape: Sequence[int],
        seq_len: int,
        device: torch.device,
        image_mode: str,
    ) -> None:
        if len(observation_shape) != 3:
            raise ValueError(f"observation_shape must be (C, H, W); got {tuple(observation_shape)}")
        if image_mode not in ("mem", "sequence"):
            raise ValueError(f"Unknown image_mode: {image_mode}")
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor
        self.image_token = processor.image_token  # e.g. '<|image_pad|>'
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        self.observation_shape = tuple(observation_shape)
        self.seq_len = seq_len
        self.device = device
        self.image_mode = image_mode
        # mem: only the last frame appears in the LLM prompt (1 image).
        # sequence: every frame appears in order (T images).
        self.num_prompt_images = 1 if image_mode == "mem" else seq_len

        # --- Image grid cache: feed a dummy through image_processor once to
        #     pin down image_grid_thw. With fixed observation dims this is
        #     constant and we can avoid recomputing it.
        C, H, W = self.observation_shape
        dummy_tensor = torch.zeros(C, H, W, dtype=torch.float32)
        img_ref = self.image_processor(images=[dummy_tensor], return_tensors="pt", do_rescale=False)
        single_grid = img_ref["image_grid_thw"].to(device)  # (1, 3)
        self.cached_single_grid = single_grid
        self.cached_all_image_grid_thw = single_grid.repeat(seq_len, 1)
        # Prompt-image grid: 1 row for mem, T rows for sequence (per batch element).
        self.cached_prompt_image_grid_thw = single_grid.repeat(self.num_prompt_images, 1)
        merge_length = self.image_processor.merge_size**2
        self.num_image_tokens = int(single_grid[0].prod().item() // merge_length)

        # --- Chat template cache: render the template once with a sentinel
        #     in place of the prompt and the image placeholder pre-expanded
        #     to ``num_image_tokens`` copies. Per call we only ``str.replace``
        #     the sentinel and tokenize, which is far cheaper than
        #     ``processor.__call__``.
        dummy_pil = Image.new("RGB", (W, H))  # apply_chat_template just needs structure
        content = [{"type": "text", "text": _PROMPT_SENTINEL}]
        # mem: 1 image in the prompt; sequence: T images, fed in temporal order.
        for _ in range(self.num_prompt_images):
            content.append({"type": "image", "image": dummy_pil})
        messages = [[{"role": "user", "content": content}]]
        templated = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if isinstance(templated, list):
            templated = templated[0]
        # Pre-expand each image placeholder in-place. ``str.replace`` (no count)
        # does not recurse on substitutions, so each <image_pad> token becomes
        # exactly ``num_image_tokens`` copies regardless of how many images
        # appear in the template.
        self.template_text: str = templated.replace(
            self.image_token, self.image_token * self.num_image_tokens
        )

    def __call__(self, images: torch.Tensor, task_prompts: list[str]) -> dict:
        """Build VLM inputs.

        Args:
            images: (B, T, C, H, W) float tensor in [0, 1].
            task_prompts: list of length B; one prompt per batch element.
        """
        if images.ndim != 5:
            raise ValueError(f"expected (B, T, C, H, W); got {tuple(images.shape)}")
        B, T = images.shape[:2]
        if T != self.seq_len:
            raise ValueError(f"T mismatch: cache expects seq_len={self.seq_len}, got T={T}")
        if len(task_prompts) != B:
            raise ValueError(f"task_prompts length {len(task_prompts)} != batch size {B}")

        device = images.device

        # --- Text path: tokenize fresh, but skip chat-template + placeholder
        #     expansion by reusing the pre-rendered template_text.
        per_batch = [self.template_text.replace(_PROMPT_SENTINEL, p) for p in task_prompts]
        text_inputs = self.tokenizer(per_batch, return_tensors="pt", padding=True)

        # --- Image path: tensors straight into image_processor, no PIL.
        flat = images.reshape(B * T, *images.shape[2:])
        img_out = self.image_processor(
            images=[flat[i] for i in range(B * T)],
            return_tensors="pt",
            do_rescale=False,
        )
        all_pixel_values = img_out["pixel_values"].to(device).to(torch.bfloat16)
        if B == 1:
            all_image_grid_thw = self.cached_all_image_grid_thw
        else:
            all_image_grid_thw = self.cached_single_grid.expand(B * T, -1)

        # image_grid_thw lists every image actually present in the LLM prompt:
        # 1 per batch element for mem, T per batch element for sequence.
        if self.num_prompt_images == 1:
            prompt_image_grid_thw = self.cached_single_grid.expand(B, -1)
        else:
            prompt_image_grid_thw = self.cached_prompt_image_grid_thw.repeat(B, 1)

        input_ids = text_inputs["input_ids"].to(device)
        return {
            "input_ids": input_ids,
            "attention_mask": text_inputs["attention_mask"].to(device),
            # Multimodal RoPE needs to know which positions are image tokens. The
            # processor returns this alongside input_ids; building the prompt by
            # hand means building it too, and it is exactly where the image token
            # sits.
            "mm_token_type_ids": (input_ids == self.image_token_id).long(),
            "image_grid_thw": prompt_image_grid_thw,
            "all_pixel_values": all_pixel_values,
            "all_image_grid_thw": all_image_grid_thw,
            "seq_len": T,
        }
