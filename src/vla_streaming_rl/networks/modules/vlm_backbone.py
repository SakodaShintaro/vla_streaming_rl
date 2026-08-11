# SPDX-License-Identifier: MIT
import torch
from peft import LoraConfig, get_peft_model
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
    else:
        # Without LoRA the VLM (ViT + LLM) is a frozen feature extractor: activations
        # still carry gradient to whatever is attached to it (e.g. the recurrent
        # temporal adapters inside the ViT), but its own weights are never updated.
        model.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(model_id, device_map=device)

    return model, processor
