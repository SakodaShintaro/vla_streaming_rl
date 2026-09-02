# SPDX-License-Identifier: MIT
# Both lines work around transformers not finding FLA's gated delta rule kernels
# (huggingface/transformers#48148), which silently leaves 18 of the model's 24
# layers on the torch fallback. The resolver only walks attributes reachable
# from the package root, so importing the submodule is what makes the chained
# path resolve; the alias covers the name FLA 0.5.0 exports it under. Both must
# run before `modeling_qwen3_5` is imported, which is when its decorators bind.
import fla.ops.gated_delta_rule as _gated_delta_rule

_gated_delta_rule.recurrent_gated_delta_rule = _gated_delta_rule.fused_recurrent_gated_delta_rule

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

    processor = AutoProcessor.from_pretrained(model_id, device_map=device)

    return model, processor
