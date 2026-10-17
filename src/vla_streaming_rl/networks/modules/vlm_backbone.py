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
from torch import nn
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)


def load_model(
    model_id: str, load_in_4bit: bool, device: torch.device
) -> tuple[nn.Module, AutoProcessor]:
    """Load Qwen3.5 model and processor."""

    bnb_config = None
    if load_in_4bit:
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
    processor = AutoProcessor.from_pretrained(model_id, device_map=device)

    return model, processor
