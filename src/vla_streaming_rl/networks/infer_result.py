# SPDX-License-Identifier: MIT
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class InferResult:
    """Structured return value of a network's ``infer`` / ``infer_and_compute_loss``.

    Replaces the free-form dict whose schema silently differed between the two
    call sites (``infer`` returned extra keys that ``infer_and_compute_loss``
    omitted, forcing a ``.get(..., [])`` fallback in the agent). The fields are
    exactly those the agents consume.
    """

    action: torch.Tensor  # (B, horizon, action_dim)
    value: float
    rnn_state: torch.Tensor  # (B, ...)
    next_image: np.ndarray  # predicted next image (H, W, 3)
    next_reward: float  # predicted next reward
    action_token_ids: list  # token ids for VLM action chunk; empty for non-VLM
