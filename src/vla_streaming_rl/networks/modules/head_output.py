# SPDX-License-Identifier: MIT
from dataclasses import dataclass

import torch


@dataclass
class HeadOutput:
    """Forward output of value / diffusion-policy / state-predictor heads.

    ``output`` is the head's raw result (logits, velocity, ...); ``activation``
    is the penultimate feature used for statistical metrics.
    """

    output: torch.Tensor
    activation: torch.Tensor
