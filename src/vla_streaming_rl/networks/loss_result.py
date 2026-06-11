# SPDX-License-Identifier: MIT
from dataclasses import dataclass

import torch

from vla_streaming_rl.networks.infer_result import InferResult


@dataclass
class EligibilityTraceInfo:
    """Critic-update quantities for eligibility-trace training."""

    actor_entropy_loss: torch.Tensor
    neg_value: torch.Tensor
    delta: torch.Tensor


@dataclass
class LossResult:
    """Structured return value of a network's ``compute_loss``.

    ``activations`` and ``info`` stay plain dicts on purpose: they are dynamic
    ``name -> value`` maps (submodule features for statistical metrics, and
    scalar telemetry) whose keys vary per network configuration.
    """

    loss: torch.Tensor
    activations: dict  # submodule name -> 2D feature tensor
    info: dict  # scalar name -> float


@dataclass
class InferLossResult:
    """Structured return value of a network's ``infer_and_compute_loss``."""

    infer_result: InferResult
    loss: torch.Tensor
    activations: dict
    info: dict
    et_info: EligibilityTraceInfo
