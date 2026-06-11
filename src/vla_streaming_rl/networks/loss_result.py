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
class ActivationFeatures:
    """2D feature tensors of the four monitored submodules, fed to the
    statistical-metrics computer (stable rank / dormant ratio / ...)."""

    state: torch.Tensor  # encoder output
    actor: torch.Tensor  # policy head
    critic: torch.Tensor  # value head
    state_predictor: torch.Tensor  # sequence prediction head


@dataclass
class LossResult:
    """Structured return value of a network's ``compute_loss``.

    ``info`` stays a plain dict on purpose: it is a dynamic ``name -> float``
    telemetry map whose keys vary per policy-head / predictor configuration and
    is flattened verbatim into wandb metrics.
    """

    loss: torch.Tensor
    activations: ActivationFeatures
    info: dict  # scalar name -> float


@dataclass
class InferLossResult:
    """Structured return value of a network's ``infer_and_compute_loss``."""

    infer_result: InferResult
    loss_result: LossResult
    et_info: EligibilityTraceInfo
