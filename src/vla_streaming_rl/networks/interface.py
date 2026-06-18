# SPDX-License-Identifier: MIT
"""Shared network interface: structured result types and the abstract base class.

Every policy/value network exposes exactly three public methods —
``infer``, ``compute_loss`` and ``infer_and_compute_loss`` — returning the
structured types defined here. ``NetworkInterface`` makes that contract explicit:
a subclass that does not implement all three cannot be instantiated. Anything
else on a concrete network is an implementation detail (``_``-prefixed by
convention) and is not part of the public surface.
"""

import abc
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ActivationFeatures:
    """2D feature tensors of the four monitored submodules, captured during the
    inference forward pass and fed to the statistical-metrics computer (stable
    rank / dormant ratio / ...)."""

    state: torch.Tensor  # encoder output
    actor: torch.Tensor  # policy head
    critic: torch.Tensor  # value head
    state_predictor: torch.Tensor  # sequence prediction head


@dataclass
class InferResult:
    """Structured return value of a network's ``infer`` / ``infer_and_compute_loss``.

    Replaces the free-form dict whose schema silently differed between the two
    call sites (``infer`` returned extra keys that ``infer_and_compute_loss``
    omitted, forcing a ``.get(..., [])`` fallback in the agent). The fields are
    exactly those the agents consume.
    """

    action: torch.Tensor  # (B, horizon, action_dim)
    value_report: dict[
        str, float
    ]  # value head diagnostics (incl. "value"); see value_head.value_report
    rnn_state: torch.Tensor  # (B, ...)
    next_image: np.ndarray  # predicted next image (H, W, 3)
    next_reward: float  # predicted next reward
    action_token_ids: list  # token ids for VLM action chunk; empty for non-VLM
    activations: ActivationFeatures  # forward-pass features for statistical metrics


@dataclass
class EligibilityTraceInfo:
    """Critic-update quantities for eligibility-trace training."""

    actor_entropy_loss: torch.Tensor
    neg_value: torch.Tensor
    delta: torch.Tensor


@dataclass
class LossResult:
    """Structured return value of a network's ``compute_loss``.

    ``info`` stays a plain dict on purpose: it is a dynamic ``name -> float``
    telemetry map whose keys vary per policy-head / predictor configuration and
    is flattened verbatim into wandb metrics.
    """

    loss: torch.Tensor
    info: dict  # scalar name -> float


@dataclass
class InferLossResult:
    """Structured return value of a network's ``infer_and_compute_loss``.

    Activations now live on ``infer_result`` (captured during the inference
    forward), so they are reached via ``infer_result.activations``.
    """

    infer_result: InferResult
    loss_result: LossResult
    et_info: EligibilityTraceInfo


class NetworkInterface(nn.Module, abc.ABC):
    """Abstract base for all policy/value networks.

    The contract is the three abstract methods below. A subclass missing any of
    them raises ``TypeError`` on instantiation. ``nn.Module`` is mixed in so
    concrete networks keep full PyTorch behaviour (``parameters()``, ``.to()``,
    ``state_dict()`` …); ``ABCMeta`` derives from ``type`` so there is no
    metaclass conflict.
    """

    @abc.abstractmethod
    def init_state(self) -> torch.Tensor:
        """Initial recurrent state the agent carries between steps."""

    @abc.abstractmethod
    def tokenize_task_prompt(self, task_prompt: str) -> list[int]:
        """Token ids for a task-prompt string (empty for non-VLM networks)."""

    @abc.abstractmethod
    def infer(
        self,
        s_seq: torch.Tensor,
        obs_z_seq: torch.Tensor,
        a_seq: torch.Tensor,
        r_seq: torch.Tensor,
        rnn_state: torch.Tensor,
        task_prompts: list[str],
    ) -> InferResult:
        """Single-step inference: the action to take, its value report, the
        carried RNN state and predicted next state. Batch size is 1.

        TODO: the agreed direction is to collapse these arguments into a single
        ``data`` object (matching ``compute_loss`` / ``infer_and_compute_loss``).
        That also needs the streaming call site to fold the live ``rnn_state``
        and ``task_prompts`` into the data window, so it is a separate refactor."""

    @abc.abstractmethod
    def compute_loss(self, data) -> LossResult:
        """Training loss over a replay batch."""

    @abc.abstractmethod
    def infer_and_compute_loss(self, data) -> InferLossResult:
        """Combined inference + loss in one forward, sharing the encoder pass."""
