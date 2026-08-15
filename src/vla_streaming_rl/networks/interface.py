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

import torch
import torch.nn as nn

from vla_streaming_rl.replay_buffer import ReplayBufferData


@dataclass
class InferInput:
    """Inputs to a network's ``infer`` (the live, single-step inference call).

    Unlike ``compute_loss`` / ``infer_and_compute_loss`` — which read a replay
    batch (:class:`ReplayBufferData`) — inference assembles its window from the
    buffer's latest frames but carries the *live* recurrent state and task prompt
    held by the agent. Those two therefore are explicit fields here rather than
    read off the buffer. Batch size is 1.
    """

    s_seq: torch.Tensor  # (B, T, *obs_shape) image observation window
    a_seq: torch.Tensor  # (B, T, action_dim)
    r_seq: torch.Tensor  # (B, T, 1)
    rnn_state: torch.Tensor  # live recurrent state carried by the agent
    task_prompts: list[str]  # one prompt per batch element
    velocity_x_seq: torch.Tensor  # (B, T, 1)
    velocity_y_seq: torch.Tensor  # (B, T, 1)
    velocity_z_seq: torch.Tensor  # (B, T, 1)
    episode_return_seq: torch.Tensor  # (B, T, 1)
    pass_mark_seq: torch.Tensor  # (B, T, 1)
    global_step_seq: torch.Tensor  # (B, T, 1)
    episode_step_seq: torch.Tensor  # (B, T, 1)
    health_seq: torch.Tensor  # (B, T, 1)


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
    value_report: dict[str, float]  # value head diagnostics (incl. "value")
    rnn_state: torch.Tensor  # (B, ...)
    next_image_latent: torch.Tensor  # predicted next image, in encoder latent space
    next_reward_latent: torch.Tensor  # predicted next reward, in encoder latent space
    activations: ActivationFeatures  # forward-pass features for statistical metrics
    features: torch.Tensor


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

    The contract is the five abstract methods below. A subclass missing any of
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
    def infer(self, data: InferInput) -> InferResult:
        """Single-step inference: the action to take, its value report, the
        carried RNN state and predicted next state. Batch size is 1."""

    @abc.abstractmethod
    def compute_loss(self, data: ReplayBufferData) -> LossResult:
        """Training loss over a replay batch."""

    @abc.abstractmethod
    def infer_and_compute_loss(self, data: ReplayBufferData) -> InferLossResult:
        """Combined inference + loss in one forward, sharing the encoder pass."""
