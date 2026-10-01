# SPDX-License-Identifier: MIT
"""Shared network interface: structured result types and the abstract base class.

Every policy/value network exposes the forward passes — ``infer``,
``compute_loss`` and ``infer_and_compute_loss`` — returning the structured
types defined here, plus the observation-side hooks the agents drive
(``stored_image_shape`` / ``to_stored_image`` / ``observe_scalar_obs``).
``NetworkInterface`` makes that contract explicit: a subclass that does not
implement all of them cannot be instantiated. Anything else on a concrete
network is an implementation detail (``_``-prefixed by convention) and is not
part of the public surface.
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
    buffer's latest frames but carries the *live* recurrent state held by the
    agent, which therefore is an explicit field here rather than read off the
    buffer. Batch size is 1.
    """

    s_seq: torch.Tensor  # (B, T, *obs_shape) image observation window
    a_seq: torch.Tensor  # (B, T, action_dim)
    r_seq: torch.Tensor  # (B, T, 1)
    rnn_state: torch.Tensor  # live recurrent state carried by the agent
    system_token_ids_seq: torch.Tensor  # (B, T, max_prompt_tokens)
    turn_token_ids_seq: torch.Tensor  # (B, T, max_prompt_tokens)
    reply_token_ids_seq: torch.Tensor  # (B, T, max_prompt_tokens)
    velocity_x_seq: torch.Tensor  # (B, T, 1)
    velocity_y_seq: torch.Tensor  # (B, T, 1)
    velocity_z_seq: torch.Tensor  # (B, T, 1)
    episode_return_seq: torch.Tensor  # (B, T, 1)
    pass_mark_seq: torch.Tensor  # (B, T, 1)
    remaining_return_seq: torch.Tensor  # (B, T, 1)
    global_step_seq: torch.Tensor  # (B, T, 1)
    episode_step_seq: torch.Tensor  # (B, T, 1)
    health_seq: torch.Tensor  # (B, T, 1)
    subtask_activations_seq: torch.Tensor  # (B, T, *subtask_shape)
    subtask_age_seq: torch.Tensor  # (B, T, 1)


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
    """Structured return value of a network's ``infer_and_compute_loss``."""

    infer_result: InferResult
    loss_result: LossResult
    et_info: EligibilityTraceInfo


class NetworkInterface(nn.Module, abc.ABC):
    """Abstract base for all policy/value networks.

    The contract is the abstract methods below. A subclass missing any of
    them raises ``TypeError`` on instantiation. ``nn.Module`` is mixed in so
    concrete networks keep full PyTorch behaviour (``parameters()``, ``.to()``,
    ``state_dict()`` …); ``ABCMeta`` derives from ``type`` so there is no
    metaclass conflict.
    """

    # 高レベル方策のサブタスクの活性を読むネットワークは、1ステップ分を保存する形を宣言する。
    # エージェントはこれでバッファの欄の大きさを決め、``to_stored_subtask`` を通した
    # ものを入れる。読まないネットワークは何も持たない
    subtask_shape: tuple[int, int] = (0, 0)

    @abc.abstractmethod
    def init_state(self) -> torch.Tensor:
        """Initial recurrent state the agent carries between steps."""

    @abc.abstractmethod
    def stored_image_shape(self) -> tuple[int, ...]:
        """Shape of the per-step image tensor the replay buffer stores, which
        is the raw observation's or that of a representation the network
        pre-encodes it into (see ``to_stored_image``)."""

    @abc.abstractmethod
    def to_stored_image(self, image: torch.Tensor) -> torch.Tensor:
        """The stored form of one raw observation image, matching
        ``stored_image_shape``."""

    @abc.abstractmethod
    def to_stored_subtask(self, activations: torch.Tensor) -> torch.Tensor:
        """高レベル方策の1ステップ分のサブタスクの活性を、``subtask_shape`` の保存形に
        したもの。"""

    @abc.abstractmethod
    def observe_scalar_obs(
        self,
        velocity_x: float,
        velocity_y: float,
        velocity_z: float,
        episode_return: float,
        pass_mark: float,
        remaining_return: float,
        global_step: float,
        episode_step: float,
        health: float,
    ) -> None:
        """This tick's raw scalar observations, handed over before ``infer``
        so a network can track their running statistics."""

    @abc.abstractmethod
    def tokenize(self, text: str) -> list[int]:
        """Token ids of a prompt string for the replay buffer (empty for
        non-VLM networks)."""

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
