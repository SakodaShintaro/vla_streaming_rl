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

import numpy as np
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


@dataclass(frozen=True)
class HighLevelPolicyOutput:
    """高レベル方策がいま持っている返答。返答を書いたステップから次を書くまで同じ
    ものを返し、``age`` だけが進む。高レベル方策を持たないネットワークでは空。"""

    # (*subtask_shape)。返答の <subtask> 区間の活性を、1ステップに読む幅へ均したもの
    activations: torch.Tensor
    # 何ステップ前に書いたか。書いたステップで 0
    age: int
    text: str
    # 書いたときの会話と、この返答。描画用
    exchange: list[dict]
    # この返答が判定した、前の返答のサブタスクの達成度（yes の確率）。前のサブタスクが
    # ないか、返答に判定がなければ None
    achieved: float | None
    input_tokens: int
    output_tokens: int
    msec: float


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

    # A chain-of-thought stream is opt-in: a network that runs one declares the
    # shape of what it hands out per step, and the agents size the replay
    # buffer's chain field from it and store what ``advance_high_level`` returns
    # verbatim. Everything else contributes no tokens.
    subtask_shape: tuple[int, int] = (0, 0)

    def advance_high_level(
        self, episode_started: bool, window: ReplayBufferData
    ) -> HighLevelPolicyOutput:
        """このステップの高レベル方策の返答。高レベル方策を持たないネットワークでは空
        （``networks/actor_critic_with_action_value.py`` を参照）。``window`` はバッファの
        最新の行で、このステップが最後。プロンプトをそこから読む高レベル方策のため。"""
        del episode_started, window
        return HighLevelPolicyOutput(
            activations=torch.zeros(self.subtask_shape),
            age=0,
            text="",
            exchange=[],
            achieved=None,
            input_tokens=0,
            output_tokens=0,
            msec=0.0,
        )

    def render_panels(self, reply: HighLevelPolicyOutput) -> dict[str, np.ndarray]:
        """Named RGB panels this network contributes to the render strip, drawn
        from this step's ``reply``. The agents pass these through verbatim, so
        the stable-panel contract of :class:`agents.base.StepResult` applies:
        the same keys with the same shapes on every step of a run."""
        del reply
        return {}

    def render_texts(self, reply: HighLevelPolicyOutput) -> dict[str, str]:
        """Named free-form text this network contributes to the episode log,
        the readable counterpart of ``render_panels``."""
        del reply
        return {}

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
