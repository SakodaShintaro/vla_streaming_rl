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
    scalar_obs_seq: torch.Tensor  # (B, T, SCALAR_OBS_DIM)、並びは SCALAR_OBS_NAMES
    # このステップの観測の画像そのもの（保存した形ではない）を IMAGE_OBS_NAMES の並びで。
    # 会話にフレームを積むネットワークが読む
    frames: list[torch.Tensor]
    # このステップのタスクのプロンプトと、フレームに添える数値のターン
    system_text: str
    turn_text: str
    # バッファを貯めている間のステップか。このあいだの行動（一様乱数や、返答に書いた行動）は
    # ネットワークが決める
    warmup: bool


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
    # このステップの行に保存しておく状態の元（``stored_state_shape`` の形）。学習はこれを
    # 読み戻して状態を作る。保存しないネットワークでは None
    stored_state: torch.Tensor | None
    # このステップにネットワークが書いた文と、それに添える1行（読んだ・書いたトークン数、
    # かかった時間など）。文を書かないネットワークや、このステップに書かなかったときは
    # どちらも None
    text: str | None
    text_status: str | None
    # 最後に書いた文が指す、このステップの行動（正規化した行動空間、(action_dim,)）。学習は
    # これを読み戻して使う。文を書かないネットワークでは None
    reply_action: torch.Tensor | None
    # その文を書いたときに実際に読んだ会話と、その文。描画用。文を書かなかったときは None
    exchange: list[dict] | None
    # 行動の候補を比べて選んだときの、その経緯の1行と、候補ごとの
    # （名前、行動の説明、行動価値、選ばれたか）。描画用。候補を比べないネットワークでは None
    selection_status: str | None
    selection_rows: list[tuple[str, str, float, bool]] | None


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

    @abc.abstractmethod
    def init_state(self) -> torch.Tensor:
        """Initial recurrent state the agent carries between steps."""

    @abc.abstractmethod
    def stored_image_shape(self) -> tuple[int, ...]:
        """Shape of the per-step image tensor the replay buffer stores, which
        is the raw observation's or that of a representation the network
        pre-encodes it into (see ``to_stored_image``)."""

    @abc.abstractmethod
    def to_stored_image(self, images: list[torch.Tensor]) -> torch.Tensor:
        """1ステップの観測の画像（IMAGE_OBS_NAMES の並び）をまとめて保存する形、
        ``stored_image_shape`` の形。"""

    @abc.abstractmethod
    def answer(self, request: str) -> str:
        """エピソードの会話の続きに ``request`` を足して、自由な文の返答を書かせたもの。
        文を書かないネットワークでは空文字列。"""

    @abc.abstractmethod
    def stored_state_shape(self) -> tuple[int, ...]:
        """行動を決めたステップの行に保存する状態の元の形。保存しないネットワークでは
        大きさ 0 の形。"""

    @abc.abstractmethod
    def observe_scalar_obs(self, scalar_obs: np.ndarray) -> None:
        """This tick's raw scalar observations (``SCALAR_OBS_NAMES`` order),
        handed over before ``infer`` so a network can track their running
        statistics."""

    @abc.abstractmethod
    def infer(self, data: InferInput) -> InferResult:
        """Single-step inference: the action to take, its value report, the
        carried RNN state and predicted next state. Batch size is 1."""

    @abc.abstractmethod
    def compute_loss(self, data: ReplayBufferData) -> LossResult:
        """Training loss over a replay batch."""

    @abc.abstractmethod
    def infer_and_compute_loss(
        self,
        data: ReplayBufferData,
        frames: list[torch.Tensor],
        system_text: str,
        turn_text: str,
    ) -> InferLossResult:
        """Combined inference + loss in one forward, sharing the encoder pass.
        ``frames``, ``system_text`` and ``turn_text`` are the newest tick's, as in
        :class:`InferInput`."""
