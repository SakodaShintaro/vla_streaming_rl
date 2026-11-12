# SPDX-License-Identifier: MIT
"""Shared network interface: structured result types and the abstract base class.

Every policy/value network exposes the forward passes — ``infer``,
``compute_replay_loss`` and ``compute_streaming_loss`` — returning the structured
types defined here, plus the observation-side hooks the agents drive
(``stored_image_shape`` / ``represent`` / ``remember`` …).
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

    Unlike ``compute_replay_loss`` — which reads a replay batch
    (:class:`ReplayBufferData`) — inference assembles its window from the
    buffer's latest frames. Batch size is 1.
    """

    s_seq: torch.Tensor  # (B, T, *obs_shape) image observation window
    a_seq: torch.Tensor  # (B, T, action_dim)
    r_seq: torch.Tensor  # (B, T, 1)
    scalar_obs_seq: torch.Tensor  # (B, T, SCALAR_OBS_DIM)、並びは SCALAR_OBS_NAMES
    # バッファを貯めている間のステップか。このあいだは一様乱数の行動を返す
    warmup: bool


@dataclass
class InferResult:
    """Structured return value of a network's ``infer``."""

    action: torch.Tensor  # (B, horizon, action_dim)
    value_report: dict[str, float]  # value head diagnostics (incl. "value")
    features: torch.Tensor


@dataclass
class LossResult:
    """Structured return value of a network's ``compute_replay_loss`` / ``compute_streaming_loss``.

    ``info`` stays a plain dict on purpose: it is a dynamic ``name -> float``
    telemetry map whose keys vary per policy-head / predictor configuration and
    is flattened verbatim into wandb metrics.
    """

    loss: torch.Tensor
    info: dict  # scalar name -> float
    td_error: torch.Tensor  # (B,) 窓ごとの TD 誤差。リプレイの優先度の更新に使う


class NetworkInterface(nn.Module, abc.ABC):
    """Abstract base for all policy/value networks.

    The contract is the abstract methods below. A subclass missing any of
    them raises ``TypeError`` on instantiation. ``nn.Module`` is mixed in so
    concrete networks keep full PyTorch behaviour (``parameters()``, ``.to()``,
    ``state_dict()`` …); ``ABCMeta`` derives from ``type`` so there is no
    metaclass conflict.
    """

    @abc.abstractmethod
    def stored_image_shape(self) -> tuple[int, ...]:
        """バッファに保存する1ステップの表現の形。"""

    @abc.abstractmethod
    def represent(
        self,
        images: list[torch.Tensor],
        scalar_obs: np.ndarray,
        prev_action: np.ndarray,
        reward: float,
        instruction: str,
        reset_state: bool,
    ) -> torch.Tensor:
        """1ステップの観測の画像（IMAGE_OBS_NAMES の並び）と方策が読む指示の文を VLM に読ませた
        もの。LoRA 付きの GDN の層がなければ ``stored_image_shape`` の形の表現でそのまま保存し、あればLoRA 付きの GDN の層の
        入力で、``remember`` に通したものを保存する。エージェントは毎ステップ順に呼び、
        ``reset_state`` はエピソードの流れを空にするステップ（エピソードの始め）で真。
        ``prev_action`` はこの観測に至った正規化済みの行動 (action_dim,)、``reward`` は
        そのとき受け取った報酬。"""

    @abc.abstractmethod
    def reset_memory(self) -> None:
        """LoRA 付きの GDN の層の再帰の状態と、RTRL で運んでいるその感度を空にする。"""

    @abc.abstractmethod
    def remember(self, step_tokens: torch.Tensor) -> torch.Tensor:
        """このティックの LoRA 付きの GDN の層の入力を層に読ませて、再帰の状態とその感度を
        1ティック進め、画像のトークンの出力を返す。保存するのはその出力。"""

    @abc.abstractmethod
    def infer(self, data: InferInput) -> InferResult:
        """Single-step inference: the action to take, its value report and the
        state the heads read. Batch size is 1."""

    @abc.abstractmethod
    def compute_replay_loss(self, data: ReplayBufferData) -> LossResult:
        """Training loss over a replay batch."""

    @abc.abstractmethod
    def compute_streaming_loss(self, image: torch.Tensor, data: ReplayBufferData) -> LossResult:
        """最新の窓 ``data`` での損失。``image`` は前のティックで ``remember`` が返した計算グラフ
        付きの出力。"""
