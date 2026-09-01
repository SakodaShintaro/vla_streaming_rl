# SPDX-License-Identifier: MIT
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from vla_streaming_rl.agents.prompt import PromptBuilder


@dataclass
class StepResult:
    """Structured return value of an agent's ``select_action`` / ``step``.

    Replaces the old free-form ``info`` dict, which mixed three unrelated
    concerns and forced the trainer to know each agent's internal keys.
    The concerns are now split:

    - ``action``: the env action to execute this tick.
    - ``metrics``: flat scalar telemetry, logged verbatim to wandb.
    - ``panels``: named RGB image panels (HxWx3) for the render strip. The
      trainer concatenates whatever panels are present, in insertion order,
      so an agent can contribute extra panels (e.g. the next-frame prediction
      or a bird's-eye trajectory view) without the trainer knowing about them.
      Stable-panel contract: an agent must emit the SAME set of panel keys with
      the SAME shapes on every step of a run (use a blank placeholder when the
      content does not exist yet, e.g. before the first prediction). The render
      frames are encoded into a single video, which requires a constant size.
    - ``texts``: named free-form text, written to the episode's ``texts.tsv``.
      What a panel shows as pixels this keeps as characters, so a chain of
      thought can be read back and searched after the run. Every agent puts the
      language it composed under ``"episode_text"`` -- what holds for the whole
      episode, and what the trainer captions the render with -- which is empty
      when the agent composes no language at all.
    """

    action: np.ndarray
    metrics: dict[str, float]
    panels: dict[str, np.ndarray]
    texts: dict[str, str]


class Agent(ABC):
    """The contract the trainer drives: ``step`` on every tick while learning,
    ``select_action`` when acting without learning.

    The agent grid is orthogonal: the learning rule (off-policy / on-policy /
    streaming) is the class, which ``agent_type`` names, and the network it
    optimizes is a constructor argument.

    ``prompt_builder`` is the third axis: the env publishes state, the agent
    turns it into the language its policy reads."""

    def __init__(
        self, horizon: int, reset_on_episode_end: bool, prompt_builder: PromptBuilder
    ) -> None:
        self.horizon = int(horizon)
        self.reset_on_episode_end = bool(reset_on_episode_end)
        self.prompt_builder = prompt_builder

    @abstractmethod
    def select_action(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult: ...

    @abstractmethod
    def step(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult:
        """Act on this tick and learn from it, the way this learning mode learns."""

    @abstractmethod
    def on_episode_end(self, score: float) -> dict: ...

    @abstractmethod
    def optimizer_state_dict(self) -> dict:
        """Optimizer states to checkpoint. The trainer stores and restores this
        verbatim, so an agent is free to hold one optimizer or several."""

    @abstractmethod
    def load_optimizer_state_dict(self, state: dict) -> None: ...

    @abstractmethod
    def _preprocess(self, obs: dict[str, Any], info: dict): ...

    @abstractmethod
    def _to_env_action(self, net_action): ...
