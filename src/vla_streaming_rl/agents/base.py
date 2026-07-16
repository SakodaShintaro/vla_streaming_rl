# SPDX-License-Identifier: MIT
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, final

import numpy as np
from torch import nn


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
    """

    action: np.ndarray
    metrics: dict[str, float]
    panels: dict[str, np.ndarray]


class Agent(ABC):
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        for name, member in vars(Agent).items():
            if getattr(member, "__final__", False) and name in cls.__dict__:
                raise TypeError(f"{cls.__name__} may not override final method {name!r} of Agent")

    def __init__(self, learning_mode: str, horizon: int) -> None:
        if learning_mode not in ("off_policy", "streaming"):
            raise ValueError(f"Unknown learning_mode: {learning_mode!r}")
        self.learning_mode = learning_mode
        self.horizon = int(horizon)

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

    @final
    def step(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult:
        if self.learning_mode == "off_policy":
            return self._step_offpolicy(global_step, obs, reward, terminated, truncated, info)
        return self._step_streaming(global_step, obs, reward, terminated, truncated, info)

    @final
    def _step_offpolicy(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult:
        train_metrics = {}
        if global_step == self.learning_starts:
            print(f"Start training at global step {global_step}.")
        if (
            global_step >= self.learning_starts
            and global_step % self.horizon == 0
            and self.rb is not None
            and self.rb.num_stored() >= self.batch_size + self.rb.seq_len
        ):
            data = self.rb.sample(self.batch_size)
            data.rewards = self.reward_processor.normalize(data.rewards)
            data.velocities = self.velocity_processor.normalize(data.velocities)
            result = self.network.compute_loss(data)
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            result.loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            train_metrics = result.info
        step_result = self.select_action(global_step, obs, reward, terminated, truncated, info)
        step_result.metrics.update(train_metrics)
        return step_result

    @abstractmethod
    def _step_streaming(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult: ...

    @abstractmethod
    def on_episode_end(self, score: float, feedback_text: str) -> dict: ...

    @abstractmethod
    def _preprocess(self, obs: dict[str, Any], info: dict): ...

    @abstractmethod
    def _to_env_action(self, net_action): ...

    @abstractmethod
    def _panels(self, obs: np.ndarray, reward: float) -> dict: ...
