# SPDX-License-Identifier: MIT
from abc import ABC, abstractmethod
from typing import final

import numpy as np
import torch

from vla_streaming_rl.agents.step_result import StepResult


class Agent(ABC):
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        for name, member in vars(Agent).items():
            if getattr(member, "__final__", False) and name in cls.__dict__:
                raise TypeError(f"{cls.__name__} may not override final method {name!r} of Agent")

    def __init__(self, learning_mode: str) -> None:
        if learning_mode not in ("off_policy", "streaming"):
            raise ValueError(f"Unknown learning_mode: {learning_mode!r}")
        self.learning_mode = learning_mode

    @final
    def step(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
        info: dict,
    ) -> StepResult:
        if self.learning_mode == "off_policy":
            return self._step_offpolicy(
                global_step, obs, reward, terminated, truncated, task_prompt, info
            )
        return self._step_streaming(
            global_step, obs, reward, terminated, truncated, task_prompt, info
        )

    @final
    @torch.no_grad()
    def select_action(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
        info: dict,
    ) -> StepResult:
        action, metrics = self._act(
            global_step, obs, reward, terminated, truncated, task_prompt, info
        )
        return StepResult(action=action, metrics=metrics, panels=self._panels(obs, reward))

    @abstractmethod
    def _step_offpolicy(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
        info: dict,
    ) -> StepResult: ...

    @abstractmethod
    def _step_streaming(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
        info: dict,
    ) -> StepResult: ...

    @abstractmethod
    def on_episode_end(self, score: float, feedback_text: str) -> dict: ...

    @abstractmethod
    def _train_offpolicy(self, global_step: int) -> dict: ...

    @abstractmethod
    def _act(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
        info: dict,
    ) -> tuple[np.ndarray, dict]: ...

    @abstractmethod
    def _preprocess(self, obs: np.ndarray, info: dict, task_prompt: str): ...

    @abstractmethod
    def _to_env_action(self, net_action): ...

    @abstractmethod
    def _panels(self, obs: np.ndarray, reward: float) -> dict: ...
