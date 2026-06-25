# SPDX-License-Identifier: MIT
from abc import ABC, abstractmethod
from typing import final

import numpy as np
from torch import nn

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

    @abstractmethod
    def select_action(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
        info: dict,
    ) -> StepResult: ...

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
    def _step_offpolicy(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
        info: dict,
    ) -> StepResult:
        train_metrics = {}
        if global_step == self.learning_starts:
            print(f"Start training at global step {global_step}.")
        if global_step >= self.learning_starts and self.rb is not None:
            data = self.rb.sample(self.batch_size)
            data.rewards = self.reward_processor.normalize(data.rewards)
            result = self.network.compute_loss(data)
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            result.loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            train_metrics = result.info
        step_result = self.select_action(
            global_step, obs, reward, terminated, truncated, task_prompt, info
        )
        step_result.metrics.update(train_metrics)
        return step_result

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
    def _preprocess(self, obs: np.ndarray, info: dict, task_prompt: str): ...

    @abstractmethod
    def _to_env_action(self, net_action): ...

    @abstractmethod
    def _panels(self, obs: np.ndarray, reward: float) -> dict: ...
