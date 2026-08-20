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


def no_reward_shaping(reward: float, obs: dict, episode_done: bool) -> float:
    """The reward shaper of an environment whose reward is trained on as it comes."""
    del obs, episode_done
    return reward


class Agent(ABC):
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        for name, member in vars(Agent).items():
            if getattr(member, "__final__", False) and name in cls.__dict__:
                raise TypeError(f"{cls.__name__} may not override final method {name!r} of Agent")

    def __init__(self, learning_mode: str, horizon: int) -> None:
        assert learning_mode in ("off_policy", "on_policy", "streaming"), (
            f"Unknown learning_mode: {learning_mode!r}"
        )
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
        if self.learning_mode == "on_policy":
            return self._step_onpolicy(global_step, obs, reward, terminated, truncated, info)
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

    def _step_onpolicy(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult:
        """Learning-mode hook: the agent buffers its own on-policy rollout and
        updates from it when the rollout is full. Only agents that declare
        ``learning_mode='on_policy'`` reach this, so it is a hook rather than
        part of the abstract contract every agent has to satisfy."""
        raise NotImplementedError(f"{type(self).__name__} does not implement on_policy learning")

    def _step_streaming(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult:
        """Learning-mode hook: inference and the update share one forward pass
        on every tick. See ``_step_onpolicy`` for why this is not abstract."""
        raise NotImplementedError(f"{type(self).__name__} does not implement streaming learning")

    @abstractmethod
    def on_episode_end(self, score: float, feedback_text: str) -> dict: ...

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
