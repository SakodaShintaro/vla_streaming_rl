# SPDX-License-Identifier: MIT
"""Gymnasium adapter for the RoboMME memory-manipulation benchmark.

RoboMME ships its tasks as GPU-batched ManiSkill/SAPIEN environments registered
under a task id (e.g. ``PickXtimes``) whose ``reset`` / ``step`` return batched
CUDA tensors (``num_envs`` leading dim). Its ``BenchmarkEnvBuilder`` stacks
demonstration/oracle wrappers on top for imitation learning; those replay
scripted trajectories and are unusable for online RL, so this wrapper bypasses
them and drives one raw ``gym.make(task_id)`` env with ``num_envs=1``, exposing
it through the single-env Gymnasium API with the same multi-modal observation
contract as ``LiberoEnv``: ``{"image", "wrist_image", "proprio"}``.

The language instruction is published in ``info`` under ``task_prompt`` and
lifted into ``obs["language"]`` downstream by ``LanguageObsWrapper``.

``robomme`` is installed editable from the ``external/robomme_benchmark``
submodule (whose ``pyproject`` was relaxed off the ManiSkill fork / ``torch==2.9.1``
pins to ``mani-skill>=3.0.1`` / ``torch>=2.7`` so it resolves against this repo's
stack; its tasks register cleanly against the official ManiSkill). Importing
``robomme.robomme_env`` registers the RoboMME task envs.

RoboMME's ``compute_dense_reward`` is intentionally zeroed (the benchmark scores
success rate, not reward), so the learning reward here is the sparse task-success
signal from ``evaluate`` (published in ``info["success"]``); richer shaping is
left to the agent, mirroring ``LiberoEnv``.
"""

from typing import Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import robomme.robomme_env  # noqa: F401
import torch
from robomme.robomme_env.utils import task_goal

# Keys under which the language instruction is published in the info dict.
INFO_KEY_TASK_PROMPT = "task_prompt"
INFO_KEY_SUCCESS = "success"

# ManiSkill obs keys for the RoboMME panda_wristcam robot.
_SENSOR_DATA = "sensor_data"
_FRONT_CAMERA = "base_camera"
_WRIST_CAMERA = "hand_camera"
_RGB = "rgb"
_AGENT = "agent"
_QPOS = "qpos"

# Proprioception is the full arm+gripper joint configuration (7 arm + 2 gripper),
# the natural state for RoboMME's joint-space control.
_PROPRIO_DIM = 9

# Absolute joint-position control (7 arm joints + 1 gripper). This matches the
# RoboMME benchmark's ``action_space="joint_angle"`` (its ``BenchmarkEnvBuilder``
# builds the base env with ``control_mode="pd_joint_pos"``), so a policy trained
# here speaks the same action interface the official evaluation expects.
_CONTROL_MODE = "pd_joint_pos"


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _scalar_bool(x: torch.Tensor) -> bool:
    return bool(torch.as_tensor(x).reshape(-1)[0].item())


class RobommeEnv(gym.Env):
    """Single-env Gymnasium view over one RoboMME memory task."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, task_id: str, resolution: int, horizon: int, seed: int) -> None:
        super().__init__()
        self._task_id = task_id
        self._horizon = horizon

        # The task instance is fixed at construction: RoboMME derives the task
        # difficulty and the number of pick-and-place repetitions from ``seed``
        # inside ``__init__``, so a fresh instance is one fixed variant (object
        # poses still re-randomize on every ``reset``).
        self._env = gym.make(
            task_id,
            num_envs=1,
            obs_mode="rgb",
            control_mode=_CONTROL_MODE,
            render_mode="rgb_array",
            sensor_configs={"width": resolution, "height": resolution},
            seed=seed,
        )

        self._task_prompt = ""
        self._last_front = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        self.last_wrist_image = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        self.last_proprio = np.zeros(_PROPRIO_DIM, dtype=np.float32)
        self.last_instruction = ""

        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=255, shape=(resolution, resolution, 3), dtype=np.uint8
                ),
                "wrist_image": gym.spaces.Box(
                    low=0, high=255, shape=(resolution, resolution, 3), dtype=np.uint8
                ),
                "proprio": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(_PROPRIO_DIM,), dtype=np.float32
                ),
            }
        )
        # Expose the raw controller action space (absolute joint-position limits
        # for the 7 arm joints + gripper) rather than a fixed [-1, 1] box, so the
        # bounds a policy sees are the true joint limits.
        self.action_space = self._env.action_space
        self._action_dim = int(np.prod(self.action_space.shape))

        self._step_count = 0

    def _extract(self, obs: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        front = np.ascontiguousarray(_to_numpy(obs[_SENSOR_DATA][_FRONT_CAMERA][_RGB])[0])
        wrist = np.ascontiguousarray(_to_numpy(obs[_SENSOR_DATA][_WRIST_CAMERA][_RGB])[0])
        proprio = _to_numpy(obs[_AGENT][_QPOS])[0].astype(np.float32)

        self._last_front = front
        # Cached so an agent holding the env can read the modalities alongside
        # publishing them, matching ``LiberoEnv``.
        self.last_wrist_image = wrist
        self.last_proprio = proprio
        self.last_instruction = self._task_prompt

        observation = {"image": front, "wrist_image": wrist, "proprio": proprio}
        info = {INFO_KEY_TASK_PROMPT: self._task_prompt}
        return observation, info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs, _ = self._env.reset(seed=seed)
        # The language goal reads task attributes (repeat count, target colour)
        # that are only populated once the scene is loaded, so it is resolved
        # after ``reset``. ``get_language_goal`` returns phrasing variants.
        self._task_prompt = task_goal.get_language_goal(self._env, self._task_id)[0]
        self._step_count = 0
        return self._extract(obs)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        batched = np.asarray(action, dtype=np.float32).reshape(1, self._action_dim)
        obs, _, _, _, raw_info = self._env.step(batched)

        success = _scalar_bool(raw_info[INFO_KEY_SUCCESS])
        terminated = success
        truncated = (not terminated) and (self._step_count >= self._horizon)

        observation, info = self._extract(obs)
        info["is_success"] = float(success)
        return observation, float(success), terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._last_front

    def close(self) -> None:
        self._env.close()
