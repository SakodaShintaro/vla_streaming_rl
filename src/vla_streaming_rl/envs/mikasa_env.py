# SPDX-License-Identifier: MIT
"""Gymnasium adapter for the MIKASA-Robo-VLA memory benchmark.

MIKASA-Robo-VLA ships GPU-batched ManiSkill/SAPIEN environments whose ``reset``
and ``step`` return batched CUDA tensors (``num_envs`` leading dim). This wrapper
drives one such environment with ``num_envs=1`` and exposes it through the
single-env Gymnasium API the trainer expects, with the same multi-modal
observation contract as ``LiberoEnv``:
``{"image": base camera, "wrist_image": hand camera, "proprio": state}``.

The language instruction is published in ``info`` under ``task_prompt`` and
lifted into ``obs["language"]`` downstream by ``LanguageObsWrapper``.

The package lives under ``external/MIKASA-Robo`` and is put on ``sys.path`` here
rather than installed as a dependency: its own ``pyproject`` pins
``mani_skill==3.0.0b15`` / ``gymnasium==0.29.1``, which would conflict with the
resolved ``mani-skill>=3.0.1`` / ``gymnasium 1.x`` stack. Only the source is
reused; the pins are ignored.
"""

import sys
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import torch

_MIKASA_ROOT = Path(__file__).resolve().parents[3] / "external" / "MIKASA-Robo"
if str(_MIKASA_ROOT) not in sys.path:
    sys.path.insert(0, str(_MIKASA_ROOT))

import mikasa_robo_suite.vla.memory_envs  # noqa: E402,F401

INFO_KEY_TASK_PROMPT = "task_prompt"
INFO_KEY_SUCCESS = "success"

_ACTION_DIM = 7
_PROPRIO_DIM = 8

_SENSOR_DATA = "sensor_data"
_BASE_CAMERA = "base_camera"
_HAND_CAMERA = "hand_camera"
_RGB = "rgb"
_AGENT = "agent"
_QPOS = "qpos"
_EXTRA = "extra"
_TCP_POSE = "tcp_pose"


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _scalar_bool(x: torch.Tensor) -> bool:
    return bool(x.reshape(-1)[0].item())


def _quat_wxyz_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert a SAPIEN ``wxyz`` quaternion to a 3-vector axis-angle rotation.

    Matches the LIBERO pi0.5 proprio convention (``quat2axisangle``), so the
    8-D state vector is structurally identical across the two benchmarks.
    """
    quat = quat.astype(np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    quat = quat / norm
    w = float(np.clip(quat[0], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(max(1.0 - w * w, 0.0))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = quat[1:] / sin_half
    return (axis * angle).astype(np.float32)


class MikasaEnv(gym.Env):
    """Single-env Gymnasium view over one MIKASA-Robo-VLA memory task."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_id: str,
        resolution: int,
        control_mode: str,
        reward_mode: str,
    ) -> None:
        super().__init__()
        self._resolution = resolution
        self._env = gym.make(
            env_id,
            num_envs=1,
            obs_mode="rgb",
            control_mode=control_mode,
            reward_mode=reward_mode,
            render_mode="all",
            sim_backend="gpu",
        )
        self._task_prompt = str(self._env.unwrapped.LANGUAGE_INSTRUCTION)
        self._last_render = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        self.last_wrist_image = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        self.last_proprio = np.zeros(_PROPRIO_DIM, dtype=np.float32)
        self.last_instruction = self._task_prompt

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
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(_ACTION_DIM,), dtype=np.float32
        )

    def _camera(self, obs: dict[str, Any], name: str) -> np.ndarray:
        rgb = _to_numpy(obs[_SENSOR_DATA][name][_RGB])[0]
        return cv2.resize(rgb, (self._resolution, self._resolution), interpolation=cv2.INTER_AREA)

    def _proprio(self, obs: dict[str, Any]) -> np.ndarray:
        qpos = _to_numpy(obs[_AGENT][_QPOS])[0]
        tcp = _to_numpy(obs[_EXTRA][_TCP_POSE])[0]
        pos = tcp[0:3]
        axisangle = _quat_wxyz_to_axisangle(tcp[3:7])
        gripper = qpos[-2:]
        return np.concatenate([pos, axisangle, gripper]).astype(np.float32)

    def _extract(self, obs: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        image = self._camera(obs, _BASE_CAMERA)
        wrist = self._camera(obs, _HAND_CAMERA)
        proprio = self._proprio(obs)
        self.last_wrist_image = wrist
        self.last_proprio = proprio
        self.last_instruction = self._task_prompt
        observation = {"image": image, "wrist_image": wrist, "proprio": proprio}
        info = {INFO_KEY_TASK_PROMPT: self._task_prompt}
        return observation, info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs, _ = self._env.reset(seed=seed) if seed is not None else self._env.reset()
        return self._extract(obs)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        batched = np.asarray(action, dtype=np.float32).reshape(1, _ACTION_DIM)
        obs, reward, terminated, truncated, info = self._env.step(batched)
        observation, extracted_info = self._extract(obs)
        extracted_info["is_success"] = float(_scalar_bool(info[INFO_KEY_SUCCESS]))
        return (
            observation,
            float(_to_numpy(reward).reshape(-1)[0]),
            _scalar_bool(terminated),
            _scalar_bool(truncated),
            extracted_info,
        )

    def render(self) -> np.ndarray:
        self._last_render = _to_numpy(self._env.render())[0]
        return self._last_render

    def close(self) -> None:
        self._env.close()
