# SPDX-License-Identifier: MIT
"""Gymnasium adapter for the RoboMemArena robotic-memory benchmark.

RoboMemArena ships a LIBERO fork (robosuite / MuJoCo) plus 26 BDDL memory tasks.
Its tasks are driven exactly like ``LiberoEnv``'s: a ``robosuite``
``OffScreenRenderEnv`` built from a BDDL file, using the legacy gym API (``reset``
returns only ``obs``; ``step`` returns a 4-tuple). This wrapper exposes it through
the Gymnasium API the trainer expects, with the same multi-modal observation
contract as ``LiberoEnv``:
``{"image": agentview, "wrist_image": wrist, "proprio": state}``.

Unlike the LIBERO benchmark suite, RoboMemArena success is not robosuite's
built-in ``done`` but a goal monitor parsed from the BDDL ``(:goal ...)`` clause;
that check (and the language prompt) is reused verbatim from the fork's
``eval_common`` so the learning signal matches the benchmark scoring.

The fork is installed nowhere: its source lives under
``external/RoboMemArena`` and is put on ``sys.path`` here (the same treatment
``carla_bootstrap`` gives ``external/Bench2Drive``). Every runtime dependency
(``robosuite`` / MuJoCo / ``bddl``) is already resolved for the LIBERO env, so no
extra package is needed. A fork-local LIBERO config is written under
``~/.robomemarena_libero`` so importing the fork's ``libero`` neither prompts on
stdin nor clobbers the pip-``libero`` config that ``LiberoEnv`` relies on.
"""

import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RMA_ROOT = _REPO_ROOT / "external" / "RoboMemArena" / "evaluation_benchmark"
_FORK_ROOT = _RMA_ROOT / "libero_fork"
_SCRIPTS_ROOT = _RMA_ROOT / "scripts"

INFO_KEY_TASK_PROMPT = "task_prompt"

_ROT180 = (slice(None, None, -1), slice(None, None, -1))

_OBS_AGENTVIEW = "agentview_image"
_OBS_WRIST = "robot0_eye_in_hand_image"
_OBS_EEF_POS = "robot0_eef_pos"
_OBS_EEF_QUAT = "robot0_eef_quat"
_OBS_GRIPPER_QPOS = "robot0_gripper_qpos"

_ACTION_DIM = 7
_PROPRIO_DIM = 8


def _bootstrap_runtime() -> None:
    """Prepare the fork's LIBERO runtime before ``import libero``.

    Writes a fork-local config (so the fork does not prompt on stdin or touch the
    pip-``libero`` config) and puts the fork and its eval scripts on ``sys.path``.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    config_dir = os.path.expanduser("~/.robomemarena_libero")
    config_file = os.path.join(config_dir, "config.yaml")
    if not os.path.exists(config_file):
        benchmark_root = str(_FORK_ROOT / "libero")
        path_dict = {
            "benchmark_root": benchmark_root,
            "bddl_files": os.path.join(benchmark_root, "bddl_files"),
            "init_states": os.path.join(benchmark_root, "init_files"),
            "datasets": os.path.join(benchmark_root, "..", "datasets"),
            "assets": os.path.join(benchmark_root, "assets"),
        }
        os.makedirs(config_dir, exist_ok=True)
        with open(config_file, "w") as f:
            yaml.dump(path_dict, f)
    os.environ["LIBERO_CONFIG_PATH"] = config_dir

    for path in (str(_FORK_ROOT), str(_SCRIPTS_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


_bootstrap_runtime()

from eval_common import (  # noqa: E402
    _build_goal_monitor_dict,
    _resolve_bddl_path,
    _resolve_task_id,
    check_goal_success,
    get_prompt,
)
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from robosuite.utils.transform_utils import quat2axisangle  # noqa: E402


class RoboMemArenaEnv(gym.Env):
    """Single-camera-observation Gymnasium view over one RoboMemArena task.

    Args:
        task_id: RoboMemArena task selector — an int / ``"taskN"`` (1..26) or a
            direct ``.bddl`` path, resolved by the fork's ``_resolve_bddl_path``.
        resolution: square camera height/width in pixels.
        horizon: max controllable steps before the episode is truncated.
        settle_steps: zero-action steps applied right after ``reset`` to let the
            physics settle before control starts.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self, task_id: str | int, resolution: int, horizon: int, settle_steps: int
    ) -> None:
        super().__init__()
        self._resolution = resolution
        self._horizon = horizon
        self._settle_steps = settle_steps

        self._bddl_path = _resolve_bddl_path(task_id)
        _, task_key = _resolve_task_id(task_id)
        self._task_prompt = get_prompt(task_key, self._bddl_path.stem)
        self._monitor_dict = _build_goal_monitor_dict(self._bddl_path)

        self._env = OffScreenRenderEnv(
            bddl_file_name=str(self._bddl_path),
            camera_heights=resolution,
            camera_widths=resolution,
            ignore_done=True,
            reward_shaping=True,
            control_freq=20,
            initialization_noise=None,
        )

        self._last_agentview = np.zeros((resolution, resolution, 3), dtype=np.uint8)
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

        self._step_count = 0

    def _extract(self, obs: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        agentview = np.ascontiguousarray(obs[_OBS_AGENTVIEW][_ROT180])
        wrist = np.ascontiguousarray(obs[_OBS_WRIST][_ROT180])
        proprio = np.concatenate(
            [
                obs[_OBS_EEF_POS],
                quat2axisangle(obs[_OBS_EEF_QUAT]),
                obs[_OBS_GRIPPER_QPOS],
            ]
        ).astype(np.float32)
        self._last_agentview = agentview
        self.last_wrist_image = wrist
        self.last_proprio = proprio
        self.last_instruction = self._task_prompt
        observation = {"image": agentview, "wrist_image": wrist, "proprio": proprio}
        info = {INFO_KEY_TASK_PROMPT: self._task_prompt}
        return observation, info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._env.seed(seed)
        obs = self._env.reset()
        zero_action = np.zeros(_ACTION_DIM, dtype=np.float32)
        for _ in range(self._settle_steps):
            obs, _, _, _ = self._env.step(zero_action)
        self._step_count = 0
        return self._extract(obs)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        obs, _, _, _ = self._env.step(np.asarray(action, dtype=np.float32))

        success = check_goal_success(self._env, self._monitor_dict)
        terminated = bool(success)
        truncated = (not terminated) and (self._step_count >= self._horizon)

        observation, info = self._extract(obs)
        info["is_success"] = float(success)
        return observation, float(success), terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._last_agentview

    def close(self) -> None:
        self._env.close()
