# SPDX-License-Identifier: MIT
"""Gymnasium adapter for the LIBERO manipulation benchmark.

LIBERO ships a robosuite (MuJoCo) simulator that uses the legacy gym API
(``reset`` returns only ``obs``; ``step`` returns a 4-tuple with a single
``done`` flag). This wrapper exposes it through the Gymnasium API the trainer
expects.

Observation contract (transitional): the trainer / replay buffer / renderer in
this repo only handle a single RGB frame ``(H, W, 3)``, so ``step`` / ``reset``
return the agentview camera image as the observation. Everything a richer VLA
policy (e.g. pi0.5) needs — the wrist camera, proprioceptive state and the
language instruction — is smuggled through the ``info`` dict under fixed keys.
This is deliberately a stop-gap until the observation pipeline is generalized to
carry multi-modal observations natively.
"""

import importlib.util
import os
from typing import Any

import gymnasium as gym
import numpy as np
import yaml

# Keys under which the extra (non-image-observation) modalities are published in
# the info dict. A policy that needs them reads these explicitly.
INFO_KEY_TASK_PROMPT = "task_prompt"
INFO_KEY_WRIST_IMAGE = "wrist_image"
INFO_KEY_PROPRIO = "proprio"

# The LIBERO pi0.5 checkpoint was trained with the HuggingFace-LIBERO camera
# convention: every camera frame is rotated 180° (both height and width flipped).
# This must match exactly or the policy sees a mirrored world and reaches in the
# wrong direction. See lerobot ``LiberoEnvProcessor._process_observation``
# (``torch.flip(img, dims=[2, 3])``). A plain vertical flip is NOT equivalent —
# it leaves the image horizontally mirrored relative to training.
_ROT180 = (slice(None, None, -1), slice(None, None, -1))

# robosuite obs keys.
_OBS_AGENTVIEW = "agentview_image"
_OBS_WRIST = "robot0_eye_in_hand_image"
_OBS_EEF_POS = "robot0_eef_pos"
_OBS_EEF_QUAT = "robot0_eef_quat"
_OBS_GRIPPER_QPOS = "robot0_gripper_qpos"

# Action dimensionality of the default LIBERO OSC_POSE controller
# (end-effector delta pose + gripper).
_ACTION_DIM = 7

# Sentinel for "sample a fresh task from the suite on every reset".
RANDOM_TASK_ID = -1


def _bootstrap_libero_runtime() -> None:
    """Make ``import libero.libero`` work non-interactively and headlessly.

    LIBERO prompts on stdin the first time ``libero.libero`` is imported (to
    choose a dataset path) and writes ``~/.libero/config.yaml``. Online RL needs
    no demonstration datasets — only the packaged bddl/init/assets — so we
    pre-write that config pointing at the installed package, suppressing the
    prompt. We also select the EGL backend so MuJoCo renders without a display.
    """
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    config_file = os.path.expanduser("~/.libero/config.yaml")
    if os.path.exists(config_file):
        return

    # Locate the installed ``libero.libero`` package without importing it
    # (importing it is exactly what triggers the stdin prompt).
    spec = importlib.util.find_spec("libero")
    package_root = os.path.join(os.path.dirname(spec.origin), "libero")
    path_dict = {
        "benchmark_root": package_root,
        "bddl_files": os.path.join(package_root, "bddl_files"),
        "init_states": os.path.join(package_root, "init_files"),
        "datasets": os.path.join(package_root, "..", "datasets"),
        "assets": os.path.join(package_root, "assets"),
    }
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, "w") as f:
        yaml.dump(path_dict, f)


# LIBERO / robosuite are imported at module top level (not lazily inside the
# methods) because nothing imports this module at import time: the sole importer
# is ``make_libero_env``, which already defers ``import libero_env`` until the
# LIBERO env is actually requested. So the "repo stays importable without LIBERO"
# guarantee lives there, and these imports only run once the user opts in.
# ``_bootstrap_libero_runtime`` must run *before* ``import libero.libero`` (the
# import is what triggers LIBERO's stdin config prompt), so it is called here.
_bootstrap_libero_runtime()
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from robosuite.utils.transform_utils import quat2axisangle  # noqa: E402


class LiberoEnv(gym.Env):
    """Single-camera-observation Gymnasium view over one LIBERO task suite.

    Args:
        task_suite_name: one of the LIBERO suites
            (``libero_spatial`` / ``libero_object`` / ``libero_goal`` /
            ``libero_90`` / ``libero_10``).
        task_id: index of the task within the suite, or ``RANDOM_TASK_ID`` to
            draw a new task on every reset.
        resolution: square camera height/width in pixels (kept divisible by the
            VAE stride upstream).
        settle_steps: number of zero-action steps applied right after
            ``set_init_state`` to let the physics settle before control starts.
        horizon: max controllable steps before the episode is truncated.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task_suite_name: str,
        task_id: int,
        resolution: int,
        settle_steps: int,
        horizon: int,
        perturb_robot_joint_std: float,
        seed: int,
    ) -> None:
        super().__init__()
        self._benchmark_dict = benchmark.get_benchmark_dict()
        self._task_suite_name = task_suite_name
        self._task_suite = self._benchmark_dict[task_suite_name]()
        self._libero_bddl_root = get_libero_path("bddl_files")
        self._task_id = task_id
        self._resolution = resolution
        self._settle_steps = settle_steps
        self._horizon = horizon
        # LIBERO-PRO-style initial-state perturbation: per-arm-joint Gaussian
        # jitter (radians) applied to the robot's initial joint configuration at
        # reset. LIBERO replays a fixed pool of demonstration init states that
        # the policy has memorized, saturating the success rate; perturbing the
        # robot's starting pose breaks that memorization and gives RL headroom.
        # 0.0 reproduces the stock (unperturbed) behavior. See ``_perturb_robot``.
        self._perturb_robot_joint_std = float(perturb_robot_joint_std)

        self._np_random = np.random.default_rng(seed)

        # The robosuite env is rebuilt only when the task changes (constructing
        # it is expensive). Track the currently-loaded task id.
        self._loaded_task_id = -1
        self._env = None
        self._task_prompt = ""
        self._last_agentview = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        self.last_wrist_image = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        self.last_proprio = np.zeros(8, dtype=np.float32)
        self.last_instruction = ""

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(resolution, resolution, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(_ACTION_DIM,), dtype=np.float32
        )

        self._step_count = 0

    def _resolve_task_id(self) -> int:
        if self._task_id == RANDOM_TASK_ID:
            return int(self._np_random.integers(0, self._task_suite.n_tasks))
        return self._task_id

    def _build_env_for_task(self, task_id: int) -> None:
        task = self._task_suite.get_task(task_id)
        bddl_file = f"{self._libero_bddl_root}/{task.problem_folder}/{task.bddl_file}"
        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": self._resolution,
            "camera_widths": self._resolution,
        }
        if self._env is not None:
            self._env.close()
        self._env = OffScreenRenderEnv(**env_args)
        self._task_prompt = task.language
        self._loaded_task_id = task_id

    def _extract(self, obs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        agentview = np.ascontiguousarray(obs[_OBS_AGENTVIEW][_ROT180])
        wrist = np.ascontiguousarray(obs[_OBS_WRIST][_ROT180])
        # 8-D proprio matching the LIBERO pi0.5 checkpoint's state feature:
        # end-effector position, orientation as axis-angle, and gripper width.
        proprio = np.concatenate(
            [
                obs[_OBS_EEF_POS],
                quat2axisangle(obs[_OBS_EEF_QUAT]),
                obs[_OBS_GRIPPER_QPOS],
            ]
        ).astype(np.float32)
        self._last_agentview = agentview
        # Cached so an agent holding the env (e.g. the pi0.5 agent) can read the
        # modalities that don't fit the single-RGB observation slot, alongside
        # publishing them in info. See the module docstring's stop-gap note.
        self.last_wrist_image = wrist
        self.last_proprio = proprio
        self.last_instruction = self._task_prompt
        info = {
            INFO_KEY_TASK_PROMPT: self._task_prompt,
            INFO_KEY_WRIST_IMAGE: wrist,
            INFO_KEY_PROPRIO: proprio,
        }
        return agentview, info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        task_id = self._resolve_task_id()
        if task_id != self._loaded_task_id:
            self._build_env_for_task(task_id)

        self._env.reset()
        init_states = self._task_suite.get_task_init_states(task_id)
        init_state_id = int(self._np_random.integers(0, len(init_states)))
        obs = self._env.set_init_state(init_states[init_state_id])

        # Perturb the robot's starting joint configuration *after* the fixed
        # init state is loaded (it overwrites the robot qpos) and *before* the
        # settle steps, so the zero-action settle holds the new pose.
        self._perturb_robot()

        zero_action = np.zeros(_ACTION_DIM, dtype=np.float32)
        for _ in range(self._settle_steps):
            obs, _, _, _ = self._env.step(zero_action)

        self._step_count = 0
        return self._extract(obs)

    def _perturb_robot(self) -> None:
        """Add Gaussian jitter to the robot's initial arm-joint positions.

        No-op when ``perturb_robot_joint_std == 0``. The OSC controller derives
        its goal from the current pose on every step (delta input), so a
        zero-action settle holds the perturbed configuration rather than snapping
        back; the joint velocities are zeroed so the settle starts at rest.
        """
        if self._perturb_robot_joint_std <= 0.0:
            return
        robot = self._env.env.robots[0]
        sim = self._env.sim
        pos_idx = robot._ref_joint_pos_indexes
        noise = self._np_random.normal(0.0, self._perturb_robot_joint_std, size=len(pos_idx))
        sim.data.qpos[pos_idx] += noise
        sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
        sim.forward()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        obs, reward, done, _ = self._env.step(np.asarray(action, dtype=np.float32))

        # LIBERO reward is sparse success (reward_shaping disabled): a positive
        # reward means the task goal was satisfied this step.
        success = reward > 0.0
        terminated = bool(success)
        truncated = (not terminated) and (bool(done) or self._step_count >= self._horizon)

        observation, info = self._extract(obs)
        info["is_success"] = float(success)
        return observation, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._last_agentview

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
