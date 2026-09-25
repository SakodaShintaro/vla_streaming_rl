# SPDX-License-Identifier: MIT
"""Animal-AI Gymnasium environment.

Wraps Animal-AI v5 (which exposes a Unity ML-Agents BehaviorSpec interface)
as a single-agent gym.Env. The action is Box(-1, 1, shape=(2,)) for parity with
the project's other environments (CARLA, GUI games):
    dim 0: forward (+1) / back (-1)
    dim 1: rotate right (+1) / left (-1)
How Unity consumes it depends on `continuous_action`:
  - False: the official MultiDiscrete([3, 3]) branches, reached by
           discretizing with a +/-1/3 dead-zone (0=noop, 1=forward/right,
           2=back/left). Works with any Animal-AI binary.
  - True:  the value is passed through as a throttle / turn rate. This needs a
           binary rebuilt from animal-ai-unity with the hybrid action spec
           (2 continuous actions alongside the branches); +/-1 there reproduces
           the discrete branches exactly, so it is a strict superset.

Everything about talking to Unity lives in `AnimalAIEnv`. The only thing that
differs between the ways we run Animal-AI is *which arena each episode loads*,
so that lives in an `ArenaSelector` (see `animalai_curriculum.py`, whose
docstring describes the modes); the render-panel drawing lives in
`animalai_render.py`.

`end_at_pass_mark` ends an episode as soon as a collected reward carries the
return to the arena's pass mark, which Unity itself never does (see `step`).
"""

import random
from pathlib import Path

import gymnasium as gym
import numpy as np
from animalai import AnimalAIEnvironment
from gymnasium import spaces
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)

from vla_streaming_rl.envs.animalai_curriculum import Arena, ArenaSelector, parse_arena
from vla_streaming_rl.envs.animalai_render import (
    RENDER_SIZE_PX,
    draw_header,
    fit_square,
    render_progress,
    render_topdown,
)

# Environment parameter the rebuilt binary reads to pick continuous over
# discrete actions (see TrainingAgent.ReadAction in animal-ai-unity).
CONTINUOUS_ACTIONS_KEY = "continuousActions"


def _aai_environment_class(extra_args: list[str]) -> type:
    """`AnimalAIEnvironment` subclass that actually passes `extra_args` to Unity.

    `AnimalAIEnvironment.__init__` rebuilds the player's command line from
    `executable_args` and discards whatever the caller handed it as
    `additional_args`, so the switches the rebuilt binary reads at startup
    (`--topDownCamera`, `--topDownResolution`) never arrive. `executable_args`
    is a staticmethod, so the arguments are bound to a class rather than an
    instance.
    """

    class _AnimalAIEnvironmentWithArgs(AnimalAIEnvironment):
        @staticmethod
        def executable_args(*args) -> list[str]:
            return AnimalAIEnvironment.executable_args(*args) + extra_args

    return _AnimalAIEnvironmentWithArgs


# Shaping coefficients.
GOAL_BONUS = 0.5
RAMPS_COEF = 0.01
BACK_MOVE_COEF = 0.001
PASS_MARK_BONUS = 1.0


def _to_discrete(value: float) -> int:
    """Map a continuous control in [-1, 1] to AAI's {0=noop, 1, 2}."""
    if value >= 1.0 / 3.0:
        return 1
    elif value <= -1.0 / 3.0:
        return 2
    return 0


class AnimalAIEnv(gym.Env):
    """Animal-AI environment with continuous Box action space.

    `selector` decides which arena each episode loads; see
    `animalai_curriculum.py`.
    """

    _PHYSICS_STEP_SEC = 0.02
    _DECISION_PERIOD = 5

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 30,
        "decision_fps": 1.0 / (_DECISION_PERIOD * _PHYSICS_STEP_SEC),
    }

    def __init__(
        self,
        resolution: int,
        seed: int,
        base_port: int,
        binary_path: str,
        continuous_action: bool,
        topdown_camera: bool,
        topdown_resolution: int,
        colored_walls: bool,
        end_at_pass_mark: bool,
        selector: ArenaSelector,
    ):
        super().__init__()
        self.continuous_action = continuous_action
        self.topdown_camera = topdown_camera
        self.topdown_resolution = topdown_resolution
        self.colored_walls = colored_walls
        self.end_at_pass_mark = end_at_pass_mark
        self.selector = selector

        self.binary_path = str(Path(binary_path).expanduser())
        self.resolution = resolution
        self.seed_value = seed
        # Add jitter so parallel envs don't fight over the same socket.
        self.base_port = base_port + random.randint(0, 1000)
        self.render_mode = "rgb_array"

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(resolution, resolution, 3), dtype=np.uint8
        )

        self._aai: AnimalAIEnvironment | None = None
        self._behavior_name: str | None = None
        # How many continuous actions the binary accepts: 0 for an official
        # build, 2 for one rebuilt with the hybrid action spec. Read off the
        # behavior spec at launch.
        self._continuous_size = 0
        self._latest_image: np.ndarray | None = None
        # The overhead camera's view, on the binaries that have one. It is not
        # part of `observation_space`: `render` draws it, the policy never sees it.
        self._latest_topdown_image: np.ndarray | None = None
        self.global_step = 0
        self.episode_step = 0
        self.arena_name: str = ""
        self.pass_mark: float = 0.0
        self._arena: Arena | None = None
        self._arena_items: list[dict] = []
        self._episode_return = 0.0
        # (x, y, z) agent position in arena coords; populated each step from
        # the AAI vector observation. None before the first reset.
        self._agent_xyz: tuple[float, float, float] | None = None
        # (pitch, yaw, roll) agent orientation in degrees, from the AAI vector
        # observation's Unity euler angles. None before the first reset.
        self._agent_rotation: tuple[float, float, float] | None = None
        # (vx, vy, vz) agent velocity from the AAI vector observation.
        self._agent_velocity: np.ndarray = np.zeros(3, dtype=np.float32)
        # Agent health from the AAI vector observation: it decays at a rate set
        # by the arena's `t`, refills whenever a reward is collected, and the
        # episode ends when it hits 0 -- the real "time left" of the episode.
        self._agent_health: float = 0.0

    @property
    def is_exhausted(self) -> bool:
        """Whether the selector has served its whole playlist, which is what ends
        a run over a finite arena order ("sequential", "eval"). The modes that
        never run out answer False forever and leave the run to `step_limit`."""
        return self.selector.is_exhausted

    def set_global_step(self, global_step: int) -> None:
        """Resume hook: restore the step counter the curriculum schedules on."""
        self.global_step = int(global_step)

    def get_curriculum_state(self) -> dict:
        return self.selector.state()

    def set_curriculum_state(
        self, arena_attempts: dict, arena_successes: dict, progress: dict
    ) -> None:
        self.selector.load_state(arena_attempts, arena_successes, progress)

    def _ensure_started(self):
        if self._aai is not None:
            return
        # Queued before the constructor's first reset, which is when the
        # channel's messages reach Unity.
        parameters_channel = EnvironmentParametersChannel()
        parameters_channel.set_float_parameter(
            CONTINUOUS_ACTIONS_KEY, float(self.continuous_action)
        )
        environment_class = _aai_environment_class(
            [
                "--topDownCamera",
                str(int(self.topdown_camera)),
                "--topDownResolution",
                str(self.topdown_resolution),
                "--coloredWalls",
                str(int(self.colored_walls)),
            ]
        )
        self._aai = environment_class(
            side_channels=[parameters_channel],
            file_name=self.binary_path,
            arenas_configurations=str(self.selector.arenas[0].path),
            seed=self.seed_value,
            play=False,
            useCamera=True,
            resolution=self.resolution,
            useRayCasts=False,
            # One decision per 5 Academy steps (= 5 physics steps of 0.02 s),
            # so an arena's `t` costs t*5 physics steps and an episode advances
            # in 0.1 s of simulated time. This is the only one of these knobs
            # that changes what the agent experiences.
            decisionPeriod=self._DECISION_PERIOD,
            # Run the simulation as fast as it will go: `timescale` matches the
            # paper's training scripts, and no frame rate cap. Neither changes
            # the agent's experience -- the physics step is fixed at 0.02 s and
            # decisions are counted in Academy steps, not frames -- so these
            # only buy wall-clock speed (~9x over the default timescale=1).
            timescale=300,
            targetFrameRate=-1,
            # Left at the AAI default (real time drives frame duration) as the
            # paper's scripts had it. Setting it would only change how the same
            # physics steps are split across frames.
            captureFrameRate=0,
            # `no_graphics=True` (the alternative for headless) disables the
            # renderer entirely and produces a solid-color image, which is
            # unusable for vision policies.
            no_graphics=False,
            base_port=self.base_port,
            inference=False,
            use_YAML=True,
        )
        self._behavior_name = next(iter(self._aai.behavior_specs.keys()))
        spec = self._aai.behavior_specs[self._behavior_name]
        # ML-Agents sorts an agent's sensors by name, so the observations arrive
        # as ["CameraSensor", "TopDownCameraSensor", "VectorSensor"] -- first
        # person, overhead, scalars. The overhead camera is there only on a
        # rebuilt binary running with topdown_camera, which leaves the scalars
        # last either way.
        self._observation_count = len(spec.observation_specs)
        assert self._observation_count in (2, 3), self._observation_count
        self._vector_size = spec.observation_specs[self._observation_count - 1].shape[0]
        assert self._vector_size == 10, (
            f"{self.binary_path} emits a {self._vector_size}-element vector observation; "
            "the agent's rotation needs a binary rebuilt from animal-ai-unity."
        )
        self._continuous_size = spec.action_spec.continuous_size
        assert self._continuous_size == 2 or not self.continuous_action, (
            f"{self.binary_path} exposes {self._continuous_size} continuous actions; "
            "continuous_action=True needs a binary rebuilt from animal-ai-unity."
        )

    def _decode_obs(self, obs_chw_float: np.ndarray) -> np.ndarray:
        # AAI emits float32 in [0, 1] with shape (3, H, W).
        return (obs_chw_float.transpose(1, 2, 0) * 255.0).astype(np.uint8)

    def _read_observation(self, steps) -> None:
        # Vector obs layout (useCamera=True, useRayCasts=False): the last
        # observation is [health, vx, vy, vz, x, y, z, pitch, yaw, roll].
        vec = steps.obs[self._observation_count - 1][0]
        self._latest_image = self._decode_obs(steps.obs[0][0])
        # Watched in `render`, never handed to the policy.
        self._latest_topdown_image = (
            self._decode_obs(steps.obs[1][0]) if self._observation_count == 3 else None
        )
        self._agent_health = float(vec[0])
        self._agent_xyz = (float(vec[4]), float(vec[5]), float(vec[6]))
        self._agent_rotation = (float(vec[7]), float(vec[8]), float(vec[9]))
        self._agent_velocity = np.array(
            [float(vec[1]), float(vec[2]), float(vec[3])], dtype=np.float32
        )

    def _shape_reward(self, reward: float, episode_done: bool) -> float:
        """Reaching a goal is worth more than the arena says, climbing is encouraged,
        walking backwards is discouraged, and an episode ends with a bonus for having
        cleared the arena's pass mark or a penalty for not."""
        velocity = self._agent_velocity
        if reward > 0.1:
            reward += GOAL_BONUS
        if velocity[1] > 0.01:
            reward += float(velocity[1]) * RAMPS_COEF
        if velocity[2] < 0:
            reward += float(velocity[2]) * BACK_MOVE_COEF
        if episode_done:
            cleared = self._episode_return >= self.pass_mark
            reward += PASS_MARK_BONUS if cleared else -PASS_MARK_BONUS
        return reward

    def _build_info(self, shaped_reward: float) -> dict:
        info = {
            "shaped_reward": shaped_reward,
            "arena_name": self.arena_name,
            "arena_yaml": str(self._arena.path),
            "pass_mark": self.pass_mark,
            "global_step": self.global_step,
            "episode_step": self.episode_step,
            "health": self._agent_health,
            "velocity": self._agent_velocity,
            "agent_xyz": self._agent_xyz,
            "agent_rotation": self._agent_rotation,
        }
        info.update(self.selector.info(self.global_step))
        return info

    def reset(self, seed: int | None, options: dict | None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._ensure_started()

        # `options={"arena_stem": name}` pins one arena instead of asking the
        # selector (scripts/collect_probe_data.py repeats a single arena).
        forced_name = options["arena_stem"] if options is not None else None
        self._arena = (
            self.selector.arena_by_name(forced_name)
            if forced_name is not None
            else self.selector.next_arena(self.global_step)
        )

        self.arena_name = self._arena.name
        self.pass_mark, self._arena_items = parse_arena(self._arena.path)
        self._aai.reset(arenas_configurations=str(self._arena.path))
        self.episode_step = 0
        self._episode_return = 0.0

        decision_steps, _ = self._aai.get_steps(self._behavior_name)
        self._read_observation(decision_steps)
        return self._latest_image, self._build_info(0.0)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.global_step += 1
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0).reshape(1, 2)
        # Both halves go out every step; which one Unity obeys was fixed at
        # launch by CONTINUOUS_ACTIONS_KEY. The slice is empty for an official
        # binary, which has no continuous actions to fill.
        action_tuple = ActionTuple(
            continuous=a[:, : self._continuous_size],
            discrete=np.array(
                [[_to_discrete(float(a[0, 0])), _to_discrete(float(a[0, 1]))]], dtype=np.int32
            ),
        )
        self._aai.set_actions(self._behavior_name, action_tuple)
        self._aai.step()

        decision_steps, terminal_steps = self._aai.get_steps(self._behavior_name)
        episode_over = len(terminal_steps) > 0
        # `interrupted` is AAI running the agent's health down to 0, i.e. a time
        # limit rather than a real terminal (goal reached / death zone).
        interrupted = episode_over and bool(terminal_steps.interrupted[0])
        steps = terminal_steps if episode_over else decision_steps
        reward = float(steps.reward[0])
        self._read_observation(steps)

        self.episode_step += 1
        self._episode_return += reward
        terminated = episode_over and not interrupted
        truncated = interrupted

        # Unity never ends an arena on the pass mark, so without this an agent
        # that has earned it keeps bleeding 1/t per step. The reward gate keeps
        # a negative pass mark from being cleared by the first step's -1/t.
        if self.end_at_pass_mark and not episode_over and reward > 0.0:
            terminated = terminated or self._episode_return >= self.pass_mark

        done = terminated or truncated
        shaped_reward = self._shape_reward(reward, done)

        success = self._episode_return >= self.pass_mark
        if done:
            # Before _build_info, so the selector reports post-episode progress.
            self.selector.on_episode_end(self._arena, success)
        info = self._build_info(shaped_reward)
        if done:
            info["success"] = bool(success)
        return self._latest_image, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """This episode's arena from above, beside the arena set's coverage.

        The arena pane is the overhead camera when the binary renders one, and
        otherwise a schematic drawn from the arena yaml -- the camera shows where
        everything is, the schematic only where everything started.
        """
        if self.render_mode != "rgb_array":
            return None
        attempts, successes = self.selector.arena_record(self._arena)
        header_lines = [
            f"{self.arena_name}  {successes}/{attempts}",
            self.selector.status(self.global_step),
            f"step:{self.global_step}  health:{self._agent_health:.2f}",
        ]
        arena = (
            fit_square(self._latest_topdown_image, RENDER_SIZE_PX)
            if self._latest_topdown_image is not None
            else render_topdown(self._arena_items, self._agent_xyz)
        )
        topdown = np.vstack([draw_header(header_lines, RENDER_SIZE_PX), arena])
        progress = render_progress(self.selector.progress_by_group())
        padding = np.full(
            (topdown.shape[0] - progress.shape[0], progress.shape[1], 3), 240, dtype=np.uint8
        )
        return np.hstack([topdown, np.vstack([progress, padding])])

    def close(self):
        if self._aai is not None:
            self._aai.close()
            self._aai = None


if __name__ == "__main__":
    from vla_streaming_rl.envs.animalai_curriculum import StagedSelector

    env = AnimalAIEnv(
        resolution=96,
        seed=0,
        base_port=5005,
        binary_path="~/animalai_env/4.3.2_alpha2/Linux/animalAI.x86_64",
        continuous_action=False,
        topdown_camera=False,
        topdown_resolution=96,
        colored_walls=True,
        end_at_pass_mark=True,
        selector=StagedSelector(variant="01", steps_per_stage=2_000_000, seed=0),
    )
    for episode in range(8):
        obs, info = env.reset(seed=episode, options=None)
        print(f"ep={episode} arena={info['arena_name']} stage={info['stage']}")
        total_reward = 0.0
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total_reward += reward
            if terminated or truncated:
                break
        print(f"  return={total_reward:.4f} success={info.get('success')}")
    env.close()
