# SPDX-License-Identifier: MIT
import re
from pathlib import Path

import cv2
import gymnasium as gym
import hydra
import numpy as np

REPEAT = 4


CAR_RACING_PROMPT = (
    "You control the red car in CarRacing-v3 (top-down). Stay on the gray road and avoid going onto the green grass; hug the road center when possible. "
    "Action space: steer [-1, +1] where -1 is full left and +1 is full right; accel [-1, +1] where positive is gas and negative is brake. "
)


def _car_racing_parse_action(action_text: str) -> tuple[np.ndarray, bool]:
    pattern = r"(?:t\d+:\s*)?steer=([+-]?\d*\.?\d+),\s*accel=([+-]?\d*\.?\d+)"
    matches = re.findall(pattern, action_text)
    action_array = np.zeros((len(matches), 2), dtype=np.float32)
    for i in range(len(matches)):
        action_array[i, 0] = np.clip(float(matches[i][0]), -1.0, 1.0)
        action_array[i, 1] = np.clip(float(matches[i][1]), -1.0, 1.0)
    return action_array, len(matches) > 0


def make_animalai_env(
    resolution: int,
    mode: str,
    train_arena_root: str,
    steps_per_stage: int,
    revisit_temperature: float,
) -> gym.Env:
    """`mode` selects staged / success / eval; see animalai_env's module docstring."""
    from vla_streaming_rl.envs.animalai_env import AnimalAIEnv, build_selector

    selector = build_selector(
        mode=mode,
        train_arena_root=train_arena_root,
        steps_per_stage=steps_per_stage,
        revisit_temperature=revisit_temperature,
        seed=0,
    )
    return AnimalAIEnv(resolution=resolution, seed=0, base_port=5005, selector=selector)


def make_carla_env(
    route_id: str | None,
    sequence_mode: str,
    start_index: int,
    loop: bool,
    early_off_route_m: float | None,
    early_blocked_steps: int | None,
    eval_output_dir: str | None,
) -> gym.Env:
    """Hydra `_target_` factory for the raw CARLA env (no wrappers).

    The route XML is fixed to ``CARLA_ROUTE_XML`` (bench2drive220) in code.
    ``eval_output_dir`` is injected by ``make_env`` (not the user-facing
    config) and points at the Hydra run dir / "eval". weather.xml is
    auto-located next to the route XML.
    """
    # Wire up the CARLA / Bench2Drive sys.path (normally train_carla.sh's
    # PYTHONPATH), so a run needs no bash wrapper. This must run before importing
    # ``carla_leaderboard_env`` (it imports leaderboard / srunner at module
    # load); the CARLA server itself is launched lazily by the env's shared
    # ``get_client`` (carla_bootstrap.py), reused across all trials.
    from vla_streaming_rl.envs.carla_bootstrap import setup_carla_paths

    setup_carla_paths()

    from vla_streaming_rl.envs.carla_leaderboard_env import CARLALeaderboardEnv

    CARLA_ROUTE_XML = str(
        Path(__file__).parents[2] / "external/Bench2Drive/leaderboard/data/bench2drive220.xml"
    )

    return CARLALeaderboardEnv(
        route_xml=CARLA_ROUTE_XML,
        route_id=route_id,
        sequence_mode=sequence_mode,
        start_index=start_index,
        loop=loop,
        early_off_route_m=early_off_route_m,
        early_blocked_steps=early_blocked_steps,
        eval_output_dir=eval_output_dir,
    )


def make_env(env_id: str, env_factory, result_dir) -> gym.Env:
    """Build the training env.

    ``result_dir`` (when set) is the Hydra run dir; for the CARLA env it
    is used to derive ``eval_output_dir = result_dir / "eval"`` for the
    Bench2Drive eval artifacts.
    """
    if env_id == "CarRacing-v3":
        env = gym.make(env_id, render_mode="rgb_array")
        env = env.env  # Unwrap the original TimeLimit wrapper
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000 * REPEAT)
        env = CarRacingRewardFixWrapper(env)
        env = CarRacingActionWrapper(env)
        env = ActionRepeatWrapper(env, repeat=REPEAT)
        env = AverageRewardEarlyStopWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = DictObsWrapper(env)
        env = TransposeAndNormalizeObs(env)
        env = ZeroObsOnDoneWrapper(env)
        env = ZeroScalarObsWrapper(env)
        env = StepCountInfoWrapper(env, max_episode_steps=1000)
        env = StepCountObsWrapper(env)
        env = EpisodeReturnObsWrapper(env)
        env = PromptWrapper(env, CAR_RACING_PROMPT)
        env = LanguageObsWrapper(env)
        env.unwrapped.eval_range = 20
        env.unwrapped.parse_action_text = _car_racing_parse_action
        return env

    elif env_id == "CARLA-Leaderboard-v0":
        # eval_output_dir is injected here (not via the user-facing
        # env_factory config) so the user does not have to remember a
        # path that is always result_dir/eval.
        eval_output_dir = str(result_dir / "eval") if result_dir is not None else None
        env = hydra.utils.instantiate(env_factory, eval_output_dir=eval_output_dir)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = ZeroScalarObsWrapper(env)
        env = StepCountObsWrapper(env)
        env = EpisodeReturnObsWrapper(env)
        env = LanguageObsWrapper(env)
        env.unwrapped.eval_range = 220
        return env

    elif env_id == "AnimalAI-v0":
        env = hydra.utils.instantiate(env_factory)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = DictObsWrapper(env)
        env = TransposeAndNormalizeObs(env)
        env = VelocityObsWrapper(env)
        env = PassMarkObsWrapper(env)
        env = StepCountObsWrapper(env)
        env = EpisodeReturnObsWrapper(env)
        env = LanguageObsWrapper(env)
        env.unwrapped.eval_range = 20
        return env

    else:
        raise ValueError(f"Unsupported environment: {env_id}")


class ActionRepeatWrapper(gym.Wrapper):
    """
    Repeat the same action for multiple steps
    """

    def __init__(self, env: gym.Env, repeat: int) -> None:
        super().__init__(env)
        self.repeat = repeat

    def step(self, action: np.ndarray) -> tuple:
        total_reward = 0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        return obs, total_reward, terminated, truncated, info


class AverageRewardEarlyStopWrapper(gym.Wrapper):
    """
    End episode early if average reward over last some steps is too low
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.window_size = 20
        self.recent_rewards = []

    def reset(self, **kwargs) -> tuple:
        self.recent_rewards = []
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.recent_rewards.append(reward)
        self.recent_rewards = self.recent_rewards[-self.window_size :]

        if len(self.recent_rewards) >= self.window_size:
            count = sum(r < 0.0 for r in self.recent_rewards)
            if count == self.window_size:
                truncated = True

        return obs, reward, terminated, truncated, info


class TransposeAndNormalizeObs(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        h, w = env.observation_space["image"].shape[0:2]
        spaces = dict(env.observation_space.spaces)
        spaces["image"] = gym.spaces.Box(low=0.0, high=1.0, shape=(3, h, w), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        o = obs["image"].astype(np.float32) / 255.0
        o = np.transpose(o, (2, 0, 1))
        return {**obs, "image": o}


class DictObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({"image": env.observation_space})

    def observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        return {"image": obs}


class LanguageObsWrapper(gym.Wrapper):
    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        obs["language"] = info.pop("task_prompt")
        return obs, info

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs["language"] = info.pop("task_prompt")
        return obs, reward, terminated, truncated, info


class VelocityObsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        spaces = dict(env.observation_space.spaces)
        spaces["velocity"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)

    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        obs["velocity"] = info.pop("velocity")
        return obs, info

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs["velocity"] = info.pop("velocity")
        return obs, reward, terminated, truncated, info


class PassMarkObsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        spaces = dict(env.observation_space.spaces)
        spaces["pass_mark"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)

    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        obs["pass_mark"] = np.array([info["pass_mark"]], dtype=np.float32)
        return obs, info

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs["pass_mark"] = np.array([info["pass_mark"]], dtype=np.float32)
        return obs, reward, terminated, truncated, info


class EpisodeReturnObsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        spaces = dict(env.observation_space.spaces)
        spaces["episode_return"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(spaces)
        self._episode_return = 0.0

    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        self._episode_return = 0.0
        obs["episode_return"] = np.array([self._episode_return], dtype=np.float32)
        return obs, info

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_return += float(reward)
        obs["episode_return"] = np.array([self._episode_return], dtype=np.float32)
        return obs, reward, terminated, truncated, info


class StepCountInfoWrapper(gym.Wrapper):
    """Count the step numbers an env does not report itself.

    AnimalAI and CARLA publish their own counters (their episode limit is a
    property of the arena / route), so they need only ``StepCountObsWrapper``;
    CarRacing's limit is the ``TimeLimit`` set in ``make_env``, so it is
    counted here instead. ``max_episode_steps`` is in agent steps, i.e. after
    action repeat.
    """

    def __init__(self, env: gym.Env, max_episode_steps: int) -> None:
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.global_step = 0
        self.episode_step = 0

    def _add_info(self, info: dict) -> dict:
        info["global_step"] = self.global_step
        info["episode_step"] = self.episode_step
        info["remaining_step"] = max(self.max_episode_steps - self.episode_step, 0)
        return info

    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        self.episode_step = 0
        return obs, self._add_info(info)

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_step += 1
        self.global_step += 1
        return obs, reward, terminated, truncated, self._add_info(info)


class StepCountObsWrapper(gym.Wrapper):
    """Expose the env's step counters as scalar observations.

    ``global_step`` tells the model *when* an experience happened, so the
    sequence it reads carries an order across the whole run; ``episode_step``
    and ``remaining_step`` give it the finite horizon it is acting under.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        spaces = dict(env.observation_space.spaces)
        spaces["global_step"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        spaces["episode_step"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        spaces["remaining_step"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(spaces)

    def _add_step_obs(self, obs: dict, info: dict) -> dict:
        obs["global_step"] = np.array([info["global_step"]], dtype=np.float32)
        obs["episode_step"] = np.array([info["episode_step"]], dtype=np.float32)
        obs["remaining_step"] = np.array([info["remaining_step"]], dtype=np.float32)
        return obs

    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        return self._add_step_obs(obs, info), info

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._add_step_obs(obs, info), reward, terminated, truncated, info


class ZeroScalarObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        spaces = dict(env.observation_space.spaces)
        spaces["velocity"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        spaces["pass_mark"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        self._zero_velocity = np.zeros(3, dtype=np.float32)
        self._zero_pass_mark = np.zeros(1, dtype=np.float32)

    def observation(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            **obs,
            "velocity": self._zero_velocity,
            "pass_mark": self._zero_pass_mark,
        }


class CarRacingRewardFixWrapper(gym.Wrapper):
    """
    Fix CarRacing's -100 penalty for going off-track.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Fix the -100 penalty
        if reward < -30:
            reward += 100

        return obs, reward, terminated, truncated, info


class CarRacingActionWrapper(gym.ActionWrapper):
    """
    Convert 2D action space (steer, gas_or_brake) to 3D action space (steer, gas, brake).
    - steer: [-1, +1] (unchanged)
    - gas_or_brake: [-1, +1]
      - positive: gas=value, brake=0
      - negative: gas=0, brake=abs(value)
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]).astype(np.float32),
            high=np.array([+1.0, +1.0]).astype(np.float32),
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        steer = action[0]
        gas_or_brake = action[1]
        gas_or_brake *= 0.25  # scale down
        gas = np.maximum(gas_or_brake, 0.0)
        brake = np.maximum(-gas_or_brake, 0.0)
        return np.array([steer, gas, brake], dtype=np.float32)


class ResizeObs(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, shape: tuple[int, ...]) -> None:
        super().__init__(env)
        self.shape = shape
        h, w = shape[1:]  # shape is (C, H, W), so extract H, W
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(h, w, 3), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # obs is (H, W, C), resize and return (H, W, C)
        h, w = self.shape[1:]  # target height and width
        return cv2.resize(obs, (w, h), interpolation=cv2.INTER_AREA)


class ZeroObsOnDoneWrapper(gym.ObservationWrapper):
    """
    Zero out observations when episode is terminated or truncated.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def observation(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return obs

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Zero out observation if episode is done
        if terminated or truncated:
            obs = {key: np.zeros_like(value) for key, value in obs.items()}

        return obs, reward, terminated, truncated, info


class PromptWrapper(gym.Wrapper):
    """Inject a prompt string into info dict for gym environments that don't natively provide one."""

    def __init__(self, env: gym.Env, prompt: str) -> None:
        super().__init__(env)
        self.prompt = prompt

    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        info["task_prompt"] = self.prompt
        return obs, info

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["task_prompt"] = self.prompt
        return obs, reward, terminated, truncated, info
