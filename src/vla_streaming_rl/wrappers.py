# SPDX-License-Identifier: MIT

import gymnasium as gym
import numpy as np

from vla_streaming_rl.image_obs import image_obs_names


def make_env(env_id: str, env_factory, result_dir, seed: int) -> gym.Env:
    """Build the training env.

    ``result_dir`` (when set) is the Hydra run dir; for the CARLA env it
    is used to derive ``eval_output_dir = result_dir / "eval"`` for the
    Bench2Drive eval artifacts.
    """
    if env_id == "CarRacing-v3":
        REPEAT = 4
        env = gym.make(env_id, render_mode="rgb_array")
        env = env.env  # Unwrap the original TimeLimit wrapper
        # CarRacing steps its physics once per rendered frame, so one decision
        # per frame is what it offers before ActionRepeatWrapper thins it out.
        env.metadata["decision_fps"] = env.metadata["render_fps"]
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000 * REPEAT)
        env = CarRacingRewardFixWrapper(env)
        env = CarRacingActionWrapper(env)
        env = ActionRepeatWrapper(env, repeat=REPEAT)
        env = AverageRewardEarlyStopWrapper(env)
        env = UnshapedRewardWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = DictObsWrapper(env)
        env = TransposeAndNormalizeObs(env)
        env = ZeroObsOnDoneWrapper(env)
        env = ZeroScalarObsWrapper(env)
        env = StepCountInfoWrapper(env)
        env = InfoToObsWrapper(env, key="global_step", shape=(1,), pop=False)
        env = InfoToObsWrapper(env, key="episode_step", shape=(1,), pop=False)
        env = EpisodeReturnObsWrapper(env)
        env.unwrapped.eval_range = 20
        return env

    if env_id == "CARLA-Leaderboard-v0":
        from vla_streaming_rl.envs.carla import make_env as make_carla_env

        return make_carla_env(env_factory, result_dir)

    assert env_id == "AnimalAI-v0", f"Unsupported environment: {env_id}"
    from vla_streaming_rl.envs.animalai import make_env as make_animalai_env

    return make_animalai_env(env_factory, seed)


class UnshapedRewardWrapper(gym.Wrapper):
    """Publish ``info["shaped_reward"]`` for an env whose reward is trained on as it
    comes, so every env hands the agent both numbers on the same channel."""

    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        info["shaped_reward"] = 0.0
        return obs, info

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["shaped_reward"] = float(reward)
        return obs, reward, terminated, truncated, info


class ActionRepeatWrapper(gym.Wrapper):
    """
    Repeat the same action for multiple steps
    """

    def __init__(self, env: gym.Env, repeat: int) -> None:
        super().__init__(env)
        self.repeat = repeat
        # Acting once per ``repeat`` frames is what the agent's own clock runs
        # at, so the rate the wrapped env publishes no longer describes it.
        self.metadata = {
            **env.metadata,
            "decision_fps": env.metadata["decision_fps"] / repeat,
        }

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
    """観測の画像（IMAGE_OBS_NAMES のうち観測にあるもの全部）を [0, 1] の (3, H, W) にする。"""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.image_names = image_obs_names(env.observation_space)
        spaces = dict(env.observation_space.spaces)
        for name in self.image_names:
            h, w = env.observation_space[name].shape[0:2]
            spaces[name] = gym.spaces.Box(low=0.0, high=1.0, shape=(3, h, w), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        images = {
            name: np.transpose(obs[name].astype(np.float32) / 255.0, (2, 0, 1))
            for name in self.image_names
        }
        return {**obs, **images}


class DictObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({"image": env.observation_space})

    def observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        return {"image": obs}


def _with_box_obs(env: gym.Env, boxes: dict[str, tuple[int, ...]]) -> gym.spaces.Dict:
    """The env's Dict observation space with one unbounded float32 Box per
    entry of ``boxes`` added."""
    spaces = dict(env.observation_space.spaces)
    for key, shape in boxes.items():
        spaces[key] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
    return gym.spaces.Dict(spaces)


class InfoToObsWrapper(gym.Wrapper):
    """Publish one ``info`` entry as a float32 observation of the given shape,
    on reset and on every step. ``pop`` moves the entry out of ``info``
    instead of copying it."""

    def __init__(self, env: gym.Env, key: str, shape: tuple[int, ...], pop: bool) -> None:
        super().__init__(env)
        self.key = key
        self.shape = shape
        self.pop = pop
        self.observation_space = _with_box_obs(env, {key: shape})

    def _attach(self, obs: dict, info: dict) -> dict:
        value = info.pop(self.key) if self.pop else info[self.key]
        obs[self.key] = np.asarray(value, dtype=np.float32).reshape(self.shape)
        return obs

    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        return self._attach(obs, info), info

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._attach(obs, info), reward, terminated, truncated, info


class EpisodeReturnObsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = _with_box_obs(env, {"episode_return": (1,)})
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


class RemainingReturnObsWrapper(gym.ObservationWrapper):
    """Expose how much return the episode still owes its pass mark.

    ``pass_mark - episode_return`` is what "am I about to clear this arena?"
    reduces to; the two terms are already observations, but the difference is
    the one the policy and the value head actually act on. Envs without a pass
    mark report 0 through ``ZeroScalarObsWrapper`` instead.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = _with_box_obs(env, {"remaining_return": (1,)})

    def observation(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        remaining = obs["pass_mark"] - obs["episode_return"]
        return {**obs, "remaining_return": remaining.astype(np.float32)}


class StepCountInfoWrapper(gym.Wrapper):
    """Count the step numbers an env does not report itself.

    AnimalAI and CARLA publish their own counters, so their ``info`` already
    carries them; CarRacing does not, so it is counted here instead.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.global_step = 0
        self.episode_step = 0

    def _add_info(self, info: dict) -> dict:
        info["global_step"] = self.global_step
        info["episode_step"] = self.episode_step
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


class ZeroScalarObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._zeros = {
            "velocity": np.zeros(3, dtype=np.float32),
            "pass_mark": np.zeros(1, dtype=np.float32),
            "health": np.zeros(1, dtype=np.float32),
            "remaining_return": np.zeros(1, dtype=np.float32),
        }
        self.observation_space = _with_box_obs(
            env, {key: value.shape for key, value in self._zeros.items()}
        )

    def observation(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {**obs, **self._zeros}


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
