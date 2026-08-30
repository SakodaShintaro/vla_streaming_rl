# SPDX-License-Identifier: MIT
import re
from pathlib import Path

import cv2
import gymnasium as gym
import hydra
import numpy as np

CAR_RACING_PROMPT = "You control the red car in CarRacing-v3 (top-down). Stay on the gray road and avoid going onto the green grass; hug the road center when possible."


CAR_RACING_ACTION_SPEC = "steer=<value>, accel=<value> where each <value> is a float in [-1, 1]"


def _car_racing_parse_action(action_text: str) -> tuple[np.ndarray, bool]:
    pattern = r"(?:t\d+:\s*)?steer=([+-]?\d*\.?\d+),\s*accel=([+-]?\d*\.?\d+)"
    matches = re.findall(pattern, action_text)
    action_array = np.zeros((len(matches), 2), dtype=np.float32)
    for i in range(len(matches)):
        action_array[i, 0] = np.clip(float(matches[i][0]), -1.0, 1.0)
        action_array[i, 1] = np.clip(float(matches[i][1]), -1.0, 1.0)
    return action_array, len(matches) > 0


# Animal-AI's native action is MultiDiscrete([3, 3]): one move (noop / forward
# / back) and one rotation (noop / right / left) per tick. The env exposes it as
# Box(-1, 1, shape=(2,)) and discretizes back with a +/-1/3 dead-zone, so each
# named action maps onto the extreme Box value that survives that dead-zone.
# An action is a move letter followed by a rotation letter -- FN walks straight
# ahead, NL turns on the spot, FR does both. The letters say what they mean, and
# the position says which half they belong to, so the N the two halves share
# reads unambiguously.
_ANIMALAI_MOVE = {"F": 1.0, "B": -1.0, "N": 0.0}
_ANIMALAI_ROTATE = {"R": 1.0, "L": -1.0, "N": 0.0}
ANIMALAI_ACTION_CHOICES = [
    move + rotation for move in _ANIMALAI_MOVE for rotation in _ANIMALAI_ROTATE
]

ANIMALAI_PROMPT = (
    "You control an animal in a 3D arena, seen from its own point of view. "
    "Touching a green sphere scores points and ends the episode, and a larger "
    "sphere scores more. Touching a yellow sphere scores points and the episode "
    "continues. Touching a red sphere loses points and ends the episode. "
    "Entering a red zone ends the episode immediately. An orange zone drains "
    "your health, and your health also drains as time passes. "
    "What this arena asks of you follows as `Task:`. "
    "Action space: two letters, a move and a rotation applied on the same tick. "
    "The move is F (walk forward), B (walk backward) or N (stand still). "
    "The rotation is R (turn right), L (turn left) or N (no turn). "
    "So FN walks straight ahead, NL turns left on the spot, FR walks while "
    "turning right, and NN does nothing. "
    "Bring whatever you are heading for to the center of your view before you "
    "walk forward: turn on the spot (NR or NL) until it is centered, and only "
    "then move. "
    "Keep your speed down -- the third velocity component reported below should "
    "stay at about 10 or less, so stand still (NN) for a tick whenever it climbs "
    "past that. "
)
ANIMALAI_ACTION_SPEC = (
    f"two letters -- one move out of {{{', '.join(_ANIMALAI_MOVE)}}} followed by one "
    f"rotation out of {{{', '.join(_ANIMALAI_ROTATE)}}}, with no other text"
)


def _animalai_parse_action(action_text: str) -> tuple[np.ndarray, bool]:
    """Decode the move/rotation letter pair into the Box action that the env
    discretizes back into Animal-AI's native MultiDiscrete([3, 3]) pair.

    Anything but one of the nine pairs is a format violation and is reported as
    such rather than repaired here.
    """
    code = action_text.strip().upper()
    if code not in ANIMALAI_ACTION_CHOICES:
        return np.zeros((0, 2), dtype=np.float32), False
    action_array = np.array(
        [[_ANIMALAI_MOVE[code[0]], _ANIMALAI_ROTATE[code[1]]]], dtype=np.float32
    )
    return action_array, True


def _animalai_format_action(action: np.ndarray) -> str:
    """The move/rotation letter pair a Box action discretizes to, for feeding a
    taken action back to a reader that was taught the letters."""
    move = np.asarray(action, dtype=np.float32).ravel()[0]
    rotation = np.asarray(action, dtype=np.float32).ravel()[1]
    move_code = "F" if move > 1.0 / 3.0 else ("B" if move < -1.0 / 3.0 else "N")
    rotation_code = "R" if rotation > 1.0 / 3.0 else ("L" if rotation < -1.0 / 3.0 else "N")
    return move_code + rotation_code


def _format_action_vector(action: np.ndarray) -> str:
    """Fallback rendering for envs with no symbolic action language."""
    return " ".join(f"{value:+.2f}" for value in np.asarray(action, dtype=np.float32).ravel())


def make_animalai_env(
    resolution: int,
    mode: str,
    train_variant: str,
    steps_per_stage: int,
    advance_success_rate: float,
    binary_path: str,
    continuous_action: bool,
    topdown_camera: bool,
    topdown_resolution: int,
    end_at_pass_mark: bool,
) -> gym.Env:
    from vla_streaming_rl.envs.animalai_env import AnimalAIEnv, build_selector

    selector = build_selector(
        mode=mode,
        train_variant=train_variant,
        steps_per_stage=steps_per_stage,
        advance_success_rate=advance_success_rate,
        seed=0,
    )
    return AnimalAIEnv(
        resolution=resolution,
        seed=0,
        base_port=5005,
        binary_path=binary_path,
        continuous_action=continuous_action,
        topdown_camera=topdown_camera,
        topdown_resolution=topdown_resolution,
        end_at_pass_mark=end_at_pass_mark,
        selector=selector,
    )


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
        REPEAT = 4
        env = gym.make(env_id, render_mode="rgb_array")
        env = env.env  # Unwrap the original TimeLimit wrapper
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
        env = StepCountObsWrapper(env)
        env = EpisodeReturnObsWrapper(env)
        env = PromptWrapper(env, CAR_RACING_PROMPT)
        env = LanguageObsWrapper(env)
        env.unwrapped.eval_range = 20
        env.unwrapped.parse_action_text = _car_racing_parse_action
        env.unwrapped.format_action = _format_action_vector
        env.unwrapped.action_spec = CAR_RACING_ACTION_SPEC
        # Continuous: there is no finite set of action texts to enumerate.
        env.unwrapped.action_choices = []
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
        env.unwrapped.format_action = _format_action_vector
        return env

    elif env_id == "AnimalAI-v0":
        env = hydra.utils.instantiate(env_factory)
        # The env composes its own task prompt (it appends live scalars), so the
        # action encoding is injected by replacing the task text it starts from.
        env.unwrapped.prompt = ANIMALAI_PROMPT
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = DictObsWrapper(env)
        env = TransposeAndNormalizeObs(env)
        env = VelocityObsWrapper(env)
        env = PassMarkObsWrapper(env)
        env = HealthObsWrapper(env)
        env = StepCountObsWrapper(env)
        env = EpisodeReturnObsWrapper(env)
        env = RemainingReturnObsWrapper(env)
        env = LanguageObsWrapper(env)
        env.unwrapped.eval_range = 20
        env.unwrapped.parse_action_text = _animalai_parse_action
        env.unwrapped.format_action = _animalai_format_action
        env.unwrapped.action_spec = ANIMALAI_ACTION_SPEC
        env.unwrapped.action_choices = ANIMALAI_ACTION_CHOICES
        return env

    else:
        raise ValueError(f"Unsupported environment: {env_id}")


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


class HealthObsWrapper(gym.Wrapper):
    """Expose Animal-AI's agent health as a scalar observation.

    Health is what actually bounds an AAI episode: it decays at a rate set by
    the arena's `t`, refills whenever a reward is collected, and the episode
    ends when it reaches 0. It is passed through raw, as the env reports it.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        spaces = dict(env.observation_space.spaces)
        spaces["health"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)

    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        obs["health"] = np.array([info["health"]], dtype=np.float32)
        return obs, info

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs["health"] = np.array([info["health"]], dtype=np.float32)
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


class RemainingReturnObsWrapper(gym.ObservationWrapper):
    """Expose how much return the episode still owes its pass mark.

    ``pass_mark - episode_return`` is what "am I about to clear this arena?"
    reduces to; the two terms are already observations, but the difference is
    the one the policy and the value head actually act on. Envs without a pass
    mark report 0 through ``ZeroScalarObsWrapper`` instead.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        spaces = dict(env.observation_space.spaces)
        spaces["remaining_return"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        remaining = obs["pass_mark"] - obs["episode_return"]
        return {**obs, "remaining_return": remaining.astype(np.float32)}


class StepCountInfoWrapper(gym.Wrapper):
    """Count the step numbers an env does not report itself.

    AnimalAI and CARLA publish their own counters, so they need only
    ``StepCountObsWrapper``; CarRacing does not, so it is counted here instead.
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


class StepCountObsWrapper(gym.Wrapper):
    """Expose the env's step counters as scalar observations.

    ``global_step`` tells the model *when* an experience happened, so the
    sequence it reads carries an order across the whole run; ``episode_step``
    tells it how far into the current episode it is.
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
        self.observation_space = gym.spaces.Dict(spaces)

    def _add_step_obs(self, obs: dict, info: dict) -> dict:
        obs["global_step"] = np.array([info["global_step"]], dtype=np.float32)
        obs["episode_step"] = np.array([info["episode_step"]], dtype=np.float32)
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
        spaces["health"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        spaces["remaining_return"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(spaces)
        self._zero_velocity = np.zeros(3, dtype=np.float32)
        self._zero_pass_mark = np.zeros(1, dtype=np.float32)
        self._zero_health = np.zeros(1, dtype=np.float32)
        self._zero_remaining_return = np.zeros(1, dtype=np.float32)

    def observation(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            **obs,
            "velocity": self._zero_velocity,
            "pass_mark": self._zero_pass_mark,
            "health": self._zero_health,
            "remaining_return": self._zero_remaining_return,
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
