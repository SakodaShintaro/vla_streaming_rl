# SPDX-License-Identifier: MIT
import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import gymnasium as gym
import hydra
import minigrid
import numpy as np

from vla_streaming_rl.envs.color_panel_env import ColorPanelEnv
from vla_streaming_rl.envs.letter_tracing_env import LetterTracingEnv
from vla_streaming_rl.envs.stl10_panel_env import STL10PanelEnv
from vla_streaming_rl.envs.tracking_square_env import TrackingSquareEnv

REPEAT = 4

# bench2drive220 is the fixed route set for the CARLA env; random-route mode is
# no longer used, so the route XML is wired in here instead of via config/CLI.
# Absolute path resolved against the repo root (three levels up from this file).
CARLA_ROUTE_XML = str(
    Path(__file__).parents[2] / "external/Bench2Drive/leaderboard/data/bench2drive220.xml"
)


CAR_RACING_PROMPT = (
    "You control the red car in CarRacing-v3 (top-down). Stay on the gray road and avoid going onto the green grass; hug the road center when possible. "
    "Action space: steer [-1, +1] where -1 is full left and +1 is full right; accel [-1, +1] where positive is gas and negative is brake. "
)

MINIGRID_PROMPT = "Navigate the MiniGrid memory corridor. Remember the object seen at the start and choose the matching one at the end. "

BABYAI_GOTO_LOCAL_PROMPT_PREFIX = (
    "Navigate a small MiniGrid room to reach the target object. Mission: "
)

HOPPER_PROMPT = "Control a 2D one-legged hopper. Keep it balanced and hopping forward as fast as possible without falling."


def _color_panel_parse_action(action_text: str) -> tuple[np.ndarray, bool]:
    pattern = r"(?:t\d+:\s*)?dx=([+-]?\d*\.?\d+),\s*dy=([+-]?\d*\.?\d+),\s*button=([+-]?\d*\.?\d+)"
    matches = re.findall(pattern, action_text)
    action_array = np.zeros((len(matches), 3), dtype=np.float32)
    for i in range(len(matches)):
        action_array[i, 0] = np.clip(float(matches[i][0]), -1.0, 1.0)
        action_array[i, 1] = np.clip(float(matches[i][1]), -1.0, 1.0)
        action_array[i, 2] = np.clip(float(matches[i][2]), -1.0, 1.0)
    return action_array, len(matches) > 0


def _car_racing_parse_action(action_text: str) -> tuple[np.ndarray, bool]:
    pattern = r"(?:t\d+:\s*)?steer=([+-]?\d*\.?\d+),\s*accel=([+-]?\d*\.?\d+)"
    matches = re.findall(pattern, action_text)
    action_array = np.zeros((len(matches), 2), dtype=np.float32)
    for i in range(len(matches)):
        action_array[i, 0] = np.clip(float(matches[i][0]), -1.0, 1.0)
        action_array[i, 1] = np.clip(float(matches[i][1]), -1.0, 1.0)
    return action_array, len(matches) > 0


def make_animalai_env() -> gym.Env:
    """Hydra `_target_` factory for the raw AnimalAI env (no wrappers).

    v5 has no auto-download; place the unzipped Linux build at
    ~/animalai_env/Linux/animalAI.x86_64.
    """
    from vla_streaming_rl.envs.animalai_env import AnimalAIEnv

    # resolution must be divisible by 8 (Wan VAE encode/decode is stride-8).
    return AnimalAIEnv(
        resolution=96,
        seed=0,
        base_port=5005,
    )


def make_libero_env(
    task_suite_name: str,
    task_id: int,
    resolution: int,
    settle_steps: int,
    horizon: int,
    perturb_robot_joint_std: float,
) -> gym.Env:
    """Hydra `_target_` factory for the raw LIBERO env (no wrappers).

    The wrist camera, proprioceptive state and language instruction are
    published in the info dict (see ``LiberoEnv``); ``make_env`` only adds the
    image-observation wrappers shared with the other pixel envs.
    """
    from vla_streaming_rl.envs.libero_env import LiberoEnv

    return LiberoEnv(
        task_suite_name=task_suite_name,
        task_id=task_id,
        resolution=resolution,
        settle_steps=settle_steps,
        horizon=horizon,
        perturb_robot_joint_std=perturb_robot_joint_std,
        seed=0,
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
    if env_id == "BabyAI-GoToLocal-v0":
        env = gym.make(env_id, render_mode="rgb_array", highlight=False)
        env = minigrid.wrappers.RGBImgObsWrapper(env, tile_size=32)
        env = minigrid.wrappers.ImgObsWrapper(env)
        env = ReduceActionSpaceWrapper(env, n_actions=3)
        env = DiscreteToContinuousWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TransposeAndNormalizeObs(env)
        env = ZeroObsOnDoneWrapper(env)
        env = MissionPromptWrapper(env, BABYAI_GOTO_LOCAL_PROMPT_PREFIX)
        env.unwrapped.eval_range = 100
        return env

    elif env_id == "MiniGrid-MemoryS9-v0":
        env = gym.make(env_id, agent_view_size=3, render_mode="rgb_array")
        env = minigrid.wrappers.RGBImgPartialObsWrapper(env, tile_size=32)
        env = minigrid.wrappers.ImgObsWrapper(env)
        env = ReduceActionSpaceWrapper(env, n_actions=3)
        env = DiscreteToContinuousWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TransposeAndNormalizeObs(env)
        env = ZeroObsOnDoneWrapper(env)
        env = PromptWrapper(env, MINIGRID_PROMPT)
        env.unwrapped.eval_range = 100
        return env

    elif env_id == "CarRacing-v3":
        env = gym.make(env_id, render_mode="rgb_array")
        env = env.env  # Unwrap the original TimeLimit wrapper
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000 * REPEAT)
        env = CarRacingRewardFixWrapper(env)
        env = CarRacingActionWrapper(env)
        env = ActionRepeatWrapper(env, repeat=REPEAT)
        env = AverageRewardEarlyStopWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TransposeAndNormalizeObs(env)
        env = ZeroObsOnDoneWrapper(env)
        env = PromptWrapper(env, CAR_RACING_PROMPT)
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
        env.unwrapped.eval_range = 220
        return env

    elif env_id == "AnimalAI-v0":
        env = hydra.utils.instantiate(env_factory)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TransposeAndNormalizeObs(env)
        env.unwrapped.eval_range = 20
        return env

    elif env_id == "LetterTracing-v0":
        env = LetterTracingEnv(render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TransposeAndNormalizeObs(env)
        env.unwrapped.eval_range = 20
        return env

    elif env_id == "ColorPanel-v0":
        env = ColorPanelEnv(render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TransposeAndNormalizeObs(env)
        env.unwrapped.eval_range = 20
        env.unwrapped.parse_action_text = _color_panel_parse_action
        return env

    elif env_id == "STL10Panel-v0":
        data_dir = Path.home() / "data/stl-10/train"
        env = STL10PanelEnv(render_mode="rgb_array", data_dir=data_dir)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TransposeAndNormalizeObs(env)
        env.unwrapped.eval_range = 20
        env.unwrapped.parse_action_text = _color_panel_parse_action
        return env

    elif env_id == "TrackingSquare-v0":
        env = TrackingSquareEnv(render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TransposeAndNormalizeObs(env)
        env.unwrapped.eval_range = 40
        env.unwrapped.parse_action_text = _color_panel_parse_action
        return env

    elif env_id == "LIBERO-v0":
        # The env already publishes task_prompt (the language instruction) plus
        # the wrist camera and proprio in info, so no PromptWrapper is needed.
        env = hydra.utils.instantiate(env_factory)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TransposeAndNormalizeObs(env)
        env = ZeroObsOnDoneWrapper(env)
        env.unwrapped.eval_range = 100
        return env

    elif env_id == "Hopper-v5":
        env = gym.make(env_id, render_mode="rgb_array")
        env = PixelObsWrapper(env, height=96, width=96)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TransposeAndNormalizeObs(env)
        env = ZeroObsOnDoneWrapper(env)
        env = PromptWrapper(env, HOPPER_PROMPT)
        env.unwrapped.eval_range = 20
        return env

    else:
        raise ValueError(f"Unsupported environment: {env_id}")


class DiscreteToContinuousWrapper(gym.Wrapper):
    """
    Convert discrete action space to continuous.
    Example: Discrete(4) -> Box(-1.0, 1.0, (4,))
    The max action is selected.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(env.action_space.n,), dtype=np.float32
        )

    def step(self, action: np.ndarray) -> tuple:
        # Convert continuous action to discrete by selecting the max action
        discrete_action = np.argmax(action)
        return self.env.step(discrete_action)


class ReduceActionSpaceWrapper(gym.Wrapper):
    """
    Reduce discrete action space to only relevant actions.
    For MiniGrid Memory environments, reduce to 3 actions: turn left, turn right, move forward.
    """

    def __init__(self, env: gym.Env, n_actions: int) -> None:
        super().__init__(env)
        self.n_actions = n_actions
        self.action_space = gym.spaces.Discrete(n_actions)

    def step(self, action: np.ndarray | int) -> tuple:
        # Convert action if it's an array
        if isinstance(action, np.ndarray):
            action = action[0] if len(action) > 0 else action
        return self.env.step(action)


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
        h, w = env.observation_space.shape[0:2]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3, h, w), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        o = obs.astype(np.float32) / 255.0
        # Convert from (H, W, C) to (C, H, W)
        o = np.transpose(o, (2, 0, 1))
        return o


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


class PixelObsWrapper(gym.ObservationWrapper):
    """Replace state-vector observation with rendered RGB pixels."""

    def __init__(self, env: gym.Env, height: int, width: int) -> None:
        super().__init__(env)
        self.height = height
        self.width = width
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        img = self.env.render()
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return img


class ZeroObsOnDoneWrapper(gym.ObservationWrapper):
    """
    Zero out observations when episode is terminated or truncated.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Zero out observation if episode is done
        if terminated or truncated:
            obs = np.zeros_like(obs)

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


class MissionPromptWrapper(gym.Wrapper):
    """Inject a prompt built from the BabyAI mission string into info dict."""

    def __init__(self, env: gym.Env, prefix: str) -> None:
        super().__init__(env)
        self.prefix = prefix
        self._prompt = ""

    def reset(self, **kwargs) -> tuple:
        with redirect_stdout(io.StringIO()):
            obs, info = self.env.reset(**kwargs)
        self._prompt = self.prefix + self.unwrapped.mission
        info["task_prompt"] = self._prompt
        return obs, info

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["task_prompt"] = self._prompt
        return obs, reward, terminated, truncated, info
