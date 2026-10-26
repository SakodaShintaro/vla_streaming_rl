# SPDX-License-Identifier: MIT
"""Animal-AI: the env, its arena curriculum and its render panels, and the
factories that build them for the trainer."""

import gymnasium as gym
import hydra

from vla_streaming_rl.wrappers import (
    EpisodeReturnObsWrapper,
    InfoToObsWrapper,
    RemainingReturnObsWrapper,
    TransposeAndNormalizeObs,
)


def make_animalai_env(
    resolution: int,
    mode: str,
    train_variant: str,
    train_levels: list[str],
    eval_levels: list[str],
    eval_variants: list[str],
    steps_per_stage: int,
    binary_path: str,
    continuous_action: bool,
    topdown_camera: bool,
    topdown_resolution: int,
    topdown_observation: bool,
    colored_walls: bool,
    end_at_pass_mark: bool,
    blink_dark_steps_increment: int,
    seed: int,
) -> gym.Env:
    from vla_streaming_rl.envs.animalai.curriculum import build_selector
    from vla_streaming_rl.envs.animalai.env import AnimalAIEnv

    selector = build_selector(
        mode=mode,
        train_variant=train_variant,
        train_levels=list(train_levels),
        eval_levels=list(eval_levels),
        eval_variants=list(eval_variants),
        steps_per_stage=steps_per_stage,
        seed=seed,
    )
    # アリーナの選択順と、Unity 側の乱数（位置 -1 の物体の配置など）の両方を
    # トップレベルのシードに従わせる。
    return AnimalAIEnv(
        resolution=resolution,
        seed=seed,
        base_port=5005,
        binary_path=binary_path,
        continuous_action=continuous_action,
        topdown_camera=topdown_camera,
        topdown_resolution=topdown_resolution,
        topdown_observation=topdown_observation,
        colored_walls=colored_walls,
        end_at_pass_mark=end_at_pass_mark,
        blink_dark_steps_increment=blink_dark_steps_increment,
        selector=selector,
    )


def make_env(env_factory, seed: int) -> gym.Env:
    """The Animal-AI env as the trainer reads it: the raw env under the wrappers
    that move the scalar observations into the observation dict."""
    env = hydra.utils.instantiate(env_factory, seed=seed)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = TransposeAndNormalizeObs(env)
    env = InfoToObsWrapper(env, key="velocity", shape=(3,), pop=True)
    env = InfoToObsWrapper(env, key="pass_mark", shape=(1,), pop=False)
    env = InfoToObsWrapper(env, key="health", shape=(1,), pop=False)
    env = InfoToObsWrapper(env, key="global_step", shape=(1,), pop=False)
    env = InfoToObsWrapper(env, key="episode_step", shape=(1,), pop=False)
    env = EpisodeReturnObsWrapper(env)
    env = RemainingReturnObsWrapper(env)
    env.unwrapped.eval_range = 20
    return env
