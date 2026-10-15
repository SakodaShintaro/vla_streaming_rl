# SPDX-License-Identifier: MIT
"""CARLA (Bench2Drive): the env, its scenario runtime and eval writer, and the
factories that build them for the trainer."""

from pathlib import Path

import gymnasium as gym
import hydra

from vla_streaming_rl.wrappers import (
    EpisodeReturnObsWrapper,
    InfoToObsWrapper,
    ZeroScalarObsWrapper,
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
    # ``envs.carla.env`` (it imports leaderboard / srunner at module
    # load); the CARLA server itself is launched lazily by the env's shared
    # ``get_client`` (bootstrap.py), reused across all trials.
    from vla_streaming_rl.envs.carla.bootstrap import setup_carla_paths

    setup_carla_paths()

    from vla_streaming_rl.envs.carla.env import CARLALeaderboardEnv

    CARLA_ROUTE_XML = str(
        Path(__file__).parents[4] / "external/Bench2Drive/leaderboard/data/bench2drive220.xml"
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


def make_env(env_factory, result_dir) -> gym.Env:
    """The CARLA env as the trainer reads it. ``result_dir`` (when set) is the
    Hydra run dir; ``eval_output_dir = result_dir / "eval"`` is derived from it
    for the Bench2Drive eval artifacts."""
    # eval_output_dir is injected here (not via the user-facing
    # env_factory config) so the user does not have to remember a
    # path that is always result_dir/eval.
    eval_output_dir = str(result_dir / "eval") if result_dir is not None else None
    env = hydra.utils.instantiate(env_factory, eval_output_dir=eval_output_dir)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = ZeroScalarObsWrapper(env)
    env = InfoToObsWrapper(env, key="global_step", shape=(1,), pop=False)
    env = InfoToObsWrapper(env, key="episode_step", shape=(1,), pop=False)
    env = EpisodeReturnObsWrapper(env)
    env.unwrapped.eval_range = 220
    return env
