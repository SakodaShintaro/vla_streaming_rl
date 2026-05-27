#!/bin/bash
# SimLingo zero-shot baseline runner for the Bench2Drive220 sweep.
# In-repo equivalent of simlingo/scripts/eval_220routes.sh, sharing
# train.py + CARLALeaderboardEnv + eval_writer with the RL training
# entry point (train_carla.sh) so the output files match. Same CARLA
# port (2000) and TM port (8000) as train_carla.sh — do not run both
# simultaneously.
#
# SimLingo's team_code + simlingo_training are vendored under
# src/vla_streaming_rl/simlingo/, and the env publishes the
# leaderboard-compatible sensor stack (rgb / gps / imu / speed) via
# info["sensors"], so no external simlingo checkout is referenced.
# The checkpoint is auto-downloaded from huggingface (RenzKa/simlingo)
# into the local HF cache on first run.
#
# Zero-shot reproduction is achieved by two overrides below:
#   simlingo.delta_scale=0      → action == base waypoints (the DACER2
#                                  diffusion policy still samples Δ but
#                                  it is multiplied by 0).
#   simlingo.learning_starts=1e9 → replay buffer never trains, so the
#                                  per-step VLM forward stays the only
#                                  GPU work and the actor / critic
#                                  weights remain at their init values.
set -eux

suffix=${1:-""}
cd $(dirname $0)

CARLA_ROOT=${CARLA_ROOT:-$HOME/CARLA_0.9.16}
B2D_ROOT=$(pwd)/external/Bench2Drive

export PYTHONPATH=${B2D_ROOT}/leaderboard:${B2D_ROOT}/scenario_runner:${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH:-}
export SCENARIO_RUNNER_ROOT=${B2D_ROOT}/scenario_runner

if pgrep -x CarlaUE4-Linux-Shipping > /dev/null; then
    echo "ERROR: CARLA is already running. Kill it first: pkill -f CarlaUE4 && sleep 5" >&2
    exit 1
fi
setsid ${CARLA_ROOT}/CarlaUE4.sh -RenderOffScreen &
CARLA_PGID=$!
trap 'kill -TERM -- -$CARLA_PGID 2>/dev/null; kill 0' EXIT

route_xml=${B2D_ROOT}/leaderboard/data/bench2drive220.xml

# Same train.py as train_carla.sh, but with agent=simlingo. The
# delta_scale=0 + learning_starts=1e9 overrides reduce SimLingoAgent to
# the upstream pretrained policy (base waypoints → PID), so the eval
# numbers match a zero-shot run.
uv run python scripts/train.py \
  agent=simlingo \
  env=carla \
  exp_name=simlingo$suffix \
  env_factory.route_xml=${route_xml} \
  env_factory.sequence_mode=sequential \
  env_factory.start_index=0 \
  env_factory.loop=false \
  simlingo.delta_scale=0 \
  simlingo.learning_starts=1000000000 \
