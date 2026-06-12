#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

CARLA_ROOT=${CARLA_ROOT:-$HOME/CARLA_0.9.16}
B2D_ROOT=$(pwd)/external/Bench2Drive
export PYTHONPATH=${B2D_ROOT}/leaderboard:${B2D_ROOT}/scenario_runner:${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH:-}
export SCENARIO_RUNNER_ROOT=${B2D_ROOT}/scenario_runner

# Refuse to start if CARLA is already running. Stale CARLA processes carry
# over actor IDs / TM state / sensor stream sessions across runs, which
# silently corrupts behavior — kill any existing CARLA manually before
# running this script (e.g. `pkill -f CarlaUE4 && sleep 5`).
if pgrep -f CarlaUE4 > /dev/null; then
    echo "ERROR: CARLA is already running. Kill it first: pkill -f CarlaUE4 && sleep 5" >&2
    exit 1
fi
setsid ${CARLA_ROOT}/CarlaUE4.sh -RenderOffScreen &
CARLA_PGID=$!
trap 'kill -TERM -- -$CARLA_PGID 2>/dev/null; kill 0' EXIT

# Bench2Drive 220-scenario driver: walk every route in
# leaderboard/data/bench2drive220.xml (positions 0..219, equivalent to
# simlingo's bench2drive_split/bench2drive_{00..219}.xml ordering) in XML
# order, reloading the world on town change. Training stops after the
# 220th scenario completes. Eval artifacts (Driving Score / Success Rate /
# Efficiency / Comfort) are auto-written under the Hydra run dir's eval/
# subdir;
sequence_mode=sequential
start_index=0

uv run python scripts/train.py \
  agent=cnn_streaming \
  env=carla \
  exp_name=carla$suffix \
  env_factory.sequence_mode=${sequence_mode} \
  env_factory.start_index=${start_index} \
