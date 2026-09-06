#!/bin/bash
set -eux

# Usage: ./train_animalai.sh <agent> <exp_name>
agent=${1}
exp_name=${2}
cd $(dirname $0)

export GRPC_VERBOSITY=ERROR

# Fetch the Unity player this run needs, which is a no-op once it is installed.
# The version and the install path live there and in configs/env/animalai.yaml,
# so this script names neither.
./setup_animalai.sh

# The Unity player writes one CSV row per step into a queue its writer cannot
# drain at this throughput, which puts the host out of memory after a few hours.
# Disable it once with: uv run --with dnfile python local/patch_env_logging.py
#
# Off-screen rendering needs an X server (--no-graphics-monitor still requires
# DISPLAY for the GL context). Local DISPLAY=:0 works; for true headless,
# wrap with xvfb-run.
: "${DISPLAY:=:0}"
export DISPLAY

uv run python scripts/train.py \
  agent=${agent} \
  env=animalai \
  exp_name=${exp_name} \
  resume_dir=null
