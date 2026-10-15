#!/bin/bash
set -eux

# Usage: ./train_animalai.sh <difficulty> <agent> <exp_name> [hydra overrides...]
#   difficulty: easy   -> level 01 の variant 01 だけ（簡易な検証用）
#               normal -> configs/env/animalai.yaml のデフォルト通り
difficulty=${1}
agent=${2}
exp_name=${3}
shift 3
cd $(dirname $0)

export GRPC_VERBOSITY=ERROR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

case ${difficulty} in
  easy)
    difficulty_overrides=(
      'env_factory.train_levels=["01"]'
      'env_factory.train_variant="01"'
      # level 01 の 30 アリーナをそれぞれ 10 回ずつ回す量
      episode_limit=300
    )
    ;;
  normal)
    difficulty_overrides=()
    ;;
  *)
    echo "Unknown difficulty: ${difficulty} (easy or normal)" >&2
    exit 1
    ;;
esac

uv run python scripts/train.py \
  agent=${agent} \
  env=animalai \
  exp_name=${exp_name} \
  resume_dir=null \
  "${difficulty_overrides[@]}" \
  "$@"
