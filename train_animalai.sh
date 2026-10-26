#!/bin/bash
set -eux

# Usage: ./train_animalai.sh <difficulty> <agent> <exp_name> [hydra overrides...]
#   difficulty: easy    -> level 01 の variant 01 だけ（簡易な検証用）
#               1301    -> 13-01-01 だけ
#               14      -> 暗転の課題 14-01-01（最初の 1 ステップだけ見える）
#               blink   -> level 01 を点滅の記憶の課題にしたもの（env_factory.blink_dark_steps_increment）
#               normal  -> configs/env/animalai.yaml のデフォルト通り
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
      episode_limit=1200
    )
    ;;
  one_lap)
    difficulty_overrides=(
      'env_factory.train_levels=["01"]'
      'env_factory.train_variant="01"'
      episode_limit=30
    )
    ;;
  1301)
    difficulty_overrides=(
      'env_factory.train_levels=["13-01"]'
      'env_factory.train_variant="01"'
      episode_limit=1200
    )
    ;;
  14)
    # 暗転の課題（scripts/make_blackout_arenas.py で作る 14-01-01）
    difficulty_overrides=(
      'env_factory.train_levels=["14"]'
      'env_factory.train_variant="01"'
      episode_limit=1200
      # 1 エピソードは最大 100 ステップなので、最初に見た 1 枚が窓から外れないようにする
      actor_critic.seq_len=100
    )
    ;;
  blink)
    # 点滅の記憶の課題に作り変えたレベル 01（configs/env/animalai.yaml の blink_dark_steps_increment）
    difficulty_overrides=(
      'env_factory.train_levels=["01"]'
      'env_factory.train_variant="01"'
      env_factory.blink_dark_steps_increment=1
      episode_limit=1200
      actor_critic.seq_len=100
    )
    ;;
  normal)
    difficulty_overrides=()
    ;;
  *)
    echo "Unknown difficulty: ${difficulty} (easy, one_lap, 1301, 14, blink or normal)" >&2
    exit 1
    ;;
esac

uv run python scripts/train.py \
  agent=${agent} \
  env=animalai \
  exp_name=${exp_name} \
  resume_dir=null \
  "wandb_group='${difficulty}'" \
  "${difficulty_overrides[@]}" \
  "$@"
