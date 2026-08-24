#!/bin/bash
set -eux

cd $(dirname $0)

# ==== Select setup via argument =======================================
# Usage: ./exp.sh {car_racing}

SETUP=${1:-}

case "$SETUP" in
  car_racing)
    AGENT=vlm_streaming; ENV=car_racing
    STREAM_LR="actor_lr=1e-6 critic_lr=1e-5"
    OFF16_LR="actor_lr=1e-5 critic_lr=1e-5"
    OFF1_LR="actor_lr=1e-6 critic_lr=1e-5"
    ;;
  *)
    echo "Usage: $0 {car_racing}" >&2
    exit 1
    ;;
esac

# ==== Run the 3 learning-mode variants ================================

result_dir=~/data/$(date +%Y%m%d_%H%M%S)_${AGENT}_${ENV}

# Streaming
uv run python scripts/train.py \
  agent=$AGENT \
  env=$ENV \
  agent_type=streaming \
  batch_size=1 \
  $STREAM_LR \
  result_dir=$result_dir \
  exp_name=streaming

# OffPolicy batch_size 16
uv run python scripts/train.py \
  agent=$AGENT \
  env=$ENV \
  agent_type=off_policy \
  batch_size=16 \
  $OFF16_LR \
  result_dir=$result_dir \
  exp_name=off_policy_bs16

# OffPolicy batch_size 1
uv run python scripts/train.py \
  agent=$AGENT \
  env=$ENV \
  agent_type=off_policy \
  batch_size=1 \
  $OFF1_LR \
  result_dir=$result_dir \
  exp_name=off_policy_bs1
