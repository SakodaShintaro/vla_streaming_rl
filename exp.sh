#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

# ==== Select ONE setup (uncomment its block) ==========================

# --- CarRacing-Qwen ---
AGENT=standard; ENV=car_racing
STREAM_LR="actor_lr=1e-6 critic_lr=1e-6"
OFF16_LR="actor_lr=1e-5 critic_lr=1e-5"
OFF1_LR="actor_lr=1e-6 critic_lr=1e-6"

# --- CARLA-SimLingo ---
# AGENT=simlingo; ENV=carla_special_case
# STREAM_LR="actor_lr=2e-7 critic_lr=5e-7"
# OFF16_LR="actor_lr=2e-6 critic_lr=5e-6"
# OFF1_LR="actor_lr=2e-7 critic_lr=5e-7"

# --- LIBERO-pi0.5 ---
# AGENT=libero_pi05; ENV=libero
# STREAM_LR="actor_lr=2e-7 critic_lr=5e-7"
# OFF16_LR="actor_lr=2e-6 critic_lr=5e-6"
# OFF1_LR="actor_lr=2e-7 critic_lr=5e-7"

# ==== Run the 3 learning-mode variants ================================

# Streaming
uv run python scripts/train.py \
  agent=$AGENT \
  env=$ENV \
  learning_mode=streaming \
  batch_size=1 \
  $STREAM_LR \
  exp_name=${AGENT}_streaming$suffix

# OffPolicy batch_size 16
uv run python scripts/train.py \
  agent=$AGENT \
  env=$ENV \
  learning_mode=off_policy \
  batch_size=16 \
  $OFF16_LR \
  exp_name=${AGENT}_off_policy_bs16$suffix

# OffPolicy batch_size 1
uv run python scripts/train.py \
  agent=$AGENT \
  env=$ENV \
  learning_mode=off_policy \
  batch_size=1 \
  $OFF1_LR \
  exp_name=${AGENT}_off_policy_bs1$suffix
