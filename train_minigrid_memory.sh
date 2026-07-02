#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py \
  agent=standard \
  learning_mode=off_policy \
  batch_size=16 \
  actor_lr=1e-5 \
  critic_lr=1e-5 \
  env=minigrid_memory \
  exp_name=pathwise$suffix
