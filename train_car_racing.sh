#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py \
  agent=standard \
  network_class=actor_critic_with_action_value \
  learning_mode=off_policy \
  batch_size=16 \
  actor_lr=1e-5 \
  critic_lr=1e-5 \
  env=car_racing \
  exp_name=${suffix}
