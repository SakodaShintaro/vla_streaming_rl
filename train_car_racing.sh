#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py \
  agent=cnn_off_policy_bs16 \
  env=car_racing \
  exp_name=${suffix}
