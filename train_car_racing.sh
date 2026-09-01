#!/bin/bash
set -eux

# Usage: ./train_car_racing.sh <agent> <exp_name>
agent=${1}
exp_name=${2}
cd $(dirname $0)

uv run python scripts/train.py \
  agent=${agent} \
  env=car_racing \
  exp_name=${exp_name}
