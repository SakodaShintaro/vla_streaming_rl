#!/bin/bash
set -eux

# Usage: ./train_carla.sh <agent> <exp_name>
agent=${1}
exp_name=${2}
cd $(dirname $0)

uv run python scripts/train.py \
  agent=${agent} \
  env=carla_special_case \
  exp_name=${exp_name}
