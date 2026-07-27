#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py \
  agent=standard \
  env=carla_special_case \
  exp_name=standard${suffix}
