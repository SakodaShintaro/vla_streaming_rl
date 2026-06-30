#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

# AGENT=cnn_off_policy_bs16
# AGENT=cnn_streaming
# AGENT=vlm_off_policy_bs16
AGENT=vlm_streaming

uv run python scripts/train.py \
  agent=${AGENT} \
  env=carla_special_case \
  exp_name=${AGENT}${suffix}
