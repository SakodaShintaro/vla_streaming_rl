#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

# AGENT=cnn_off_policy_bs16
# AGENT=cnn_streaming
# AGENT=base_on_policy
# AGENT=vlm_off_policy_bs16
AGENT=vlm_streaming

uv run python scripts/train.py \
  agent=${AGENT} \
  env=car_racing \
  exp_name=${AGENT}${suffix}
