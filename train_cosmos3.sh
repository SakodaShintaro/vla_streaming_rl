#!/bin/bash
# SPDX-License-Identifier: MIT
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py \
  agent=cosmos_edge \
  env=carla_special_case \
  exp_name=cosmos3$suffix
