#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py \
  agent=libero_pi05 \
  env=mikasa \
  exp_name=mikasa_pi05$suffix
