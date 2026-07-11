#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py \
  agent=libero_pi05 \
  env=robomemarena \
  exp_name=robomemarena_pi05$suffix
