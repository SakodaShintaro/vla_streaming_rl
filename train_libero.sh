#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

# pi0.5 (LeRobot PI05Policy) online return-weighted fine-tuning on LIBERO.
# MUJOCO_GL=egl lets robosuite render headlessly; the env also sets it itself.
MUJOCO_GL=egl uv run python scripts/train.py \
  agent=libero_pi05 \
  env=libero \
  exp_name=libero_pi05$suffix
