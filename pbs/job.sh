#!/bin/bash
# SPDX-License-Identifier: MIT
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -P gch51673
set -eux
cd ${PBS_O_WORKDIR}
set +x; source ~/.secrets; set -x

# fla's Triton kernel for the backward of the gated delta rule is wrong on Hopper GPUs (H200) with
# the Triton that torch 2.8 pins, and fla refuses to run it. With tilelang and an nvcc present it
# uses its TileLang kernel there instead, so install tilelang into this venv only (uv.lock is left
# alone) and give the container the host's CUDA toolkit for nvcc.
CUDA_HOME=/apps/cuda/12.8.1

singularity exec --nv \
  --bind ${HOME}:${HOME} \
  --bind ${CUDA_HOME}:${CUDA_HOME} \
  --env CUDA_HOME="${CUDA_HOME}" \
  --env WANDB_API_KEY="${WANDB_API_KEY}" \
  --env HF_TOKEN="${HF_TOKEN}" \
  ${HOME}/work/vla_streaming_rl/pbs/container/xvfb.sif \
  bash -c "
    Xvfb :1 -screen 0 1024x768x24 &
    export DISPLAY=:1
    cd ${HOME}/work/vla_streaming_rl
    uv sync
    uv pip install tilelang
    ./${script} ${args}
  "
