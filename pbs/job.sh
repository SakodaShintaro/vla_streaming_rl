#!/bin/bash
# SPDX-License-Identifier: MIT
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -P gch51673
set -eux
cd ${PBS_O_WORKDIR}
source ~/.secrets

singularity exec --nv \
  --bind ${HOME}:${HOME} \
  --env WANDB_API_KEY="${WANDB_API_KEY}" \
  --env HF_TOKEN="${HF_TOKEN}" \
  ${HOME}/work/vla_streaming_rl/pbs/container/xvfb.sif \
  bash -c "
    Xvfb :1 -screen 0 1024x768x24 &
    export DISPLAY=:1
    cd ${HOME}/work/vla_streaming_rl
    uv sync
    ./${script} ${exp_name}
  "
