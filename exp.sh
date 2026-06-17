#!/bin/bash
set -eux

ENV_PATH=${1:?"Usage: $0 <env_config_path> <sweep_name>  (e.g. configs/env/gui.yaml my_sweep)"}
SWEEP_NAME=${2:?"Usage: $0 <env_config_path> <sweep_name>  (e.g. configs/env/gui.yaml my_sweep)"}
ENV_PATH=$(readlink -f "$ENV_PATH")
cd $(dirname $0)

ENV_NAME=$(basename "$ENV_PATH" .yaml)
ENV_ID=$(grep '^env_id:' "$ENV_PATH" | awk '{print $2}')

sed "s|value: placeholder|value: ${ENV_NAME}|" configs/exp.yaml \
  | wandb sweep --name "$SWEEP_NAME" /dev/stdin --project "vla_streaming_rl_${ENV_ID}"
