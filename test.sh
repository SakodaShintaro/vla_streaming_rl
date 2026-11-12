#!/bin/bash
set -eux

# debug runs still create a Hydra run dir, so keep it out of results/
RESULT_DIR=/tmp/vla_streaming_rl_test

uv run python scripts/train.py exp_name=test debug=true result_dir=${RESULT_DIR} batch_size=2
uv run python scripts/train.py exp_name=test debug=true result_dir=${RESULT_DIR} batch_size=2 env=animalai streaming_update=true
