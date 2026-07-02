#!/bin/bash
set -eux

RESULT_DIR=${1:?"Usage: $0 <result_dir> [suffix]"}
RESULT_DIR=$(readlink -f $RESULT_DIR)
suffix=${2:-""}
cd $(dirname $0)

WANDB_GROUP="train_all_$(date +%Y%m%d_%H%M%S)$suffix"

mkdir -p $RESULT_DIR

# Ensure the results directory is empty
if [ -d "$RESULT_DIR" ] && [ "$(ls -A $RESULT_DIR)" ]; then
  echo "Error: $RESULT_DIR is not empty. Please move or delete the existing results before running this script."
  exit 1
fi

git show -s > $RESULT_DIR/git_show.txt
git diff > $RESULT_DIR/git_diff.txt

# Streaming, with eligibility trace, learning rate 5e-6
uv run python scripts/train.py \
  agent=standard \
  env=car_racing \
  exp_name=vlm_streaming$suffix \
  learning_rate=5e-6 \
  result_dir=$RESULT_DIR \
  wandb_group=$WANDB_GROUP

# Off-policy, batch size 16, learning rate 1e-5
uv run python scripts/train.py \
  agent=standard \
  learning_mode=off_policy \
  batch_size=16 \
  actor_lr=1e-5 \
  critic_lr=1e-5 \
  env=car_racing \
  exp_name=vlm_off_policy_bs16$suffix \
  result_dir=$RESULT_DIR \
  wandb_group=$WANDB_GROUP

# Off-policy, batch size 1, learning rate 2e-6
uv run python scripts/train.py \
  agent=standard \
  learning_mode=off_policy \
  batch_size=1 \
  actor_lr=2e-6 \
  critic_lr=2e-6 \
  env=car_racing \
  exp_name=vlm_off_policy_bs1$suffix \
  result_dir=$RESULT_DIR \
  wandb_group=$WANDB_GROUP

# Comparison
uv run python scripts/train.py \
  agent=standard \
  network_class=actor_critic_with_action_value \
  learning_mode=off_policy \
  batch_size=16 \
  actor_lr=1e-5 \
  critic_lr=1e-5 \
  env=car_racing \
  exp_name=no_vlm_off_policy_bs16$suffix \
  result_dir=$RESULT_DIR \
  wandb_group=$WANDB_GROUP
