#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

# agent=standard defaults to vlm streaming. Variant overrides (uncomment one):
# NAME=cnn_off_policy_bs16; OVERRIDES="network_class=actor_critic_with_action_value learning_mode=off_policy batch_size=16 actor_lr=1e-5 critic_lr=1e-5"
# NAME=cnn_streaming; OVERRIDES="network_class=actor_critic_with_action_value learning_mode=streaming"
# NAME=vlm_off_policy_bs16; OVERRIDES="learning_mode=off_policy batch_size=16 actor_lr=1e-5 critic_lr=1e-5"
NAME=vlm_streaming; OVERRIDES=""

uv run python scripts/train.py \
  agent=standard \
  ${OVERRIDES} \
  env=car_racing \
  exp_name=${NAME}${suffix}
