#!/bin/bash
set -eux

uv run python scripts/train.py exp_name=test debug=true batch_size=2 learning_mode=off_policy network_class=actor_critic_with_action_value
uv run python scripts/train.py exp_name=test debug=true batch_size=2 learning_mode=streaming network_class=actor_critic_with_action_value
uv run python scripts/train.py exp_name=test debug=true batch_size=2 learning_mode=streaming network_class=vlm_actor_critic_with_action_value
uv run python scripts/train.py exp_name=test debug=true agent=simlingo env=carla_special_case
uv run python scripts/train.py exp_name=test debug=true agent=libero_pi05 env=libero
