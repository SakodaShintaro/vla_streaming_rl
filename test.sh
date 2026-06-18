#!/bin/bash
set -eux

uv run python scripts/train.py exp_name=test debug=true batch_size=2 agent_type=off_policy network_class=actor_critic_with_action_value
uv run python scripts/train.py exp_name=test debug=true batch_size=2 agent_type=streaming network_class=actor_critic_with_action_value
uv run python scripts/train.py exp_name=test debug=true batch_size=2 agent_type=streaming network_class=vlm_actor_critic_with_action_value
uv run python scripts/train.py agent=simlingo env=carla_special_case exp_name=test debug=true
uv run python scripts/train.py agent=libero_pi05 env=libero exp_name=test debug=true
