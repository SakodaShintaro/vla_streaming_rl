#!/bin/bash
set -eux

uv run python scripts/train.py exp_name=test debug=true batch_size=2 learning_mode=off_policy network_class=actor_critic_with_action_value
uv run python scripts/train.py exp_name=test debug=true batch_size=2 learning_mode=streaming network_class=actor_critic_with_action_value
uv run python scripts/train.py exp_name=test debug=true batch_size=2 learning_mode=streaming network_class=vlm_actor_critic_with_action_value
uv run python scripts/train.py exp_name=test debug=true batch_size=2 learning_mode=streaming network_class=vlm_actor_critic_with_action_value env=animalai
