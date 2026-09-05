#!/bin/bash
set -eux

# debug runs still create a Hydra run dir, so keep it out of results/
RESULT_DIR=/tmp/vla_streaming_rl_test

uv run python scripts/train.py exp_name=test debug=true result_dir=${RESULT_DIR} batch_size=2 agent_type=off_policy network_class=actor_critic_with_action_value
uv run python scripts/train.py exp_name=test debug=true result_dir=${RESULT_DIR} batch_size=2 agent_type=streaming network_class=actor_critic_with_action_value
uv run python scripts/train.py exp_name=test debug=true result_dir=${RESULT_DIR} batch_size=2 agent_type=off_policy network_class=animal_actor_critic
uv run python scripts/train.py exp_name=test debug=true result_dir=${RESULT_DIR} batch_size=2 agent_type=streaming network_class=animal_actor_critic
uv run python scripts/train.py exp_name=test debug=true result_dir=${RESULT_DIR} batch_size=2 agent_type=streaming network_class=vlm_actor_critic_with_action_value
uv run python scripts/train.py exp_name=test debug=true result_dir=${RESULT_DIR} batch_size=2 agent_type=streaming network_class=vlm_actor_critic_with_action_value env=animalai
uv run python scripts/train.py exp_name=test debug=true result_dir=${RESULT_DIR} batch_size=2 agent_type=streaming network_class=actor_critic_with_action_value cot_tokens_num=4
uv run python scripts/train.py exp_name=test debug=true result_dir=${RESULT_DIR} batch_size=2 agent_type=off_policy network_class=actor_critic_with_action_value cot_tokens_num=4
uv run python scripts/train.py exp_name=test debug=true result_dir=${RESULT_DIR} batch_size=2 agent_type=off_policy network_class=actor_critic_with_action_value cot_tokens_num=0
