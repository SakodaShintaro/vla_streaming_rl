#!/bin/bash
set -eux

# Usage: ./train_animalai.sh <agent> <exp_name>
#
# Every variant is one agent config; nothing else differs between them:
#   vlm_off_policy_bs16     : standard agent, VLM actor-critic, off-policy
#   animal_ppo              : the Animal-AI Olympics winning recurrent PPO
agent=${1}
exp_name=${2}
cd $(dirname $0)

export GRPC_VERBOSITY=ERROR

# Animal-AI v5 does not auto-download the Unity binary. Place the unzipped
# Linux build at $HOME/animalai_env/Linux/animalAI.x86_64 (downloaded from
# https://github.com/Kinds-of-Intelligence-CFI/animal-ai/releases).
AAI_BINARY="${HOME}/animalai_env/Linux/animalAI.x86_64"
if [ ! -x "${AAI_BINARY}" ]; then
    echo "ERROR: AAI binary not found or not executable: ${AAI_BINARY}" >&2
    echo "Download Linux.zip from animal-ai releases, unzip to ~/animalai_env/, chmod +x." >&2
    exit 1
fi

# The Unity player writes one CSV row per step into a queue its writer cannot
# drain at this throughput, which puts the host out of memory after a few hours.
# Disable it once with: uv run --with dnfile python local/patch_env_logging.py
#
# Off-screen rendering needs an X server (--no-graphics-monitor still requires
# DISPLAY for the GL context). Local DISPLAY=:0 works; for true headless,
# wrap with xvfb-run.
: "${DISPLAY:=:0}"
export DISPLAY

uv run python scripts/train.py \
  agent=${agent} \
  env=animalai \
  exp_name=${exp_name} \
  resume_dir=null
