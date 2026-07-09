#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

# Animal-AI v5 does not auto-download the Unity binary. Place the unzipped
# Linux build at $HOME/animalai_env/Linux/animalAI.x86_64 (downloaded from
# https://github.com/Kinds-of-Intelligence-CFI/animal-ai/releases).
AAI_BINARY="${HOME}/animalai_env/Linux/animalAI.x86_64"
if [ ! -x "${AAI_BINARY}" ]; then
    echo "ERROR: AAI binary not found or not executable: ${AAI_BINARY}" >&2
    echo "Download Linux.zip from animal-ai releases, unzip to ~/animalai_env/, chmod +x." >&2
    exit 1
fi

# Off-screen rendering needs an X server (--no-graphics-monitor still requires
# DISPLAY for the GL context). Local DISPLAY=:0 works; for true headless,
# wrap with xvfb-run.
: "${DISPLAY:=:0}"
export DISPLAY

uv run python scripts/train.py \
  agent=standard \
  network_class=actor_critic_with_action_value \
  learning_mode=streaming \
  actor_lr=1e-7 \
  critic_lr=1e-7 \
  env=animalai \
  exp_name=animalai${suffix}
