#!/bin/bash
# SPDX-License-Identifier: MIT
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
  agent=cosmos_edge \
  env=animalai \
  reward_processor_type=scaling \
  frame_stride=1 \
  exp_name=cosmos3_animalai${suffix}
