#!/bin/bash
# SPDX-License-Identifier: MIT
set -eux

cd $(dirname $0)

# The Animal-AI Unity player rebuilt from
#   https://github.com/SakodaShintaro/animal-ai-unity (branch feat/continuous_actions)
# so it also accepts continuous actions and renders the top-down camera. In the
# discrete mode it is step-for-step identical to the official 4.3.x release.
# configs/env/animalai.yaml points env_factory.binary_path at what this installs.
AAI_VERSION=4.3.2_alpha1
AAI_DIR="$HOME/animalai_env/$AAI_VERSION"
AAI_URL="https://github.com/SakodaShintaro/animal-ai-unity/releases/download/${AAI_VERSION}/Linux.zip"

# The zip is rooted at Linux/, the same way the official release unpacks.
if [ ! -f "$AAI_DIR/Linux/animalAI.x86_64" ]; then
    mkdir -p "$AAI_DIR"
    curl -L --fail -o "$AAI_DIR/Linux.zip" "$AAI_URL"
    unzip -q -o "$AAI_DIR/Linux.zip" -d "$AAI_DIR"
    rm -f "$AAI_DIR/Linux.zip"
    chmod +x "$AAI_DIR/Linux/animalAI.x86_64"
fi
