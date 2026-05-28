#!/bin/bash
set -eux

cd $(dirname $0)

# CARLA 0.9.16 — pyproject.toml references the wheel at
#   ../../CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-cp312-cp312-manylinux_2_31_x86_64.whl
# so the binary distribution must live at $HOME/CARLA_0.9.16.
CARLA_VERSION=0.9.16
CARLA_DIR="$HOME/CARLA_${CARLA_VERSION}"
CARLA_URL="https://tiny.carla.org/carla-0-9-16-linux"
ADDMAPS_URL="https://tiny.carla.org/additional-maps-0-9-16-linux"

# CARLA tarball is flat (CHANGELOG, CarlaUE4/, Engine/, ...), so extract into
# a pre-created CARLA_0.9.16/ directory.
if [ ! -f "$CARLA_DIR/CarlaUE4.sh" ]; then
    mkdir -p "$CARLA_DIR"
    curl -L --fail "$CARLA_URL" | tar -xz -C "$CARLA_DIR"
fi

# AdditionalMaps tarball is rooted at CarlaUE4/, so it overlays the CARLA root.
# Drop it into Import/ and run ImportAssets.sh — the standard CARLA workflow.
# Town06 ships only in AdditionalMaps, so its presence is the idempotency check.
if [ ! -f "$CARLA_DIR/CarlaUE4/Content/Carla/Maps/Town06.umap" ]; then
    mkdir -p "$CARLA_DIR/Import"
    curl -L --fail -o "$CARLA_DIR/Import/AdditionalMaps_${CARLA_VERSION}.tar.gz" "$ADDMAPS_URL"
    (cd "$CARLA_DIR" && ./ImportAssets.sh)
    rm -f "$CARLA_DIR/Import/AdditionalMaps_${CARLA_VERSION}.tar.gz"
fi
