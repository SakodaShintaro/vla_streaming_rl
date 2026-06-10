#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py \
  agent=simlingo \
  env=carla_special_case \
  exp_name=simlingo$suffix \
  env_factory.route_xml="$(pwd)/external/Bench2Drive/leaderboard/data/bench2drive220.xml"
