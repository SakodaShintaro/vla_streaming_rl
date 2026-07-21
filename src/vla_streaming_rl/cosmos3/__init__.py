# SPDX-License-Identifier: MIT
# Cosmos3 model/pipeline classes come from the pinned git build of diffusers
# (see pyproject.toml); this package holds only the project-local RL adapter.
from vla_streaming_rl.cosmos3.policy import CosmosEdgePolicy

__all__ = ["CosmosEdgePolicy"]
