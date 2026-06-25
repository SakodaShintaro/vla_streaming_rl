# SPDX-License-Identifier: MIT
"""Vendored VLAC (Vision-Language-Action-Critic) integration.

``evo_vlac`` is the vendored upstream package (InternVL2 + ms-swift), patched for
transformers>=5. :func:`load_vlac_critic` overlays the bundled remote-code
patches and builds the critic; :class:`VlacRewardRelabeler` turns it into a dense
PBRS shaping reward for the LIBERO replay buffer.
"""

from vla_streaming_rl.networks.vlac.loader import load_vlac_critic
from vla_streaming_rl.networks.vlac.relabel import VlacRewardRelabeler

__all__ = ["load_vlac_critic", "VlacRewardRelabeler"]
