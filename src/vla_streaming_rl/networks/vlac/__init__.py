# SPDX-License-Identifier: MIT
"""Vendored VLAC (Vision-Language-Action-Critic) integration.

``evo_vlac`` is the vendored upstream package (InternVL2 + ms-swift), patched for
transformers>=5. :class:`GAC_model` builds the critic (pulling its weights from
the HF hub on first use); :class:`VlacRewardRelabeler` turns it into a dense PBRS
shaping reward for the LIBERO replay buffer.
"""

from vla_streaming_rl.networks.vlac.relabel import VlacRewardRelabeler

__all__ = ["VlacRewardRelabeler"]
