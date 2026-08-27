# SPDX-License-Identifier: MIT
"""The two vision-tower helpers this repo needs, on the API that replaced them.

``fast_pos_embed_interpolate`` and ``rot_pos_emb`` on Qwen's vision tower are
deprecated. Both encoders here drive the tower block by block rather than calling
its ``forward``, so they need exactly what those two returned. Keeping the
replacements in one file means the next upstream move touches one place instead
of every encoder.
"""

import torch
from torch import nn
from transformers.vision_utils import (
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)


def interpolated_pos_embed(visual: nn.Module, grid_thw: torch.Tensor) -> torch.Tensor:
    """The tower's learned position embedding, interpolated onto this grid."""
    indices, weights = get_vision_interpolation_indices_and_weights(
        grid_thw,
        num_grid_per_side=visual.num_grid_per_side,
        mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners,
        spatial_merge_size=visual.config.spatial_merge_size,
    )
    return (visual.pos_embed(indices) * weights[:, :, None]).sum(1)


def rotary_pos_embed(visual: nn.Module, grid_thw: torch.Tensor) -> torch.Tensor:
    """The tower's rotary embedding for this grid's patch positions."""
    return visual.rotary_pos_emb(get_vision_position_ids(grid_thw, visual.spatial_merge_size))
