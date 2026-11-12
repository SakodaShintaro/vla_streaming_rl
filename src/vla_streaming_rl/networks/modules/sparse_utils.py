# SPDX-License-Identifier: MIT
"""
Sparse network utilities for implementing one-shot random pruning
based on "Network Sparsity Unlocks the Scaling Potential of Deep Reinforcement Learning"
"""

import torch
import torch.nn as nn


def create_random_mask(shape: tuple, sparsity: float, device: torch.device) -> torch.Tensor:
    """
    Create a random binary mask for pruning

    Args:
        shape: Shape of the tensor to mask
        sparsity: Fraction of weights to zero out (0.0 to 1.0)
        device: Device to create mask on

    Returns:
        Binary mask tensor
    """
    mask = torch.ones(shape, device=device, dtype=torch.float32)

    if sparsity > 0.0:
        # Create random mask
        rand_tensor = torch.rand(shape, device=device)
        # Set weights to 0 where random values are below sparsity threshold
        mask = (rand_tensor >= sparsity).float()

    return mask


def create_sparse_init_mask(shape: tuple, sparsity: float, device: torch.device) -> torch.Tensor:
    """
    Create a SparseInit mask (SparseInit of "Streaming Deep Reinforcement Learning Finally
    Works"): each output unit (row) keeps its own random subset of incoming weights and
    zeros the rest, so every input still reaches some units.

    Args:
        shape: Shape of the tensor (output_dim, input_dim) for FC layers
        sparsity: Proportion of each row's incoming weights to zero out (0.0 to 1.0)
        device: Device to create mask on

    Returns:
        Binary mask tensor (1 for kept weights, 0 for zeroed weights)
    """
    assert len(shape) == 2, f"SparseInit only supports 2D tensors, got shape {shape}"

    output_dim, fan_in = shape
    # 行ごとに fan_in のうち sparsity の割合を 0 にする。行ごとに独立な順列の先頭を選ぶ
    n_zero_inputs = int(sparsity * fan_in)
    zero_input_indices = torch.rand(output_dim, fan_in, device=device).argsort(dim=1)[
        :, :n_zero_inputs
    ]
    mask = torch.ones(shape, device=device, dtype=torch.float32)
    mask.scatter_(1, zero_input_indices, 0.0)
    return mask


def apply_one_shot_pruning(
    module: nn.Module,
    overall_sparsity: float = 0.9,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """
    Apply one-shot random pruning to a neural network module

    Args:
        module: Neural network module to prune
        overall_sparsity: Overall target sparsity (0.0 to 1.0)
        device: Device for computations

    Returns:
        SparseMask object containing all layer masks
    """
    use_sparse_init = True
    if device is None:
        device = next(module.parameters()).device

    masks = {}

    for name, param in module.named_parameters():
        if "weight" in name and param.dim() >= 2:  # Only prune weight parameters with 2+ dimensions
            shape = param.shape

            if use_sparse_init:
                # Use SparseInit - structured sparsity by input dimensions
                mask = create_sparse_init_mask(shape, overall_sparsity, device)
            else:
                # Use uniform sparsity
                mask = create_random_mask(shape, overall_sparsity, device)

            masks[name] = mask

            # Apply mask immediately
            param.data *= mask

            # Calculate actual sparsity
            actual_sparsity = (mask == 0).float().mean().item()
            print(
                f"Layer {name}: target sparsity={overall_sparsity:.3f}, actual sparsity={actual_sparsity:.3f}"
            )

    return masks
