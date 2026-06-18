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
    Create a SparseInit mask that zeros out complete input dimensions.
    Based on Algorithm 1 from "Streaming Deep Reinforcement Learning Finally Works"

    Args:
        shape: Shape of the tensor (output_dim, input_dim) for FC layers
        sparsity: Sparsity level (0.0 to 1.0) - proportion of input dimensions to zero out
        device: Device to create mask on

    Returns:
        Binary mask tensor
    """
    assert len(shape) == 2, f"SparseInit only supports 2D tensors, got shape {shape}"

    _, input_dim = shape
    fan_in = input_dim

    # n ← s × fan_in (number of input dimensions to zero out)
    n_zero_inputs = int(sparsity * fan_in)

    # Create permutation set P of size fan_in
    input_perm = torch.randperm(fan_in, device=device)

    # Index set I of size n (subset of P) - input dimensions to zero out
    zero_input_indices = input_perm[:n_zero_inputs]

    # Create mask (1 for kept weights, 0 for zeroed weights)
    mask = torch.ones(shape, device=device, dtype=torch.float32)
    # Wi,j ← 0, ∀i ∈ I, ∀j (zero out selected input dimensions)
    mask[:, zero_input_indices] = 0

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


def apply_masks_during_training(module: nn.Module) -> None:
    """
    Apply masks to ensure pruned weights stay zero during training
    This should be called after each optimizer step

    Args:
        module: Neural network module
    """
    if not hasattr(module, "sparse_mask") or module.sparse_mask is None:
        return

    masks = module.sparse_mask
    for name, param in module.named_parameters():
        if name in masks:
            param.data *= masks[name].to(param.device)
