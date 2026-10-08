# SPDX-License-Identifier: MIT
"""Saving and loading the trainable weights, shared by every entry script."""

from pathlib import Path

import torch


def _unwrap_compiled(network: torch.nn.Module) -> torch.nn.Module:
    return network._orig_mod if hasattr(network, "_orig_mod") else network


def _persistent_buffer_names(module: torch.nn.Module) -> set[str]:
    """state_dict に載る buffer の名前。観測の正規化の統計や温度のように、学習中に
    更新されるがパラメータではないものを含む。"""
    buffer_names = {name for name, _ in module.named_buffers()}
    return {name for name in module.state_dict() if name in buffer_names}


def save_checkpoint(result_dir: Path, network, agent) -> None:
    """Save trainable weights and persistent buffers (checkpoint.pt) and
    optimizer states (optimizer.pt)."""
    module = _unwrap_compiled(network)
    trainable_state = {
        name: param.detach().cpu()
        for name, param in module.named_parameters()
        if param.requires_grad
    }
    state = module.state_dict()
    for name in _persistent_buffer_names(module):
        trainable_state[name] = state[name].detach().cpu()
    torch.save(trainable_state, result_dir / "checkpoint.pt")
    torch.save(agent.optimizer_state_dict(), result_dir / "optimizer.pt")


def load_checkpoint_weights(checkpoint_path: Path, network: torch.nn.Module) -> None:
    """Load a checkpoint.pt into the network and verify nothing was dropped.

    ``missing`` includes two harmless cases: frozen params (never saved) and
    names that alias a shared tensor already restored under a different key
    (e.g. tied lm_head/embed_tokens). Every trainable missing name is resolved
    to its tensor and flagged only if that exact tensor was never touched by
    any key actually in the checkpoint."""
    trainable_state = torch.load(checkpoint_path, map_location="cuda")
    module = _unwrap_compiled(network)
    missing, unexpected = module.load_state_dict(trainable_state, strict=False)
    assert not unexpected, f"checkpoint parameters not found in the network: {unexpected[:5]}"

    loaded_ids = set()
    for name, param in module.named_parameters(remove_duplicate=False):
        if name in trainable_state:
            loaded_ids.add(id(param))
    name_to_param = dict(module.named_parameters(remove_duplicate=False))
    unaccounted_missing = [
        name
        for name in missing
        if name in name_to_param
        and name_to_param[name].requires_grad
        and id(name_to_param[name]) not in loaded_ids
    ]
    assert not unaccounted_missing, (
        f"trainable network parameters not found in the checkpoint: {unaccounted_missing[:5]}"
    )
    # 正規化の統計などが欠けたまま読むと、学習時と違う入力で動いてしまう
    missing_buffers = sorted(_persistent_buffer_names(module) - set(trainable_state))
    assert not missing_buffers, (
        f"network buffers not found in the checkpoint: {missing_buffers[:5]}"
    )
    print(f"Loaded {len(trainable_state)} weight tensors from {checkpoint_path}")
