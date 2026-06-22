# SPDX-License-Identifier: MIT
"""Load the vendored VLAC critic (InternVL2-based) in this repository's
environment (transformers>=5 / ms-swift>=4.2).

The patched model remote-code lives under ``model_patches/<model_name>`` and is
overlaid onto the downloaded weights directory before loading, so the
transformers-5 port survives a fresh ``VLAC-2B`` download on any machine. The
``evo_vlac`` package is vendored under this directory and imported as a
subpackage, so no external install is required.
"""

import glob
import os
import shutil

_VLAC_DIR = os.path.dirname(os.path.abspath(__file__))
_PATCH_ROOT = os.path.join(_VLAC_DIR, "model_patches")
_MODULES_CACHE = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")


def _overlay_model_patches(model_path: str) -> None:
    """Copy the repo's patched remote-code over the weights dir and drop the
    stale transformers_modules cache so the patched code is re-imported."""
    model_name = os.path.basename(os.path.normpath(model_path))
    patch_dir = os.path.join(_PATCH_ROOT, model_name)
    if not os.path.isdir(patch_dir):
        raise FileNotFoundError(f"No bundled VLAC patches for {model_name!r} under {_PATCH_ROOT}")
    for fname in os.listdir(patch_dir):
        shutil.copy2(os.path.join(patch_dir, fname), os.path.join(model_path, fname))
    for cached in glob.glob(os.path.join(_MODULES_CACHE, "*" + model_name.replace("-", "*") + "*")):
        shutil.rmtree(cached, ignore_errors=True)


def load_vlac_critic(
    model_path: str,
    device_map: str,
    tag: str,
    temperature: float,
    top_k: int,
):
    """Overlay the patched remote-code, build the vendored ``GAC_model`` critic,
    initialise it on ``device_map`` and return the ready-to-infer instance."""
    _overlay_model_patches(model_path)
    from vla_streaming_rl.networks.vlac.evo_vlac import GAC_model

    critic = GAC_model(tag=tag)
    critic.init_model(model_path=model_path, device_map=device_map)
    critic.temperature = temperature
    critic.top_k = top_k
    critic.set_config()
    critic.set_system_prompt()
    return critic
