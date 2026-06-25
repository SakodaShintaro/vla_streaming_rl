# SPDX-License-Identifier: MIT
"""Load the vendored VLAC critic (InternVL2-based) in this repository's
environment (transformers>=5).

The transformers-5 port of the InternVL2 / InternLM2 model code is vendored
under ``evo_vlac/internvl_vlac`` and loaded directly (no ``trust_remote_code``,
no remote-code overlay). Only the weights are pulled from the HF hub into the
local cache on first use.
"""

import torch

_VLAC_REPO_ID = "InternRobotics/VLAC"


def _resolve_model_path() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=_VLAC_REPO_ID)


def load_vlac_critic(
    tag: str,
    temperature: float,
    top_k: int,
):
    """Build the vendored ``GAC_model`` critic, initialise it on ``device_map``
    and return the ready-to-infer instance."""
    model_path = _resolve_model_path()
    from vla_streaming_rl.networks.vlac.evo_vlac import GAC_model

    critic = GAC_model(tag=tag)
    critic.init_model(model_path=model_path, device_map=torch.device("cuda"))
    critic.temperature = temperature
    critic.top_k = top_k
    critic.set_config()
    critic.set_system_prompt()
    return critic
