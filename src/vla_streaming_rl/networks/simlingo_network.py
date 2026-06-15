# SPDX-License-Identifier: MIT
"""The learnable network behind :class:`SimLingoAgent`.

Bundles the (frozen) SimLingo VLM driving model, the trainable waypoint heads
(the deterministic policy μ), and the Q critic into a single ``nn.Module`` so
the agent receives it from outside via ``build_network`` — exactly like every
other agent — instead of constructing the model itself. The tokenizer / config
byproducts the agent's inference pipeline needs are exposed as attributes.
"""

from collections.abc import Callable
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn
from transformers import Qwen2Tokenizer

from vla_streaming_rl.networks.value_head import DistributionalValueHead
from vla_streaming_rl.simlingo.models.driving import DrivingModel
from vla_streaming_rl.simlingo.models.encoder.internvl2_vendored.configuration_internvl_chat import (
    InternVLChatConfig,
)
from vla_streaming_rl.simlingo.utils.internvl2_utils import (
    SIMLINGO_ADDITIONAL_SPECIAL_TOKENS,
)

# SimLingo's driving adaptor emits 20 route waypoints + 10 speed waypoints, each
# 2-D — see DrivingAdaptor in simlingo_training/models/adaptors.py.
_ROUTE_LEN = 20
_SPEED_WPS_LEN = 10
_WP_DIM = 2
# Number of waypoint-query positions in ``driving_features`` (route then speed).
# The DrivingAdaptor heads consume one feature vector per query.
_NUM_WP_QUERIES = _ROUTE_LEN + _SPEED_WPS_LEN
_ACTION_DIM = _NUM_WP_QUERIES * _WP_DIM

_HF_REPO_ID = "RenzKa/simlingo"
_HF_CKPT_NAME = "pytorch_model.pt"


class SimLingoNetwork(nn.Module):
    """SimLingo VLM + waypoint heads (policy μ) + Q critic.

    ``forward(driving_input)`` is the VLM forward, returning SimLingo's
    ``(pred_speed_wps, pred_route, language, driving_features)``. The whole VLM
    is frozen except the two waypoint heads (``actor_heads``); those and the
    ``critic`` are the only trainable parameters.

    Exposed attributes used by ``SimLingoAgent``:
      - ``actor_heads`` / ``critic`` / ``driving_adaptor`` — for training.
      - ``feature_dim`` — shape.
      - ``tokenizer`` / ``num_image_token`` / ``cfg`` — to build the agent's
        prompt / inference pipeline.
    """

    def __init__(
        self,
        *,
        value_head_factory: Callable[[int], DistributionalValueHead],
        device: torch.device,
    ) -> None:
        super().__init__()
        config_path = str(self._resolve_checkpoint())
        print(f"Config path: {config_path}")

        # load config from the checkpoint's .hydra folder
        config_load_path = Path(config_path).parent.parent.parent / ".hydra" / "config.yaml"
        with open(config_load_path, "r") as file:
            cfg = OmegaConf.load(file)
        cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
        self.cfg = cfg

        # tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(cfg.model.vision_model.variant)
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(SIMLINGO_ADDITIONAL_SPECIAL_TOKENS)}
        )
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer

        # ``AutoConfig`` + ``trust_remote_code=True`` resolves to
        # ``InternVLChatConfig``; use the vendored class directly.
        tmp_config = InternVLChatConfig.from_pretrained(cfg.model.vision_model.variant)
        image_size = tmp_config.force_image_size or tmp_config.vision_config.image_size
        patch_size = tmp_config.vision_config.patch_size
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (tmp_config.downsample_ratio**2)
        )
        cache_dir = f"pretrained/{(cfg.model.vision_model.variant.split('/')[1])}"

        # DrivingModel's submodules are built in the default dtype; SimLingo runs
        # in bfloat16. The waypoint heads are promoted to float32 below.
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        model_kwargs = {k: v for k, v in cfg.model.items() if k != "_target_"}
        self.vlm = DrivingModel(
            cfg_data_module=cfg.data_module,
            processor=tokenizer,
            cache_dir=cache_dir,
            **model_kwargs,
        ).to(device)
        torch.set_default_dtype(default_dtype)
        self.vlm.load_state_dict(torch.load(config_path))

        # Freeze the whole VLM, then re-enable just the two SimLingo waypoint
        # heads — those *are* the deterministic policy μ and the only part of
        # the VLM that learns. Promote them to float32 so the tiny actor LR isn't
        # swallowed by bf16 rounding and they line up with the float32 critic.
        for p in self.vlm.parameters():
            p.requires_grad_(False)
        self.driving_adaptor = self.vlm.adaptors.driving
        self.actor_heads = nn.ModuleList(
            [self.driving_adaptor.route_head, self.driving_adaptor.speed_wps_head]
        ).float()
        for p in self.actor_heads.parameters():
            p.requires_grad_(True)

        self.feature_dim = int(self.vlm.language_model.hidden_size)
        self.critic = value_head_factory(self.feature_dim).to(device)

    @staticmethod
    def _resolve_checkpoint() -> Path:
        """Pull the SimLingo checkpoint from HF and return its local path.

        ``snapshot_download`` populates the HF cache; we pick the single
        ``pytorch_model.pt`` inside (excluding the blob-store hardlinks, which
        point to the same file but live under a content-addressed path that
        breaks SimLingo's ``Path(...).parent.parent.parent / .hydra/config.yaml``
        lookup).
        """
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(_HF_REPO_ID))
        candidates = [p for p in snapshot.rglob(_HF_CKPT_NAME) if "/blobs/" not in str(p)]
        if not candidates:
            raise RuntimeError(f"no {_HF_CKPT_NAME} in HF snapshot of {_HF_REPO_ID} at {snapshot}")
        return candidates[0]

    def forward(self, driving_input):
        """VLM forward → ``(pred_speed_wps, pred_route, language, features)``."""
        return self.vlm(driving_input)
