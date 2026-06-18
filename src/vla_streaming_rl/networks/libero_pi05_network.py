# SPDX-License-Identifier: MIT
"""The learnable network behind :class:`LiberoPi05Agent`.

Wraps LeRobot's pi0.5 (``PI05Policy``) flow-matching VLA together with its
pretrained pre/post-processor pipelines (which carry the LIBERO dataset
normalization statistics) in a single ``nn.Module``, mirroring how
``SimLingoNetwork`` bundles the SimLingo VLM + heads for ``SimLingoAgent``.

The PaliGemma vision/LLM backbone is frozen; only the action expert and its
projections stay trainable, so online RL fine-tuning updates the small expert
on top of a fixed perception stack. The processor pipelines and the feature
keys the agent's inference path needs are exposed as attributes.
"""

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from torch import nn

# LeRobot observation/action keys for the LIBERO pi0.5 checkpoint.
OBS_IMAGE_AGENTVIEW = "observation.images.image"
OBS_IMAGE_WRIST = "observation.images.image2"
OBS_STATE = "observation.state"
TASK_KEY = "task"
ACTION_KEY = "action"


class LiberoPi05Network(nn.Module):
    """pi0.5 policy (frozen backbone + trainable action expert) for LIBERO.

    Exposed attributes used by ``LiberoPi05Agent``:
      - ``policy`` — the ``PI05Policy`` (inference + per-sample flow-matching loss).
      - ``preprocessor`` / ``postprocessor`` — pipelines that normalize/tokenize
        a raw observation dict and un-normalize predicted actions.
      - ``trainable_parameters`` — the action-expert params the agent optimizes.
      - ``state_dim`` / ``action_dim`` / ``chunk_size`` — shapes the agent needs
        to assemble batches and slice executed action chunks.
    """

    def __init__(
        self,
        *,
        device: torch.device,
    ) -> None:
        super().__init__()
        checkpoint_repo = "lerobot/pi05_libero_finetuned_quantiles_v044"
        cfg = PreTrainedConfig.from_pretrained(checkpoint_repo)
        cfg.device = str(device)
        # Freeze the PaliGemma vision + LLM; keep only the action expert and its
        # projections trainable. PI05Pytorch reads these flags at construction.
        cfg.freeze_vision_encoder = True
        cfg.train_expert_only = True
        # torch.compile (max-autotune) is disabled: it traces a fixed shape and
        # breaks on the loss path. Gradient checkpointing is kept ON: even with a
        # frozen backbone, the training forward runs the full PaliGemma in the
        # same graph as the trainable action expert, so its activations are
        # retained for the expert's backward. Recomputing them (checkpointing)
        # rather than storing them is what makes the 4B model's update fit in
        # GPU memory. (Requires the policy be in train() mode at loss time, which
        # ``_apply_checkpoint`` gates on — see LiberoPi05Agent._train.)
        cfg.compile_model = False
        cfg.gradient_checkpointing = True
        self.cfg = cfg

        self.policy = PI05Policy.from_pretrained(checkpoint_repo, config=cfg).to(device)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=checkpoint_repo
        )

        self.state_dim = int(cfg.input_features[OBS_STATE].shape[0])
        self.action_dim = int(cfg.output_features[ACTION_KEY].shape[0])
        self.chunk_size = int(cfg.chunk_size)

        # ``train_expert_only`` already cleared requires_grad on the frozen
        # backbone; collect what remains so the agent's optimizer owns exactly
        # the trainable expert parameters.
        self.trainable_parameters = [p for p in self.policy.parameters() if p.requires_grad]

    def forward(self, batch: dict, reduction: str) -> tuple[torch.Tensor, dict]:
        """Per-sample (``reduction="none"``) or mean flow-matching loss."""
        return self.policy.forward(batch, reduction=reduction)
