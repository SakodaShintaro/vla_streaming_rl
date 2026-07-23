# SPDX-License-Identifier: MIT
"""Differentiable action policy on top of the (frozen) Cosmos3-Edge pipeline.

The stock ``Cosmos3OmniPipeline.__call__`` is a ``@torch.no_grad`` inference path
that jointly denoises future video + action with CFG. For DACER2-style policy RL
we need a *differentiable* action chunk whose gradient reaches only the action
projection heads (``action_proj_in`` / ``action_proj_out`` / ``action_modality_embed``),
with the 28-layer backbone, the VAE and the text/vision projections all frozen.

``CosmosEdgePolicy`` subclasses the pipeline and adds two methods:

* ``encode`` (no-grad): builds the joint-sequence packing for one observation —
  text + vision-conditioning latents + fresh action noise — reusing the pipeline's
  own ``_prepare_*`` helpers, and captures a pooled backbone state for the critic.
* ``sample_action_chunk`` (grad): re-runs the joint denoise for a few steps with
  gradient enabled on the action tokens only (vision is stepped under no-grad and
  detached each step, so it is fixed context), returning the raw action chunk.

Rollout actions for the env still come from the validated ``__call__`` inference
path; this module is used only inside the training loss.
"""

from dataclasses import dataclass

import numpy as np
import torch
from diffusers import Cosmos3OmniPipeline, CosmosActionCondition

# Trainable action-head submodule name prefixes on the transformer.
ACTION_HEAD_PREFIXES = ("action_proj_in", "action_proj_out", "action_modality_embed")


@dataclass
class CosmosEdgeEncoding:
    """Static packing + initial latents for one observation (all detached)."""

    fwd_static: dict
    vision_latents: torch.Tensor
    action_latents: torch.Tensor
    vision_condition_mask: torch.Tensor
    action_condition_mask: torch.Tensor
    action_domain_id: torch.Tensor
    raw_action_dim: int
    num_noisy_vision_tokens: int
    num_noisy_action_tokens: int
    state: torch.Tensor


class CosmosEdgePolicy(Cosmos3OmniPipeline):
    def freeze_backbone(self) -> list:
        """Freeze everything except the action heads; return the trainable params."""
        self.vae.requires_grad_(False)
        for name, param in self.transformer.named_parameters():
            param.requires_grad_(name.startswith(ACTION_HEAD_PREFIXES))
        return [p for p in self.transformer.parameters() if p.requires_grad]

    @property
    def pooled_state_dim(self) -> int:
        return int(self.transformer.config.hidden_size)

    def _pack_static(
        self, frames, prompt, action_cond, num_inference_steps, generator, device, dtype
    ):
        """Replicate the cond-only setup of ``__call__`` (no CFG / sound / safety)."""
        num_frames = action_cond.chunk_size + 1
        probe = self.video_processor.preprocess_video(frames)
        from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import (
            _ACTION_RESOLUTION_BINS,
            VideoProcessor,
        )

        height, width = VideoProcessor.classify_height_width_bin(
            int(probe.shape[-2]),
            int(probe.shape[-1]),
            ratios=_ACTION_RESOLUTION_BINS[str(action_cond.resolution_tier)],
        )
        cond_input_ids, _ = self.tokenize_prompt(
            prompt,
            None,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=24.0,
            use_system_prompt=False,
            add_resolution_template=True,
            add_duration_template=True,
            action_mode=action_cond.mode,
            action_view_point=action_cond.view_point,
        )
        text_seg = self._prepare_text_segment(cond_input_ids, device=device)
        (
            vision_latents,
            _sound,
            action_latents,
            fps_vision,
            _fps_sound,
            vision_condition_mask,
            _scm,
            action_condition_mask,
            action_domain_id,
            _img_size,
            raw_action_dim,
            action_cond_frames,
        ) = self.prepare_latents(
            image=None,
            video=frames,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=24.0,
            generator=generator,
            device=device,
            dtype=dtype,
            enable_sound=False,
            action=action_cond,
        )
        vc_idx = (
            torch.nonzero(vision_condition_mask[:, 0, 0] > 0, as_tuple=False).flatten().tolist()
        )
        vision_seg = self._prepare_vision_segment(
            input_vision_tokens=vision_latents,
            has_image_condition=bool(vc_idx),
            mrope_offset=text_seg["vision_start_temporal_offset"],
            vision_fps=fps_vision,
            curr=text_seg["und_len"],
            device=device,
            condition_frame_indexes=vc_idx,
        )
        action_seg = self._prepare_action_segment(
            input_action_tokens=action_latents,
            condition_frame_indexes=action_cond_frames,
            mrope_offset=text_seg["vision_start_temporal_offset"],
            action_fps=fps_vision,
            curr=text_seg["und_len"] + vision_seg["num_vision_tokens"],
            device=device,
        )
        position_ids = torch.cat(
            [
                text_seg["text_mrope_ids"],
                vision_seg["vision_mrope_ids"],
                action_seg["action_mrope_ids"],
            ],
            dim=1,
        )
        sequence_length = (
            text_seg["und_len"] + vision_seg["num_vision_tokens"] + action_seg["action_len"]
        )
        fwd_static = dict(
            input_ids=text_seg["input_ids"],
            text_indexes=text_seg["text_indexes"],
            position_ids=position_ids,
            und_len=text_seg["und_len"],
            sequence_length=sequence_length,
            vision_token_shapes=vision_seg["vision_token_shapes"],
            vision_sequence_indexes=vision_seg["vision_sequence_indexes"],
            vision_mse_loss_indexes=vision_seg["vision_mse_loss_indexes"],
            vision_noisy_frame_indexes=vision_seg["vision_noisy_frame_indexes"],
            action_token_shapes=action_seg["action_token_shapes"],
            action_sequence_indexes=action_seg["action_sequence_indexes"],
            action_mse_loss_indexes=action_seg["action_mse_loss_indexes"],
            action_noisy_frame_indexes=action_seg["action_noisy_frame_indexes"],
        )
        meta = dict(
            num_noisy_vision_tokens=vision_seg["num_noisy_vision_tokens"],
            num_noisy_action_tokens=action_seg["num_noisy_action_tokens"],
            raw_action_dim=int(raw_action_dim),
        )
        return (
            fwd_static,
            vision_latents,
            action_latents,
            vision_condition_mask,
            action_condition_mask,
            action_domain_id,
            fps_vision,
            meta,
        )

    def _forward_step(
        self,
        fwd_static,
        vision_tokens,
        action_tokens,
        action_domain_id,
        t,
        num_noisy_vision,
        num_noisy_action,
        device,
    ):
        dtype = self.transformer.dtype
        vision_timesteps = torch.full((num_noisy_vision,), float(t), device=device)
        action_timesteps = torch.full((num_noisy_action,), float(t), device=device)
        preds_vision, _s, preds_action = self.transformer(
            **fwd_static,
            vision_tokens=[vision_tokens.to(dtype)],
            vision_timesteps=vision_timesteps,
            action_tokens=[action_tokens.to(dtype)],
            action_timesteps=action_timesteps,
            action_domain_ids=[action_domain_id],
            return_dict=False,
        )
        return preds_vision, preds_action

    @torch.no_grad()
    def encode(
        self,
        frames,
        prompt,
        chunk_size,
        domain_name,
        resolution_tier,
        num_inference_steps,
        generator,
    ) -> CosmosEdgeEncoding:
        device = self._get_execution_device()
        dtype = self.transformer.dtype
        action_cond = CosmosActionCondition(
            mode="policy",
            chunk_size=chunk_size,
            domain_name=domain_name,
            resolution_tier=resolution_tier,
            video=frames,
            view_point="ego_view",
        )
        (
            fwd_static,
            vision_latents,
            action_latents,
            vision_condition_mask,
            action_condition_mask,
            action_domain_id,
            _fps,
            meta,
        ) = self._pack_static(
            frames, prompt, action_cond, num_inference_steps, generator, device, dtype
        )

        # Capture a pooled backbone state for the critic via a hook on the final
        # understanding-stream norm, from one forward at the initial (noisy) step.
        captured = {}
        handle = self.transformer.norm.register_forward_hook(
            lambda mod, inp, out: captured.__setitem__("und", out)
        )
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        t0 = self.scheduler.timesteps[0].item()
        self._forward_step(
            fwd_static,
            vision_latents,
            action_latents,
            action_domain_id,
            t0,
            meta["num_noisy_vision_tokens"],
            meta["num_noisy_action_tokens"],
            device,
        )
        handle.remove()
        state = captured["und"].float().mean(dim=0)  # [hidden_size]

        return CosmosEdgeEncoding(
            fwd_static=fwd_static,
            vision_latents=vision_latents,
            action_latents=action_latents,
            vision_condition_mask=vision_condition_mask,
            action_condition_mask=action_condition_mask,
            action_domain_id=action_domain_id,
            raw_action_dim=meta["raw_action_dim"],
            num_noisy_vision_tokens=meta["num_noisy_vision_tokens"],
            num_noisy_action_tokens=meta["num_noisy_action_tokens"],
            state=state,
        )

    def sample_action_chunk(
        self, enc: CosmosEdgeEncoding, num_steps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable action chunk: Euler-integrate the flow with grad on the
        action tokens; vision is stepped under no-grad (fixed context). Returns the
        raw action chunk ``[chunk_size, raw_action_dim]`` and the (detached) final
        predicted vision latents (the world model's future-frame prediction, for
        decoding/visualization)."""
        device = self._get_execution_device()
        vision_latents = enc.vision_latents.clone()
        action_latents = torch.randn_like(enc.action_latents)
        raw = enc.raw_action_dim
        dt = -1.0 / num_steps
        for step in range(num_steps):
            t = (1.0 + step * dt) * self.scheduler.config.num_train_timesteps
            preds_vision, preds_action = self._forward_step(
                enc.fwd_static,
                vision_latents,
                action_latents,
                enc.action_domain_id,
                t,
                enc.num_noisy_vision_tokens,
                enc.num_noisy_action_tokens,
                device,
            )
            _vv, _vs, v_action = self._mask_velocity_predictions(
                preds_vision,
                None,
                vision_condition_mask=[enc.vision_condition_mask],
                preds_action=preds_action,
                action_condition_mask=[enc.action_condition_mask],
                raw_action_dim=raw,
            )
            # Vision advances under no-grad (fixed context; never trained).
            with torch.no_grad():
                v_vision = preds_vision[0] * (1.0 - enc.vision_condition_mask[0]).to(
                    preds_vision[0].dtype
                )
                vision_latents = (vision_latents + dt * v_vision).detach()
            # Action carries gradient into the action heads.
            action_latents = action_latents + dt * v_action.to(action_latents.dtype)
        return action_latents[:, :raw].contiguous(), vision_latents

    @torch.no_grad()
    def decode_vision_latents(self, vision_latents: torch.Tensor) -> np.ndarray:
        """Decode the predicted (5D) video latents to the farthest future RGB frame.

        Mirrors the pipeline's postprocess/decode: de-normalize by the VAE latent
        statistics, VAE-decode, postprocess to numpy, and return the last frame
        ``(H, W, 3)`` uint8 (the farthest future prediction of the world model)."""
        dtype = self.vae.dtype
        mean = self._vae_latents_mean.to(device=vision_latents.device, dtype=dtype)
        inv_std = self._vae_latents_inv_std.to(device=vision_latents.device, dtype=dtype)
        z_raw = vision_latents.to(dtype) / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
        decoded = self.vae.decode(z_raw).sample
        video = self.video_processor.postprocess_video(decoded, output_type="np")[0]  # (T, H, W, 3)
        return (video[-1] * 255.0).clip(0, 255).astype(np.uint8)

    def action_velocity(
        self, enc: CosmosEdgeEncoding, action_tokens: torch.Tensor, flow_time: float
    ) -> torch.Tensor:
        """One differentiable action-velocity forward at ``flow_time`` in [0, 1].

        ``action_tokens`` are padded action latents ``[chunk, action_dim]``; the
        vision context is the (fixed) conditioning latents. Returns the masked
        action velocity ``[chunk, action_dim]`` for the flow-matching loss.
        """
        device = self._get_execution_device()
        t = flow_time * self.scheduler.config.num_train_timesteps
        preds_vision, preds_action = self._forward_step(
            enc.fwd_static,
            enc.vision_latents,
            action_tokens,
            enc.action_domain_id,
            t,
            enc.num_noisy_vision_tokens,
            enc.num_noisy_action_tokens,
            device,
        )
        _vv, _vs, v_action = self._mask_velocity_predictions(
            preds_vision,
            None,
            vision_condition_mask=[enc.vision_condition_mask],
            preds_action=preds_action,
            action_condition_mask=[enc.action_condition_mask],
            raw_action_dim=enc.raw_action_dim,
        )
        return v_action
