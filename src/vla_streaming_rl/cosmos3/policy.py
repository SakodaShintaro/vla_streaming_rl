# SPDX-License-Identifier: MIT
"""Differentiable action policy on top of the (frozen) Cosmos3-Edge pipeline.

The stock ``Cosmos3OmniPipeline.__call__`` is a ``@torch.no_grad`` inference path
that jointly denoises future video + action with CFG. For DACER2-style policy RL
we need a *differentiable* action chunk whose gradient reaches only the action
projection heads (``action_proj_in`` / ``action_proj_out`` / ``action_modality_embed``),
with the 28-layer backbone, the VAE and the text/vision projections all frozen.

``CosmosEdgePolicy`` subclasses the pipeline. ``setup`` fixes the per-run
generation configuration once; after that two methods do the work:

* ``encode`` (no-grad): builds the joint-sequence packing for one observation —
  text + vision-conditioning latents + fresh action noise — reusing the pipeline's
  own ``_prepare_*`` helpers, and captures a pooled backbone state for the critic.
* ``sample_action_chunk`` (grad): Euler-integrates the flow for a few steps with
  gradient on the action tokens only. ``encode`` anchors the first few latent frames
  as the clean history clip; the remaining (future) vision frames are noisy and are
  generated jointly with the action (vision stepped under no-grad, action with grad).
  Returns the raw action chunk plus the vision latents holding history + generated
  future — so the action is a velocity-grounded, prompt-steered future trajectory.

Both rollout (``infer``) and the training loss sample through this method.
"""

from dataclasses import dataclass

import numpy as np
import torch
from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import (
    _ACTION_RESOLUTION_BINS,
    _EMBODIMENT_TO_DOMAIN_ID,
    VideoProcessor,
)

# Trainable action-head submodule name prefixes on the transformer.
ACTION_HEAD_PREFIXES = ("action_proj_in", "action_proj_out", "action_modality_embed")


@dataclass
class CosmosEdgeEncoding:
    """Static packing + initial latents for one observation (all detached)."""

    fwd_static: dict
    vision_latents: torch.Tensor
    action_latents: torch.Tensor
    x0_action: torch.Tensor  # clean action values for anchored (history) rows; 0 elsewhere
    vision_condition_mask: torch.Tensor
    action_condition_mask: torch.Tensor
    action_domain_id: torch.Tensor
    raw_action_dim: int
    num_noisy_vision_tokens: int
    num_noisy_action_tokens: int
    num_history_vision_tokens: int  # leading (clean history frame) tokens in the vision segment
    state: torch.Tensor  # pooled backbone state for the critic; filled by ``encode``


class CosmosEdgePolicy(Cosmos3OmniPipeline):
    def setup(
        self,
        chunk_size: int,
        domain_name: str,
        resolution_tier: int,
        num_inference_steps: int,
        num_cond_latent_frames: int,
        fps: float,
    ) -> None:
        """Fix the per-run generation configuration once (called right after
        ``from_pretrained``), so ``encode`` only takes per-observation inputs."""
        self.chunk_size = chunk_size
        self.domain_name = domain_name
        self.resolution_tier = resolution_tier
        self.num_inference_steps = num_inference_steps
        self.num_cond_latent_frames = num_cond_latent_frames
        self.fps = fps

    def freeze_backbone(self) -> list:
        """Freeze everything except the action heads; return the trainable params."""
        self.vae.requires_grad_(False)
        for name, param in self.transformer.named_parameters():
            param.requires_grad_(name.startswith(ACTION_HEAD_PREFIXES))
        return [p for p in self.transformer.parameters() if p.requires_grad]

    @property
    def pooled_state_dim(self) -> int:
        return int(self.transformer.config.hidden_size)

    def _pack_static(self, frames, prompt, history_action) -> CosmosEdgeEncoding:
        """Replicate the cond-only setup of ``__call__`` (no CFG / sound / safety).

        ``frames`` is the recent history clip; the first ``num_cond_latent_frames``
        latent frames are anchored as the clean past+current history, the rest stay
        noisy and are generated jointly with the action. ``policy`` mode leaves the
        action tokens all-noisy (generated) and tags the prompt with the policy
        caption template — the right label for "generate future + action from
        observation"; the vision anchoring (which ``policy`` would otherwise set to
        frame 0 only) is overridden below. The returned encoding's ``state`` is a
        placeholder filled by ``encode``."""
        device = self._get_execution_device()
        dtype = self.transformer.dtype
        action_cond = CosmosActionCondition(
            mode="policy",
            chunk_size=self.chunk_size,
            domain_name=self.domain_name,
            resolution_tier=self.resolution_tier,
            video=frames,
            view_point="ego_view",
        )
        num_frames = self.chunk_size + 1
        probe = self.video_processor.preprocess_video(frames)

        height, width = VideoProcessor.classify_height_width_bin(
            int(probe.shape[-2]),
            int(probe.shape[-1]),
            ratios=_ACTION_RESOLUTION_BINS[str(self.resolution_tier)],
        )
        cond_input_ids, _ = self.tokenize_prompt(
            prompt,
            None,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=self.fps,
            use_system_prompt=False,
            add_resolution_template=True,
            add_duration_template=True,
            action_mode=action_cond.mode,
            action_view_point=action_cond.view_point,
        )
        text_seg = self._prepare_text_segment(cond_input_ids, device=device)
        # ``prepare_latents`` would hand back (vision_latents, action_latents,
        # fps_vision, vision_condition_mask, action_condition_mask,
        # action_domain_id, raw_action_dim, ...), but the history-conditioned
        # anchoring below immediately overwrites vision_latents/vision_condition_mask
        # and action_condition_mask, and action_latents' *values* are never read
        # (only its shape/dtype, via zeros_like/randn_like) — only fps_vision,
        # action_domain_id and raw_action_dim actually survive. Calling
        # ``prepare_latents`` for those would pay for a second, wasted VAE encode
        # of the same padded-to-canvas conditioning video (it runs its own
        # ``_prepare_action_video_conditioning`` + ``_encode_video`` internally for
        # a single-frame anchor we don't use), so compute the three trivial
        # survivors directly instead.
        fps_vision = self.fps
        action_domain_id = torch.tensor(
            [_EMBODIMENT_TO_DOMAIN_ID[self.domain_name]], dtype=torch.long, device=device
        )
        raw_action_dim = action_cond.raw_action_dim
        action_latents = torch.zeros(
            self.chunk_size, self.transformer.action_dim, device=device, dtype=dtype
        )
        # History-conditioned joint generation: anchor the first
        # ``num_cond_latent_frames`` latent frames as the clean past+current history
        # clip; the remaining latent frames stay noisy and are generated jointly with
        # the action. VAE temporal factor is 4, so N latent frames == the first
        # ``(N-1) * 4 + 1`` pixel frames of the conditioning clip.
        vision_tensor, action_image_size, _, _ = self._prepare_action_video_conditioning(
            frames, self.resolution_tier, num_frames, device, dtype
        )
        x0_vision = self._remove_action_video_padding_from_latent(
            self._encode_video(vision_tensor).contiguous().float(), action_image_size
        ).to(dtype)
        latent_t = x0_vision.shape[2]
        vision_condition_mask = torch.zeros((latent_t, 1, 1), device=device, dtype=dtype)
        vision_condition_mask[: self.num_cond_latent_frames, 0, 0] = 1.0
        vision_latents = vision_condition_mask * x0_vision + (
            1.0 - vision_condition_mask
        ) * torch.randn_like(x0_vision)
        vc_idx = list(range(self.num_cond_latent_frames))
        vision_seg = self._prepare_vision_segment(
            input_vision_tokens=vision_latents,
            has_image_condition=bool(vc_idx),
            mrope_offset=text_seg["vision_start_temporal_offset"],
            vision_fps=fps_vision,
            curr=text_seg["und_len"],
            device=device,
            condition_frame_indexes=vc_idx,
        )
        # Anchor the history action rows with the given real ego-pose deltas (clean),
        # so only the future rows are generated. ``history_action`` is the first
        # ``n_fix`` rows in raw av dim; pad to the model's action_dim.
        chunk = action_latents.shape[0]
        x0_action = torch.zeros_like(action_latents)
        n_fix = history_action.shape[0]
        x0_action[:n_fix, : history_action.shape[1]] = history_action.to(action_latents)
        action_condition_mask = torch.zeros((chunk, 1), device=device, dtype=action_latents.dtype)
        action_condition_mask[:n_fix, 0] = 1.0
        action_seg = self._prepare_action_segment(
            input_action_tokens=action_latents,
            condition_frame_indexes=list(range(n_fix)),
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
        return CosmosEdgeEncoding(
            fwd_static=fwd_static,
            vision_latents=vision_latents,
            action_latents=action_latents,
            x0_action=x0_action,
            vision_condition_mask=vision_condition_mask,
            action_condition_mask=action_condition_mask,
            action_domain_id=action_domain_id,
            raw_action_dim=int(raw_action_dim),
            num_noisy_vision_tokens=vision_seg["num_noisy_vision_tokens"],
            num_noisy_action_tokens=action_seg["num_noisy_action_tokens"],
            num_history_vision_tokens=self.num_cond_latent_frames
            * (vision_seg["num_vision_tokens"] // latent_t),
            state=torch.zeros(0),
        )

    def _forward_step(self, enc: CosmosEdgeEncoding, vision_tokens, action_tokens, t):
        device = self._get_execution_device()
        dtype = self.transformer.dtype
        vision_timesteps = torch.full((enc.num_noisy_vision_tokens,), float(t), device=device)
        action_timesteps = torch.full((enc.num_noisy_action_tokens,), float(t), device=device)
        preds_vision, _s, preds_action = self.transformer(
            **enc.fwd_static,
            vision_tokens=[vision_tokens.to(dtype)],
            vision_timesteps=vision_timesteps,
            action_tokens=[action_tokens.to(dtype)],
            action_timesteps=action_timesteps,
            action_domain_ids=[enc.action_domain_id],
            return_dict=False,
        )
        return preds_vision, preds_action

    @torch.no_grad()
    def encode(self, frames, prompt, history_action) -> CosmosEdgeEncoding:
        """Pack one observation (history clip + prompt + real history action rows)
        and capture the pooled backbone state for the critic, via a hook on the
        final generation-stream norm from one forward at the initial (noisy)
        step. The state is the mean of the final-layer tokens of the clean history
        vision frames (the leading tokens of the generation stream). The
        understanding stream is causal over the text prompt only and never attends
        to vision tokens, so it carries no observation information; the generation
        tokens attend to the full joint sequence (prompt included).
        ``sample_action_chunk`` then produces the future ego-pose trajectory
        (grounded in the observed velocity + the navigation prompt)."""
        enc = self._pack_static(frames, prompt, history_action)

        captured = {}
        handle = self.transformer.norm_moe_gen.register_forward_hook(
            lambda mod, inp, out: captured.__setitem__("gen", out)
        )
        self.scheduler.set_timesteps(self.num_inference_steps, device=self._get_execution_device())
        t0 = self.scheduler.timesteps[0].item()
        self._forward_step(enc, enc.vision_latents, enc.action_latents, t0)
        handle.remove()
        # Vision tokens lead the generation stream, frame-major; the first
        # ``num_history_vision_tokens`` belong to the anchored (noise-free) history
        # frames, so the pooled state is deterministic given the observation.
        enc.state = (
            captured["gen"][: enc.num_history_vision_tokens].float().mean(dim=0)
        )  # [hidden_size]
        return enc

    def sample_action_chunk(
        self, enc: CosmosEdgeEncoding, num_steps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable action chunk: Euler-integrate the flow with grad on the
        action tokens; the future (non-history) vision frames are generated under
        no-grad while the anchored history frames stay fixed. Returns the raw action
        chunk ``[chunk_size, raw_action_dim]`` and the vision latents holding the
        history + generated future (the last frame is the farthest future prediction,
        for decoding/visualization)."""
        vision_latents = enc.vision_latents.clone()
        # Anchored (history) action rows start at their clean real value and are held
        # fixed (masked velocity == 0); the future rows start from noise and denoise.
        action_latents = enc.action_condition_mask * enc.x0_action + (
            1.0 - enc.action_condition_mask
        ) * torch.randn_like(enc.action_latents)
        raw = enc.raw_action_dim
        # Padding channels beyond the raw action dim are zero at train time (the
        # stock pipeline zeroes them at init and after every step); keep them clean.
        # The masked velocity is already zero there, so zeroing once at init holds.
        action_latents[:, raw:] = 0.0
        dt = -1.0 / num_steps
        for step in range(num_steps):
            t = (1.0 + step * dt) * self.scheduler.config.num_train_timesteps
            preds_vision, preds_action = self._forward_step(enc, vision_latents, action_latents, t)
            v_vision, _vs, v_action = self._mask_velocity_predictions(
                preds_vision,
                None,
                vision_condition_mask=[enc.vision_condition_mask],
                preds_action=preds_action,
                action_condition_mask=[enc.action_condition_mask],
                raw_action_dim=raw,
            )
            # Vision advances as fixed (detached) context — the world model's
            # future-frame prediction. Use the per-frame-masked velocity from
            # ``_mask_velocity_predictions`` (conditioning frame kept, noisy frames
            # denoised); never trained. NB: ``vision_condition_mask`` is per-frame
            # along the temporal axis, so it must be broadcast over the whole 5D
            # latent, not indexed at frame 0.
            with torch.no_grad():
                vision_latents = (
                    vision_latents + dt * v_vision.detach().to(vision_latents.dtype)
                ).detach()
            # Action carries gradient into the action heads.
            action_latents = action_latents + dt * v_action.to(action_latents.dtype)
        return action_latents[:, :raw].contiguous(), vision_latents

    @torch.no_grad()
    def decode_vision_latents(
        self, vision_latents: torch.Tensor, frame_index: int | slice
    ) -> np.ndarray:
        """Decode the (5D) video latents and return frame(s) ``frame_index`` as
        ``(H, W, 3)`` uint8 (or ``(N, H, W, 3)`` if ``frame_index`` is a slice).

        Mirrors the pipeline's postprocess/decode: de-normalize by the VAE latent
        statistics, VAE-decode, postprocess to numpy. Frame 0 is the conditioning
        frame (a VAE reconstruction of the current observation — a sanity check on
        the encode/decode path); the last frame is the world model's farthest
        future prediction."""
        dtype = self.vae.dtype
        mean = self._vae_latents_mean.to(device=vision_latents.device, dtype=dtype)
        inv_std = self._vae_latents_inv_std.to(device=vision_latents.device, dtype=dtype)
        z_raw = vision_latents.to(dtype) / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
        decoded = self.vae.decode(z_raw).sample
        video = self.video_processor.postprocess_video(decoded, output_type="np")[0]  # (T, H, W, 3)
        return (video[frame_index] * 255.0).clip(0, 255).astype(np.uint8)

    def action_velocity(
        self, enc: CosmosEdgeEncoding, action_tokens: torch.Tensor, flow_time: float
    ) -> torch.Tensor:
        """One differentiable action-velocity forward at ``flow_time`` in [0, 1].

        ``action_tokens`` are padded action latents ``[chunk, action_dim]``; the
        vision context is the (fixed) conditioning latents. Returns the masked
        action velocity ``[chunk, action_dim]`` for the flow-matching loss.
        """
        t = flow_time * self.scheduler.config.num_train_timesteps
        preds_vision, preds_action = self._forward_step(enc, enc.vision_latents, action_tokens, t)
        _vv, _vs, v_action = self._mask_velocity_predictions(
            preds_vision,
            None,
            vision_condition_mask=[enc.vision_condition_mask],
            preds_action=preds_action,
            action_condition_mask=[enc.action_condition_mask],
            raw_action_dim=enc.raw_action_dim,
        )
        return v_action
