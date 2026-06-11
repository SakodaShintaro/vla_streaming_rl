# SPDX-License-Identifier: MIT
"""Action-independent goal-image predictor backed by the trained Self-Forcing
world model.

External interface:
- ``__init__(...)``: configure
- ``step(obs)`` -> goal frame for this step
- ``reset()``: drop rolling state (e.g. on episode boundary)

Everything else is private.

Internally we keep a rolling buffer of finalized **latents** (length = K_lat),
not pixel frames, and the Wan VAE encoder's causal feat_map cache is persisted
across calls. Each ``step(obs)`` accumulates pixels into the in-flight latent
position and only invokes the encoder when a full latent's worth of pixels is
ready (1 pixel for the seed, then 4 pixels per subsequent latent). DiT
inference fires every ``predict_interval`` env steps once the latent buffer is
full; the resulting (fpb*4)-frame predicted block is cached and the goal frame
returned by ``step`` advances through the block over the next predict_interval
calls so the displayed lookahead stays constant at
(block_pix + 1 - predict_interval) frames.

When ``enabled=False``, the heavy world-model pipeline is not loaded and every
``step`` call returns a pre-allocated black image.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from vla_streaming_rl.self_forcing.model.inference_model import CausalInferencePipeline
from vla_streaming_rl.self_forcing.utils.misc import (
    CachedTextEncoder,
    load_generator_state_dict,
    resolve_checkpoint_path,
)

_INFERENCE_KEY_ORDER = ("generator_ema", "generator", "model")


class WorldModelGoalPredictor:
    def __init__(
        self,
        *,
        enabled: bool,
        config_path: str,
        checkpoint_path: str | None,
        device: torch.device,
        num_context_blocks: int,
    ) -> None:
        self._enabled = bool(enabled)
        self._device = device

        config = OmegaConf.load(config_path)
        self._fpb = int(config.num_frame_per_block)
        self._fixed_caption = config.b2d_caption
        self._K_lat = int(num_context_blocks) * self._fpb
        self.target_h = int(config.pixel_height)
        self.target_w = int(config.pixel_width)
        # block_pix and predict_interval are read-only public attributes so
        # callers (infer_valid.py, render code) can derive lookahead/cycle
        # info without duplicating the formula.
        self.block_pix = self._fpb * 4
        self.predict_interval = int(config.predict_interval)

        if not (1 <= self.predict_interval <= self.block_pix):
            raise ValueError(
                f"predict_interval must be in [1, {self.block_pix}] (block size); "
                f"got {self.predict_interval}"
            )

        self._step_counter: int = 0
        self._pix_acc: list[torch.Tensor] = []  # pixels for the in-flight latent
        self._latent_buffer: list[torch.Tensor] = []  # finalized latents, len <= K_lat
        self._seed_done: bool = False
        # Full predicted block; black until first inference completes.
        self._latest_block = np.zeros(
            (self.block_pix, self.target_h, self.target_w, 3), dtype=np.uint8
        )

        self._pipeline: CausalInferencePipeline | None = None
        self._vae_enc_cache: list | None = None
        if self._enabled:
            self._pipeline = self._build_pipeline(config, checkpoint_path)
            self._vae_enc_cache = self._pipeline.vae.model.make_encoder_cache()

    @property
    def enabled(self) -> bool:
        """Whether the world model is loaded. Callers gate the goal panel on
        this (a static, per-run value) so the panel set stays stable across the
        run rather than appearing once the latent buffer fills."""
        return self._enabled

    def reset(self) -> None:
        """Drop the rolling state (e.g. on episode end)."""
        self._step_counter = 0
        self._pix_acc.clear()
        self._latent_buffer.clear()
        self._seed_done = False
        self._latest_block = np.zeros_like(self._latest_block)
        if self._enabled:
            self._vae_enc_cache = self._pipeline.vae.model.make_encoder_cache()

    @torch.inference_mode()
    def step(self, obs: np.ndarray) -> np.ndarray:
        """Push one observation frame and return the goal frame for this step.

        The goal is the model's prediction of the world
        (block_pix + 1 - predict_interval) pixel frames into the future. With
        block_pix=12: predict_interval=1 → 12 frames lookahead;
        predict_interval=6 → 7 frames lookahead; predict_interval=12 → 1 frame
        lookahead.

        Args:
            obs: (3, H, W) float in [0, 1] — the current env observation.

        Returns:
            (H, W, 3) uint8 RGB image at Wan native resolution. Black until the
            latent buffer first fills; or always black when ``enabled=False``.
        """
        if self._enabled:
            self._push(obs)
            if (
                len(self._latent_buffer) >= self._K_lat
                and self._step_counter % self.predict_interval == 0
            ):
                self._latest_block = self._run_inference()

        cycle_step = self._step_counter % self.predict_interval
        goal_frame_idx = (self.block_pix - self.predict_interval) + cycle_step
        goal = self._latest_block[goal_frame_idx]
        self._step_counter += 1
        return goal

    def _build_pipeline(self, config, checkpoint_path: str | None) -> CausalInferencePipeline:
        from peft import LoraConfig, get_peft_model

        # Wan VAE downsamples 8x spatially, so latent dims = pixel dims / 8.
        pipeline = CausalInferencePipeline(
            timestep_shift=config.timestep_shift,
            num_frame_per_block=self._fpb,
            context_noise=config.context_noise,
            latent_height=self.target_h // 8,
            latent_width=self.target_w // 8,
        )
        base_ckpt_path = resolve_checkpoint_path(config.generator_ckpt)
        pipeline.generator.load_state_dict(
            load_generator_state_dict(base_ckpt_path, prefer_keys=_INFERENCE_KEY_ORDER),
            strict=True,
        )

        lora_cfg = config.lora
        if not bool(lora_cfg.enabled):
            raise ValueError("config.lora.enabled must be true.")
        pipeline.generator.model.requires_grad_(False)
        peft_cfg = LoraConfig(
            r=int(lora_cfg.rank),
            lora_alpha=int(lora_cfg.alpha),
            lora_dropout=float(lora_cfg.dropout),
            target_modules=list(lora_cfg.target_modules),
            bias="none",
        )
        pipeline.generator.model = get_peft_model(pipeline.generator.model, peft_cfg)

        if checkpoint_path:
            sd = load_generator_state_dict(
                resolve_checkpoint_path(checkpoint_path), prefer_keys=_INFERENCE_KEY_ORDER
            )
            missing, unexpected = pipeline.generator.load_state_dict(sd, strict=False)
            print(
                f"[WorldModelGoalPredictor] finetune load: "
                f"{len(missing)} missing, {len(unexpected)} unexpected"
            )

        pipeline = pipeline.to(dtype=torch.bfloat16)
        pipeline.generator.to(device=self._device)
        pipeline.vae.to(device=self._device)
        pipeline.text_encoder.to(device=self._device)
        cached_text = pipeline.text_encoder([self._fixed_caption])
        pipeline.text_encoder = CachedTextEncoder(cached_text, device=self._device)
        torch.cuda.empty_cache()
        return pipeline

    def _push(self, obs: np.ndarray) -> None:
        t = torch.from_numpy(obs).to(device=self._device, dtype=torch.float32)
        if t.shape[1:] != (self.target_h, self.target_w):
            t = F.interpolate(
                t.unsqueeze(0),
                size=(self.target_h, self.target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        t = (t * 2.0 - 1.0).to(dtype=torch.bfloat16)
        self._pix_acc.append(t)

        # Seed call: 1 pixel -> 1 latent (Wan VAE special case).
        # Subsequent calls: 4 pixels -> 1 latent.
        threshold = 1 if not self._seed_done else 4
        if len(self._pix_acc) >= threshold:
            self._encode_chunk()

    @torch.inference_mode()
    def _encode_chunk(self) -> None:
        pix = torch.stack(self._pix_acc, dim=1).unsqueeze(0)  # (1, C, T, H, W)
        scale = self._pipeline.vae._scale(pix.device, pix.dtype)
        mu = self._pipeline.vae.model.encode(pix, scale, cache=self._vae_enc_cache)
        # mu: (1, 16, T_lat, H_lat, W_lat); permute to (1, T_lat, 16, H_lat, W_lat).
        mu = mu.permute(0, 2, 1, 3, 4).to(dtype=torch.bfloat16)
        for t_idx in range(mu.shape[1]):
            self._latent_buffer.append(mu[:, t_idx : t_idx + 1].contiguous())
        if len(self._latent_buffer) > self._K_lat:
            del self._latent_buffer[: len(self._latent_buffer) - self._K_lat]
        self._pix_acc.clear()
        self._seed_done = True

    @torch.inference_mode()
    def _run_inference(self) -> np.ndarray:
        ctx_latent = torch.cat(self._latent_buffer[-self._K_lat :], dim=1)
        # (1, K_lat, 16, H_lat, W_lat)
        noise = torch.randn(
            (1, self._fpb, 16, ctx_latent.shape[-2], ctx_latent.shape[-1]),
            device=self._device,
            dtype=torch.bfloat16,
        )
        _, all_lat = self._pipeline.inference(
            noise=noise,
            text_prompts=[self._fixed_caption],
            initial_latent=ctx_latent,
        )
        decode_seq = torch.cat([ctx_latent, all_lat[:, -self._fpb :]], dim=1)
        video = self._pipeline.vae.decode_to_pixel(decode_seq)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        out = (video[0].permute(0, 2, 3, 1).cpu().float() * 255).clamp(0, 255).to(torch.uint8)
        return out.numpy()[-self.block_pix :]
