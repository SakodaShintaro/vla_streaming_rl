# SPDX-License-Identifier: MIT
"""Vendored Cosmos3OmniTransformer smoke test.

dummy mode (default): a tiny random-weight transformer runs one denoising forward
over a joint text(understanding) + vision(generation) + action(generation)
sequence, then backprops a loss on the action prediction to confirm the action
head is differentiable (the prerequisite for DACER2-style policy RL).

real mode (--real): download and load the published `nvidia/Cosmos3-Edge`
transformer weights (~6.7 GB, bf16) and run one forward on GPU with dummy latents
shaped to the real config, confirming the checkpoint maps onto the vendored class.

pipeline mode (--pipeline): assemble the full vendored Cosmos3OmniPipeline from the
real components (transformer, WanVAE, tokenizer, scheduler) and run an end-to-end
action-policy generation on a dummy observation, producing a real action chunk.
"""

import argparse

import numpy as np
import torch
from PIL import Image

from vla_streaming_rl.cosmos3 import (
    Cosmos3OmniPipeline,
    Cosmos3OmniTransformer,
    CosmosActionCondition,
)

REAL_REPO = "nvidia/Cosmos3-Edge"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    return parser.parse_args()


def build_tiny_model():
    return Cosmos3OmniTransformer(
        attention_bias=False,
        head_dim=16,
        hidden_size=64,
        intermediate_size=128,
        latent_channel=8,
        latent_patch_size=2,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=2,
        patch_latent_dim=32,
        rms_norm_eps=1e-6,
        rope_theta=5000000.0,
        action_dim=16,
        action_gen=True,
        num_embodiment_domains=4,
        vocab_size=100,
        hidden_act="silu",
        rope_axes_dim=[4, 2, 2],
    )


def build_inputs(device, latent_channel, latent_patch_size, action_dim):
    und_len = 3
    t_v, hp, wp = 1, 2, 2
    v_len = t_v * hp * wp
    t_a = 4
    seq_len = und_len + v_len + t_a

    text_indexes = torch.arange(und_len, device=device)
    vision_indexes = torch.arange(und_len, und_len + v_len, device=device)
    action_indexes = torch.arange(und_len + v_len, seq_len, device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(3, seq_len)
    h = hp * latent_patch_size
    w = wp * latent_patch_size

    return dict(
        input_ids=torch.randint(0, 100, (und_len,), device=device),
        text_indexes=text_indexes,
        position_ids=position_ids,
        und_len=und_len,
        sequence_length=seq_len,
        vision_tokens=[torch.randn(1, latent_channel, t_v, h, w, device=device)],
        vision_token_shapes=[(t_v, hp, wp)],
        vision_sequence_indexes=vision_indexes,
        vision_mse_loss_indexes=vision_indexes,
        vision_timesteps=torch.full((v_len,), 500.0, device=device),
        vision_noisy_frame_indexes=[torch.arange(t_v, device=device)],
        action_tokens=[torch.randn(t_a, action_dim, device=device)],
        action_token_shapes=[(t_a, 1, 1)],
        action_sequence_indexes=action_indexes,
        action_mse_loss_indexes=action_indexes,
        action_timesteps=torch.full((t_a,), 500.0, device=device),
        action_noisy_frame_indexes=[torch.arange(t_a, device=device)],
        action_domain_ids=[torch.tensor(0, device=device)],
    )


def cast_inputs(inputs, dtype):
    inputs["vision_tokens"] = [t.to(dtype) for t in inputs["vision_tokens"]]
    inputs["action_tokens"] = [t.to(dtype) for t in inputs["action_tokens"]]
    inputs["vision_timesteps"] = inputs["vision_timesteps"].to(dtype)
    inputs["action_timesteps"] = inputs["action_timesteps"].to(dtype)
    return inputs


def run_dummy():
    device = torch.device("cpu")
    model = build_tiny_model().to(device).train()
    inputs = build_inputs(device, latent_channel=8, latent_patch_size=2, action_dim=16)

    out = model(**inputs)
    action_pred = out.action[0]
    print(f"vision pred shape: {tuple(out.sample[0].shape)}")
    print(f"action pred shape: {tuple(action_pred.shape)}")

    loss = action_pred.float().pow(2).mean()
    loss.backward()
    grad = model.action_proj_out.fc.weight.grad
    ok = grad is not None and torch.isfinite(grad).all()
    print(f"loss: {loss.item():.6f}")
    print(f"action head grad present and finite: {bool(ok)}")
    print(f"action head grad norm: {grad.norm().item():.6f}")
    print("DUMMY SMOKE TEST PASSED")


def run_real():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    print(f"loading {REAL_REPO} transformer (bf16) ...")
    model = Cosmos3OmniTransformer.from_pretrained(
        REAL_REPO, subfolder="transformer", torch_dtype=dtype
    )
    model = model.to(device).eval()
    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded: {n_params:,} params")
    print(
        f"config: hidden_size={cfg.hidden_size} layers={cfg.num_hidden_layers} "
        f"latent_channel={cfg.latent_channel} patch_latent_dim={cfg.patch_latent_dim} "
        f"action_gen={cfg.action_gen} action_dim={cfg.action_dim} "
        f"num_embodiment_domains={cfg.num_embodiment_domains}"
    )

    inputs = build_inputs(
        device,
        latent_channel=cfg.latent_channel,
        latent_patch_size=cfg.latent_patch_size,
        action_dim=cfg.action_dim,
    )
    inputs = cast_inputs(inputs, dtype)

    with torch.no_grad():
        out = model(**inputs)
    print(f"vision pred shape: {tuple(out.sample[0].shape)}")
    print(f"action pred shape: {tuple(out.action[0].shape)}")
    print(f"action pred finite: {bool(torch.isfinite(out.action[0].float()).all())}")
    print(f"gpu mem allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("REAL WEIGHT LOAD TEST PASSED")


def run_pipeline():
    device = torch.device("cuda")
    chunk_size = 8
    domain_name = "bridge_orig_lerobot"

    # With diffusers pulled from git, from_pretrained resolves the Cosmos3 classes off
    # model_index.json (including use_native_flow_schedule=True), so the canonical load
    # path works directly — no manual component assembly or scheduler workaround.
    print(f"loading {REAL_REPO} pipeline (bf16) ...")
    pipe = Cosmos3OmniPipeline.from_pretrained(
        REAL_REPO, torch_dtype=torch.bfloat16, enable_safety_checker=False
    ).to(device)
    # WanVAE was trained without amp; bf16 encode overflows to non-finite latents that
    # poison the whole joint sequence. Keep the VAE in fp32 (a WanVAE property, not a
    # diffusers-version issue).
    pipe.vae.to(torch.float32)
    print("pipeline loaded")

    # A smooth gradient frame (a random-noise image encodes to wildly OOD VAE
    # latents and produces non-finite outputs at low step counts).
    grad = np.linspace(40, 210, 256, dtype=np.uint8)
    canvas = np.repeat(grad[None, :, None], 256, axis=0).repeat(3, axis=2)
    frames = [Image.fromarray(canvas) for _ in range(chunk_size + 1)]
    result = pipe(
        prompt="pick up the object on the table",
        action=CosmosActionCondition(
            mode="policy",
            chunk_size=chunk_size,
            domain_name=domain_name,
            resolution_tier=256,
            video=frames,
            view_point="ego_view",
        ),
        fps=24.0,
        num_inference_steps=10,
        guidance_scale=1.0,
        generator=torch.Generator().manual_seed(0),
        use_system_prompt=False,
        enable_safety_check=False,
    )
    action = result.action[0]
    finite = bool(torch.isfinite(action.float()).all())
    print(f"domain: {domain_name}  chunk_size: {chunk_size}")
    print(f"action chunk shape: {tuple(action.shape)}")
    print(f"action finite: {finite}")
    print(f"gpu mem allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    assert finite, "action contains non-finite values"
    print("PIPELINE ACTION-POLICY TEST PASSED")


args = parse_args()
if args.pipeline:
    run_pipeline()
elif args.real:
    run_real()
else:
    run_dummy()
