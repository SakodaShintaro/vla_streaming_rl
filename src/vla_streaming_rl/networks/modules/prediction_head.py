# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn
import torch.nn.functional as F

from .flux_dit import FluxDiT
from .reward_processor import RewardProcessor


class StatePredictionHead(nn.Module):
    def __init__(
        self,
        image_latent_shape: tuple[int, int, int],
        reward_processor: RewardProcessor,
        action_dim: int,
        predictor_hidden_dim: int,
        predictor_block_num: int,
        predictor_type: str,
    ) -> None:
        super().__init__()
        self.image_latent_shape = image_latent_shape
        self.reward_processor = reward_processor
        self.predictor_type = predictor_type
        hidden_image_dim = image_latent_shape[0]
        self.state_predictor = FluxDiT(
            in_channels=hidden_image_dim,
            out_channels=hidden_image_dim,
            vec_in_dim=action_dim,
            context_in_dim=hidden_image_dim,
            hidden_size=predictor_hidden_dim,
            mlp_ratio=4.0,
            num_heads=8,
            depth_double_blocks=predictor_block_num,
            depth_single_blocks=predictor_block_num,
            axes_dim=[64],
            theta=10000,
            qkv_bias=True,
            use_r_time=(predictor_type == "mean_flow"),
        )

    def _loss_flow_matching(
        self,
        predictor_state: torch.Tensor,
        action: torch.Tensor,
        x1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        x0 = torch.randn_like(x1)
        shape_t = (x0.shape[0],) + (1,) * (len(x0.shape) - 1)
        t = torch.rand(shape_t, device=x1.device)
        xt = (1.0 - t) * x0 + t * x1
        pred_dict = self.state_predictor.forward(xt, t, predictor_state, action, None)
        pred_vt = pred_dict.output
        vt = x1 - x0
        loss = F.mse_loss(pred_vt, vt)
        return loss, pred_dict.activation, {"seq_loss": loss.item()}

    def _loss_mean_flow(
        self,
        predictor_state: torch.Tensor,
        action: torch.Tensor,
        x1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        B = x1.shape[0]
        device = x1.device
        x0 = torch.randn_like(x1)
        # Logit-normal time sampling (MeanFlow paper recommendation): μ=-0.4, σ=1.0
        # biases t toward smaller values where modelling is harder.
        normal_samples = torch.randn(B, 2, device=device) * 1.0 + (-0.4)
        time_samples = torch.sigmoid(normal_samples)
        # Lower time t is closer to noise (=0); upper time r is closer to target (=1).
        t = time_samples.min(dim=1).values
        r = time_samples.max(dim=1).values
        # r = t for 25% of samples (paper-recommended) so the network also learns FM at r=t.
        fm_mask = torch.rand(B, device=device) < 0.25
        r = torch.where(fm_mask, t, r)

        v = x1 - x0
        t_b = t.view(-1, 1, 1)
        xt = (1.0 - t_b) * x0 + t_b * x1

        def f(x: torch.Tensor, t_: torch.Tensor) -> torch.Tensor:
            return self.state_predictor.forward(x, t_, predictor_state, action, r).output

        # JVP computes total derivative along the trajectory: du/dt = ∂u/∂x · v + ∂u/∂t.
        # SDPA MATH backend because Flash Attention lacks double-backward / forward-mode AD.
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            u, du_dt = torch.func.jvp(f, (xt, t), (v, torch.ones_like(t)))

        # In our convention (t ≤ r) the identity gives u_target = v + (r - t) * du/dt.
        gap = (r - t).view(-1, 1, 1)
        u_target = v + gap * du_dt
        # Adaptive loss weighting (MeanFlow paper): without it (r-t)·du/dt self-amplifies
        # and training diverges.
        delta_sq = (u - u_target.detach()).pow(2)
        w = 1.0 / (delta_sq.detach().mean(dim=(1, 2), keepdim=True) + 1e-3)
        loss = (w * delta_sq).mean()

        # Tracking-only: restrict to the r=t subset, where u_target = v exactly.
        per_sample_uv_sq = (u - v).detach().pow(2).mean(dim=(1, 2))
        fm_count = fm_mask.sum().clamp(min=1)
        fm_mse = (per_sample_uv_sq * fm_mask.float()).sum() / fm_count
        du_dt_mag = du_dt.detach().abs().mean()

        activation = torch.zeros((B, 1), device=device)
        info = {
            "seq_loss": loss.item(),
            "mf_fm_mse": fm_mse.item(),
            "mf_du_dt": du_dt_mag.item(),
        }
        return loss, activation, info

    def encode_target(
        self, next_image_latent: torch.Tensor, next_reward: torch.Tensor
    ) -> torch.Tensor:
        """Build the flow-matching regression target ``x1`` (B, H'*W'+1, C')
        from the next image's latent (B, C', H', W'), which the caller encodes
        without gradient (fixed target), and the next reward through this
        head's reward processor, appended as the final position.
        """
        target_state_next = next_image_latent.flatten(2).permute(0, 2, 1)  # (B, H'*W', C')

        target_reward_next = self.reward_processor.encode(next_reward).squeeze(1)  # (B, C')
        return torch.cat([target_state_next, target_reward_next.unsqueeze(1)], dim=1)

    def compute_loss(
        self,
        predictor_state: torch.Tensor,
        action: torch.Tensor,
        next_image_latent: torch.Tensor,
        next_reward: torch.Tensor,
        detach_predictor: bool,
        disable_state_predictor: bool,
    ) -> tuple[torch.Tensor, dict]:
        """Flow-matching transition loss: train the predictor so that
        ``(predictor_state, action)`` maps to the encoded next ``(image, reward)``.

        Mirrors ``PolicyHead.compute_actor_loss`` / ``ValueHead.compute_critic_loss``
        — the head owns its loss and logging. ``detach_predictor`` stops the
        gradient into the encoder; ``disable_state_predictor`` skips the
        predictor entirely and returns a zero loss. The caller supplies the
        already-prepared ``predictor_state`` (its layout is network-specific).
        """
        if disable_state_predictor:
            dummy_loss = torch.tensor(0.0, device=predictor_state.device, requires_grad=True)
            return dummy_loss, {"seq_loss": 0.0}

        if detach_predictor:
            predictor_state = predictor_state.detach()

        # Context layout: (B, tokens, C) where C is the image channel dim
        # (no-op for callers already in that shape).
        C = self.image_latent_shape[0]
        predictor_state = predictor_state.view(predictor_state.size(0), -1, C)

        x1 = self.encode_target(next_image_latent, next_reward)
        if self.predictor_type == "flow_matching":
            loss, _, info = self._loss_flow_matching(predictor_state, action, x1)
        elif self.predictor_type == "mean_flow":
            loss, _, info = self._loss_mean_flow(predictor_state, action, x1)
        else:
            raise ValueError(f"Unknown predictor_type: {self.predictor_type}")
        return loss, info
