# SPDX-License-Identifier: MIT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .epona.flux_dit import FluxDiT
from .image_processor import ImageProcessor
from .reward_processor import RewardProcessor


class StatePredictionHead(nn.Module):
    def __init__(
        self,
        image_processor: ImageProcessor,
        reward_processor: RewardProcessor,
        action_dim: int,
        predictor_hidden_dim: int,
        predictor_block_num: int,
        predictor_type: str,
    ) -> None:
        super().__init__()
        self.image_processor = image_processor
        self.reward_processor = reward_processor
        self.predictor_type = predictor_type
        hidden_image_dim = image_processor.output_shape[0]
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

    def _sample_flow_matching(
        self,
        z: torch.Tensor,
        state_curr: torch.Tensor,
        action_curr: torch.Tensor,
        predictor_step_num: int,
    ) -> torch.Tensor:
        B = z.shape[0]
        device = z.device
        dt = 1.0 / predictor_step_num
        curr_time = torch.zeros((B,), device=device)
        activation = None
        for _ in range(predictor_step_num):
            head_out = self.state_predictor.forward(z, curr_time, state_curr, action_curr, None)
            activation = head_out.activation
            z = z + dt * head_out.output
            curr_time = curr_time + dt
        return z, activation

    def _sample_mean_flow(
        self,
        z: torch.Tensor,
        state_curr: torch.Tensor,
        action_curr: torch.Tensor,
        predictor_step_num: int,
    ) -> torch.Tensor:
        B = z.shape[0]
        device = z.device
        # Integrate from t=0 (noise) to t=1 (target) in `predictor_step_num` jumps.
        # At each segment [t, r] (t<r) the network predicts the average velocity u,
        # and we step z_r = z_t + (r - t) * u(z_t, t, r).
        time_steps = torch.linspace(0.0, 1.0, predictor_step_num + 1, device=device)
        activation = None
        for i in range(predictor_step_num):
            t = torch.full((B,), time_steps[i].item(), device=device)
            r = torch.full((B,), time_steps[i + 1].item(), device=device)
            head_out = self.state_predictor.forward(z, t, state_curr, action_curr, r)
            activation = head_out.activation
            z = z + (time_steps[i + 1].item() - time_steps[i].item()) * head_out.output
        return z, activation

    @torch.inference_mode()
    def predict_next_state(
        self,
        state_curr: torch.Tensor,
        action_curr: torch.Tensor,
        observation_space_shape: tuple[int, ...],
        predictor_step_num: int,
        disable_state_predictor: bool,
    ) -> tuple[np.ndarray, float, torch.Tensor]:
        if disable_state_predictor:
            H = observation_space_shape[1]
            W = observation_space_shape[2]
            next_image = np.zeros((H, W, 3), dtype=np.float32)
            next_reward = 0.0
            dummy_activation = torch.zeros((state_curr.size(0), 1), device=state_curr.device)
            return next_image, next_reward, dummy_activation

        device = state_curr.device
        B = state_curr.size(0)
        C = self.image_processor.output_shape[0]
        H = self.image_processor.output_shape[1]
        W = self.image_processor.output_shape[2]
        state_curr = state_curr.view(B, -1, C)
        noise_shape = (B, (H * W) + 1, C)
        normal = torch.distributions.Normal(
            torch.zeros(noise_shape, device=device),
            torch.ones(noise_shape, device=device),
        )
        next_hidden_state = normal.sample().to(device)
        next_hidden_state = torch.clamp(next_hidden_state, -3.0, 3.0)

        if self.predictor_type == "flow_matching":
            next_hidden_state, activation = self._sample_flow_matching(
                next_hidden_state, state_curr, action_curr, predictor_step_num
            )
        elif self.predictor_type == "mean_flow":
            next_hidden_state, activation = self._sample_mean_flow(
                next_hidden_state, state_curr, action_curr, predictor_step_num
            )
        else:
            raise ValueError(f"Unknown predictor_type: {self.predictor_type}")

        image_part = next_hidden_state[:, :-1, :]
        image_part = image_part.permute(0, 2, 1).view(B, C, H, W)
        next_image = self.image_processor.decode(image_part)
        next_image = next_image.detach().cpu().numpy()
        next_image = next_image.squeeze(0)
        next_image = next_image.transpose(1, 2, 0)

        reward_part = next_hidden_state[:, -1, :]
        next_reward = self.reward_processor.decode(reward_part)

        return next_image, next_reward.item(), activation

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

    def compute_loss(
        self,
        predictor_state: torch.Tensor,
        action: torch.Tensor,
        x1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        if self.predictor_type == "flow_matching":
            return self._loss_flow_matching(predictor_state, action, x1)
        if self.predictor_type == "mean_flow":
            return self._loss_mean_flow(predictor_state, action, x1)
        raise ValueError(f"Unknown predictor_type: {self.predictor_type}")
