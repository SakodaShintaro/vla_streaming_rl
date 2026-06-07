# SPDX-License-Identifier: MIT
import math

import torch
from torch import nn
from torch.distributions import Beta, Categorical
from torch.nn import functional as F

from .blocks import SimbaBlock
from .sparse_utils import apply_one_shot_pruning


def _euler_denoise(
    noise: torch.Tensor,
    denoising_time: float,
    denoising_steps: int,
    predict_fn,
) -> torch.Tensor:
    """Euler denoising via flow matching (forward: 0 -> denoising_time).

    Args:
        predict_fn: callable(x_t, t) -> clean data x_1 (x-prediction).
            Velocity is derived as v = (x_1_pred - x_t) / (1 - t).

    Returns:
        tanh(x_T), same shape as noise
    """
    x_t = noise
    dt = denoising_time / denoising_steps
    time_val = 0.0
    for _ in range(denoising_steps):
        t = torch.full((noise.shape[0],), time_val, device=noise.device)
        pred = predict_fn(x_t, t)
        denom = max(1e-4, 1.0 - time_val)
        v = (pred - x_t) / denom
        x_t = x_t + dt * v
        time_val += dt
    return torch.tanh(x_t)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        block_num: int,
        denoising_time: float,
        sparsity: float,
        horizon: int,
        denoising_steps: int,
        dacer_loss_weight: float,
    ) -> None:
        super().__init__()
        time_embedding_size = 256
        self.horizon = horizon
        total_action_dim = action_dim * horizon
        self.fc_in = nn.Linear(state_dim + total_action_dim + time_embedding_size, hidden_dim)
        self.fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc_out = nn.Linear(hidden_dim, total_action_dim)
        self.action_dim = action_dim
        self.denoising_steps = denoising_steps
        self.denoising_time = denoising_time
        self.dacer_loss_weight = dacer_loss_weight
        self.t_embedder = TimestepEmbedder(time_embedding_size)
        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )

    def forward(
        self, a: torch.Tensor, t: torch.Tensor, state: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            a: flattened action (B, horizon * action_dim)
            t: timestep (B,)
            state: state embedding (B, state_dim)
        """
        result_dict = {}

        t = self.t_embedder(t)
        x = torch.cat([a, t, state], 1)
        x = self.fc_in(x)

        x = self.fc_mid(x)
        x = self.norm(x)

        result_dict["activation"] = x

        x = self.fc_out(x)
        x = torch.tanh(x)
        result_dict["output"] = x
        return result_dict

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bs = x.size(0)
        noise = torch.randn(bs, self.horizon, self.action_dim, device=x.device)

        def predict_fn(x_t, t):
            x_flat = x_t.view(bs, -1)
            return self.forward(x_flat, t, x)["output"].view(bs, self.horizon, self.action_dim)

        action = _euler_denoise(noise, self.denoising_time, self.denoising_steps, predict_fn)

        dummy_log_p = torch.zeros((bs, 1), device=x.device)
        return action, dummy_log_p

    def compute_actor_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        *,
        value_head,
        detach_actor: bool,
    ) -> tuple[torch.Tensor, dict, dict]:
        """Advantage-maximizing actor loss + DACER2 score matching
        (https://arxiv.org/abs/2505.23426).
        """
        del action_chunk
        if detach_actor:
            state = state.detach()
        action, log_pi = self.get_action(state)
        B, horizon, action_dim = action.shape
        device = action.device

        # Advantage term: maximize E[Q(s, π(s))] via the dueling advantage stream.
        for param in value_head.parameters():
            param.requires_grad_(False)
        advantage_dict = value_head.get_advantage(state, action)
        advantage = value_head.to_value(advantage_dict["output"]).view(-1, 1)
        actor_loss = -advantage.mean()
        for param in value_head.parameters():
            param.requires_grad_(True)

        # DACER2 score matching: regress the policy's x-prediction onto a
        # Q-gradient-derived velocity target, with logarithmic time weighting.
        eps = 1e-4
        t = torch.rand((B, 1), device=device) * (1 - eps) + eps
        w_t = torch.exp(0.4 * t + (-1.8))

        actions = action.view(B, -1).clone().detach().requires_grad_(True)
        actions_chunk = actions.view(B, horizon, action_dim)
        q_output_dict = value_head(state, actions_chunk)
        q_values = value_head.to_value(q_output_dict["output"]).unsqueeze(-1)
        q_grad = torch.autograd.grad(q_values.sum(), actions, create_graph=True)[0]

        noise = torch.randn_like(actions).clamp(-3.0, 3.0)
        a_t = (1.0 - t) * noise + t * actions
        with torch.no_grad():
            v_target = (1 - t) / t * q_grad + 1 / t * actions
            v_target = v_target / (v_target.norm(dim=1, keepdim=True) + 1e-8)
            v_target = w_t * v_target
            # Convert to x-space (policy outputs the clean action prediction).
            target = a_t + (1 - t) * v_target

        pred_dict = self.forward(a_t, t.squeeze(1), state)
        pred = pred_dict["output"]
        dacer_loss = F.mse_loss(pred, target)
        total_loss = actor_loss + dacer_loss * self.dacer_loss_weight

        activations_dict = {
            "actor": pred_dict["activation"],
            "critic": advantage_dict["activation"],
        }
        info_dict = {
            "actor_loss": actor_loss.item(),
            "dacer_loss": dacer_loss.item(),
            "advantage": advantage.mean().item(),
            "log_pi": log_pi.mean().item(),
        }
        return total_loss, activations_dict, info_dict


class CFGDiffusionPolicy(nn.Module):
    """
    Diffusion Policy with Classifier Free Guidance (CFGRL/pistar06)

    Training: Drop condition I (positive/negative) with condition_drop_prob probability
    Inference: Adjust guidance strength with beta (cfgrl_beta)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        block_num: int,
        denoising_time: float,
        sparsity: float,
        cfgrl_beta: float,
        horizon: int,
        denoising_steps: int,
        condition_drop_prob: float,
    ) -> None:
        super().__init__()
        self.cfgrl_beta = cfgrl_beta
        self.condition_drop_prob = condition_drop_prob
        self.horizon = horizon
        time_embedding_size = 256
        condition_embedding_size = 64
        total_action_dim = action_dim * horizon
        # Condition I embedding: 0=negative, 1=positive, 2=unconditional(dropout)
        self.condition_embedding = nn.Embedding(3, condition_embedding_size)

        self.fc_in = nn.Linear(
            state_dim + total_action_dim + time_embedding_size + condition_embedding_size,
            hidden_dim,
        )
        self.fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc_out = nn.Linear(hidden_dim, total_action_dim)
        self.action_dim = action_dim
        self.denoising_steps = denoising_steps
        self.denoising_time = denoising_time
        self.t_embedder = TimestepEmbedder(time_embedding_size)
        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )

    def forward(
        self,
        a: torch.Tensor,
        t: torch.Tensor,
        state: torch.Tensor,
        condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            a: flattened action (B, horizon * action_dim)
            t: timestep (B,)
            state: state embedding (B, state_dim)
            condition: condition I (B,) - 0=negative, 1=positive, 2=unconditional
        """
        result_dict = {}

        t_emb = self.t_embedder(t)
        cond_emb = self.condition_embedding(condition)  # (B, condition_embedding_size)
        x = torch.cat([a, t_emb, state, cond_emb], 1)
        x = self.fc_in(x)

        x = self.fc_mid(x)
        x = self.norm(x)

        result_dict["activation"] = x

        x = self.fc_out(x)
        x = torch.tanh(x)
        result_dict["output"] = x
        return result_dict

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inference using CFG.

        Guidance in output space: out = (1 - beta) * out_uncond + beta * out_positive
        Higher beta increases guidance towards positive advantage direction.
        """
        bs = x.size(0)
        device = x.device
        noise = torch.randn(bs, self.horizon, self.action_dim, device=device)

        # Condition labels: 1=positive, 2=unconditional
        cond_positive = torch.ones((bs,), dtype=torch.long, device=device)
        cond_uncond = torch.full((bs,), 2, dtype=torch.long, device=device)

        def predict_fn(x_t, t):
            x_flat = x_t.view(bs, -1)
            out_pos = self.forward(x_flat, t, x, cond_positive)["output"]
            out_unc = self.forward(x_flat, t, x, cond_uncond)["output"]
            out = (1 - self.cfgrl_beta) * out_unc + self.cfgrl_beta * out_pos
            return out.view(bs, self.horizon, self.action_dim)

        action = _euler_denoise(noise, self.denoising_time, self.denoising_steps, predict_fn)

        dummy_log_p = torch.zeros((bs, 1), device=device)
        return action, dummy_log_p

    def compute_actor_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        *,
        value_head: nn.Module,
        detach_actor: bool,
    ) -> tuple[torch.Tensor, dict, dict]:
        """CFGRL/pistar06: condition on advantage sign, drop with condition_drop_prob."""
        if detach_actor:
            state = state.detach()
        batch_size = state.shape[0]
        device = state.device
        action_flat = action_chunk.view(batch_size, -1)

        with torch.no_grad():
            advantage_dict = value_head.get_advantage(state, action_chunk)
            advantage = value_head.to_value(advantage_dict["output"]).view(-1)
            threshold = advantage.median()
            condition = (advantage >= threshold).long()
            drop_mask = torch.rand(batch_size, device=device) < self.condition_drop_prob
            condition = torch.where(drop_mask, torch.full_like(condition, 2), condition)

        eps = 1e-4
        t = torch.rand((batch_size, 1), device=device) * (1 - eps) + eps
        noise = torch.randn_like(action_flat)
        noise = torch.clamp(noise, -3.0, 3.0)
        a_t = (1.0 - t) * noise + t * action_flat
        actor_output_dict = self.forward(a_t, t.squeeze(1), state, condition)
        pred = actor_output_dict["output"]
        actor_loss = F.mse_loss(pred, action_flat)

        positive_ratio = (condition == 1).float().mean().item()
        negative_ratio = (condition == 0).float().mean().item()
        uncond_ratio = (condition == 2).float().mean().item()
        activations_dict = {"actor": actor_output_dict["activation"]}
        info_dict = {
            "actor_loss": actor_loss.item(),
            "dacer_loss": 0.0,
            "log_pi": 0.0,
            "advantage": advantage.mean().item(),
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "uncond_ratio": uncond_ratio,
        }
        return actor_loss, activations_dict, info_dict


class MeanFlowPolicy(nn.Module):
    """One-step policy network u_θ(a_t, r, t, s) trained with the SOM objective.

    Inference uses a single network evaluation:
        a = ε - u_θ(ε, r=0, t=1, s),  ε ~ N(0, I)
    matching the SOM/MeanFlow convention (t=1 noise, t=0 action).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        block_num: int,
        horizon: int,
        sparsity: float,
        som_alpha: float,
        som_w: float,
    ) -> None:
        super().__init__()
        time_embedding_size = 256
        self.horizon = horizon
        self.action_dim = action_dim
        self.som_alpha = som_alpha
        self.som_w = som_w
        total_action_dim = action_dim * horizon
        self.fc_in = nn.Linear(state_dim + total_action_dim + 2 * time_embedding_size, hidden_dim)
        self.fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc_out = nn.Linear(hidden_dim, total_action_dim)
        self.t_embedder = TimestepEmbedder(time_embedding_size)
        # Gap (t - r) embedding; at r=t it is r_embedder(0), recovering FM.
        self.r_embedder = TimestepEmbedder(time_embedding_size)
        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )

    def forward(
        self,
        a: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Args:
        a: noisy action (B, horizon*action_dim).
        t: upper time (B,) — closer to noise in SOM convention (r ≤ t).
        r: lower time (B,).
        state: (B, state_dim).
        """
        result_dict = {}
        t_emb = self.t_embedder(t)
        r_emb = self.r_embedder(t - r)
        x = torch.cat([a, t_emb, r_emb, state], 1)
        x = self.fc_in(x)
        x = self.fc_mid(x)
        x = self.norm(x)
        result_dict["activation"] = x
        x = self.fc_out(x)
        result_dict["output"] = x
        return result_dict

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bs = x.size(0)
        device = x.device
        eps = torch.randn(bs, self.horizon, self.action_dim, device=device)
        eps_flat = eps.view(bs, -1)
        t = torch.ones(bs, device=device)
        r = torch.zeros(bs, device=device)
        u = self.forward(eps_flat, t, r, x)["output"]
        action_flat = eps_flat - u
        action = torch.tanh(action_flat).view(bs, self.horizon, self.action_dim)
        dummy_log_p = torch.zeros((bs, 1), device=device)
        return action, dummy_log_p

    def compute_actor_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        *,
        value_head: nn.Module,
        detach_actor: bool,
    ) -> tuple[torch.Tensor, dict, dict]:
        """Score-based One-step MeanFlow Policy Optimization (SOM, arXiv:2605.23365).

        The target velocity for MeanFlow is derived from the Q-function via an
        iDEM-style Monte Carlo score estimator over a Boltzmann policy
        p0(a|s) ∝ exp(α Q(s,a)) under the VP-SDE forward kernel
        a_t = μ_t a_0 + σ_t ε. The MeanFlow identity is then enforced via a JVP.
        Convention here matches the SOM paper: t=1 is noise, t=0 is action, r ≤ t.
        """
        # som_alpha (Boltzmann inverse-temperature) and som_w (PF-ODE score
        # scale) are the two "greediness" knobs, exposed via config for sweeps.
        # Remaining defaults follow the SOM paper (Appendix C / Table 3).
        som_alpha = self.som_alpha
        som_w = self.som_w
        som_mc_K = 100
        som_beta_min = 0.1
        som_beta_max = 20.0

        if detach_actor:
            state = state.detach()
        B, horizon, action_dim = action_chunk.shape
        D = horizon * action_dim
        device = state.device
        a0 = action_chunk.view(B, D)

        # Sample time pair: clamp away from boundaries for VP-SDE numerical safety.
        t_eps = 1e-3
        t = torch.rand(B, device=device) * (1 - 2 * t_eps) + t_eps
        r = torch.rand(B, device=device) * t
        # 25% FM mode (r = t) so the network also learns the boundary case.
        fm_mask = torch.rand(B, device=device) < 0.25
        r = torch.where(fm_mask, t, r)

        # VP-SDE coefficients with linear β schedule β(t) = β_min + t(β_max - β_min).
        int_beta = som_beta_min * t + 0.5 * (som_beta_max - som_beta_min) * t * t
        mu_t = torch.exp(-0.5 * int_beta)
        sigma_t = torch.sqrt(1.0 - mu_t * mu_t + 1e-8)
        beta_t = som_beta_min + t * (som_beta_max - som_beta_min)

        # Forward noising kernel sample.
        noise = torch.randn(B, D, device=device)
        at = mu_t.unsqueeze(1) * a0 + sigma_t.unsqueeze(1) * noise

        # === Q-derived score estimator (iDEM, Eq. 5) ===
        at_grad = at.detach().clone().requires_grad_(True)
        proposal_std = (sigma_t / mu_t).unsqueeze(0).unsqueeze(2)  # (1, B, 1)
        delta = torch.randn(som_mc_K, B, D, device=device) * proposal_std
        a0_proposal = at_grad.unsqueeze(0) / mu_t.unsqueeze(0).unsqueeze(2) + delta
        state_exp = state.detach().unsqueeze(0).expand(som_mc_K, B, -1).reshape(-1, state.shape[1])
        a0_chunk_mc = a0_proposal.reshape(som_mc_K * B, horizon, action_dim)
        q_dict = value_head(state_exp, a0_chunk_mc)
        q_vals = value_head.to_value(q_dict["output"]).view(som_mc_K, B)
        # GRPO-style batch-wise normalization for scale-invariant training.
        q_norm = (q_vals - q_vals.mean()) / (q_vals.std() + 1e-6)
        log_sum_exp = torch.logsumexp(som_alpha * q_norm, dim=0)
        score = torch.autograd.grad(log_sum_exp.sum(), at_grad, create_graph=False)[0]

        # === PF-ODE target velocity (Eq. 8): unit-normalized score scaled by w ===
        score_norm = score / (score.norm(dim=1, keepdim=True) + 1e-6)
        v_t = -0.5 * beta_t.unsqueeze(1) * (at + som_w * score_norm)
        v_t = v_t.detach()

        # === MeanFlow identity via JVP: du/dt along (at, t) with tangents (v_t, 1) ===
        def f(at_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            return self.forward(at_in, t_in, r, state)["output"]

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            u, du_dt = torch.func.jvp(f, (at.detach(), t), (v_t, torch.ones_like(t)))

        gap = (t - r).unsqueeze(1)
        u_target = v_t - gap * du_dt
        # Adaptive per-sample weighting (MeanFlow paper) to stabilize (t-r)·du/dt.
        delta_sq = (u - u_target.detach()).pow(2)
        w_loss = 1.0 / (delta_sq.detach().mean(dim=1, keepdim=True) + 1e-3)
        actor_loss = (w_loss * delta_sq).mean()

        activations_dict = {"actor": torch.zeros((B, 1), device=device)}
        info_dict = {
            "actor_loss": actor_loss.item(),
            "dacer_loss": 0.0,
            "log_pi": 0.0,
            "advantage": q_vals.mean().item(),
            "som_score_norm": score.norm(dim=1).mean().item(),
            "som_du_dt": du_dt.detach().abs().mean().item(),
        }
        return actor_loss, activations_dict, info_dict


class BetaPolicy(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        total_action_dim = action_dim * horizon
        self.policy_enc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.alpha_head = nn.Linear(hidden_dim, total_action_dim)
        self.beta_head = nn.Linear(hidden_dim, total_action_dim)

    def forward(self, x: torch.Tensor, action: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """
        Args:
            x: state embedding (B, state_dim)
            action: action chunk (B, horizon, action_dim) or None for sampling
        """
        bs = x.size(0)
        policy_x = self.policy_enc(x)
        alpha = self.alpha_head(policy_x).exp() + 1  # (B, horizon * action_dim)
        beta = self.beta_head(policy_x).exp() + 1  # (B, horizon * action_dim)

        dist = Beta(alpha, beta)
        if action is None:
            action_01 = dist.sample()  # (B, horizon * action_dim)
            action_flat = action_01 * 2.0 - 1.0
        else:
            # action: (B, horizon, action_dim) -> (B, horizon * action_dim)
            action_flat = action.view(bs, -1)
            action_01 = (action_flat + 1.0) / 2.0

        # Sum log prob over all dimensions (horizon * action_dim)
        total_action_dim = self.action_dim * self.horizon
        a_logp = (
            dist.log_prob(action_01).sum(dim=1, keepdim=True)
            - torch.log(torch.tensor(2.0, device=policy_x.device)) * total_action_dim
        )

        # Reshape to (B, horizon, action_dim)
        action_out = action_flat.view(bs, self.horizon, self.action_dim)

        return {
            "action": action_out,  # (B, horizon, action_dim)
            "a_logp": a_logp,
            "entropy": dist.entropy().sum(dim=1, keepdim=True),  # sum over all dims
            "activation": policy_x,
        }

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.forward(x, None)
        return result["action"], result["a_logp"]

    def compute_actor_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        *,
        value_head: nn.Module,
        detach_actor: bool,
    ) -> tuple[torch.Tensor, dict, dict]:
        """REINFORCE-style policy-gradient loss used inside ActorCriticWithActionValue."""
        del action_chunk
        if detach_actor:
            state = state.detach()
        policy_output = self.forward(state, None)
        action = policy_output["action"]
        log_pi = policy_output["a_logp"]
        entropy = policy_output["entropy"]

        advantage_dict = value_head.get_advantage(state, action)
        advantage = value_head.to_value(advantage_dict["output"]).view(-1, 1)
        actor_loss = -(log_pi * advantage.detach()).mean() - 0.02 * entropy.mean()

        activations_dict = {
            "actor": policy_output["activation"],
            "critic": advantage_dict["activation"],
        }
        info_dict = {
            "actor_loss": actor_loss.item(),
            "dacer_loss": 0.0,
            "log_pi": log_pi.mean().item(),
            "advantage": advantage.mean().item(),
            "entropy": entropy.mean().item(),
        }
        return actor_loss, activations_dict, info_dict


class CategoricalPolicy(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.policy_enc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        # Output logits for each timestep in the horizon
        self.logits_head = nn.Linear(hidden_dim, action_dim * horizon)

    def forward(self, x: torch.Tensor, action: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """
        Args:
            x: state embedding (B, state_dim)
            action: action chunk (B, horizon, action_dim) or None for sampling
        """
        bs = x.size(0)
        policy_x = self.policy_enc(x)
        logits = self.logits_head(policy_x)  # (B, horizon * action_dim)
        logits = logits.view(bs, self.horizon, self.action_dim)  # (B, horizon, action_dim)

        # Create independent categorical distribution for each timestep
        dist = Categorical(logits=logits)  # batch shape (B, horizon)

        if action is None:
            action_idx = dist.sample()  # (B, horizon)
            a_logp = dist.log_prob(action_idx).sum(dim=1, keepdim=True)  # sum over horizon
            # Convert to one-hot: (B, horizon, action_dim)
            action_onehot = F.one_hot(action_idx, num_classes=self.action_dim).float()
            action_out = action_onehot * 2.0 - 1.0
        else:
            # action: (B, horizon, action_dim) - one-hot encoded
            action_idx = action.argmax(dim=2)  # (B, horizon)
            a_logp = dist.log_prob(action_idx).sum(dim=1, keepdim=True)  # sum over horizon
            action_out = action

        return {
            "action": action_out,  # (B, horizon, action_dim)
            "a_logp": a_logp,
            "entropy": dist.entropy().sum(dim=1, keepdim=True),  # sum over horizon
            "activation": policy_x,
        }

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.forward(x, None)
        return result["action"], result["a_logp"]
