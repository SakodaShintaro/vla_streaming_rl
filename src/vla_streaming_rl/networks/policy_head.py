# SPDX-License-Identifier: MIT
import math

import torch
from torch import nn
from torch.distributions import Beta, Categorical
from torch.nn import functional as F

from .blocks import SimbaBlock
from .diffusion_utils import euler_denoise
from .sparse_utils import apply_one_shot_pruning


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

        action = euler_denoise(noise, self.denoising_time, self.denoising_steps, predict_fn)

        dummy_log_p = torch.zeros((bs, 1), device=x.device)
        return action, dummy_log_p


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
    ) -> None:
        super().__init__()
        self.cfgrl_beta = cfgrl_beta
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

        action = euler_denoise(noise, self.denoising_time, self.denoising_steps, predict_fn)

        dummy_log_p = torch.zeros((bs, 1), device=device)
        return action, dummy_log_p


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
