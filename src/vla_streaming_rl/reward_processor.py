# SPDX-License-Identifier: MIT
import gymnasium as gym
import numpy as np
import torch
from torch import nn


class RewardProcessor:
    """Reward Processor."""

    def __init__(self, processing_type: str, reward_scale: float) -> None:
        self.return_rms = gym.wrappers.utils.RunningMeanStd(shape=())
        self.epsilon = 1e-8
        self.type = processing_type
        self.reward_scale = reward_scale
        assert self.reward_scale > 0.0

    def update(self, reward: float) -> None:
        """Update the running mean and std with the new reward."""
        self.return_rms.update(np.array([reward]))

    def normalize(self, reward: torch.Tensor) -> torch.Tensor:
        """Normalize the reward."""
        if self.type == "none":
            result = reward
        elif self.type == "const":
            result = reward * self.reward_scale
        elif self.type == "scaling":
            result = reward / np.sqrt(self.return_rms.var + self.epsilon)
            result *= self.reward_scale
        elif self.type == "centering":
            result = (reward - self.return_rms.mean) / np.sqrt(self.return_rms.var + self.epsilon)
            result *= self.reward_scale
        else:
            msg = "Invalid normalizer type"
            raise ValueError(msg)

        MAX_VALUE = 10.0
        result = torch.clamp(result, -MAX_VALUE, MAX_VALUE)
        return result


class RunningNormalizer(nn.Module):
    """Per-component running mean/std normalizer for a fixed-width vector.

    統計はネットワークの buffer として持つので、チェックポイントに保存され、評価や
    再開のときに学習時と同じ正規化が戻る。更新は gymnasium の RunningMeanStd と同じ
    並列の式で、1件ずつ足す。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float64))
        # RunningMeanStd と同じく、最初の1件で割り算が壊れないよう小さな件数から始める
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))
        self.epsilon = 1e-8

    @torch.no_grad()
    def update(self, x: np.ndarray) -> None:
        sample = torch.as_tensor(x, dtype=torch.float64, device=self.mean.device)
        delta = sample - self.mean
        total = self.count + 1.0
        self.mean += delta / total
        self.var.copy_((self.var * self.count + delta.pow(2) * self.count / total) / total)
        self.count.copy_(total)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device, x.dtype)
        std = torch.sqrt(self.var.to(x.device, x.dtype) + self.epsilon)
        return (x - mean) / std


if __name__ == "__main__":
    rp_scaling = RewardProcessor("scaling", 1.0)
    rp_centering = RewardProcessor("centering", 1.0)
    rewards = [0.5, 10.0, 2.0, 3.0, 4.0, 5.0, -4.0, -10.0, 0.0, 1.0, -1.0, -50.0]
    for r in rewards:
        rp_scaling.update(r)
        rp_centering.update(r)
        r_tensor = torch.tensor(r)
        norm_r_scaling = rp_scaling.normalize(r_tensor).item()
        norm_r_centering = rp_centering.normalize(r_tensor).item()
        print(f"{r=:+6.2f}, {norm_r_scaling=:+6.2f}, {norm_r_centering=:+6.2f}")
