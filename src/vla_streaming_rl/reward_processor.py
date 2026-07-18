# SPDX-License-Identifier: MIT
import gymnasium as gym
import numpy as np
import torch

_CARLA_IDLE_PENALTY = 0.1
_CARLA_CLIP = 1.0


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
        elif self.type == "carla":
            # Exact-zero check: in CARLA the per-step reward is the delta
            # of ``score_composed``, which is 0.0 only when route progress
            # stalls and no new infraction fires — i.e. a genuine stuck
            # state, not an "almost zero" continuous reward.
            result = torch.where(
                reward == 0.0, torch.full_like(reward, -_CARLA_IDLE_PENALTY), reward
            )
            result = torch.clamp(result, -_CARLA_CLIP, _CARLA_CLIP)
            return result
        else:
            msg = "Invalid normalizer type"
            raise ValueError(msg)

        MAX_VALUE = 10.0
        result = torch.clamp(result, -MAX_VALUE, MAX_VALUE)
        return result


class RunningNormalizer:
    """Per-component running mean/std normalizer for a fixed-width vector."""

    def __init__(self, dim: int) -> None:
        self.return_rms = gym.wrappers.utils.RunningMeanStd(shape=(dim,))
        self.epsilon = 1e-8

    def update(self, x: np.ndarray) -> None:
        self.return_rms.update(x[None, :])

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.return_rms.mean, dtype=x.dtype, device=x.device)
        std = torch.sqrt(
            torch.as_tensor(self.return_rms.var, dtype=x.dtype, device=x.device) + self.epsilon
        )
        return (x - mean) / std


if __name__ == "__main__":
    rp_scaling = RewardProcessor("scaling", 1.0)
    rp_centering = RewardProcessor("centering", 1.0)
    rp_carla = RewardProcessor("carla", 1.0)
    rewards = [0.5, 10.0, 2.0, 3.0, 4.0, 5.0, -4.0, -10.0, 0.0, 1.0, -1.0, -50.0]
    for r in rewards:
        rp_scaling.update(r)
        rp_centering.update(r)
        r_tensor = torch.tensor(r)
        norm_r_scaling = rp_scaling.normalize(r_tensor).item()
        norm_r_centering = rp_centering.normalize(r_tensor).item()
        norm_r_carla = rp_carla.normalize(r_tensor).item()
        print(
            f"{r=:+6.2f}, {norm_r_scaling=:+6.2f}, {norm_r_centering=:+6.2f}, {norm_r_carla=:+6.2f}"
        )
