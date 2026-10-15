# SPDX-License-Identifier: MIT
import gymnasium as gym
import numpy as np
import torch


class RewardScaler:
    """学習に使う報酬を、見てきた値の標準偏差で割って揃える。

    統計は ``update`` に渡した値（毎ステップの報酬か、エピソードの収益か）から取り、
    割ったあとの値は ±``MAX_VALUE`` に収める。"""

    MAX_VALUE = 10.0

    def __init__(self) -> None:
        self.return_rms = gym.wrappers.utils.RunningMeanStd(shape=())
        self.epsilon = 1e-8

    def update(self, reward: float) -> None:
        """Update the running mean and std with the new reward."""
        self.return_rms.update(np.array([reward]))

    def normalize(self, reward: torch.Tensor) -> torch.Tensor:
        result = reward / np.sqrt(self.return_rms.var + self.epsilon)
        return torch.clamp(result, -self.MAX_VALUE, self.MAX_VALUE)
