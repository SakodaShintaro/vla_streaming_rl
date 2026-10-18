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


class FixedRewardScaler:
    """学習に使う報酬に、決まった倍率 ``scale`` を掛ける。

    :class:`RewardScaler` と違い、倍率は学習の間ずっと変わらず、値も切らない。見てきた報酬で
    倍率が動くと同じ遷移の TD の目標が動き続け、切ると大きさの違う報酬が同じ値に張り付く。
    ``update`` は :class:`RewardScaler` と同じ呼び方にするためのもので、何もしない。"""

    def __init__(self, scale: float) -> None:
        assert scale > 0.0, scale
        self.scale = scale

    def update(self, reward: float) -> None:
        del reward

    def normalize(self, reward: torch.Tensor) -> torch.Tensor:
        return reward * self.scale


def build_reward_scaler(mode: str, fixed_scale: float) -> RewardScaler | FixedRewardScaler:
    """``mode`` が "running" なら見てきた報酬の標準偏差で割って切るもの、"fixed" なら
    ``fixed_scale`` を掛けるだけのもの。"""
    if mode == "running":
        return RewardScaler()
    assert mode == "fixed", f"Unknown reward_scale_mode: {mode!r}"
    return FixedRewardScaler(fixed_scale)
