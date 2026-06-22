# SPDX-License-Identifier: MIT
"""Turn a VLAC trajectory critic into a dense per-step shaping reward.

VLAC scores an image trajectory into a cumulative completion ``value_list``
(0..``value_scale``, monotone for a successful rollout). Treating the normalised
value as a potential ``Phi``, potential-based reward shaping gives a dense
per-step reward ``alpha * (gamma * Phi(s_{t+1}) - Phi(s_t))`` that is added on
top of the existing sparse terminal reward without changing the optimal policy.
"""

import numpy as np


class VlacRewardRelabeler:
    """Compute PBRS dense rewards for one episode's observation frames."""

    def __init__(
        self,
        critic,
        skip: int,
        alpha: float,
        gamma: float,
        value_scale: float,
        batch_num: int,
    ) -> None:
        self._critic = critic
        self._skip = skip
        self._alpha = alpha
        self._gamma = gamma
        self._value_scale = value_scale
        self._batch_num = batch_num

    def potentials(self, frames: list, task: str) -> np.ndarray:
        """Per-frame normalised potential ``Phi``, length ``len(frames)``.

        VLAC samples the value every ``skip`` frames; the sparse nodes are
        linearly interpolated back up to one value per frame."""
        _, value_list = self._critic.get_trajectory_critic(
            task=task,
            image_list=frames,
            ref_image_list=None,
            batch_num=self._batch_num,
            ref_num=0,
            skip=self._skip,
        )
        value = np.asarray([float(v) for v in value_list], dtype=np.float32) / self._value_scale
        node_idx = np.arange(len(value), dtype=np.float32) * self._skip
        node_idx = np.minimum(node_idx, len(frames) - 1)
        all_idx = np.arange(len(frames), dtype=np.float32)
        return np.interp(all_idx, node_idx, value).astype(np.float32)

    def dense_rewards(self, frames: list, task: str) -> np.ndarray:
        """PBRS dense reward per env step, length ``len(frames) - 1``."""
        phi = self.potentials(frames, task)
        return self._alpha * (self._gamma * phi[1:] - phi[:-1])
