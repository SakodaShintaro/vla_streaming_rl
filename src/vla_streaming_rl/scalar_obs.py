# SPDX-License-Identifier: MIT
"""エージェントが毎ステップ観測から読み、画像と並べてネットワークに渡すスカラー。

並びはここだけで決める。エージェントは :func:`read_scalar_obs` で1本のベクトルにして
バッファに積み、ネットワークはその並びのまま読む。"""

from typing import Any

import numpy as np

SCALAR_OBS_NAMES = (
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "episode_return",
    "pass_mark",
    "remaining_return",
    "global_step",
    "episode_step",
    "health",
)
SCALAR_OBS_DIM = len(SCALAR_OBS_NAMES)


def read_scalar_obs(obs: dict[str, Any]) -> np.ndarray:
    """観測の dict から、``SCALAR_OBS_NAMES`` の並びのベクトル (SCALAR_OBS_DIM,)。"""
    scalar_obs = np.concatenate(
        [
            obs["velocity"],
            obs["episode_return"],
            obs["pass_mark"],
            obs["remaining_return"],
            obs["global_step"],
            obs["episode_step"],
            obs["health"],
        ]
    ).astype(np.float32)
    assert scalar_obs.shape == (SCALAR_OBS_DIM,), scalar_obs.shape
    return scalar_obs
