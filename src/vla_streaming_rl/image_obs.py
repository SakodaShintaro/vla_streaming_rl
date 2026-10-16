# SPDX-License-Identifier: MIT
"""エージェントが毎ステップ観測から読み、ネットワークに渡す画像。

並びはここだけで決める。環境はこのうちの一部だけを観測に持ち（"image" はどの環境も持つ）、
エージェントは観測にあるものをこの並びで読んで、ネットワークに渡す。"""

import gymnasium as gym

# "image" は一人称の視点、"topdown_image" は Animal-AI のアリーナを真上から見た鳥瞰図
IMAGE_OBS_NAMES = ("image", "topdown_image")


def image_obs_names(observation_space: gym.spaces.Dict) -> tuple[str, ...]:
    """観測の空間が持つ画像の名前を ``IMAGE_OBS_NAMES`` の並びで。"""
    names = tuple(name for name in IMAGE_OBS_NAMES if name in observation_space.spaces)
    assert names[0] == "image", names
    return names
