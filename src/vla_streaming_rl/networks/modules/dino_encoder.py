# SPDX-License-Identifier: MIT
import math

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from transformers import Dinov2Model

from vla_streaming_rl.networks.modules.reward_embed import embed_reward
from vla_streaming_rl.networks.modules.self_attention import (
    SpatialTransformerBlock,
    get_fourier_embeds_from_coordinates,
)
from vla_streaming_rl.networks.modules.vlm_encoder import STATE_SCALAR_INDICES

# DINOv2 の事前学習の画像の正規化（ImageNet の平均と標準偏差）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoEncoder(nn.Module):
    """観測の画像を1枚ずつ凍結した DINOv2 に読ませ、最後の層のパッチの特徴をそのまま
    1ステップ分の状態にするエンコーダ。

    ``represent`` は画像を ``resolution`` 四方に縮めて DINOv2 に通し、最後の層（正規化の後）の
    パッチのトークンの格子を ``token_pool`` x ``token_pool`` ずつ平均してバッファに保存する。
    DINOv2 は学習しないので、バッファの表現は古くならない。

    ``to_state`` は保存した表現を位置ごとに ``token_dim`` へ射影し、同じステップの前の行動・
    報酬・スカラーの観測をそれぞれ ``token_dim`` 幅のトークンにしたものを並べ、
    ``token_mixer_layers`` 層の Transformer でトークンどうしを混ぜて、ヘッドが読む状態にする
    （VLM のエンコーダの ``to_state`` と同じ）。時間方向には何もしない。
    """

    def __init__(
        self,
        image_shapes: list[tuple[int, ...]],
        action_dim: int,
        dino_config: DictConfig,
        device: str,
    ) -> None:
        super().__init__()
        self.device = device
        self.model = Dinov2Model.from_pretrained(dino_config.model_id).to(device)
        self.model.eval().requires_grad_(False)
        self.register_buffer(
            "pixel_mean", torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        )
        # パッチの格子は resolution / patch_size 四方で、それを token_pool 四方ずつ平均する
        self.resolution = dino_config.resolution
        self.token_pool = dino_config.token_pool
        patch_size = self.model.config.patch_size
        assert self.resolution % (patch_size * self.token_pool) == 0, self.resolution
        self.grid = self.resolution // patch_size
        self.images_num = len(image_shapes)
        self.num_positions = self.images_num * (self.grid // self.token_pool) ** 2
        self.feature_dim = self.model.config.hidden_size

        self.token_dim = dino_config.token_dim
        self.state_out_proj = nn.Linear(self.feature_dim, self.token_dim).to(device)
        # スカラーをトークンにする sin/cos の角周波数。周期を scalar_max_period から
        # scalar_min_period まで対数等間隔にとり、sin と cos で token_dim 幅にする
        assert self.token_dim % 2 == 0, self.token_dim
        periods = torch.logspace(
            math.log10(dino_config.scalar_max_period),
            math.log10(dino_config.scalar_min_period),
            self.token_dim // 2,
        )
        self.register_buffer("scalar_frequencies", (2.0 * math.pi / periods).to(device))
        # 表現の位置、前の行動の成分ごと、報酬、スカラーの観測ごとに1トークン
        tokens_num = self.num_positions + action_dim + 1 + len(STATE_SCALAR_INDICES)
        self.output_dim = tokens_num * self.token_dim
        # 位置の埋め込みは 0、残差は layer scale で小さく始める
        self.token_position = nn.Parameter(
            torch.zeros(1, tokens_num, self.token_dim, device=device)
        )
        self.token_mixer = nn.Sequential(
            *[
                SpatialTransformerBlock(
                    self.token_dim,
                    dino_config.token_mixer_heads,
                    tokens_num,
                    dino_config.token_mixer_layer_scale_init,
                )
                for _ in range(dino_config.token_mixer_layers)
            ]
        ).to(device)

    def representation_shape(self) -> tuple[int, ...]:
        """``represent`` が返し、バッファに保存する表現の形。"""
        return (self.num_positions, self.feature_dim)

    @torch.no_grad()
    def represent(self, images: list[torch.Tensor]) -> torch.Tensor:
        """このティックの [0, 1] の画像 (3, H, W) の並びの表現 (num_positions, feature_dim)。
        画像ごとに、パッチの格子を平均でまとめたものを、画像の順に、それぞれラスター順に並べる。"""
        assert len(images) == self.images_num, (len(images), self.images_num)
        pixels = torch.stack(images).to(self.device, torch.float32)
        pixels = F.interpolate(
            pixels,
            size=(self.resolution, self.resolution),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        pixels = (pixels - self.pixel_mean) / self.pixel_std
        # 最初のトークンは CLS で、残りがラスター順のパッチ
        patches = self.model(pixel_values=pixels).last_hidden_state[:, 1:]
        grid = patches.view(self.images_num, self.grid, self.grid, -1).permute(0, 3, 1, 2)
        grid = F.avg_pool2d(grid, self.token_pool)
        return grid.flatten(2).transpose(1, 2).reshape(self.num_positions, self.feature_dim)

    def to_state(
        self,
        image: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        scalar_obs: torch.Tensor,
    ) -> torch.Tensor:
        """保存した表現 (B, num_positions, feature_dim) を射影したものに、同じステップの前の
        行動 (B, action_dim)、報酬 (B, 1)、スカラーの観測 (B, SCALAR_OBS_DIM) をそれぞれ
        トークンにしたものを並べ、ヘッドが読む状態 (B, output_dim) にする。スカラーは周期の違う
        sin/cos にし、小さい違いも大きい値の位置も別のベクトルにする。"""
        projected = self.state_out_proj(image.to(self.device, torch.float32))
        device = projected.device
        action_tokens = get_fourier_embeds_from_coordinates(
            self.token_dim, actions.to(device, torch.float32).unsqueeze(1)
        )[:, 0]  # (B, action_dim, token_dim)
        reward_token = embed_reward(rewards.to(device, torch.float32), self.token_dim)
        scalars = scalar_obs.to(device, torch.float32)[:, STATE_SCALAR_INDICES]
        angles = scalars.unsqueeze(-1) * self.scalar_frequencies
        scalar_tokens = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        tokens = torch.cat([projected, action_tokens, reward_token, scalar_tokens], dim=1)
        return self.token_mixer(tokens + self.token_position).flatten(start_dim=1)
