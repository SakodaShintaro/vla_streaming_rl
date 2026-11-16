# SPDX-License-Identifier: MIT
import math

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from vla_streaming_rl.networks.modules.reward_embed import embed_reward
from vla_streaming_rl.networks.modules.self_attention import get_fourier_embeds_from_coordinates
from vla_streaming_rl.networks.modules.vlm_encoder import STATE_SCALAR_INDICES


class ChannelAttention(nn.Module):
    """空間で平均したチャネルから、チャネルごとの [0, 1] の重みを作る。"""

    def __init__(self, depth: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(depth, depth // 4, 1, bias=False)
        self.expand = nn.Conv2d(depth // 4, depth, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3), keepdim=True)
        return torch.sigmoid(self.expand(F.elu(self.reduce(pooled))))


class FixupAttentionBlock(nn.Module):
    """Fixup の初期化をしたチャネル注意つきの残差ブロック。2つ目の畳み込みは 0 で始まるので、
    初期のブロックは恒等写像。"""

    def __init__(self, depth: int, height: int, width: int, stage: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
        # 元の実装は入力の高さ x 幅 x チャネル x 段の番号で分散を決める
        nn.init.trunc_normal_(
            self.conv1.weight, std=math.sqrt(2.0 / (height * width * depth * stage))
        )
        nn.init.zeros_(self.conv2.weight)
        self.attention = ChannelAttention(depth)
        self.biases = nn.Parameter(torch.zeros(4))
        self.multiplier = nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.elu(x) + self.biases[0]) + self.biases[1]
        out = out * self.attention(out)
        out = self.conv2(F.elu(out) + self.biases[2])
        return x + out * self.multiplier + self.biases[3]


class FixupAttentionCNN(nn.Module):
    """Animal-AI Olympics 2019 の 1 位（Trrrrr）の提出した方策の画像のバックボーン
    （``animal_conv_residual_fixup_attention``）。段ごとに 3x3 の畳み込み、幅 3・歩幅 2 の
    max pooling、残差ブロック 2 つで、最後に ELU。"""

    def __init__(self, resolution: int, depths: list[int]) -> None:
        super().__init__()
        layers = []
        channels, size = 3, resolution
        for stage, depth in enumerate(depths, start=1):
            layers.append(nn.Conv2d(channels, depth, 3, padding=1))
            layers.append(nn.MaxPool2d(3, stride=2, padding=1))
            size = math.ceil(size / 2)
            layers.append(FixupAttentionBlock(depth, size, size, stage))
            layers.append(FixupAttentionBlock(depth, size, size, stage))
            channels = depth
        layers.append(nn.ELU())
        self.layers = nn.Sequential(*layers)
        self.out_channels = channels
        self.out_size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CNNEncoder(nn.Module):
    """観測の画像を学習する CNN で読み、1 本の LSTM でステップをまたいで状態を作るエンコーダ。
    Animal-AI Olympics 2019 の 1 位の提出した方策（``animal_a2c_network_lstm6``）と同じ形。

    ``represent`` は画像を ``resolution`` に縮めた 0〜255 の値をバッファに保存する。CNN は
    学習するので、推論でも学習でも ``window_states`` が保存した画素から今の CNN で読み直す。

    ``window_states`` は窓 (B, T) の画像をステップごとに CNN に通して特徴の格子を平らにし、
    全結合で ``hidden_dim`` にする。前の行動・報酬・スカラーの観測は周期の違う sin/cos にして
    全結合で ``scalar_hidden_dim`` にする。2つを並べて全結合で ``hidden_dim`` にしたものを、
    LSTM で窓の頭から時間方向に読む。窓の頭の LSTM の状態は 0。状態は LSTM の出力。
    """

    def __init__(
        self,
        image_shapes: list[tuple[int, ...]],
        action_dim: int,
        cnn_config: DictConfig,
        device: str,
    ) -> None:
        super().__init__()
        self.device = device
        self.seq_len = cnn_config.seq_len
        self.resolution = cnn_config.resolution
        self.images_num = len(image_shapes)
        self.backbone = FixupAttentionCNN(self.resolution, list(cnn_config.depths)).to(device)
        image_feature_dim = self.images_num * self.backbone.out_channels * self.backbone.out_size**2
        self.image_fc = nn.Linear(image_feature_dim, cnn_config.hidden_dim).to(device)
        # スカラーを埋め込む sin/cos の角周波数。周期を scalar_max_period から
        # scalar_min_period まで対数等間隔にとり、sin と cos で scalar_embed_dim 幅にする
        self.scalar_embed_dim = cnn_config.scalar_embed_dim
        assert self.scalar_embed_dim % 2 == 0, self.scalar_embed_dim
        periods = torch.logspace(
            math.log10(cnn_config.scalar_max_period),
            math.log10(cnn_config.scalar_min_period),
            self.scalar_embed_dim // 2,
        )
        self.register_buffer("scalar_frequencies", (2.0 * math.pi / periods).to(device))
        scalars_num = action_dim + 1 + len(STATE_SCALAR_INDICES)
        self.scalar_fc = nn.Linear(
            scalars_num * self.scalar_embed_dim, cnn_config.scalar_hidden_dim
        ).to(device)
        self.joint_fc = nn.Linear(
            cnn_config.hidden_dim + cnn_config.scalar_hidden_dim, cnn_config.hidden_dim
        ).to(device)
        self.lstm = nn.LSTM(cnn_config.hidden_dim, cnn_config.lstm_units, batch_first=True).to(
            device
        )
        self.output_dim = cnn_config.lstm_units

    def representation_shape(self) -> tuple[int, ...]:
        """``represent`` が返し、バッファに保存する表現の形。"""
        return (self.images_num, 3, self.resolution, self.resolution)

    @torch.no_grad()
    def represent(self, images: list[torch.Tensor]) -> torch.Tensor:
        """[0, 1] の画像 (3, H, W) の並びを ``resolution`` に縮めた 0〜255 の整数値
        (images_num, 3, resolution, resolution)。bfloat16 のバッファでも整数値は正確に残る。"""
        assert len(images) == self.images_num, (len(images), self.images_num)
        stacked = torch.stack(images).to(self.device, torch.float32)
        resized = F.interpolate(stacked, size=(self.resolution, self.resolution), mode="area")
        return (resized * 255.0).round()

    def window_states(
        self,
        images: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        scalar_obs: torch.Tensor,
    ) -> torch.Tensor:
        """窓の画素 (B, T, images_num, 3, resolution, resolution)、前の行動 (B, T, action_dim)、
        報酬 (B, T, 1)、スカラーの観測 (B, T, SCALAR_OBS_DIM) から、ステップごとの状態
        (B, T, output_dim)。ステップ t の状態は窓の頭から t までを読んだもの。"""
        B, T = images.shape[:2]
        pixels = images.flatten(0, 2).to(self.device, torch.float32) / 255.0
        features = self.backbone(pixels).reshape(B * T, -1)
        image_hidden = F.elu(self.image_fc(features))
        scalar_hidden = F.elu(
            self.scalar_fc(
                self._scalar_embeds(
                    actions.flatten(0, 1), rewards.flatten(0, 1), scalar_obs.flatten(0, 1)
                )
            )
        )
        joint = F.elu(self.joint_fc(torch.cat([image_hidden, scalar_hidden], dim=-1)))
        out, _ = self.lstm(joint.view(B, T, -1))
        return out

    def _scalar_embeds(
        self, actions: torch.Tensor, rewards: torch.Tensor, scalar_obs: torch.Tensor
    ) -> torch.Tensor:
        """前の行動 (B, action_dim)、報酬 (B, 1)、スカラーの観測 (B, SCALAR_OBS_DIM) をそれぞれ
        ``scalar_embed_dim`` 幅にして平らに並べたもの。スカラーは周期の違う sin/cos にし、小さい
        違いも大きい値の位置も別のベクトルにする（VLM のエンコーダの ``to_state`` と同じ）。"""
        device = self.device
        action_embeds = get_fourier_embeds_from_coordinates(
            self.scalar_embed_dim, actions.to(device, torch.float32).unsqueeze(1)
        )[:, 0]  # (B, action_dim, scalar_embed_dim)
        reward_embed = embed_reward(rewards.to(device, torch.float32), self.scalar_embed_dim)
        scalars = scalar_obs.to(device, torch.float32)[:, STATE_SCALAR_INDICES]
        angles = scalars.unsqueeze(-1) * self.scalar_frequencies
        scalar_embeds = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return torch.cat([action_embeds, reward_embed, scalar_embeds], dim=1).flatten(1)
