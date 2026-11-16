# SPDX-License-Identifier: MIT
import math

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from huggingface_hub import snapshot_download
from omegaconf import DictConfig
from torch import nn

from vla_streaming_rl.networks.modules.reward_embed import embed_reward
from vla_streaming_rl.networks.modules.self_attention import (
    SpatialTransformerBlock,
    get_fourier_embeds_from_coordinates,
)
from vla_streaming_rl.networks.modules.vlm_encoder import STATE_SCALAR_INDICES

# VAE が時間方向にまとめるフレームの数。最初のフレームを1枚で、以降を 4 枚ずつ潜在の1フレームにする
VAE_TEMPORAL_STRIDE = 4


class WanEncoder(nn.Module):
    """観測のフレームの流れを凍結した Wan2.1 の VAE に動画として読ませ、最新の潜在をそのまま
    1ステップ分の状態にするエンコーダ。

    VAE の encoder は時間方向に因果的で、畳み込みごとに直前のフレームの活性をキャッシュに
    持てば、動画を途切れなく続きから読める。ただし潜在の1フレームは 4 枚ごとにしかできないので、
    読み始めを 1 ティックずつずらした 4 本の流れを持つ。流れ k はティック k のフレームを1枚で
    読み始め、以降 4 枚たまるごとに読む。ティック t には流れ t mod 4 がちょうど t までの 4 枚を
    読み終え、現在のフレームを含む潜在ができる。流れは途中で空にしない。

    ``represent`` はその潜在（チャネルごとに正規化したもの）を ``latent_patch`` x
    ``latent_patch`` のパッチに切り、パッチごとの値を並べたベクトルを位置ごとのトークンとして
    バッファに保存する。VAE は学習しないので、バッファの表現は古くならない。

    ``to_state`` は保存した表現を位置ごとに ``token_dim`` へ射影し、同じステップの前の行動・
    報酬・スカラーの観測をそれぞれ ``token_dim`` 幅のトークンにしたものを並べ、
    ``token_mixer_layers`` 層の Transformer でトークンどうしを混ぜて、ヘッドが読む状態にする
    （VLM のエンコーダの ``to_state`` と同じ）。
    """

    def __init__(
        self,
        image_shapes: list[tuple[int, ...]],
        action_dim: int,
        wan_config: DictConfig,
        device: str,
    ) -> None:
        super().__init__()
        self.device = device
        model_dir = snapshot_download(wan_config.model_id, allow_patterns=["Wan2.1_VAE.pth"])
        self.vae = AutoencoderKLWan.from_single_file(
            f"{model_dir}/Wan2.1_VAE.pth",
            config=wan_config.diffusers_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        ).to(device)
        self.vae.eval().requires_grad_(False)
        self.register_buffer(
            "latents_mean",
            torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1),
            persistent=False,
        )
        self.encoder_conv_num = self.vae._cached_conv_counts["encoder"]
        # 流れごとの、VAE の encoder の畳み込みのキャッシュ（読み始める前は None）と、
        # まだ読んでいないフレーム
        self.stream_caches: list[list | None] = [None] * VAE_TEMPORAL_STRIDE
        self.stream_pending: list[list[torch.Tensor]] = [[] for _ in range(VAE_TEMPORAL_STRIDE)]
        self.stream_tick = 0

        # VAE は空間を 8 分の 1 にし、潜在をさらに latent_patch 四方のパッチに切る
        self.latent_patch = wan_config.latent_patch
        assert len({tuple(shape) for shape in image_shapes}) == 1, image_shapes
        self.images_num = len(image_shapes)
        for shape in image_shapes:
            assert shape[1] % (8 * self.latent_patch) == 0, shape
            assert shape[2] % (8 * self.latent_patch) == 0, shape
        _, height, width = image_shapes[0]
        self.num_positions = (
            self.images_num
            * (height // (8 * self.latent_patch))
            * (width // (8 * self.latent_patch))
        )
        self.feature_dim = self.vae.config.z_dim * self.latent_patch**2

        self.token_dim = wan_config.token_dim
        self.state_out_proj = nn.Linear(self.feature_dim, self.token_dim).to(device)
        # スカラーをトークンにする sin/cos の角周波数。周期を scalar_max_period から
        # scalar_min_period まで対数等間隔にとり、sin と cos で token_dim 幅にする
        assert self.token_dim % 2 == 0, self.token_dim
        periods = torch.logspace(
            math.log10(wan_config.scalar_max_period),
            math.log10(wan_config.scalar_min_period),
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
                    wan_config.token_mixer_heads,
                    tokens_num,
                    wan_config.token_mixer_layer_scale_init,
                )
                for _ in range(wan_config.token_mixer_layers)
            ]
        ).to(device)

    def representation_shape(self) -> tuple[int, ...]:
        """``represent`` が返し、バッファに保存する表現の形。"""
        return (self.num_positions, self.feature_dim)

    @torch.no_grad()
    def represent(self, images: list[torch.Tensor]) -> torch.Tensor:
        """このティックの [0, 1] の画像 (3, H, W) の並びを流れに足し、現在のフレームを含む
        潜在の表現 (num_positions, feature_dim) を返す。画像ごとのパッチを、画像の順に、それぞれ
        ラスター順に並べる。"""
        assert len(images) == self.images_num, (len(images), self.images_num)
        # (images_num, 3, 1, H, W) の [-1, 1] の動画の1フレーム
        frame = torch.stack(images).to(self.device, torch.float32).unsqueeze(2) * 2.0 - 1.0
        stream = self.stream_tick % VAE_TEMPORAL_STRIDE
        if self.stream_caches[stream] is None:
            # 読み始めはこのフレームを1枚で読む
            self.stream_caches[stream] = [None] * self.encoder_conv_num
        else:
            assert len(self.stream_pending[stream]) == VAE_TEMPORAL_STRIDE - 1
            self.stream_pending[stream].append(frame)
            frame = torch.cat(self.stream_pending[stream], dim=2)
            self.stream_pending[stream] = []
        for other in range(VAE_TEMPORAL_STRIDE):
            if other != stream and self.stream_caches[other] is not None:
                self.stream_pending[other].append(frame[:, :, -1:])
        encoded = self.vae.quant_conv(
            self.vae.encoder(frame, feat_cache=self.stream_caches[stream], feat_idx=[0])
        )
        self.stream_tick += 1
        # 分布の平均（latent_dist.mode() と同じ）。潜在のフレームは1枚
        latents = encoded[:, : self.vae.config.z_dim, -1]
        latents = (latents - self.latents_mean) / self.latents_std
        patches = F.pixel_unshuffle(latents, self.latent_patch)  # (images_num, feature_dim, h, w)
        return patches.flatten(2).transpose(1, 2).reshape(self.num_positions, self.feature_dim)

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
