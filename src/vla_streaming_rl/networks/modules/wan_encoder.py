# SPDX-License-Identifier: MIT
import math

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from huggingface_hub import hf_hub_download, snapshot_download
from omegaconf import DictConfig
from torch import nn
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel

from vla_streaming_rl.networks.modules.reward_embed import embed_reward
from vla_streaming_rl.networks.modules.self_attention import (
    SpatialTransformerBlock,
    get_fourier_embeds_from_coordinates,
)
from vla_streaming_rl.networks.modules.vlm_encoder import STATE_SCALAR_INDICES

# Wan の文の埋め込みの長さ。パイプラインは文のトークンの後ろを 0 で埋めてこの長さにそろえる
TEXT_LENGTH = 512


def _umt5_state_dict(original: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Wan の元の形式の umt5-xxl のエンコーダの重みを、transformers の UMT5EncoderModel の
    名前にしたもの。Wan の FFN は fc1(x) * gelu(gate(x)) で、UMT5 の gated-gelu の
    wi_1(x) * gelu(wi_0(x)) と同じ。位置のバイアスは層ごとに持つ。"""
    # 語の埋め込みは shared と encoder.embed_tokens が同じものを指す
    converted = {
        "shared.weight": original["token_embedding.weight"],
        "encoder.embed_tokens.weight": original["token_embedding.weight"],
        "encoder.final_layer_norm.weight": original["norm.weight"],
    }
    num_layers = 1 + max(int(name.split(".")[1]) for name in original if name.startswith("blocks."))
    for i in range(num_layers):
        src, dst = f"blocks.{i}.", f"encoder.block.{i}.layer."
        converted[dst + "0.layer_norm.weight"] = original[src + "norm1.weight"]
        for name in ["q", "k", "v", "o"]:
            converted[dst + f"0.SelfAttention.{name}.weight"] = original[
                src + f"attn.{name}.weight"
            ]
        converted[dst + "0.SelfAttention.relative_attention_bias.weight"] = original[
            src + "pos_embedding.embedding.weight"
        ]
        converted[dst + "1.layer_norm.weight"] = original[src + "norm2.weight"]
        converted[dst + "1.DenseReluDense.wi_0.weight"] = original[src + "ffn.gate.0.weight"]
        converted[dst + "1.DenseReluDense.wi_1.weight"] = original[src + "ffn.fc1.weight"]
        converted[dst + "1.DenseReluDense.wo.weight"] = original[src + "ffn.fc2.weight"]
    assert len(converted) == len(original) + 1, (len(converted), len(original))
    return converted


@torch.no_grad()
def encode_prompt(text: str, model_dir: str, diffusers_id: str, device: str) -> torch.Tensor:
    """Wan のパイプラインと同じ文の埋め込み (1, TEXT_LENGTH, 4096)。文のトークンの分だけ
    umt5-xxl のエンコーダの出力で、残りは 0。エンコーダは使い終わったら捨てる。"""
    tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/google/umt5-xxl")
    config = UMT5Config.from_json_file(hf_hub_download(diffusers_id, "text_encoder/config.json"))
    with torch.device("meta"):
        text_encoder = UMT5EncoderModel(config)
    original = torch.load(
        f"{model_dir}/models_t5_umt5-xxl-enc-bf16.pth",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    text_encoder.load_state_dict(_umt5_state_dict(original), assign=True, strict=True)
    text_encoder = text_encoder.to(device, torch.bfloat16).eval()
    inputs = tokenizer(
        [text],
        padding="max_length",
        max_length=TEXT_LENGTH,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    ).to(device)
    hidden = text_encoder(inputs.input_ids, inputs.attention_mask).last_hidden_state
    length = int(inputs.attention_mask.sum())
    embedding = torch.zeros_like(hidden)
    embedding[:, :length] = hidden[:, :length]
    del text_encoder
    torch.cuda.empty_cache()
    return embedding


class WanEncoder(nn.Module):
    """観測の画像を凍結した Wan2.1 の動画の拡散モデルに読ませた特徴を表現にし、その上の RNN で
    ステップをまたいで状態を作るエンコーダ。

    ``represent`` は画像ごとに、現在のフレームまでの過去 ``clip_frames`` 枚を動画として VAE で
    潜在にし、少しノイズを足した潜在を DiT の ``feature_block`` 番目の層まで通す（文の条件は
    指示の文で固定）。DiT はフレームどうしにも注意するので、最後の潜在のフレームの活性は前の
    フレームとのつながりを読んだものになる。その活性のパッチの格子を ``token_pool`` x
    ``token_pool`` ずつ平均してバッファに保存する。Wan は学習しないので、バッファの表現は
    古くならない。

    ``window_states`` は窓 (B, T) の表現を、ステップごとに射影して前の行動・報酬・スカラーの
    観測のトークンと並べ、トークンを混ぜる Transformer の層で混ぜてから、トークンの位置ごとに
    GRU で窓の頭から時間方向に読む。窓の頭の GRU の状態は 0。
    """

    def __init__(
        self,
        image_shapes: list[tuple[int, ...]],
        action_dim: int,
        wan_config: DictConfig,
        instruction: str,
        device: str,
    ) -> None:
        super().__init__()
        self.device = device
        self.seq_len = wan_config.seq_len
        self.token_pool = wan_config.token_pool
        # VAE は最初のフレームを1枚で、以降を 4 枚ずつ潜在の1フレームにする
        self.clip_frames = wan_config.clip_frames
        assert (self.clip_frames - 1) % 4 == 0, self.clip_frames
        model_dir = snapshot_download(wan_config.model_id)
        self.vae = AutoencoderKLWan.from_single_file(
            f"{model_dir}/Wan2.1_VAE.pth",
            config=wan_config.diffusers_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        ).to(device)
        self.transformer = WanTransformer3DModel.from_single_file(
            f"{model_dir}/diffusion_pytorch_model.safetensors",
            config=wan_config.diffusers_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        ).to(device)
        # 特徴を取る層より後ろの層は使わないので捨てる
        assert 0 <= wan_config.feature_block < len(self.transformer.blocks)
        self.transformer.blocks = self.transformer.blocks[: wan_config.feature_block + 1]
        self.vae.eval().requires_grad_(False)
        self.transformer.eval().requires_grad_(False)
        self.register_buffer(
            "latents_mean",
            torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "text_embedding",
            encode_prompt(instruction, model_dir, wan_config.diffusers_id, device).to(
                torch.bfloat16
            ),
            persistent=False,
        )
        # 流れのマッチングの補間 x_t = (1 - sigma) x_0 + sigma * noise。ノイズは固定し、同じ画像は
        # いつも同じ表現になるようにする
        self.noise_level = wan_config.noise_level
        generator = torch.Generator().manual_seed(0)
        # VAE は空間を 8 分の 1、DiT のパッチは 2 なので、パッチの格子は画像の 16 分の 1
        self.images_num = len(image_shapes)
        self.grids = []
        for shape in image_shapes:
            assert shape[1] % (16 * self.token_pool) == 0, shape
            assert shape[2] % (16 * self.token_pool) == 0, shape
            self.grids.append((shape[1] // 16, shape[2] // 16))
        assert len({tuple(shape) for shape in image_shapes}) == 1, image_shapes
        latent_shape = (
            1,
            self.vae.config.z_dim,
            1 + (self.clip_frames - 1) // 4,
            image_shapes[0][1] // 8,
            image_shapes[0][2] // 8,
        )
        self.register_buffer(
            "noise", torch.randn(latent_shape, generator=generator).to(device), persistent=False
        )
        self.num_positions = sum(h * w for h, w in self.grids) // self.token_pool**2
        self.feature_dim = self.transformer.config.num_attention_heads * (
            self.transformer.config.attention_head_dim
        )

        self.token_dim = wan_config.token_dim
        # 特徴の大きさはそろっていないので、正規化してから射影する
        self.representation_norm = nn.RMSNorm(self.feature_dim, eps=1e-6).to(device)
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
        self.rnn = nn.GRU(
            self.token_dim, self.token_dim, num_layers=wan_config.rnn_layers, batch_first=True
        ).to(device)

    def representation_shape(self) -> tuple[int, ...]:
        """``represent`` が返し、バッファに保存する表現の形。"""
        return (self.num_positions, self.feature_dim)

    @torch.no_grad()
    def represent(self, clips: list[torch.Tensor]) -> torch.Tensor:
        """画像ごとの [0, 1] の過去のフレームの並び (clip_frames, 3, H, W)（最後が現在）の表現
        (num_positions, feature_dim)。画像ごとに、最後の潜在のフレームのパッチの格子を平均で
        まとめたものを、画像の順に、それぞれラスター順に並べる。"""
        assert len(clips) == self.images_num, (len(clips), self.images_num)
        assert clips[0].shape[0] == self.clip_frames, clips[0].shape
        video = torch.stack(clips).to(self.device, torch.float32).transpose(1, 2) * 2.0 - 1.0
        latents = self.vae.encode(video).latent_dist.mode()
        latents = (latents - self.latents_mean) / self.latents_std
        latents = (1.0 - self.noise_level) * latents + self.noise_level * self.noise
        features = self._block_features(latents.to(torch.bfloat16))  # (N, frames * h * w, C)
        pooled = []
        for feature, (h, w) in zip(features, self.grids, strict=True):
            # パッチは潜在のフレームの順に並ぶので、最後の h * w 個が現在のフレームを含む潜在
            grid = feature[-h * w :].view(1, h, w, -1).permute(0, 3, 1, 2).float()
            grid = F.avg_pool2d(grid, self.token_pool)
            pooled.append(grid.flatten(2).transpose(1, 2)[0])
        return torch.cat(pooled, dim=0)

    def window_states(
        self,
        images: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        scalar_obs: torch.Tensor,
    ) -> torch.Tensor:
        """窓の表現 (B, T, num_positions, feature_dim)、前の行動 (B, T, action_dim)、報酬
        (B, T, 1)、スカラーの観測 (B, T, SCALAR_OBS_DIM) から、ステップごとの状態
        (B, T, output_dim)。ステップ t の状態は窓の頭から t までを読んだもの。"""
        B, T = images.shape[:2]
        projected = self.state_out_proj(
            self.representation_norm(images.flatten(0, 1).to(self.device, torch.float32))
        )
        tokens = self._step_tokens(
            projected, actions.flatten(0, 1), rewards.flatten(0, 1), scalar_obs.flatten(0, 1)
        )
        mixed = self.token_mixer(tokens + self.token_position)  # (B * T, N, D)
        N, D = mixed.shape[1:]
        sequences = mixed.view(B, T, N, D).transpose(1, 2).reshape(B * N, T, D)
        recurrent, _ = self.rnn(sequences)
        out = (sequences + recurrent).view(B, N, T, D).transpose(1, 2)
        return out.reshape(B, T, N * D)

    def _step_tokens(
        self,
        image_tokens: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        scalar_obs: torch.Tensor,
    ) -> torch.Tensor:
        """射影した画像のトークン (B, num_positions, token_dim) に、同じステップの前の行動
        (B, action_dim)、報酬 (B, 1)、スカラーの観測 (B, SCALAR_OBS_DIM) をそれぞれトークンに
        したものを並べる。スカラーは周期の違う sin/cos にし、小さい違いも大きい値の位置も
        別のベクトルにする（VLM のエンコーダの ``to_state`` と同じ）。"""
        device = image_tokens.device
        action_tokens = get_fourier_embeds_from_coordinates(
            self.token_dim, actions.to(device, torch.float32).unsqueeze(1)
        )[:, 0]  # (B, action_dim, token_dim)
        reward_token = embed_reward(rewards.to(device, torch.float32), self.token_dim)
        scalars = scalar_obs.to(device, torch.float32)[:, STATE_SCALAR_INDICES]
        angles = scalars.unsqueeze(-1) * self.scalar_frequencies
        scalar_tokens = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return torch.cat([image_tokens, action_tokens, reward_token, scalar_tokens], dim=1)

    def _block_features(self, latents: torch.Tensor) -> torch.Tensor:
        """潜在 (N, C, F, H, W) を DiT の残した層まで通した活性 (N, patches, feature_dim)。
        ``WanTransformer3DModel.forward`` の層の手前までと同じ計算。"""
        transformer = self.transformer
        N = latents.shape[0]
        rotary_emb = transformer.rope(latents)
        hidden = transformer.patch_embedding(latents).flatten(2).transpose(1, 2).contiguous()
        timestep = torch.full((N,), self.noise_level * 1000.0, device=latents.device)
        _, timestep_proj, text, _ = transformer.condition_embedder(
            timestep, self.text_embedding.expand(N, -1, -1), None, timestep_seq_len=None
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        for block in transformer.blocks:
            hidden = block(hidden, text, timestep_proj, rotary_emb)
        return hidden
