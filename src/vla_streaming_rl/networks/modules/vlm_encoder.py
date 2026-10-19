# SPDX-License-Identifier: MIT
import math

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from vla_streaming_rl.networks.modules.vlm_backbone import load_model
from vla_streaming_rl.scalar_obs import SCALAR_OBS_NAMES

# VLM のチャットテンプレートの書式
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

# 状態に数値のまま入れるスカラーの観測の、SCALAR_OBS_NAMES での位置。global_step は
# エピソードの中の状況を表さず、学習の間ずっと増え続けるので入れない
STATE_SCALAR_INDICES = [
    index for index, name in enumerate(SCALAR_OBS_NAMES) if name != "global_step"
]


class VLMEncoder(nn.Module):
    """画像の並びと指示の文を凍結した VLM に読ませ、1ステップ分の状態を作るエンコーダ。

    処理は2段に分かれる。``represent`` は画像と文を1つの user ターンとして VLM に読ませ、
    全位置の活性を全層（埋め込みと各層の出力）で平均し、画像ごとにトークンの格子を
    ``image_pool_size`` 四方に平均プーリングしたものと、末尾の ``num_last_tokens`` 個の位置を
    並べた表現を返す。末尾はチャットテンプレートの固定の部分で、画像と指示を読んだ後の活性を
    持つ。VLM は凍結しているので、この表現はバッファに保存して使い回せる。

    ``forward`` は保存した表現を位置ごとに ``state_out_dim`` へ射影し、同じステップのスカラーの
    観測（global_step を除く）をそれぞれ sin/cos の特徴で ``state_out_dim`` 幅のトークンにした
    ものを並べて、ヘッドが読む状態にする。学習するのはこの射影だけ。
    """

    def __init__(self, *, images_num: int, vla_config: DictConfig, device: str) -> None:
        super().__init__()
        self.device = device
        self.vlm, self.processor = load_model(
            model_id=vla_config.model_id,
            load_in_4bit=vla_config.load_in_4bit,
            device=torch.device(device),
        )
        self.vlm.eval().requires_grad_(False)
        self.vlm_hidden_size = self.vlm.config.text_config.hidden_size
        self.image_pool_size = vla_config.image_pool_size
        self.num_last_tokens = vla_config.num_last_tokens
        self.image_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.merge_size = self.processor.image_processor.merge_size
        # 観測の画像（一人称の視点と、あれば鳥瞰図）ごとにプーリングした格子を並べる
        self.images_num = images_num
        self.num_positions = images_num * self.image_pool_size**2 + self.num_last_tokens
        self.state_out_proj = nn.Linear(self.vlm_hidden_size, vla_config.state_out_dim).to(device)
        # スカラーをトークンにする sin/cos の角周波数。周期を scalar_max_period から
        # scalar_min_period まで対数等間隔にとり、sin と cos で state_out_dim 幅にする
        assert vla_config.state_out_dim % 2 == 0, vla_config.state_out_dim
        periods = torch.logspace(
            math.log10(vla_config.scalar_max_period),
            math.log10(vla_config.scalar_min_period),
            vla_config.state_out_dim // 2,
        )
        self.register_buffer("scalar_frequencies", (2.0 * math.pi / periods).to(device))
        self.output_dim = (
            self.num_positions + len(STATE_SCALAR_INDICES)
        ) * vla_config.state_out_dim

    def representation_shape(self) -> tuple[int, ...]:
        """``represent`` が返す表現の形。射影は学習するので、その手前を保存する。"""
        return (self.num_positions, self.vlm_hidden_size)

    def represent(self, images: list[torch.Tensor], text: str) -> torch.Tensor:
        """[0, 1] の画像 (3, H, W) の並びと文を VLM に読ませた、全位置の活性の全層平均から、
        画像ごとにトークンの格子を ``image_pool_size`` 四方にプーリングしたものと末尾の
        ``num_last_tokens`` 個の位置を並べた表現 (num_positions, hidden_size)。"""
        assert len(images) == self.images_num, (len(images), self.images_num)
        placeholders = IMAGE_PLACEHOLDER * len(images)
        prompt = f"{IM_START}user\n{placeholders}{text}{IM_END}\n{IM_START}assistant\n"
        inputs = self.processor(
            text=[prompt], images=list(images), return_tensors="pt", do_rescale=False
        ).to(self.device)
        outputs = self.vlm(**inputs, output_hidden_states=True, logits_to_keep=1)
        # 全層を一度に積まず、足し込みながら平均する
        layer_mean = sum(depth[0].to(torch.float32) for depth in outputs.hidden_states) / len(
            outputs.hidden_states
        )
        # 画像のトークンは、画像の順に、それぞれ統合後の格子のラスター順に並ぶ
        image_tokens = layer_mean[inputs["input_ids"][0] == self.image_pad_id]
        grids = (inputs["image_grid_thw"][:, 1:] // self.merge_size).tolist()
        pooled = [
            F.adaptive_avg_pool2d(
                tokens.T.reshape(1, self.vlm_hidden_size, height, width), self.image_pool_size
            )
            .flatten(start_dim=2)[0]
            .T
            for (height, width), tokens in zip(
                grids, image_tokens.split([h * w for h, w in grids]), strict=True
            )
        ]
        return torch.cat([*pooled, layer_mean[-self.num_last_tokens :]]).to(torch.bfloat16)

    def forward(self, representation: torch.Tensor, scalar_obs: torch.Tensor) -> torch.Tensor:
        """保存した表現 (B, num_positions, hidden_size) を射影したものに、同じステップの
        スカラーの観測 (B, SCALAR_OBS_DIM) を sin/cos の特徴でトークンにしたものを並べ、ヘッドが
        読む状態 (B, output_dim) にする。

        スカラー1つを1次元のまま連結すると、表現の次元に比べて割合が小さく埋もれやすく、残りの
        収益の 1 と 2 のような小さい違いも1つの入力のわずかな差にしかならない。周期の短い成分は
        小さい違いを別のベクトルにし、周期の長い成分は大きい値のおおよその位置を表す。変換は
        固定で、学習中に変わらない。"""
        projected = self.state_out_proj(representation.to(torch.float32))
        scalars = scalar_obs.to(projected.device, torch.float32)[:, STATE_SCALAR_INDICES]
        angles = scalars.unsqueeze(-1) * self.scalar_frequencies
        scalar_tokens = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return torch.cat([projected, scalar_tokens], dim=1).flatten(start_dim=1)
