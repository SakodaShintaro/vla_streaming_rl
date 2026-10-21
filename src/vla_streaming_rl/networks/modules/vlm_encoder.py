# SPDX-License-Identifier: MIT
import math

import torch
from omegaconf import DictConfig
from torch import nn

from vla_streaming_rl.networks.modules.reward_embed import embed_reward
from vla_streaming_rl.networks.modules.self_attention import (
    SpatialTransformerBlock,
    get_fourier_embeds_from_coordinates,
)
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
    """画像の並びと指示の文を VLM に読ませ、1ステップ分の状態を作るエンコーダ。

    処理は2段に分かれる。``represent`` は画像と文を1つの user ターンとして VLM に読ませ、
    画像のトークンの位置の活性を ``[image_layer_start, image_layer_stop)`` の層（0 は言語モデルに
    入る前の埋め込みで、ViT の出力そのもの）で平均した、画像ごとのトークンの格子と、末尾の
    ``num_last_tokens`` 個の位置の活性の全層平均を並べた表現を返す。末尾はチャットテンプレートの固定の部分で、画像と指示を読んだ後の活性を
    持つ。VLM のうち ViT（PatchMerger まで）だけを学習し、言語モデルは凍結したまま勾配を
    通す。ViT が学習で変わるので、表現は保存せず、学習のたびに画像から作り直す。

    ``forward`` は保存した表現を位置ごとに ``state_out_dim`` へ射影し、同じステップの前の行動・
    報酬・スカラーの観測（global_step を除く）をそれぞれ ``state_out_dim`` 幅のトークンにした
    ものを並べ、``token_mixer_layers`` 層の Transformer でトークンどうしを混ぜて、ヘッドが
    読む状態にする。前の行動と報酬のトークンと、混ぜる層は :class:`SpatialTemporalEncoder` と
    同じ作り方にする。学習するのは ViT と射影と混ぜる層。
    """

    def __init__(
        self,
        *,
        image_shapes: list[tuple[int, ...]],
        action_dim: int,
        vla_config: DictConfig,
        device: str,
    ) -> None:
        super().__init__()
        self.device = device
        self.vlm, self.processor = load_model(
            model_id=vla_config.model_id,
            load_in_4bit=vla_config.load_in_4bit,
            device=torch.device(device),
        )
        self.vlm.eval().requires_grad_(False)
        # ViT だけを学習する。bf16 のままでは小さい学習率の更新が丸めで消えるので float32 に
        # する。出力は言語モデルの埋め込みの型に揃えてから入る
        self.vlm.model.visual.float().requires_grad_(True)
        self.vlm_hidden_size = self.vlm.config.text_config.hidden_size
        self.num_last_tokens = vla_config.num_last_tokens
        self.image_layers = slice(vla_config.image_layer_start, vla_config.image_layer_stop)
        self.image_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        # 観測の画像（一人称の視点と、あれば鳥瞰図）ごとのトークンの格子を並べる。格子の
        # 大きさは画像の大きさからプロセッサが決める（256px なら統合後 8x8）
        self.image_shapes = [tuple(shape) for shape in image_shapes]
        self.images_num = len(self.image_shapes)
        image_processor = self.processor.image_processor
        image_tokens_num = sum(
            int(
                image_processor(images=[torch.zeros(shape)], return_tensors="pt", do_rescale=False)[
                    "image_grid_thw"
                ][0].prod()
            )
            // image_processor.merge_size**2
            for shape in self.image_shapes
        )
        self.num_positions = image_tokens_num + self.num_last_tokens
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
        self.state_out_dim = vla_config.state_out_dim
        # 表現の位置、前の行動の成分ごと、報酬、スカラーの観測ごとに1トークン
        tokens_num = self.num_positions + action_dim + 1 + len(STATE_SCALAR_INDICES)
        self.output_dim = tokens_num * vla_config.state_out_dim
        # ヘッドに渡す前に、トークンどうしを Transformer の層で混ぜる
        # 位置の埋め込みは 0、残差は layer scale で小さく始めるので、
        # 学習の始めは混ぜないのとほぼ同じ。層の数が 0 なら混ぜない
        self.token_position = nn.Parameter(
            torch.zeros(1, tokens_num, vla_config.state_out_dim, device=device)
        )
        self.token_mixer = nn.Sequential(
            *[
                SpatialTransformerBlock(
                    vla_config.state_out_dim,
                    vla_config.token_mixer_heads,
                    tokens_num,
                    vla_config.token_mixer_layer_scale_init,
                )
                for _ in range(vla_config.token_mixer_layers)
            ]
        ).to(device)

    def represent(self, images: torch.Tensor, text: str) -> torch.Tensor:
        """[0, 1] の画像の並び (B, images_num, 3, H, W) と文を VLM に読ませ、画像のトークンの
        活性の ``image_layers`` の層での平均と、末尾の ``num_last_tokens`` 個の位置の活性の
        全層平均を並べた表現 (B, num_positions, hidden_size)。どのサンプルも同じ文と同じ大きさの画像なので、
        詰め物なしで同じ長さの列になる。"""
        batch_size = images.shape[0]
        assert images.shape[1] == self.images_num, (images.shape, self.images_num)
        placeholders = IMAGE_PLACEHOLDER * self.images_num
        prompt = f"{IM_START}user\n{placeholders}{text}{IM_END}\n{IM_START}assistant\n"
        inputs = self.processor(
            text=[prompt] * batch_size,
            images=list(images.flatten(end_dim=1)),
            return_tensors="pt",
            do_rescale=False,
        ).to(self.device)
        outputs = self.vlm(**inputs, output_hidden_states=True, logits_to_keep=1)
        # 全層を一度に積まず、足し込みながら平均する
        last_start = inputs["input_ids"].shape[1] - self.num_last_tokens
        last_tokens = sum(
            depth[:, last_start:].to(torch.float32) for depth in outputs.hidden_states
        ) / len(outputs.hidden_states)
        # 画像のトークンは、画像の順に、それぞれ統合後の格子のラスター順に並ぶ
        image_position = inputs["input_ids"][0] == self.image_pad_id
        image_layers = outputs.hidden_states[self.image_layers]
        image_tokens = sum(
            depth[:, image_position].to(torch.float32) for depth in image_layers
        ) / len(image_layers)
        return torch.cat([image_tokens, last_tokens], dim=1)

    def forward(
        self,
        representation: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        scalar_obs: torch.Tensor,
    ) -> torch.Tensor:
        """``represent`` の表現 (B, num_positions, hidden_size) を射影したものに、同じステップの
        前の行動 (B, action_dim)、報酬 (B, 1)、スカラーの観測 (B, SCALAR_OBS_DIM) をそれぞれ
        トークンにしたものを並べ、ヘッドが読む状態 (B, output_dim) にする。

        スカラー1つを1次元のまま連結すると、表現の次元に比べて割合が小さく埋もれやすく、残りの
        収益の 1 と 2 のような小さい違いも1つの入力のわずかな差にしかならない。周期の短い成分は
        小さい違いを別のベクトルにし、周期の長い成分は大きい値のおおよその位置を表す。変換は
        固定で、学習中に変わらない。"""
        projected = self.state_out_proj(representation)
        device = projected.device
        action_tokens = get_fourier_embeds_from_coordinates(
            self.state_out_dim, actions.to(device, torch.float32).unsqueeze(1)
        )[:, 0]  # (B, action_dim, state_out_dim)
        reward_token = embed_reward(rewards.to(device, torch.float32), self.state_out_dim)
        scalars = scalar_obs.to(device, torch.float32)[:, STATE_SCALAR_INDICES]
        angles = scalars.unsqueeze(-1) * self.scalar_frequencies
        scalar_tokens = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        tokens = torch.cat([projected, action_tokens, reward_token, scalar_tokens], dim=1)
        return self.token_mixer(tokens + self.token_position).flatten(start_dim=1)
