# SPDX-License-Identifier: MIT
import functools
import math

import torch
import torch.nn.functional as F
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
    """画像の並びと指示の文を凍結した VLM に読ませ、1ステップ分のトークンを作るエンコーダ。

    処理は2段に分かれる。``represent`` は画像と文を1つの user ターンとして VLM に読ませ、
    画像のトークンの位置の活性を ``[image_layer_start, image_layer_stop)`` の層（0 は言語モデルに
    入る前の埋め込みで、ViT の出力そのもの）で平均した、画像ごとのトークンの格子と、末尾の
    ``num_last_tokens`` 個の位置の活性の全層平均と、指示のトークンを並べた表現を返す。末尾は
    チャットテンプレートの固定の部分で、画像と指示を読んだ後の活性を持つ。指示のトークンは
    文だけを VLM に読ませ、``instruction_layers`` の層ごとに、文のトークンを順に
    ``instruction_segments`` 個の区間に分けて平均したもの。平均を1つにすると、指示どうしの
    小さい違いが潰れる。VLM は凍結しているので、この表現はバッファに保存して使い回せる。

    ``forward`` は保存した表現を位置ごとに ``token_dim`` へ射影し、同じステップの前の行動・
    報酬・スカラーの観測（global_step を除く）をそれぞれ ``token_dim`` 幅のトークンにした
    ものを並べ、``token_mixer_layers`` 層の Transformer でステップの中のトークンどうしを混ぜる。
    学習するのは射影と混ぜる層と、欠けた入力の代わりに置くトークン。
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
        self.vlm_hidden_size = self.vlm.config.text_config.hidden_size
        self.num_last_tokens = vla_config.num_last_tokens
        self.image_layers = slice(vla_config.image_layer_start, vla_config.image_layer_stop)
        self.instruction_layers = list(vla_config.instruction_layers)
        self.instruction_segments = vla_config.instruction_segments
        self.image_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        # 観測の画像（一人称の視点と、あれば鳥瞰図）ごとのトークンの格子を並べる。格子の
        # 大きさは画像の大きさからプロセッサが決める（256px なら統合後 8x8）
        self.images_num = len(image_shapes)
        image_processor = self.processor.image_processor
        self.image_grids = [
            tuple(
                int(size) // image_processor.merge_size
                for size in image_processor(
                    images=[torch.zeros(shape)], return_tensors="pt", do_rescale=False
                )["image_grid_thw"][0, 1:]
            )
            for shape in image_shapes
        ]
        self.image_tokens_num = sum(h * w for h, w in self.image_grids)
        instruction_tokens_num = len(self.instruction_layers) * self.instruction_segments
        self.num_positions = self.image_tokens_num + self.num_last_tokens + instruction_tokens_num
        self.state_out_proj = nn.Linear(self.vlm_hidden_size, vla_config.token_dim).to(device)
        # スカラーをトークンにする sin/cos の角周波数。周期を scalar_max_period から
        # scalar_min_period まで対数等間隔にとり、sin と cos で token_dim 幅にする
        assert vla_config.token_dim % 2 == 0, vla_config.token_dim
        periods = torch.logspace(
            math.log10(vla_config.scalar_max_period),
            math.log10(vla_config.scalar_min_period),
            vla_config.token_dim // 2,
        )
        self.register_buffer("scalar_frequencies", (2.0 * math.pi / periods).to(device))
        self.token_dim = vla_config.token_dim
        # 表現の位置、前の行動の成分ごと、報酬、スカラーの観測ごとに1トークン
        self.tokens_num = self.num_positions + action_dim + 1 + len(STATE_SCALAR_INDICES)

        # トークンごとの入力の種類。画像ごと、末尾、指示、報酬、スカラーの観測の順で、
        # 学習中はこの種類ごとに落とす。前の行動はエージェント自身が出したものなので落とさず、
        # 最後の番号 groups_num を割り当てる
        self.groups_num = self.images_num + 4
        image_groups = torch.cat(
            [torch.full((h * w,), index) for index, (h, w) in enumerate(self.image_grids)]
        )
        token_group = torch.cat(
            [
                image_groups,
                torch.full((self.num_last_tokens,), self.images_num),
                torch.full((instruction_tokens_num,), self.images_num + 1),
                torch.full((action_dim,), self.groups_num),
                torch.full((1,), self.images_num + 2),
                torch.full((len(STATE_SCALAR_INDICES),), self.images_num + 3),
            ]
        )
        self.register_buffer("token_group", token_group.to(device), persistent=False)
        # 落とした位置に置くトークン。位置の埋め込みは後から足すので、1つを共有する
        self.absent_token = nn.Parameter(torch.zeros(vla_config.token_dim, device=device))

        # トークンどうしを Transformer の層で混ぜる
        # 位置の埋め込みは 0、残差は layer scale で小さく始めるので、
        # 学習の始めは混ぜないのとほぼ同じ。層の数が 0 なら混ぜない
        self.token_position = nn.Parameter(
            torch.zeros(1, self.tokens_num, vla_config.token_dim, device=device)
        )
        self.token_mixer = nn.Sequential(
            *[
                SpatialTransformerBlock(
                    vla_config.token_dim,
                    vla_config.token_mixer_heads,
                    self.tokens_num,
                    vla_config.token_mixer_layer_scale_init,
                )
                for _ in range(vla_config.token_mixer_layers)
            ]
        ).to(device)

    def representation_shape(self) -> tuple[int, ...]:
        """``represent`` が返す表現の形。射影は学習するので、その手前を保存する。"""
        return (self.num_positions, self.vlm_hidden_size)

    @torch.no_grad()
    def represent(self, images: list[torch.Tensor], text: str) -> torch.Tensor:
        """[0, 1] の画像 (3, H, W) の並びと文を VLM に読ませ、画像のトークンの活性の
        ``image_layers`` の層での平均と、末尾の ``num_last_tokens`` 個の位置の活性の全層平均と、
        指示のトークンを並べた表現 (num_positions, hidden_size)。"""
        assert len(images) == self.images_num, (len(images), self.images_num)
        placeholders = IMAGE_PLACEHOLDER * len(images)
        prompt = f"{IM_START}user\n{placeholders}{text}{IM_END}\n{IM_START}assistant\n"
        inputs = self.processor(
            text=[prompt], images=list(images), return_tensors="pt", do_rescale=False
        ).to(self.device)
        outputs = self.vlm(**inputs, output_hidden_states=True, logits_to_keep=1)
        # 全層を一度に積まず、足し込みながら平均する
        last_start = inputs["input_ids"].shape[1] - self.num_last_tokens
        last_tokens = sum(
            depth[0, last_start:].to(torch.float32) for depth in outputs.hidden_states
        ) / len(outputs.hidden_states)
        # 画像のトークンは、画像の順に、それぞれ統合後の格子のラスター順に並ぶ
        image_position = inputs["input_ids"][0] == self.image_pad_id
        image_layers = outputs.hidden_states[self.image_layers]
        image_tokens = sum(
            depth[0, image_position].to(torch.float32) for depth in image_layers
        ) / len(image_layers)
        return torch.cat([image_tokens, last_tokens, self._instruction_tokens(text)])

    @functools.cache
    def _instruction_tokens(self, text: str) -> torch.Tensor:
        """文だけを VLM に読ませ、``instruction_layers`` の層ごとに文のトークンを順に
        ``instruction_segments`` 個の区間に分けて平均したもの
        (len(instruction_layers) * instruction_segments, hidden_size)。指示はエピソードの間
        変わらないので、文ごとに一度だけ計算する。"""
        inputs = self.processor.tokenizer(text, return_tensors="pt").to(self.device)
        assert inputs["input_ids"].shape[1] >= self.instruction_segments, inputs["input_ids"].shape
        outputs = self.vlm(**inputs, output_hidden_states=True, logits_to_keep=1)
        return torch.cat(
            [
                torch.stack(
                    [
                        segment.mean(dim=0)
                        for segment in outputs.hidden_states[layer][0]
                        .to(torch.float32)
                        .tensor_split(self.instruction_segments)
                    ]
                )
                for layer in self.instruction_layers
            ]
        )

    def image_targets(self, representation: torch.Tensor, pool: int) -> torch.Tensor:
        """保存した表現 (..., num_positions, hidden_size) の画像のトークンを、画像ごとに格子を
        ``pool`` x ``pool`` に平均で縮め、トークンごとに正規化したもの
        (..., images_num * pool * pool, hidden_size)。将来の画像を予測する的にする。"""
        leading = representation.shape[:-2]
        image_tokens = representation[..., : self.image_tokens_num, :].reshape(
            -1, self.image_tokens_num, self.vlm_hidden_size
        )
        pooled = [
            F.adaptive_avg_pool2d(
                tokens.transpose(1, 2).reshape(-1, self.vlm_hidden_size, h, w), pool
            )
            .flatten(2)
            .transpose(1, 2)
            for tokens, (h, w) in zip(
                image_tokens.split([h * w for h, w in self.image_grids], dim=1),
                self.image_grids,
                strict=True,
            )
        ]
        targets = F.layer_norm(torch.cat(pooled, dim=1), (self.vlm_hidden_size,))
        return targets.reshape(*leading, -1, self.vlm_hidden_size)

    def forward(
        self,
        representation: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        scalar_obs: torch.Tensor,
        observation_dropout: float,
        modality_dropout: float,
    ) -> torch.Tensor:
        """窓の各ステップの保存した表現 (B, T, num_positions, hidden_size) を射影したものに、
        同じステップの前の行動 (B, T, action_dim)、報酬 (B, T, 1)、スカラーの観測
        (B, T, SCALAR_OBS_DIM) をそれぞれトークンにしたものを並べ、ステップの中で混ぜた
        トークン (B, T, tokens_num, token_dim) にする。

        学習のときは、ステップごとに確率 ``observation_dropout`` で前の行動以外の入力を
        すべて、確率 ``modality_dropout`` で入力の種類ごとに、欠けたことを表すトークンに
        置き換える。今の観測が欠けたステップでは、方策は再帰の状態に残した過去の情報だけで
        行動を決めるしかなく、状態に情報を残すことを学ぶ。0 なら落とさない。

        スカラー1つを1次元のまま連結すると、表現の次元に比べて割合が小さく埋もれやすく、残りの
        収益の 1 と 2 のような小さい違いも1つの入力のわずかな差にしかならない。周期の短い成分は
        小さい違いを別のベクトルにし、周期の長い成分は大きい値のおおよその位置を表す。変換は
        固定で、学習中に変わらない。"""
        B, T = actions.shape[:2]
        projected = self.state_out_proj(representation.to(self.device, torch.float32))
        action_tokens = get_fourier_embeds_from_coordinates(
            self.token_dim, actions.to(self.device, torch.float32)
        )  # (B, T, action_dim, token_dim)
        reward_token = embed_reward(rewards.to(self.device, torch.float32), self.token_dim)
        scalars = scalar_obs.to(self.device, torch.float32)[..., STATE_SCALAR_INDICES]
        angles = scalars.unsqueeze(-1) * self.scalar_frequencies
        scalar_tokens = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        tokens = torch.cat([projected, action_tokens, reward_token, scalar_tokens], dim=2)

        # 種類ごとに落とすかを引き、落とさない前の行動の列を足してトークンへ広げる
        group_drop = (torch.rand(B, T, self.groups_num, device=self.device) < modality_dropout) | (
            torch.rand(B, T, 1, device=self.device) < observation_dropout
        )
        group_drop = torch.cat([group_drop, torch.zeros_like(group_drop[..., :1])], dim=-1)
        token_drop = group_drop[..., self.token_group].unsqueeze(-1)
        tokens = torch.where(token_drop, self.absent_token, tokens) + self.token_position
        mixed = self.token_mixer(tokens.flatten(0, 1))
        return mixed.reshape(B, T, self.tokens_num, self.token_dim)
