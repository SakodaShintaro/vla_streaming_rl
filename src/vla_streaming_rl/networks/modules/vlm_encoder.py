# SPDX-License-Identifier: MIT
import math

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import LinearAttentionCacheLayerMixin

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
    """画像の並びと指示の文を凍結した VLM に読ませ、1ステップ分の状態を作るエンコーダ。

    処理は2段に分かれる。``represent`` はこのステップの画像を VLM に読ませ、画像のトークンの
    位置の活性を ``[image_layer_start, image_layer_stop)`` の層（0 は言語モデルに入る前の
    埋め込みで、ViT の出力そのもの）で平均した、画像ごとのトークンの格子と、末尾の
    ``num_last_tokens`` 個の位置の活性の全層平均を並べた表現を返す。末尾はチャットテンプレートの
    固定の部分で、画像までを読んだ後の活性を持つ。VLM は凍結しているので、この表現は
    バッファに保存して使い回せる。

    VLM はエピソードを1本の user ターンとして読み続ける。先頭に指示の文を置き、ステップごとにスカラーの観測（global_step を除く）を書いた文と画像を
    キャッシュ（GDN の層の再帰状態と、full attention の層の KV）に積み、エピソードの始めで
    空にする。GDN の層の状態はエピソードの間ずっと持ち越す。full attention の層の KV は、
    指示の文（attention sink として常に残す）と直近の ``stream_window_steps`` ステップの分だけ
    残し、それより古いステップの分は捨てる。チャットテンプレートの末尾は毎ステップ最新の画像の
    後ろに一時的につないで読み、読んだ後はキャッシュから外す。表現はそれまでのステップを反映する。

    ``forward`` は保存した表現を位置ごとに ``token_dim`` へ射影し、同じステップの前の行動・
    報酬・スカラーの観測（global_step を除く）をそれぞれ ``token_dim`` 幅のトークンにした
    ものを並べ、``token_mixer_layers`` 層の Transformer でトークンどうしを混ぜて、ヘッドが
    読む状態にする。学習するのは射影と混ぜる層。
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
        assert vla_config.stream_window_steps >= 1, vla_config.stream_window_steps
        self.stream_window_steps = vla_config.stream_window_steps
        self._reset_stream()
        self.image_layers = slice(vla_config.image_layer_start, vla_config.image_layer_stop)
        self.image_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        # 観測の画像（一人称の視点と、あれば鳥瞰図）ごとのトークンの格子を並べる。格子の
        # 大きさは画像の大きさからプロセッサが決める（256px なら統合後 8x8）
        self.images_num = len(image_shapes)
        image_processor = self.processor.image_processor
        image_tokens_num = sum(
            int(
                image_processor(images=[torch.zeros(shape)], return_tensors="pt", do_rescale=False)[
                    "image_grid_thw"
                ][0].prod()
            )
            // image_processor.merge_size**2
            for shape in image_shapes
        )
        self.num_positions = image_tokens_num + self.num_last_tokens
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
        tokens_num = self.num_positions + action_dim + 1 + len(STATE_SCALAR_INDICES)
        self.output_dim = tokens_num * vla_config.token_dim
        # ヘッドに渡す前に、トークンどうしを Transformer の層で混ぜる
        # 位置の埋め込みは 0、残差は layer scale で小さく始めるので、
        # 学習の始めは混ぜないのとほぼ同じ。層の数が 0 なら混ぜない
        self.token_position = nn.Parameter(
            torch.zeros(1, tokens_num, vla_config.token_dim, device=device)
        )
        self.token_mixer = nn.Sequential(
            *[
                SpatialTransformerBlock(
                    vla_config.token_dim,
                    vla_config.token_mixer_heads,
                    tokens_num,
                    vla_config.token_mixer_layer_scale_init,
                )
                for _ in range(vla_config.token_mixer_layers)
            ]
        ).to(device)

    def representation_shape(self) -> tuple[int, ...]:
        """``represent`` が返す表現の形。射影は学習するので、その手前を保存する。"""
        return (self.num_positions, self.vlm_hidden_size)

    @torch.no_grad()
    def represent(
        self, images: list[torch.Tensor], scalar_obs: np.ndarray, text: str, reset_state: bool
    ) -> torch.Tensor:
        """[0, 1] の画像 (3, H, W) の並びと文を VLM に読ませ、画像のトークンの活性の
        ``image_layers`` の層での平均と、末尾の ``num_last_tokens`` 個の位置の活性の全層平均を
        並べた表現 (num_positions, hidden_size)。``reset_state`` ならエピソードの流れを空にして
        から、スカラーの観測 (SCALAR_OBS_DIM,) を書いた文と画像を流れに積む。"""
        assert len(images) == self.images_num, (len(images), self.images_num)
        if reset_state:
            self._reset_stream()
        return self._represent_stream(images, scalar_obs, text)

    def _image_tokens(self, input_ids: torch.Tensor, outputs) -> torch.Tensor:
        """画像のトークンの位置の活性の ``image_layers`` の層での平均。画像のトークンは、
        画像の順に、それぞれ統合後の格子のラスター順に並ぶ。"""
        image_position = input_ids[0] == self.image_pad_id
        image_layers = outputs.hidden_states[self.image_layers]
        return sum(depth[0, image_position].to(torch.float32) for depth in image_layers) / len(
            image_layers
        )

    def _reset_stream(self) -> None:
        """エピソードの流れを空にする。キャッシュと、次に読むトークンの位置（M-RoPE の位置）、
        full attention の層の KV のうち指示の文の長さと、残しているステップごとの長さ。"""
        self.stream_cache = DynamicCache(config=self.vlm.config.text_config)
        self.stream_next_position = 0
        self.stream_prompt_len = 0
        self.stream_step_lens: list[int] = []

    def _read_prompt(self, text: str) -> None:
        """流れの先頭で user ターンを開き、指示の文を読ませてキャッシュに積む。"""
        prompt_ids = self.processor.tokenizer(f"{IM_START}user\n{text}\n", return_tensors="pt")[
            "input_ids"
        ].to(self.device)
        self.stream_prompt_len = prompt_ids.shape[1]
        self.vlm(
            input_ids=prompt_ids,
            position_ids=self._text_positions(self.stream_prompt_len),
            past_key_values=self.stream_cache,
            use_cache=True,
            logits_to_keep=1,
        )
        self.stream_next_position = self.stream_prompt_len

    def _text_positions(self, length: int) -> torch.Tensor:
        """流れの続きに読む文の M-RoPE の位置 (3, 1, length)。文では3成分とも同じ。"""
        positions = torch.arange(length, device=self.device) + self.stream_next_position
        return positions.view(1, 1, -1).expand(3, 1, -1)

    def _drop_oldest_step(self) -> None:
        """full attention の層の KV から、指示の文の直後にある最も古いステップの分を捨てる。"""
        drop_len = self.stream_step_lens.pop(0)
        keep_head = slice(0, self.stream_prompt_len)
        keep_tail = slice(self.stream_prompt_len + drop_len, None)
        for layer in self.stream_cache.layers:
            if not isinstance(layer, LinearAttentionCacheLayerMixin):
                layer.keys = torch.cat(
                    [layer.keys[..., keep_head, :], layer.keys[..., keep_tail, :]], dim=-2
                )
                layer.values = torch.cat(
                    [layer.values[..., keep_head, :], layer.values[..., keep_tail, :]], dim=-2
                )

    def _represent_stream(
        self, images: list[torch.Tensor], scalar_obs: np.ndarray, text: str
    ) -> torch.Tensor:
        """このステップのスカラーの観測の文と画像をエピソードの流れに積み、続けて
        チャットテンプレートの末尾を一時的に読ませる。"""
        if self.stream_next_position == 0:
            self._read_prompt(text)
        scalars = ", ".join(
            f"{SCALAR_OBS_NAMES[index]}: {scalar_obs[index]:.2f}" for index in STATE_SCALAR_INDICES
        )
        inputs = self.processor(
            text=[f"{scalars}\n{IMAGE_PLACEHOLDER * len(images)}"],
            images=list(images),
            return_tensors="pt",
            do_rescale=False,
        ).to(self.device)
        # キャッシュに続けて読むときは M-RoPE の位置を VLM が画像込みで作れないので、
        # この分だけで作って流れの続きの位置にずらす
        step_positions, _ = self.vlm.model.get_rope_index(
            inputs["input_ids"],
            mm_token_type_ids=inputs["mm_token_type_ids"],
            image_grid_thw=inputs["image_grid_thw"],
        )
        step_positions = step_positions + self.stream_next_position
        self.stream_next_position = int(step_positions.max()) + 1
        # プロセッサの attention_mask はこの分の長さしかなく、キャッシュに積んだ分を覆わないので
        # 渡さない（渡すとマスクがずれる）。因果的なマスクはキャッシュの長さから作られる
        step_outputs = self.vlm(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            mm_token_type_ids=inputs["mm_token_type_ids"],
            position_ids=step_positions,
            past_key_values=self.stream_cache,
            use_cache=True,
            output_hidden_states=True,
            logits_to_keep=1,
        )
        self.stream_step_lens.append(inputs["input_ids"].shape[1])
        if len(self.stream_step_lens) > self.stream_window_steps:
            self._drop_oldest_step()

        # 末尾は読んだ後でキャッシュから外す。GDN の層の状態は読む前のものを書き戻し、
        # full attention の層の KV はこの分を切り落とす
        tail_ids = self.processor.tokenizer(
            f"{IM_END}\n{IM_START}assistant\n", return_tensors="pt"
        )["input_ids"].to(self.device)
        tail_len = tail_ids.shape[1]
        assert tail_len >= self.num_last_tokens, (tail_len, self.num_last_tokens)
        tail_positions = self._text_positions(tail_len)
        linear_layers = [
            layer
            for layer in self.stream_cache.layers
            if isinstance(layer, LinearAttentionCacheLayerMixin)
        ]
        saved_states = [
            (layer.conv_states[0].clone(), layer.recurrent_states[0].clone())
            for layer in linear_layers
        ]
        tail_outputs = self.vlm(
            input_ids=tail_ids,
            position_ids=tail_positions,
            past_key_values=self.stream_cache,
            use_cache=True,
            output_hidden_states=True,
            logits_to_keep=1,
        )
        for layer, (conv_state, recurrent_state) in zip(linear_layers, saved_states, strict=True):
            layer.conv_states[0].copy_(conv_state)
            layer.recurrent_states[0].copy_(recurrent_state)
        for layer in self.stream_cache.layers:
            if not isinstance(layer, LinearAttentionCacheLayerMixin):
                layer.crop(-tail_len)

        last_tokens = sum(
            depth[0, -self.num_last_tokens :].to(torch.float32)
            for depth in tail_outputs.hidden_states
        ) / len(tail_outputs.hidden_states)
        return torch.cat([self._image_tokens(inputs["input_ids"], step_outputs), last_tokens])

    def forward(
        self,
        representation: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        scalar_obs: torch.Tensor,
    ) -> torch.Tensor:
        """保存した表現 (B, num_positions, hidden_size) を射影したものに、同じステップの
        前の行動 (B, action_dim)、報酬 (B, 1)、スカラーの観測 (B, SCALAR_OBS_DIM) をそれぞれ
        トークンにしたものを並べ、ヘッドが読む状態 (B, output_dim) にする。

        スカラー1つを1次元のまま連結すると、表現の次元に比べて割合が小さく埋もれやすく、残りの
        収益の 1 と 2 のような小さい違いも1つの入力のわずかな差にしかならない。周期の短い成分は
        小さい違いを別のベクトルにし、周期の長い成分は大きい値のおおよその位置を表す。変換は
        固定で、学習中に変わらない。"""
        projected = self.state_out_proj(representation)
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
