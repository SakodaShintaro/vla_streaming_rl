# SPDX-License-Identifier: MIT
import math

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import LinearAttentionCacheLayerMixin

from vla_streaming_rl.networks.modules.gdn_memory import LORA_TARGETS, GDNMemoryLayer
from vla_streaming_rl.networks.modules.reward_embed import embed_reward
from vla_streaming_rl.networks.modules.self_attention import (
    SpatialTransformerBlock,
    get_fourier_embeds_from_coordinates,
)
from vla_streaming_rl.networks.modules.vlm_backbone import load_model
from vla_streaming_rl.scalar_obs import SCALAR_OBS_NAMES

# VLM のチャットテンプレートの書式
IM_START = "<|im_start|>"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

# LoRA 付きの GDN の層に渡す表現で、トークンごとに最後のチャネルに入れる種類
PAD_TOKEN = 0.0
TEXT_TOKEN = 1.0
IMAGE_TOKEN = 2.0

# 状態に数値のまま入れるスカラーの観測の、SCALAR_OBS_NAMES での位置。global_step は
# エピソードの中の状況を表さず、学習の間ずっと増え続けるので入れない
STATE_SCALAR_INDICES = [
    index for index, name in enumerate(SCALAR_OBS_NAMES) if name != "global_step"
]


def step_text_items(
    scalar_obs: np.ndarray, prev_action: np.ndarray, reward: float
) -> list[tuple[str, str]]:
    """ステップの文の項目ごとの (名前, 書いた文字列)。文はこれを ", " でつないだもの。"""
    return [
        (
            "previous_action",
            "previous_action: [" + ", ".join(f"{value:.2f}" for value in prev_action) + "]",
        ),
        ("reward", f"reward: {reward:.4f}"),
    ] + [
        (SCALAR_OBS_NAMES[index], f"{SCALAR_OBS_NAMES[index]}: {scalar_obs[index]:.2f}")
        for index in STATE_SCALAR_INDICES
    ]


class VLMEncoder(nn.Module):
    """画像の並びと指示の文を凍結した VLM に読ませ、1ステップ分の状態を作るエンコーダ。

    処理は2段に分かれる。``represent`` はこのステップの画像を VLM に読ませ、画像のトークンの
    位置の活性を ``[image_layer_start, image_layer_stop)`` の層（0 は言語モデルに入る前の
    埋め込みで、ViT の出力そのもの）で平均した、画像ごとのトークンの格子を並べた表現を返す。
    VLM は凍結しているので、この表現はバッファに保存して使い回せる。

    VLM はエピソードを1本の user ターンとして読み続ける。先頭に指示の文を置き、ステップごとにスカラーの観測（global_step を除く）を書いた文と画像を
    キャッシュ（GDN の層の再帰状態と、full attention の層の KV）に積み、エピソードの始めで
    空にする。GDN の層の状態はエピソードの間ずっと持ち越す。full attention の層の KV は、
    指示の文（attention sink として常に残す）と直近の ``stream_window_steps`` ステップの分だけ
    残し、それより古いステップの分は捨てる。表現はそれまでのステップを反映する。
    ``stream_window_steps`` が 0 なら毎ステップ流れを空にして、そのステップだけを読ませる
    （GDN の層の状態も持ち越さず、履歴を使わない）。

    ``forward`` は保存した表現を位置ごとに ``token_dim`` へ射影し、同じステップの前の行動・
    報酬・スカラーの観測（global_step を除く）をそれぞれ ``token_dim`` 幅のトークンにした
    ものを並べ、``token_mixer_layers`` 層の Transformer でトークンどうしを混ぜて、ヘッドが
    読む状態にする。学習するのは射影と混ぜる層。

    ``use_memory`` なら、ヘッドが読む層（``image_layer_start`` の1層）の直前の
    ``gdn_memory_layers`` 層の GDN だけをエピソードの間の記憶にする。VLM は毎ステップ流れを
    空にして読み（``stream_window_steps`` は 0）、``represent`` はその最も下の層の入力をこのステップの
    全トークンについて返す。``remember`` はそれらの層を ``GDNMemoryLayer`` で下から順に計算し、
    画像のトークンの出力（``forward`` が読む表現）を返す。再帰の状態と RTRL のトレースはエンコーダが
    持ち、その層の LoRA は ``retrace_previous_tick`` を通した streaming の損失で RTRL で学習する。
    """

    def __init__(
        self,
        *,
        image_shapes: list[tuple[int, ...]],
        action_dim: int,
        vla_config: DictConfig,
        use_memory: bool,
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
        assert vla_config.stream_window_steps >= 0, vla_config.stream_window_steps
        self.stream_window_steps = vla_config.stream_window_steps
        self._reset_stream()
        self.image_layers = slice(vla_config.image_layer_start, vla_config.image_layer_stop)
        self.image_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.use_memory = use_memory
        if self.use_memory:
            # hidden_states[i] は層 i - 1 の出力なので、ヘッドが読む層を出すのは層 start - 1。
            # LoRA 付きの GDN の層はそこまでの gdn_memory_layers 層で、表現は最も下の層の入力
            assert vla_config.image_layer_stop == vla_config.image_layer_start + 1
            assert vla_config.gdn_memory_layers >= 1, vla_config.gdn_memory_layers
            self.memory_layer_start = vla_config.image_layer_start - vla_config.gdn_memory_layers
            memory_layer_indices = range(self.memory_layer_start, vla_config.image_layer_start)
            text_config = self.vlm.config.text_config
            for index in memory_layer_indices:
                assert text_config.layer_types[index] == "linear_attention", index
            # 記憶を持つのはその層だけにする
            assert self.stream_window_steps == 0, self.stream_window_steps
            self.max_step_tokens = vla_config.gdn_max_step_tokens
            self.memory_layers = nn.ModuleList(
                [
                    GDNMemoryLayer(
                        self.vlm.model.language_model.layers[index],
                        LORA_TARGETS,
                        vla_config.gdn_lora_rank,
                        vla_config.gdn_lora_alpha,
                    )
                    for index in memory_layer_indices
                ]
            )
            # LoRA 付きの GDN の層の再帰の状態（最新のティックを読んだ後）。RTRL のトレースは各層が持つ
            self.memory = self._initial_memory()
            # 最新とその前のティックのLoRA 付きの GDN の層の入力と、それを読む前の再帰の状態。トレースは
            # 前のティックを読んだ後の状態の感度まで進める（streaming の損失が前のティックを
            # 通し直すときに進める）
            self.latest_tick: tuple[torch.Tensor, torch.Tensor] | None = None
            self.previous_tick: tuple[torch.Tensor, torch.Tensor] | None = None
        # 観測の画像（一人称の視点と、あれば鳥瞰図）ごとのトークンの格子を並べる。格子の
        # 大きさは画像の大きさからプロセッサが決める（256px なら統合後 8x8）
        self.images_num = len(image_shapes)
        image_processor = self.processor.image_processor
        self.num_positions = sum(
            int(
                image_processor(images=[torch.zeros(shape)], return_tensors="pt", do_rescale=False)[
                    "image_grid_thw"
                ][0].prod()
            )
            // image_processor.merge_size**2
            for shape in image_shapes
        )
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
        """バッファに保存する、画像のトークンの表現の形。射影は学習するので、その手前を保存する。
        GDN の記憶があればLoRA 付きの GDN の層の出力で、なければ ``represent`` が返すものそのもの。"""
        return (self.num_positions, self.vlm_hidden_size)

    @torch.no_grad()
    def represent(
        self,
        images: list[torch.Tensor],
        scalar_obs: np.ndarray,
        prev_action: np.ndarray,
        reward: float,
        text: str,
        reset_state: bool,
    ) -> torch.Tensor:
        """[0, 1] の画像 (3, H, W) の並びと文を VLM に読ませ、画像のトークンの活性の
        ``image_layers`` の層での平均の表現 (num_positions, hidden_size)。``use_memory`` なら
        代わりに、LoRA 付きの GDN の層の入力（``_step_tokens``）。``reset_state`` か、
        履歴を使わない（``stream_window_steps`` が 0）ならエピソードの流れを空にしてから、
        前の行動 (action_dim,) と受け取った報酬とスカラーの観測 (SCALAR_OBS_DIM,) を
        書いた文と画像を流れに積む。"""
        assert len(images) == self.images_num, (len(images), self.images_num)
        if reset_state or self.stream_window_steps == 0:
            self._reset_stream()
        return self._represent_stream(images, scalar_obs, prev_action, reward, text)

    def _image_tokens(self, input_ids: torch.Tensor, outputs) -> torch.Tensor:
        """画像のトークンの位置の活性の ``image_layers`` の層での平均。画像のトークンは、
        画像の順に、それぞれ統合後の格子のラスター順に並ぶ。"""
        image_position = input_ids[0] == self.image_pad_id
        image_layers = outputs.hidden_states[self.image_layers]
        return sum(depth[0, image_position].to(torch.float32) for depth in image_layers) / len(
            image_layers
        )

    def _step_tokens(self, input_ids: torch.Tensor, outputs) -> torch.Tensor:
        """このステップの全トークンのLoRA 付きの GDN の層の入力を先頭に詰め、最後のチャネルにトークンの
        種類（文、画像、詰め物）を入れたもの (max_step_tokens, hidden_size + 1)。"""
        hidden = outputs.hidden_states[self.memory_layer_start][0].to(torch.float32)
        length = hidden.shape[0]
        assert length <= self.max_step_tokens, (length, self.max_step_tokens)
        kinds = torch.where(input_ids[0] == self.image_pad_id, IMAGE_TOKEN, TEXT_TOKEN)
        step_tokens = torch.zeros(
            self.max_step_tokens, self.vlm_hidden_size + 1, device=self.device
        )
        step_tokens[:length, :-1] = hidden
        step_tokens[:length, -1] = kinds
        return step_tokens

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

    def step_inputs(
        self,
        images: list[torch.Tensor],
        scalar_obs: np.ndarray,
        prev_action: np.ndarray,
        reward: float,
        start_position: int,
    ) -> tuple[dict, torch.Tensor]:
        """このステップの文と画像のプロセッサの入力と、流れの位置 ``start_position`` から続ける
        M-RoPE の位置 (3, 1, L)。"""
        scalars = ", ".join(item for _, item in step_text_items(scalar_obs, prev_action, reward))
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
        return inputs, step_positions + start_position

    def _represent_stream(
        self,
        images: list[torch.Tensor],
        scalar_obs: np.ndarray,
        prev_action: np.ndarray,
        reward: float,
        text: str,
    ) -> torch.Tensor:
        """このステップの前の行動と報酬とスカラーの観測の文と画像をエピソードの流れに積む。
        自分がどう動いて何を受け取ったかを流れに残し、前のエピソードの試みも含めて記憶できるようにする。"""
        if self.stream_next_position == 0:
            self._read_prompt(text)
        inputs, step_positions = self.step_inputs(
            images, scalar_obs, prev_action, reward, self.stream_next_position
        )
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

        if not self.use_memory:
            return self._image_tokens(inputs["input_ids"], step_outputs)
        return self._step_tokens(inputs["input_ids"], step_outputs)

    def reset_memory(self) -> None:
        """LoRA 付きの GDN の層の再帰の状態を空にする。空の記憶はパラメータによらないので、RTRL の
        トレースも 0 に戻す。"""
        self.memory = self._initial_memory()
        self.reset_memory_trace()

    def reset_memory_trace(self) -> None:
        assert self.use_memory
        for layer in self.memory_layers:
            layer.reset_trace()

    @torch.no_grad()
    def remember(self, step_tokens: torch.Tensor) -> torch.Tensor:
        """``represent`` が返したこのティックのLoRA 付きの GDN の層の入力 (max_step_tokens, hidden_size + 1)
        をLoRA 付きの GDN の層に読ませ、再帰の状態を1ティック進めて、画像のトークンの出力
        (num_positions, hidden_size) を返す。トレースは進めない。"""
        tick = (step_tokens.unsqueeze(0).to(self.device), self.memory)
        image, self.memory = self._run_memory_layers(*tick, False)
        self.previous_tick, self.latest_tick = self.latest_tick, tick
        return image[0]

    def retrace_previous_tick(self) -> torch.Tensor:
        """前のティックを、それを読む前の再帰の状態からLoRA 付きの GDN の層に通し直し、RTRL のトレースを
        1ステップ進めて、画像のトークンの出力 (1, num_positions, hidden_size) を返す。勾配を
        取るかは呼び出し側が決める。どのティックも、次のティックの後にちょうど1回呼ぶ。"""
        assert self.previous_tick is not None
        image, _ = self._run_memory_layers(*self.previous_tick, True)
        return image

    def forward(
        self,
        image: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        scalar_obs: torch.Tensor,
    ) -> torch.Tensor:
        """画像のトークンの表現 (B, num_positions, hidden_size) を射影したものに、同じステップの
        前の行動 (B, action_dim)、報酬 (B, 1)、スカラーの観測 (B, SCALAR_OBS_DIM) をそれぞれ
        トークンにしたものを並べ、ヘッドが読む状態 (B, output_dim) にする。

        スカラー1つを1次元のまま連結すると、表現の次元に比べて割合が小さく埋もれやすく、残りの
        収益の 1 と 2 のような小さい違いも1つの入力のわずかな差にしかならない。周期の短い成分は
        小さい違いを別のベクトルにし、周期の長い成分は大きい値のおおよその位置を表す。変換は
        固定で、学習中に変わらない。"""
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

    def _initial_memory(self) -> torch.Tensor:
        """エピソードの始めのLoRA 付きの GDN の層の再帰の状態 (1, ...)。"""
        size = sum(layer.memory_size() for layer in self.memory_layers)
        return torch.zeros(1, size, device=self.device)

    def _run_memory_layers(
        self, step_tokens: torch.Tensor, memory: torch.Tensor, advance_trace: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """1ティックのLoRA 付きの GDN の層の入力 (B, max_step_tokens, hidden_size + 1) を、LoRA 付きの GDN の層に
        前の再帰の状態 ``memory`` (B, ...) から下から順に読ませ、
        最も上の層の画像のトークンの出力 (B, num_positions, hidden_size) と次の再帰の状態を作る。
        ``advance_trace`` なら、RTRL のトレースを1ステップ進める。再帰の状態は層ごとの S と、
        畳み込みが読む直前のトークンの入力 h を並べたもの。各層が自分の LoRA の
        トレースだけを運ぶ（層ごとに閉じた近似）。"""
        B = step_tokens.shape[0]
        kinds = step_tokens[..., -1]
        valid = kinds != PAD_TOKEN
        # 詰め物は各行の後ろにあるので、どの行の詰め物でもない範囲だけ読む
        length = int(valid.sum(dim=1).max())
        valid = valid[:, :length]
        layer_memories = memory.split([layer.memory_size() for layer in self.memory_layers], dim=1)
        x = step_tokens[:, :length, :-1]
        next_memories = []
        for layer, layer_memory in zip(self.memory_layers, layer_memories, strict=True):
            state_size = layer.num_heads * layer.key_dim * layer.value_dim
            S, h = layer_memory.split([state_size, layer_memory.shape[1] - state_size], dim=1)
            x, S, h = layer(
                x,
                valid,
                S.view(B, layer.num_heads, layer.key_dim, layer.value_dim),
                h.view(B, layer.conv_size - 1, layer.hidden_size),
                advance_trace,
            )
            next_memories += [S.flatten(start_dim=1), h.flatten(start_dim=1)]
        image = x[kinds[:, :length] == IMAGE_TOKEN].view(B, self.num_positions, -1)
        return image, torch.cat(next_memories, dim=1)
