# SPDX-License-Identifier: MIT
import time
from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torchvision.transforms.v2 import functional as TF
from transformers.cache_utils import DynamicLayer

from .vlm_backbone import load_model

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def _history_text(turn: dict) -> str:
    """1メッセージをチャットテンプレートの履歴と同じ書式で文字列にする。

    assistant の <think> ブロックはテンプレートの履歴レンダリングと同じ規則
    （最後の </think> から後ろだけ残す）で剥ぐ。テンプレート本体を使わない
    のは、会話全体ではなく新しいメッセージの差分だけを文字列にするため。"""
    parts = []
    for part in turn["content"]:
        if part["type"] == "image":
            parts.append(IMAGE_PLACEHOLDER)
        else:
            text = part["text"]
            if turn["role"] == "assistant" and "</think>" in text:
                text = text.split("</think>")[-1].lstrip("\n")
            parts.append(text)
    return f"{IM_START}{turn['role']}\n{''.join(parts)}{IM_END}\n"


def _turn_images(turn: dict) -> list:
    return [part["image"] for part in turn["content"] if part["type"] == "image"]


@dataclass(frozen=True)
class Chain:
    """One generation: the tokens, the text they decode to, the activation
    behind each token as (chain_len, layers_num, hidden_size), and what the
    write cost."""

    tokens: list[int]
    text: str
    positions: torch.Tensor
    prompt_tokens: int
    msec: float
    finished: bool


class ChainGenerator:
    """
    生成は純増分で行う。会話全体を毎回プロンプトに組み直すのではなく、前回
    から増えたメッセージだけを履歴書式の文字列にしてトークン化し、持ち越した
    キャッシュの上に積む。返答と生成プロンプト末尾のトークンは、履歴の末尾で
    取ったスナップショットへ巻き戻すことで捨て、返答は次の差分に履歴形
    （<think> 内を除いた形）で入り直す。モデルの文脈の実体はこのキャッシュで、
    builder の会話リストは差分生成と表示・パース用になる。

    キャッシュは実トークン数分だけ使う DynamicCache。全注意層の KV だけが会話長に
    比例するので、そこを「先頭の sink（システムプロンプト）＋直近
    ``window_tokens`` トークン」まで間引く。線形注意層には中間の情報も残る。
    """

    def __init__(
        self,
        high_level_config: DictConfig,
        enable_thinking: bool,
        device: torch.device,
    ) -> None:
        max_len = high_level_config.max_new_tokens
        temperature = high_level_config.temperature
        window_tokens = high_level_config.window_tokens
        assert max_len >= 1, max_len
        assert temperature >= 0.0, temperature
        assert window_tokens >= 1, window_tokens
        self.model, self.processor = load_model(
            model_id=high_level_config.model_id,
            use_lora=False,
            load_in_4bit=high_level_config.load_in_4bit,
            device=device,
        )
        self.model.eval().requires_grad_(False)
        self._token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self._cache_position = torch.zeros(1, dtype=torch.long, device=device)
        self._position_ids = torch.zeros(3, 1, 1, dtype=torch.long, device=device)
        self.eos_token_id = self.processor.tokenizer.eos_token_id
        self.max_len = max_len
        self.temperature = temperature
        self.window_tokens = window_tokens
        text_config = self.model.config.text_config
        generation_config = self.model.generation_config
        self.top_k = (
            generation_config.top_k
            if generation_config.top_k is not None
            else text_config.vocab_size
        )
        self.top_p = generation_config.top_p if generation_config.top_p is not None else 1.0
        self.enable_thinking = enable_thinking
        self.device = device
        self.hidden_size = text_config.hidden_size
        # The embedding plus every layer's output, matching `CoTStream`.
        self.layers_num = text_config.num_hidden_layers + 1
        # テンプレートが会話の末尾に付ける生成プロンプトのトークン列。
        # 会話の中身に依らない定数なので、ここで一度だけ測る。
        probe = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
        with_tail = self.processor.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        without_tail = self.processor.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=False, enable_thinking=enable_thinking
        )
        assert with_tail.startswith(without_tail), (with_tail, without_tail)
        self._tail_ids = self.processor.tokenizer.encode(
            with_tail[len(without_tail) :], add_special_tokens=False
        )
        self.reset_cache()

    def reset_cache(self) -> None:
        """会話が仕切り直されるときに呼ぶ。次の生成は会話全体を積み直す。"""
        self._cache = None
        # 全注意層のキャッシュが持つ行数。追い出しで会話のトークン数より
        # 少なくなりうるので、キャッシュ側の長さとして別に数える
        self._kv_len = 0
        # 次のトークンに割り当てる 3D 位置の起点
        self._next_pos = 0
        # sink として永久に残す先頭（システムプロンプト）のトークン数
        self._sink_len = 0
        # 前回の生成までに処理し終えた会話の最後のメッセージ
        self._last_consumed = None
        self._boundary_state: list[dict] = []
        self._boundary_kv_len = 0
        self._boundary_next_pos = 0

    @torch.inference_mode()
    def generate(self, conversation: list[dict]) -> Chain:
        """Write a reply to ``conversation``. A frame in it is whatever the
        agent hands its builder -- an 8-bit picture or a (C, H, W) float tensor
        in [0, 1] -- and reaches the processor as the latter."""
        start = time.perf_counter()
        if self._last_consumed is None:
            new_messages = list(conversation)
        else:
            index = None
            for i in range(len(conversation) - 1, -1, -1):
                if conversation[i] is self._last_consumed:
                    index = i
                    break
            assert index is not None, (
                "the last consumed message is gone from the conversation; call "
                "reset_cache() where the conversation restarts"
            )
            new_messages = list(conversation[index + 1 :])
            self._restore_boundary()
        assert len(new_messages) > 0, "generate() got no new message"

        delta_text = "".join(_history_text(turn) for turn in new_messages)
        images = [image for turn in new_messages for image in _turn_images(turn)]
        if len(images) > 0:
            inputs = self.processor(
                text=[delta_text],
                images=[
                    TF.to_dtype(TF.to_image(image), torch.float32, scale=True) for image in images
                ],
                return_tensors="pt",
                do_rescale=False,
            ).to(self.device)
        else:
            inputs = self.processor(text=[delta_text], return_tensors="pt").to(self.device)
        ids = inputs["input_ids"]
        delta_len = int(ids.shape[1])

        # 差分だけの 3D 位置を出して、これまでの位置の続きへずらす。mrope の
        # 位置割り当ては走査中の基点にしか依存しないので、全体で計算して
        # 切り出すのと同じ値になる。
        pos = (
            self.model.model.compute_3d_position_ids(
                input_ids=ids,
                inputs_embeds=None,
                image_grid_thw=(inputs["image_grid_thw"] if "image_grid_thw" in inputs else None),
                video_grid_thw=None,
                attention_mask=inputs["attention_mask"],
                past_key_values=None,
                mm_token_type_ids=(
                    inputs["mm_token_type_ids"] if "mm_token_type_ids" in inputs else None
                ),
            )
            + self._next_pos
        )
        stage = self.model(
            input_ids=ids,
            attention_mask=torch.ones(1, self._kv_len + delta_len, device=self.device),
            position_ids=pos,
            cache_position=torch.arange(self._kv_len, self._kv_len + delta_len, device=self.device),
            past_key_values=self._cache,
            use_cache=True,
            pixel_values=inputs["pixel_values"] if "pixel_values" in inputs else None,
            image_grid_thw=inputs["image_grid_thw"] if "image_grid_thw" in inputs else None,
        )
        self._cache = stage.past_key_values
        self._kv_len += delta_len
        self._next_pos = int(pos.max().item()) + 1
        if self._sink_len == 0:
            self._sink_len = len(
                self.processor.tokenizer.encode(
                    _history_text(conversation[0]), add_special_tokens=False
                )
            )
        self._evict()
        self._last_consumed = conversation[-1]
        self._save_boundary()

        tail_len = len(self._tail_ids)
        tail_positions = (
            (torch.arange(tail_len, device=self.device) + self._next_pos)
            .view(1, 1, -1)
            .expand(3, 1, -1)
        )
        outputs = self.model(
            input_ids=torch.tensor([self._tail_ids], device=self.device),
            attention_mask=torch.ones(1, self._kv_len + tail_len, device=self.device),
            position_ids=tail_positions,
            cache_position=torch.arange(self._kv_len, self._kv_len + tail_len, device=self.device),
            past_key_values=self._cache,
            use_cache=True,
            output_hidden_states=True,
        )
        kv_pos = self._kv_len + tail_len
        rope_pos = self._next_pos + tail_len

        positions = [self._last_position(outputs.hidden_states)]
        tokens = [self._sample(outputs.logits[0, -1])]
        while tokens[-1] != self.eos_token_id and len(tokens) < self.max_len:
            self._token.fill_(tokens[-1])
            self._cache_position.fill_(kv_pos)
            self._position_ids.fill_(rope_pos)
            outputs = self._forward_step(self._token, self._cache_position, self._position_ids)
            positions.append(self._last_position(outputs.hidden_states))
            tokens.append(self._sample(outputs.logits[0, -1]))
            kv_pos += 1
            rope_pos += 1
        return Chain(
            tokens=tokens,
            text=self.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
            positions=torch.stack(positions),
            prompt_tokens=self._kv_len + tail_len,
            msec=(time.perf_counter() - start) * 1000.0,
            finished=tokens[-1] == self.eos_token_id,
        )

    def _evict(self) -> None:
        """全注意層の KV を「sink ＋ 直近 window_tokens 行」まで間引く。

        線形注意層は固定サイズの再帰状態なので対象外。行を捨てるだけで
        位置は元のまま残るから、残った行どうしの相対位置は変わらない。"""
        limit = self._sink_len + self.window_tokens
        if self._kv_len <= limit:
            return
        for layer in self._cache.layers:
            if isinstance(layer, DynamicLayer):
                layer.keys = torch.cat(
                    [
                        layer.keys[..., : self._sink_len, :],
                        layer.keys[..., -self.window_tokens :, :],
                    ],
                    dim=-2,
                )
                layer.values = torch.cat(
                    [
                        layer.values[..., : self._sink_len, :],
                        layer.values[..., -self.window_tokens :, :],
                    ],
                    dim=-2,
                )
        self._kv_len = limit

    @classmethod
    def _copied_value(cls, value):
        """スナップショットに保存できる形の複製。テンソルは clone、テンソルを
        含む list / tuple / dict（線形注意層の conv_states と recurrent_states
        は dict[int, Tensor]）は中身ごと複製した同型、数値・真偽値はそのまま。
        それ以外（None など）は複製不要としてそのまま返す。"""
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, (list, tuple)):
            return type(value)(cls._copied_value(item) for item in value)
        if isinstance(value, dict):
            return {key: cls._copied_value(item) for key, item in value.items()}
        return value

    def _save_boundary(self) -> None:
        """いまのキャッシュ状態を境界スナップショットとして写し取る。

        DynamicCache の全注意層は追記のたびに cat で新しいテンソルを作るが、
        線形注意層は再帰状態を in-place に更新するので、保存も復元も参照の
        共有ではなく複製で行う。層の属性のうちモジュール以外（テンソル、
        テンソルの並び、長さなどの数値）を丸ごと控える。"""
        self._boundary_state = [
            {
                name: self._copied_value(value)
                for name, value in vars(layer).items()
                if not isinstance(value, torch.nn.Module)
            }
            for layer in self._cache.layers
        ]
        self._boundary_kv_len = self._kv_len
        self._boundary_next_pos = self._next_pos

    def _restore_boundary(self) -> None:
        """キャッシュを境界スナップショットの状態へ書き戻す。スナップショット
        自体は後続の in-place 更新から守るため、渡すのは常に複製。"""
        for layer, snapshot in zip(self._cache.layers, self._boundary_state, strict=True):
            for name, value in snapshot.items():
                setattr(layer, name, self._copied_value(value))
        self._kv_len = self._boundary_kv_len
        self._next_pos = self._boundary_next_pos

    def _forward_step(
        self, token: torch.Tensor, cache_position: torch.Tensor, position_ids: torch.Tensor
    ):
        """キャッシュの上での1デコードステップ。位置はモデル任せにせず
        呼び出し側の走行カウンタから渡す。"""
        return self.model(
            input_ids=token,
            past_key_values=self._cache,
            use_cache=True,
            output_hidden_states=True,
            cache_position=cache_position,
            position_ids=position_ids,
        )

    def _last_position(self, hidden_states) -> torch.Tensor:
        """The activation at the newest position at every depth, (layers_num,
        hidden_size)."""
        return torch.stack([depth[0, -1] for depth in hidden_states]).to(torch.bfloat16)

    def _sample(self, logits: torch.Tensor) -> int:
        """The token the logits imply: greedy at temperature 0, otherwise
        sampled under the model's own generation config -- the temperature, then
        the ``top_k`` likeliest tokens, then the fewest of those that hold
        ``top_p`` of the probability."""
        if self.temperature == 0.0:
            return int(logits.argmax().item())
        top = torch.topk(logits.float() / self.temperature, self.top_k)
        probs = torch.softmax(top.values, dim=-1)
        kept = probs * (probs.cumsum(dim=-1) - probs < self.top_p)
        return int(top.indices[torch.multinomial(kept, 1)].item())
