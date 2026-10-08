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
YES_WORDS = ("yes", "Yes", " yes", " Yes")
NO_WORDS = ("no", "No", " no", " No")


def _history_text(turn: dict, system_prefix: str) -> str:
    """system / user の1メッセージをチャットテンプレートの履歴と同じ書式で文字列に
    する。テンプレート本体を使わないのは、会話全体ではなく新しいメッセージの差分
    だけを文字列にするため。返答はこの生成器が書いたトークンのままキャッシュに
    残るので、ここで描き直すことはない。``system_prefix`` はテンプレートがシステム
    プロンプトの先頭に足す文言（思考の量の指示）。"""
    assert turn["role"] != "assistant", "a reply stays in the cache as written"
    parts = [system_prefix] if turn["role"] == "system" else []
    for part in turn["content"]:
        if part["type"] == "image":
            parts.append(IMAGE_PLACEHOLDER)
        else:
            parts.append(part["text"])
    return f"{IM_START}{turn['role']}\n{''.join(parts)}{IM_END}\n"


@dataclass(frozen=True)
class Chain:
    """書いた1つの返答。トークン、そのテキスト、各トークンを選んだ位置の活性
    (chain_len, layers_num, hidden_size)、読んだプロンプトのトークン数、かかった時間。"""

    tokens: list[int]
    text: str
    positions: torch.Tensor
    prompt_tokens: int
    msec: float


class ChainGenerator:
    """
    生成は純増分で行う。会話全体を毎回プロンプトに組み直すのではなく、前回
    から増えたメッセージだけを履歴書式の文字列にしてトークン化し、持ち越した
    キャッシュの上に積む。確定した返答は、生成プロンプトの後ろに書いたトークンを
    <|im_end|> で閉じたものをそのまま履歴として残す。テンプレートが過去の思考を
    残す形（preserve_thinking）では、これが履歴の書式そのものになる。
    モデルの文脈の実体はこのキャッシュで、builder の会話リストは差分生成と
    表示・パース用になる。

    キャッシュは実トークン数分だけ使う DynamicCache。全注意層の KV だけが会話長に
    比例するので、そこを「先頭の sink（システムプロンプト）＋直近
    ``window_tokens`` トークン」まで間引く。線形注意層には中間の情報も残る。
    """

    def __init__(
        self,
        high_level_config: DictConfig,
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
        # 思考の量。"none" なら思考を閉じた形で返答を書かせ、それ以外はテンプレートの
        # reasoning_effort（low / medium / xhigh）に渡して、思考を書いてから返答させる
        reasoning_effort = high_level_config.reasoning_effort
        assert reasoning_effort in ("none", "low", "medium", "xhigh"), reasoning_effort
        self.enable_thinking = reasoning_effort != "none"
        template_kwargs = (
            {"enable_thinking": True, "reasoning_effort": reasoning_effort}
            if self.enable_thinking
            else {"enable_thinking": False}
        )
        self.device = device
        self.hidden_size = text_config.hidden_size
        # The embedding plus every layer's output.
        self.layers_num = text_config.num_hidden_layers + 1
        # テンプレートが会話の末尾に付ける生成プロンプトのトークン列。
        # 会話の中身に依らない定数なので、ここで一度だけ測る。
        probe = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
        with_tail = self.processor.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        without_tail = self.processor.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=False, **template_kwargs
        )
        assert with_tail.startswith(without_tail), (with_tail, without_tail)
        self._tail_ids = self.processor.tokenizer.encode(
            with_tail[len(without_tail) :], add_special_tokens=False
        )
        # テンプレートがシステムプロンプトの先頭に足す文言。思考の量の指示があればそれが入る
        marker = "SYSTEM_PROMPT_BODY"
        rendered = self.processor.apply_chat_template(
            [{"role": "system", "content": [{"type": "text", "text": marker}]}] + probe,
            tokenize=False,
            add_generation_prompt=False,
            **template_kwargs,
        )
        system_start = f"{IM_START}system\n"
        assert rendered.startswith(system_start) and marker in rendered, rendered
        self._system_prefix = rendered[len(system_start) : rendered.index(marker)]
        # 終端の判定で、返答の先頭に置く書き出し。思考を書かせる形では、空の思考を閉じてから
        # タグを置く（テンプレートが思考なしの返答に付けるのと同じ形）
        self._judge_prefix = "\n</think>\n\n" if self.enable_thinking else ""
        # 返答を閉じるトークン。生成はこの <|im_end|> で止まる
        assert self.processor.tokenizer.eos_token == IM_END, self.processor.tokenizer.eos_token
        self._newline_ids = self.processor.tokenizer.encode("\n", add_special_tokens=False)
        # yes / no を書き出しうる最初のトークン。大文字と前の空白の有無を含める
        tokenizer = self.processor.tokenizer
        self._yes_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in YES_WORDS]
        self._no_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in NO_WORDS]
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
        # 前回の返答を書いた会話の最後のメッセージと、そこへ確定した返答の文字列
        self._last_consumed = None
        self._committed_reply = ""

    @torch.inference_mode()
    def generate(self, conversation: list[dict]) -> Chain:
        """``conversation`` への返答を最大 ``max_len`` トークン書き、確定してキャッシュに
        残す。会話の画像は、エージェントが builder に渡したもの（8 ビットの画像か、
        [0, 1] の (C, H, W) float テンソル）で、processor には後者として渡る。"""
        start = time.perf_counter()
        # 会話の差分と生成プロンプトを1回の forward で積む
        ids, pos, pixel_values, image_grid_thw = self._encode(self._new_messages(conversation))
        tail_pos = self._linear_positions(len(self._tail_ids), int(pos.max().item()) + 1)
        outputs = self._prefill(
            torch.cat([ids, torch.tensor([self._tail_ids], device=self.device)], dim=1),
            torch.cat([pos, tail_pos], dim=2),
            pixel_values,
            image_grid_thw,
            hidden=True,
        )
        prompt_tokens = self._kv_len
        position = self._last_position(outputs.hidden_states)
        logits = outputs.logits[0, -1]

        # 1トークンずつ書く。選んだトークンは次のステップで積むので、最後に選んだものは
        # まだキャッシュに入っていない
        tokens = []
        positions = []
        while True:
            positions.append(position)
            # 温度 0 なら貪欲。それ以外はモデルの生成設定に従い、温度を掛け、上位 top_k に絞り、
            # そのうち確率の和が top_p に届く最少のものから引く
            if self.temperature == 0.0:
                token = int(logits.argmax().item())
            else:
                top = torch.topk(logits.float() / self.temperature, self.top_k)
                probs = torch.softmax(top.values, dim=-1)
                kept = probs * (probs.cumsum(dim=-1) - probs < self.top_p)
                token = int(top.indices[torch.multinomial(kept, 1)].item())
            tokens.append(token)
            if token == self.eos_token_id or len(tokens) >= self.max_len:
                break
            # 位置はモデル任せにせず、キャッシュの長さと 3D 位置の起点から渡す
            self._token.fill_(token)
            self._cache_position.fill_(self._kv_len)
            self._position_ids.fill_(self._next_pos)
            outputs = self.model(
                input_ids=self._token,
                past_key_values=self._cache,
                use_cache=True,
                output_hidden_states=True,
                cache_position=self._cache_position,
                position_ids=self._position_ids,
            )
            self._kv_len += 1
            self._next_pos += 1
            position = self._last_position(outputs.hidden_states)
            logits = outputs.logits[0, -1]
        chain = Chain(
            tokens=tokens,
            text=self.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
            positions=torch.stack(positions),
            prompt_tokens=prompt_tokens,
            msec=(time.perf_counter() - start) * 1000.0,
        )

        # 最後に選んだトークンを、<|im_end|>（打ち切ったときだけ足す）と改行で閉じて積む
        close = (
            [tokens[-1]]
            + ([] if tokens[-1] == self.eos_token_id else [self.eos_token_id])
            + self._newline_ids
        )
        self._prefill(
            torch.tensor([close], device=self.device),
            self._linear_positions(len(close), self._next_pos),
            None,
            None,
            hidden=False,
        )
        if self._sink_len == 0:
            self._sink_len = len(
                self.processor.tokenizer.encode(
                    _history_text(conversation[0], self._system_prefix), add_special_tokens=False
                )
            )
        # 全注意層の KV を「sink ＋ 直近 window_tokens 行」まで間引く。線形注意層は固定サイズの
        # 再帰状態なので対象外。行を捨てるだけで位置は元のまま残るから、残った行どうしの
        # 相対位置は変わらない
        limit = self._sink_len + self.window_tokens
        if self._kv_len > limit:
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
        self._last_consumed = conversation[-1]
        self._committed_reply = chain.text
        return chain

    @torch.inference_mode()
    def yes_probability(self, chain: Chain, tag: str) -> float | None:
        """返答で最後の ``tag`` の直後に書いた最初のトークンの位置で、yes を選ぶ確率を
        yes と no の確率の和で割って返す。返答に ``tag`` がなければ None。最後のものを
        読むのは、思考の中で同じ文字列に触れていても、返答の本体の判定を読むため。

        その位置の活性の最終層に lm_head を掛けると、そのトークンを選んだときの
        ロジットになる。"""
        tokenizer = self.processor.tokenizer
        found = None
        for i in range(len(chain.tokens)):
            written = tokenizer.decode(chain.tokens[:i], skip_special_tokens=True)
            if written.rstrip().endswith(tag):
                found = i
        if found is None:
            return None
        head = self.model.lm_head
        return self._yes_share(head(chain.positions[found, -1].to(head.weight.dtype)))

    @torch.inference_mode()
    def yes_probability_after(self, conversation: list[dict], tag: str) -> float:
        """最新のターンへの返答を ``tag`` まで書いたところで、次に yes を選ぶ確率を
        yes と no の確率の和で割って返す。返答は書かず、キャッシュも呼ぶ前に戻す。"""
        # 呼ぶ前のキャッシュを写し取る。全注意層は追記のたびに cat で新しいテンソルを作るが、
        # 線形注意層は再帰状態を in-place に更新するので、参照ではなく複製で控える。層の
        # 属性のうちモジュール以外（テンソル、テンソルの並び、長さなどの数値）を丸ごと控える
        snapshot = [
            {
                name: self._copied_value(value)
                for name, value in vars(layer).items()
                if not isinstance(value, torch.nn.Module)
            }
            for layer in self._cache.layers
        ]
        kv_len = self._kv_len
        next_pos = self._next_pos

        ids, pos, pixel_values, image_grid_thw = self._encode(self._new_messages(conversation))
        tail_ids = self._tail_ids + self.processor.tokenizer.encode(
            self._judge_prefix + tag, add_special_tokens=False
        )
        outputs = self._prefill(
            torch.cat([ids, torch.tensor([tail_ids], device=self.device)], dim=1),
            torch.cat(
                [pos, self._linear_positions(len(tail_ids), int(pos.max().item()) + 1)], dim=2
            ),
            pixel_values,
            image_grid_thw,
            hidden=False,
        )
        probability = self._yes_share(outputs.logits[0, -1])

        for layer, saved in zip(self._cache.layers, snapshot, strict=True):
            for name, value in saved.items():
                setattr(layer, name, value)
        self._kv_len = kv_len
        self._next_pos = next_pos
        return probability

    @classmethod
    def _copied_value(cls, value):
        """テンソルは clone、テンソルを含む list / tuple / dict（線形注意層の
        conv_states と recurrent_states は dict[int, Tensor]）は中身ごと複製した同型、
        それ以外（数値、真偽値、None など）はそのまま返す。"""
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, (list, tuple)):
            return type(value)(cls._copied_value(item) for item in value)
        if isinstance(value, dict):
            return {key: cls._copied_value(item) for key, item in value.items()}
        return value

    def _new_messages(self, conversation: list[dict]) -> list[dict]:
        """前回の返答を書いた後に会話へ増えたメッセージ。前回確定した返答は書いた
        トークンのままキャッシュに入っているので除く。"""
        if self._last_consumed is None:
            return list(conversation)
        indices = [i for i, turn in enumerate(conversation) if turn is self._last_consumed]
        assert len(indices) == 1, (
            "the last consumed message is gone from the conversation; call "
            "reset_cache() where the conversation restarts"
        )
        reply = conversation[indices[0] + 1]
        assert (
            reply["role"] == "assistant" and reply["content"][0]["text"] == self._committed_reply
        ), "the conversation must hold the reply this generator committed, as written"
        new_messages = list(conversation[indices[0] + 2 :])
        assert len(new_messages) > 0, "no new message since the last reply"
        return new_messages

    def _yes_share(self, logits: torch.Tensor) -> float:
        probs = torch.softmax(logits.float(), dim=-1)
        yes = probs[self._yes_ids].sum()
        no = probs[self._no_ids].sum()
        return float(yes / (yes + no))

    def _encode(self, messages: list[dict]):
        """メッセージを履歴の書式でトークン列にし、これまでの位置の続きの 3D 位置と
        画像の入力を添えて返す。mrope の位置割り当ては走査中の基点にしか依存しない
        ので、全体で計算して切り出すのと同じ値になる。"""
        text = "".join(_history_text(turn, self._system_prefix) for turn in messages)
        images = [
            part["image"]
            for turn in messages
            for part in turn["content"]
            if part["type"] == "image"
        ]
        if len(images) == 0:
            # テキストだけなら 3 軸とも同じ値の連番になる
            ids = self.processor(text=[text], return_tensors="pt")["input_ids"].to(self.device)
            return ids, self._linear_positions(int(ids.shape[1]), self._next_pos), None, None
        inputs = self.processor(
            text=[text],
            images=[TF.to_dtype(TF.to_image(image), torch.float32, scale=True) for image in images],
            return_tensors="pt",
            do_rescale=False,
        ).to(self.device)
        ids = inputs["input_ids"]
        image_grid_thw = inputs["image_grid_thw"]
        pos = (
            self.model.model.compute_3d_position_ids(
                input_ids=ids,
                inputs_embeds=None,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                attention_mask=inputs["attention_mask"],
                past_key_values=None,
                mm_token_type_ids=(
                    inputs["mm_token_type_ids"] if "mm_token_type_ids" in inputs else None
                ),
            )
            + self._next_pos
        )
        return ids, pos, inputs["pixel_values"], image_grid_thw

    def _prefill(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        hidden: bool,
    ):
        """``input_ids`` をキャッシュの続きに積み、キャッシュの長さと次の 3D 位置を進める。"""
        length = int(input_ids.shape[1])
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones(1, self._kv_len + length, device=self.device),
            position_ids=position_ids,
            cache_position=torch.arange(self._kv_len, self._kv_len + length, device=self.device),
            past_key_values=self._cache,
            use_cache=True,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=hidden,
            # 次のトークンを選ぶ最後の位置のロジットだけを使う
            logits_to_keep=1,
        )
        self._cache = outputs.past_key_values
        self._kv_len += length
        self._next_pos = int(position_ids.max().item()) + 1
        return outputs

    def _linear_positions(self, length: int, start: int) -> torch.Tensor:
        """テキストだけの区間の 3D 位置 (3, 1, length)。3軸とも同じ値で ``start`` から数える。"""
        return (torch.arange(length, device=self.device) + start).view(1, 1, -1).expand(3, 1, -1)

    def _last_position(self, hidden_states) -> torch.Tensor:
        """The activation at the newest position at every depth, (layers_num,
        hidden_size)."""
        return torch.stack([depth[0, -1] for depth in hidden_states]).to(torch.bfloat16)
