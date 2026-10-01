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


def _history_text(turn: dict) -> str:
    """system / user の1メッセージをチャットテンプレートの履歴と同じ書式で文字列に
    する。テンプレート本体を使わないのは、会話全体ではなく新しいメッセージの差分
    だけを文字列にするため。返答はこの生成器が書いたトークンのままキャッシュに
    残るので、ここで描き直すことはない。"""
    assert turn["role"] != "assistant", "a reply stays in the cache as written"
    parts = []
    for part in turn["content"]:
        if part["type"] == "image":
            parts.append(IMAGE_PLACEHOLDER)
        else:
            parts.append(part["text"])
    return f"{IM_START}{turn['role']}\n{''.join(parts)}{IM_END}\n"


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
    キャッシュの上に積む。確定した返答は、生成プロンプトの後ろに書いたトークンを
    <|im_end|> で閉じたものをそのまま履歴として残す。テンプレートが過去の思考を
    残す形（preserve_thinking）では、これが履歴の書式そのものになる。境界は
    確定した返答の直後に取り、書きかけを進めるだけの呼び出しはそこへ巻き戻す。
    モデルの文脈の実体はこのキャッシュで、builder の会話リストは差分生成と
    表示・パース用になる。

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
        # The embedding plus every layer's output.
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
        # 前回の生成までに処理し終えた会話の最後のメッセージ
        self._last_consumed = None
        self._boundary_state: list[dict] = []
        self._boundary_kv_len = 0
        self._boundary_next_pos = 0
        # 書いている返答
        self._tokens: list[int] = []
        self._positions: list[torch.Tensor] = []
        self._kv_pos = 0
        self._rope_pos = 0
        # 次のトークンを選ぶ位置の活性とロジット
        self._next_position = None
        self._next_logits = None
        # 最後に選んだトークンがまだキャッシュに入っていないか
        self._unfed = False
        self._prompt_tokens = 0
        # 確定してキャッシュに入れた返答の文字列。builder が会話に足した返答と照合する
        self._committed_reply: str | None = None

    @torch.inference_mode()
    def generate(
        self, conversation: list[dict], prefix: list[int], budget: int, commit: bool
    ) -> Chain:
        """Write a reply to ``conversation``. A frame in it is whatever the
        agent hands its builder -- an 8-bit picture or a (C, H, W) float tensor
        in [0, 1] -- and reaches the processor as the latter.

        ``prefix`` は返答の書き出しとして生成プロンプトの後ろに積むトークン列で、
        その続きを最大 ``budget`` トークン書く。返答全体は ``max_len`` で打ち切る。
        ``commit`` が偽のときは最新のターンを境界に確定させず、返答も会話に残さない。
        次の呼び出しで境界へ巻き戻される。"""
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
            # キャッシュを境界スナップショットの状態へ書き戻す。スナップショット自体は
            # 後続の in-place 更新から守るため、渡すのは常に複製
            for layer, snapshot in zip(self._cache.layers, self._boundary_state, strict=True):
                for name, value in snapshot.items():
                    setattr(layer, name, self._copied_value(value))
            self._kv_len = self._boundary_kv_len
            self._next_pos = self._boundary_next_pos
        if self._committed_reply is not None:
            # 前回確定した返答は、書いたトークンのまま境界の手前に入っている
            reply = new_messages[0]
            assert (
                reply["role"] == "assistant"
                and reply["content"][0]["text"] == self._committed_reply
            ), "the conversation must hold the reply this generator committed, as written"
            self._last_consumed = reply
            self._committed_reply = None
            new_messages = new_messages[1:]
        assert len(new_messages) > 0, "generate() got no new message"
        assert commit or self._last_consumed is not None, (
            "a reply can only be continued after one has been committed"
        )
        if not commit and len(new_messages) > 1:
            # 最新のターンより前（行動を読めなかったときの環境の返事など）はもう変わらない
            # ので、ここで確定させて境界を進める。以降は最新のターンだけを積み直せばよい。
            settled = new_messages[:-1]
            ids, pos, pixel_values, image_grid_thw = self._encode(settled)
            self._prefill(ids, pos, pixel_values, image_grid_thw, hidden=False)
            self._settle(conversation, settled[-1])
            new_messages = new_messages[-1:]

        # 会話の差分、生成プロンプト、書きかけを1回の forward で積む。書きかけの i 番目の
        # トークンを選んだのは、その直前の位置の活性。
        ids, pos, pixel_values, image_grid_thw = self._encode(new_messages)
        tail_ids = torch.tensor([self._tail_ids + prefix], device=self.device)
        tail_pos = self._linear_positions(int(tail_ids.shape[1]), int(pos.max().item()) + 1)
        outputs = self._prefill(
            torch.cat([ids, tail_ids], dim=1),
            torch.cat([pos, tail_pos], dim=2),
            pixel_values,
            image_grid_thw,
            hidden=True,
        )
        first = int(ids.shape[1]) + len(self._tail_ids) - 1
        self._kv_pos = self._kv_len
        self._rope_pos = self._next_pos
        self._positions = [
            torch.stack([depth[0, first + i] for depth in outputs.hidden_states]).to(torch.bfloat16)
            for i in range(len(prefix))
        ]
        self._tokens = list(prefix)
        self._next_position = self._last_position(outputs.hidden_states)
        self._next_logits = outputs.logits[0, -1]
        self._unfed = False
        self._prompt_tokens = self._kv_len

        # 返答に最大 budget トークンを足す。最後に選んだトークンはキャッシュに入れずにおき、
        # 確定するときに閉じのトークンと一緒に積む
        written = 0
        while (
            not (len(self._tokens) > 0 and self._tokens[-1] == self.eos_token_id)
            and written < budget
            and len(self._tokens) < self.max_len
        ):
            if self._unfed:
                # キャッシュの上での1デコードステップ。位置はモデル任せにせず走行カウンタから渡す
                self._token.fill_(self._tokens[-1])
                self._cache_position.fill_(self._kv_pos)
                self._position_ids.fill_(self._rope_pos)
                outputs = self.model(
                    input_ids=self._token,
                    past_key_values=self._cache,
                    use_cache=True,
                    output_hidden_states=True,
                    cache_position=self._cache_position,
                    position_ids=self._position_ids,
                )
                self._next_position = self._last_position(outputs.hidden_states)
                self._next_logits = outputs.logits[0, -1]
                self._kv_pos += 1
                self._rope_pos += 1
            self._positions.append(self._next_position)
            # 温度 0 なら貪欲。それ以外はモデルの生成設定に従い、温度を掛け、上位 top_k に絞り、
            # そのうち確率の和が top_p に届く最少のものから引く
            if self.temperature == 0.0:
                token = int(self._next_logits.argmax().item())
            else:
                top = torch.topk(self._next_logits.float() / self.temperature, self.top_k)
                probs = torch.softmax(top.values, dim=-1)
                kept = probs * (probs.cumsum(dim=-1) - probs < self.top_p)
                token = int(top.indices[torch.multinomial(kept, 1)].item())
            self._tokens.append(token)
            self._unfed = True
            written += 1
        assert len(self._tokens) > 0, "nothing was written: a fresh reply needs a positive budget"
        ended = self._tokens[-1] == self.eos_token_id
        chain = Chain(
            tokens=list(self._tokens),
            text=self.processor.tokenizer.decode(self._tokens, skip_special_tokens=True).strip(),
            positions=torch.stack(self._positions),
            prompt_tokens=self._prompt_tokens,
            msec=(time.perf_counter() - start) * 1000.0,
            finished=ended,
        )

        if commit:
            # 書き終えた返答を <|im_end|> と改行で閉じてキャッシュに積み、その直後を境界として
            # 確定させる。最後に選んだトークンがまだ入っていなければ一緒に積む
            assert ended or len(self._tokens) >= self.max_len, "a committed reply must be complete"
            close = (
                ([self._tokens[-1]] if self._unfed else [])
                + ([] if ended else [self.eos_token_id])
                + self._newline_ids
            )
            self._kv_len = self._kv_pos
            self._next_pos = self._rope_pos
            self._prefill(
                torch.tensor([close], device=self.device),
                self._linear_positions(len(close), self._next_pos),
                None,
                None,
                hidden=False,
            )
            self._unfed = False
            self._settle(conversation, conversation[-1])
            self._committed_reply = chain.text
        return chain

    @torch.inference_mode()
    def yes_probability(self, chain: Chain, tag: str) -> float | None:
        """返答で ``tag`` の直後に書いた最初のトークンの位置で、yes を選ぶ確率を
        yes と no の確率の和で割って返す。返答に ``tag`` がなければ None。

        その位置の活性の最終層に lm_head を掛けると、そのトークンを選んだときの
        ロジットになる。"""
        tokenizer = self.processor.tokenizer
        for i in range(len(chain.tokens)):
            written = tokenizer.decode(chain.tokens[:i], skip_special_tokens=True)
            if written.rstrip().endswith(tag):
                head = self.model.lm_head
                return self._yes_share(head(chain.positions[i, -1].to(head.weight.dtype)))
        return None

    @torch.inference_mode()
    def yes_probability_after(self, conversation: list[dict], tag: str) -> float:
        """最新のターンへの返答を ``tag`` まで書いたところで、次に yes を選ぶ確率を
        yes と no の確率の和で割って返す。返答は書かず、会話にも確定させない。"""
        prefix = self.processor.tokenizer.encode(tag, add_special_tokens=False)
        self.generate(conversation, prefix, 0, commit=False)
        return self._yes_share(self._next_logits)

    def _yes_share(self, logits: torch.Tensor) -> float:
        probs = torch.softmax(logits.float(), dim=-1)
        yes = probs[self._yes_ids].sum()
        no = probs[self._no_ids].sum()
        return float(yes / (yes + no))

    def _encode(self, messages: list[dict]):
        """メッセージを履歴の書式でトークン列にし、これまでの位置の続きの 3D 位置と
        画像の入力を添えて返す。mrope の位置割り当ては走査中の基点にしか依存しない
        ので、全体で計算して切り出すのと同じ値になる。"""
        text = "".join(_history_text(turn) for turn in messages)
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

    def _settle(self, conversation: list[dict], last: dict) -> None:
        """``last`` までを積んだいまのキャッシュを確定させ、境界にする。"""
        if self._sink_len == 0:
            self._sink_len = len(
                self.processor.tokenizer.encode(
                    _history_text(conversation[0]), add_special_tokens=False
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
        self._last_consumed = last
        # いまのキャッシュ状態を境界スナップショットとして写し取る。全注意層は追記のたびに
        # cat で新しいテンソルを作るが、線形注意層は再帰状態を in-place に更新するので、
        # 保存も復元も参照の共有ではなく複製で行う。層の属性のうちモジュール以外
        # （テンソル、テンソルの並び、長さなどの数値）を丸ごと控える
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

    def _last_position(self, hidden_states) -> torch.Tensor:
        """The activation at the newest position at every depth, (layers_num,
        hidden_size)."""
        return torch.stack([depth[0, -1] for depth in hidden_states]).to(torch.bfloat16)
