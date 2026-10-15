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
# 本体を自由に書かせる返答で、思考の後ろに残しておくトークン数
FREE_BODY_TOKENS = 160


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
    """書いた1つの返答。トークン、そのテキスト、このステップで新しく積んだ位置（増えた
    メッセージ、生成プロンプト、書いた返答）の活性を全層で平均したもの
    (positions_num, hidden_size)、読んだプロンプトのトークン数、かかった時間。"""

    tokens: list[int]
    text: str
    states: torch.Tensor
    prompt_tokens: int
    msec: float


class ChainGenerator:
    """
    生成は純増分で行う。会話全体を毎回プロンプトに組み直すのではなく、前回
    から増えたメッセージだけを履歴書式の文字列にしてトークン化し、持ち越した
    キャッシュの上に積む。確定した返答は、生成プロンプトの後ろに書いたトークンを
    <|im_end|> で閉じたものをそのまま履歴として残す。テンプレートが過去の思考を
    残す形（preserve_thinking）では、これが履歴の書式そのものになる。
    モデルの文脈の実体はこのキャッシュで、呼び出し側の会話リストは差分生成と
    表示用になる。

    キャッシュは実トークン数分だけ使う DynamicCache。全注意層の KV だけが会話長に
    比例するので、そこを「先頭の sink ＋直近 ``window_tokens`` トークン」まで間引く。
    線形注意層には中間の情報も残る。

    思考を書かせるかどうかは、返答ごとに呼び出し側が決める。思考を書いた返答も、ほかの
    ターンと同じく直近 ``window_tokens`` トークンの中に残り、古くなれば押し出される。
    sink に入るのはシステムプロンプトだけ。

    ``candidates`` を渡すと、返答の本体（思考の後ろ）はその候補のどれか1つになる。次に
    選べるトークンを、候補のどれかの続きになるものだけに絞って書かせるので、書式の崩れた
    返答は出ない。思考が長引いたら、候補を書き切れるうちに思考を閉じさせる。候補が空なら
    本体も自由に書かせる。
    """

    def __init__(
        self,
        vla_config: DictConfig,
        candidates: list[str],
        device: torch.device,
    ) -> None:
        max_len = vla_config.max_new_tokens
        temperature = vla_config.temperature
        window_tokens = vla_config.window_tokens
        assert max_len >= 1, max_len
        assert temperature >= 0.0, temperature
        assert window_tokens >= 1, window_tokens
        self.model, self.processor = load_model(
            model_id=vla_config.model_id,
            use_lora=False,
            load_in_4bit=vla_config.load_in_4bit,
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
        # 思考を書かせる返答での思考の量。"none" なら思考は書かせられない。思考の量の段階
        # （reasoning_effort: low / medium / xhigh）を持つテンプレートには、それも渡す
        reasoning_effort = vla_config.reasoning_effort
        assert reasoning_effort in ("none", "low", "medium", "xhigh"), reasoning_effort
        self.thinking_enabled = reasoning_effort != "none"
        template_kwargs = {"enable_thinking": self.thinking_enabled}
        if self.thinking_enabled and "reasoning_effort" in self.processor.chat_template:
            template_kwargs["reasoning_effort"] = reasoning_effort
        self.device = device
        self.hidden_size = text_config.hidden_size
        # テンプレートが会話の末尾に付ける生成プロンプトのトークン列。会話の中身に依らない
        # 定数なので、ここで一度だけ測る。思考を書かせる返答に付けるものと、思考を閉じた形で
        # 書かせる返答に付けるもの
        probe = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
        self._thinking_tail_ids = self._generation_prompt_ids(probe, template_kwargs)
        self._tail_ids = self._generation_prompt_ids(probe, {"enable_thinking": False})
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
        # 返答を閉じるトークン。生成はこの <|im_end|> で止まる
        assert self.processor.tokenizer.eos_token == IM_END, self.processor.tokenizer.eos_token
        self._newline_ids = self.processor.tokenizer.encode("\n", add_special_tokens=False)
        # 返答の本体として書ける候補のトークン列
        tokenizer = self.processor.tokenizer
        self._candidate_ids = [
            tokenizer.encode(candidate, add_special_tokens=False) for candidate in candidates
        ]
        assert all(len(ids) > 0 for ids in self._candidate_ids), candidates
        # 思考を閉じるトークンと、テンプレートが思考と本体のあいだに置く書式
        think_close = tokenizer.encode("</think>", add_special_tokens=False)
        assert len(think_close) == 1, think_close
        self._think_close_id = think_close[0]
        self._think_close_ids = tokenizer.encode("\n</think>\n\n", add_special_tokens=False)
        self._after_think_ids = tokenizer.encode("\n\n", add_special_tokens=False)
        # 候補に絞るときに思考へ使えるトークン数。思考を閉じる書式と、いちばん長い候補と、
        # 返答を閉じるトークンの分を残す
        self._thinking_budget = (
            max_len
            - len(self._think_close_ids)
            - max([len(ids) for ids in self._candidate_ids] + [0])
            - 1
        )
        assert len(self._candidate_ids) == 0 or self._thinking_budget >= 1, (
            f"max_new_tokens {max_len} leaves no room to think before a candidate"
        )
        # 本体を自由に書かせるときに思考へ使えるトークン数。本体に FREE_BODY_TOKENS を残す
        self._free_thinking_budget = max_len - len(self._think_close_ids) - FREE_BODY_TOKENS
        assert self._free_thinking_budget >= 1, (
            f"max_new_tokens {max_len} leaves no room to think before a free reply"
        )
        self.reset_cache()

    def _generation_prompt_ids(self, probe: list[dict], template_kwargs: dict) -> list[int]:
        """テンプレートが ``template_kwargs`` のもとで会話の末尾に付ける生成プロンプト。"""
        with_tail = self.processor.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        without_tail = self.processor.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=False, **template_kwargs
        )
        assert with_tail.startswith(without_tail), (with_tail, without_tail)
        return self.processor.tokenizer.encode(
            with_tail[len(without_tail) :], add_special_tokens=False
        )

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
    def generate(self, conversation: list[dict], thinking: bool, constrained: bool) -> Chain:
        """``conversation`` への返答を最大 ``max_len`` トークン書き、確定してキャッシュに
        残す。``thinking`` なら、思考を書いてから返答させる。``constrained`` なら返答の
        本体を候補のどれかに絞り、そうでなければ自由に書かせる。会話の画像は、エージェントが
        builder に渡したもの（8 ビットの画像か、[0, 1] の (C, H, W) float テンソル）で、
        processor には後者として渡る。"""
        assert self.thinking_enabled or not thinking, "reasoning_effort is none"
        assert len(self._candidate_ids) > 0 or not constrained, "no candidates to choose among"
        # 思考に使えるトークン数。越えたら思考を閉じさせて、残りで本体を書かせる
        thinking_budget = self._thinking_budget if constrained else self._free_thinking_budget
        start = time.perf_counter()
        # 会話の最初の返答か
        opening = self._cache is None
        tail_ids = self._thinking_tail_ids if thinking else self._tail_ids
        # 会話の差分と生成プロンプトを1回の forward で積む
        ids, pos, pixel_values, image_grid_thw = self._encode(self._new_messages(conversation))
        tail_pos = self._linear_positions(len(tail_ids), int(pos.max().item()) + 1)
        outputs = self._prefill(
            torch.cat([ids, torch.tensor([tail_ids], device=self.device)], dim=1),
            torch.cat([pos, tail_pos], dim=2),
            pixel_values,
            image_grid_thw,
            hidden=True,
        )
        prompt_tokens = self._kv_len
        # このステップで新しく積んだ位置の活性。増えたメッセージと生成プロンプトの分
        states = [self._layer_mean(outputs.hidden_states)]
        logits = outputs.logits[0, -1]

        # 1トークンずつ書く。選んだトークンは次のステップで積むので、最後に選んだものは
        # まだキャッシュに入っていない
        tokens = []
        # 決められた通りに書かせる残りのトークン（思考を閉じる書式）と、本体として書いた分
        forced = []
        body = []
        while True:
            if thinking and len(tokens) >= thinking_budget:
                # 思考が長引いたので、本体を書き切れるうちに閉じさせる
                forced = list(self._think_close_ids)
                thinking = False
            if len(forced) > 0:
                token = forced.pop(0)
            else:
                if constrained and not thinking:
                    logits = self._constrain(logits, body)
                token = self._sample(logits)
                if constrained and not thinking:
                    body.append(token)
            tokens.append(token)
            if thinking and token == self._think_close_id:
                # モデルが自分で思考を閉じた。テンプレートの書式どおり空行を置いてから本体へ
                thinking = False
                forced = list(self._after_think_ids) if constrained else []
            if token == self.eos_token_id or len(tokens) >= self.max_len:
                break
            outputs = self._feed_token(token)
            states.append(self._layer_mean(outputs.hidden_states))
            logits = outputs.logits[0, -1]
        chain = Chain(
            tokens=tokens,
            text=self.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
            states=torch.cat(states),
            prompt_tokens=prompt_tokens,
            msec=(time.perf_counter() - start) * 1000.0,
        )

        self._close_reply(tokens[-1])
        if opening:
            self._sink_len = len(
                self.processor.tokenizer.encode(
                    _history_text(conversation[0], self._system_prefix), add_special_tokens=False
                )
            )
        self._evict()
        self._last_consumed = conversation[-1]
        self._committed_reply = chain.text
        return chain

    def _constrain(self, logits: torch.Tensor, body: list[int]) -> torch.Tensor:
        """次に選べるトークンを、書いた本体 ``body`` に続けて候補のどれかになるものだけに
        絞ったロジット。候補を書き切ったら、返答を閉じるトークンだけが残る。"""
        allowed = sorted(
            {
                ids[len(body)] if len(ids) > len(body) else self.eos_token_id
                for ids in self._candidate_ids
                if ids[: len(body)] == body
            }
        )
        masked = torch.full_like(logits, float("-inf"))
        masked[allowed] = logits[allowed]
        return masked

    def _sample(self, logits: torch.Tensor) -> int:
        """次のトークンを1つ選ぶ。温度 0 なら貪欲。それ以外はモデルの生成設定に従い、温度を
        掛け、上位 top_k に絞り、そのうち確率の和が top_p に届く最少のものから引く。"""
        if self.temperature == 0.0:
            return int(logits.argmax().item())
        top = torch.topk(logits.float() / self.temperature, self.top_k)
        probs = torch.softmax(top.values, dim=-1)
        kept = probs * (probs.cumsum(dim=-1) - probs < self.top_p)
        return int(top.indices[torch.multinomial(kept, 1)].item())

    def _feed_token(self, token: int):
        """選んだ1トークンをキャッシュの続きに積み、その出力を返す。位置はモデル任せに
        せず、キャッシュの長さと 3D 位置の起点から渡す。"""
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
        return outputs

    def _close_reply(self, last_token: int) -> None:
        """最後に選んだトークン（まだキャッシュに入っていない）を、<|im_end|>（打ち切った
        ときだけ足す）と改行で閉じて積む。"""
        close = (
            [last_token]
            + ([] if last_token == self.eos_token_id else [self.eos_token_id])
            + self._newline_ids
        )
        self._prefill(
            torch.tensor([close], device=self.device),
            self._linear_positions(len(close), self._next_pos),
            None,
            None,
            hidden=False,
        )

    def _evict(self) -> None:
        """全注意層の KV を「sink ＋ 直近 window_tokens 行」まで間引く。線形注意層は固定
        サイズの再帰状態なので対象外。行を捨てるだけで位置は元のまま残るから、残った行
        どうしの相対位置は変わらない。"""
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

    def _layer_mean(self, hidden_states) -> torch.Tensor:
        """積んだ各位置の活性を、全層（埋め込みと各層の出力）で平均したもの、
        (positions_num, hidden_size)。全層を一度に積まず、足し込みながら平均する。"""
        mean = None
        for depth in hidden_states:
            term = depth[0].to(torch.float32)
            mean = term if mean is None else mean + term
        return (mean / len(hidden_states)).to(torch.bfloat16)
