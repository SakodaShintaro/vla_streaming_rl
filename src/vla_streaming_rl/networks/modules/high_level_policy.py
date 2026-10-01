# SPDX-License-Identifier: MIT
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from vla_streaming_rl.agents.prompt import ACHIEVED_TAG, SUBTASK_RE, PromptBuilder, assistant_turn

from .chain_generator import ChainGenerator


class HighLevelPolicy:
    """凍結 VLM による高レベル方策。``steps_per_reply`` ステップごとに builder の会話へ
    返答を書いて確定し、その間は同じ返答を持ち続ける。低レベル方策には返答の
    ``<subtask>`` 区間の活性を、1ステップに読む幅へ均して渡す。"""

    def __init__(
        self,
        high_level_config: DictConfig,
        prompt_builder: PromptBuilder,
        device: torch.device,
    ) -> None:
        tokens_per_step = high_level_config.subtask_tokens_num
        steps_per_reply = high_level_config.steps_per_reply
        assert tokens_per_step >= 1, f"tokens_per_step must be positive; got {tokens_per_step}"
        assert steps_per_reply >= 1, f"steps_per_reply must be positive; got {steps_per_reply}"
        # 思考オフ。<think> を開けたままにすると、場面ではなく依頼について考えて返答を使い切る
        self.generator = ChainGenerator(high_level_config, enable_thinking=False, device=device)
        assert self.generator.max_len >= tokens_per_step, (
            f"max_len {self.generator.max_len} below {tokens_per_step}: the pool would "
            "stretch a reply shorter than one step's read"
        )
        self.tokens_per_step = tokens_per_step
        self.steps_per_reply = steps_per_reply
        # 会話はエージェントのもの。返答を書くステップでだけ読み、書いた返答をそのターンに足す
        self.prompt_builder = prompt_builder
        self.device = device
        self.reset()

    def reset(self) -> None:
        """返答を捨てる。次の advance はそのフレームで新しい返答を書く。書き込む先の
        会話をリセットするのは builder の側。"""
        self.generator.reset_cache()
        self._text = ""
        # 確定した返答が判定した、前の返答のサブタスクの達成度。エピソードの最初の
        # 返答には前のサブタスクがないので None
        self._achieved: float | None = None
        self._replies_written = 0
        self._last_conversation = []
        # 最後の返答の費用。返答を持ち続けるステップは費用を払ったステップと違うので保持する
        self._input_tokens = 0
        self._output_tokens = 0
        self._msec = 0.0
        self._activations = torch.zeros(
            (self.tokens_per_step, self.generator.layers_num, self.generator.hidden_size),
            dtype=torch.bfloat16,
            device=self.device,
        )
        # 0 は「いま書く」。エピソードの最初の advance は、そのエピソードの最初のフレームで書く
        self._until_next = 0

    def age(self) -> int:
        """いま読んでいる返答を何ステップ前に書いたか。書いたステップで 0、持ち続ける
        最後のステップで ``steps_per_reply - 1``。同じ活性でも、書いたフレームの上と
        十数ステップ後とでは意味が違うので、エンコーダは活性と一緒にこれを読む。"""
        return self.steps_per_reply - 1 - self._until_next

    @torch.inference_mode()
    def advance(self) -> torch.Tensor:
        """このステップの活性。返答を書く番なら書いて確定する。

        Returns:
            (tokens_per_step, layers_num, hidden_size) bfloat16。次の返答を書くまで同じもの。
        """
        if self._until_next == 0:
            self._last_conversation = self.prompt_builder.conversation()
            reply = self.generator.generate(
                self._last_conversation, [], self.generator.max_len, commit=True
            )
            self._achieved = (
                self.generator.yes_probability(reply, ACHIEVED_TAG)
                if self._replies_written > 0
                else None
            )
            self._replies_written += 1

            # <subtask> 区間のトークンの活性だけを読む。区間の長さによらず、区間方向に
            # 均して1ステップに読む幅にそろえる。区間がなければ 0
            tokenizer = self.generator.processor.tokenizer
            match = SUBTASK_RE.search(tokenizer.decode(reply.tokens, skip_special_tokens=True))
            rows = []
            if match is not None:
                start = 0
                for i in range(len(reply.tokens)):
                    end = len(tokenizer.decode(reply.tokens[: i + 1], skip_special_tokens=True))
                    if start < match.end(1) and end > match.start(1):
                        rows.append(i)
                    start = end
            if len(rows) == 0:
                self._activations = torch.zeros_like(self._activations)
            else:
                span = reply.positions[rows].to(torch.float32).permute(1, 2, 0)
                pooled = F.adaptive_avg_pool1d(span, self.tokens_per_step)
                self._activations = pooled.permute(2, 0, 1).to(torch.bfloat16)

            self._text = reply.text
            self._input_tokens = reply.prompt_tokens
            self._output_tokens = len(reply.tokens)
            self._msec = reply.msec
            self.prompt_builder.add_reply(reply.text)
            self._until_next = self.steps_per_reply
        self._until_next -= 1
        return self._activations

    def judge_current(self) -> float:
        """いま実行中のサブタスクの達成度を、いまのターンで判定する。エピソードが
        終わり、次の返答が来ないサブタスクのため。"""
        assert self._replies_written > 0, "there is no subtask running before the first reply"
        return self.generator.yes_probability_after(
            self.prompt_builder.conversation(), ACHIEVED_TAG
        )

    def achieved(self) -> float | None:
        """最後に確定した返答が判定した、前の返答のサブタスクの達成度（yes の確率）。
        前のサブタスクがないか、返答に判定がなければ None。"""
        return self._achieved

    def stats(self) -> dict:
        """最後の返答の費用。読んだトークン数、書いたトークン数、かかった時間。"""
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "msec": self._msec,
        }

    def text(self) -> str:
        """最後に確定した返答。"""
        return self._text

    def exchange(self) -> list[dict]:
        """最後の返答を書いたときの会話と返答。描画用。"""
        return self._last_conversation + [assistant_turn(self.text())]
