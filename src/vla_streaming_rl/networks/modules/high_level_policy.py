# SPDX-License-Identifier: MIT
import dataclasses
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from vla_streaming_rl.agents.prompt import ACHIEVED_TAG, SUBTASK_RE, PromptBuilder, assistant_turn

from .chain_generator import ChainGenerator


@dataclass(frozen=True)
class HighLevelPolicyOutput:
    """高レベル方策がいま持っている返答。返答を書いたステップから次を書くまで同じ
    ものを返し、``age`` だけが進む。"""

    # (tokens_per_step, layers_num, hidden_size)。返答の <subtask> 区間の活性を、
    # 1ステップに読む幅へ均したもの
    activations: torch.Tensor
    # 何ステップ前に書いたか。書いたステップで 0
    age: int
    text: str
    # 書いたときの会話と、この返答。描画用
    exchange: list[dict]
    # この返答が判定した、前の返答のサブタスクの達成度（yes の確率）。前のサブタスクが
    # ないか、返答に判定がなければ None
    achieved: float | None
    input_tokens: int
    output_tokens: int
    msec: float


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
        self.generator = ChainGenerator(high_level_config, device=device)
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
        # いま持っている返答。エピソードの最初の返答を書くまでは None
        self._reply: HighLevelPolicyOutput | None = None
        # 0 は「いま書く」。エピソードの最初の advance は、そのエピソードの最初のフレームで書く
        self._until_next = 0

    @torch.inference_mode()
    def advance(self) -> HighLevelPolicyOutput:
        """このステップの返答。返答を書く番なら書いて確定する。次の返答を書くまで同じ
        ものを返し、``age`` だけが進む。同じ活性でも、書いたフレームの上と十数ステップ
        後とでは意味が違うので、エンコーダは活性と一緒に ``age`` を読む。"""
        if self._until_next == 0:
            conversation = self.prompt_builder.conversation()
            chain = self.generator.generate(conversation)
            # エピソードの最初の返答には判定すべき前のサブタスクがない
            achieved = (
                self.generator.yes_probability(chain, ACHIEVED_TAG)
                if self._reply is not None
                else None
            )

            # <subtask> 区間のトークンの活性だけを読む。区間の長さによらず、区間方向に
            # 均して1ステップに読む幅にそろえる。区間がなければ 0
            tokenizer = self.generator.processor.tokenizer
            match = SUBTASK_RE.search(tokenizer.decode(chain.tokens, skip_special_tokens=True))
            rows = []
            if match is not None:
                start = 0
                for i in range(len(chain.tokens)):
                    end = len(tokenizer.decode(chain.tokens[: i + 1], skip_special_tokens=True))
                    if start < match.end(1) and end > match.start(1):
                        rows.append(i)
                    start = end
            if len(rows) == 0:
                activations = torch.zeros(
                    (self.tokens_per_step, self.generator.layers_num, self.generator.hidden_size),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
            else:
                span = chain.positions[rows].to(torch.float32).permute(1, 2, 0)
                pooled = F.adaptive_avg_pool1d(span, self.tokens_per_step)
                activations = pooled.permute(2, 0, 1).to(torch.bfloat16)

            self._reply = HighLevelPolicyOutput(
                activations=activations,
                age=0,
                text=chain.text,
                exchange=conversation + [assistant_turn(chain.text)],
                achieved=achieved,
                input_tokens=chain.prompt_tokens,
                output_tokens=len(chain.tokens),
                msec=chain.msec,
            )
            self.prompt_builder.add_reply(chain.text)
            self._until_next = self.steps_per_reply
        self._until_next -= 1
        return dataclasses.replace(self._reply, age=self.steps_per_reply - 1 - self._until_next)

    def judge_current(self) -> float:
        """いま実行中のサブタスクの達成度を、いまのターンで判定する。エピソードが
        終わり、次の返答が来ないサブタスクのため。"""
        assert self._reply is not None, "there is no subtask running before the first reply"
        return self.generator.yes_probability_after(
            self.prompt_builder.conversation(), ACHIEVED_TAG
        )
