# SPDX-License-Identifier: MIT
"""書きかけのチェーンを毎ステップ少しずつ書き進め、書き終えたところで確定する
チェーン。

エピソードの最初のステップでは ``CoTBatch`` と同じくチェーンを一度に書いて
確定する。以降は確定の次のステップから新しいチェーンを書き始め、毎ステップ
``write_tokens_per_step`` トークンずつ書き進める。``steps_per_image`` ステップ
ごとの書き進めでは、確定済みの会話の境界までキャッシュを巻き戻し、最新の
フレームのターン（まだ確定させない）、生成プロンプト、書きかけの返答を積み
直してから続きを書く。それ以外のステップはキャッシュの続きを書くだけで、新しい
フレームは読まない。

返答を書き終えたステップで確定し、行動を読ませる。そのステップで最新の
フレームを読んでいれば、キャッシュの中身をそのまま確定させる。読んでいなければ
最新のフレームの下で積み直してから確定する。``steps_per_chain`` ステップ経っても
書き終わらなければ、最新のフレームの下で残りを書き切って確定する。

書きかけの ``<subtask>`` 区間の活性は書き進めるたびに読み直されるので、
低レベル方策が読むサブタスクの表現はチェーンの確定を待たずに新しくなる。
"""

import torch
from omegaconf import DictConfig

from vla_streaming_rl.agents.prompt import PromptBuilder

from .chain_generator import Chain
from .cot_batch import CoTBatch


class CoTStream(CoTBatch):
    def __init__(
        self,
        high_level_config: DictConfig,
        prompt_builder: PromptBuilder,
        device: torch.device,
    ) -> None:
        steps_per_image = high_level_config.cot_steps_per_image
        write_tokens_per_step = high_level_config.cot_write_tokens_per_step
        assert steps_per_image >= 1, steps_per_image
        assert write_tokens_per_step >= 1, write_tokens_per_step
        self.steps_per_image = steps_per_image
        self.write_tokens_per_step = write_tokens_per_step
        super().__init__(high_level_config, prompt_builder, device)

    def reset(self) -> None:
        super().reset()
        self._draft: list[int] = []
        # 最後に確定してからのステップ数。None はエピソードの最初でまだ確定していないこと
        self._since_commit: int | None = None

    def age(self) -> int:
        """How many environment steps ago the chain now being read was committed:
        0 on the step that committed it, below ``steps_per_chain`` always."""
        assert self._since_commit is not None, "age() is read after advance()"
        return self._since_commit

    @torch.inference_mode()
    def advance(self) -> torch.Tensor:
        """This environment step's activations: the chain is advanced every step,
        on the latest frame every ``steps_per_image`` steps, and committed on the
        step it is finished.

        Returns:
            (tokens_per_step, layers_num, hidden_size) bfloat16.
        """
        if self._since_commit is None:
            # エピソードの最初のフレームでは、そのフレームについて一度に書き切る
            self._commit(self._draft)
            return self._activations
        self._since_commit += 1
        if self._since_commit >= self.steps_per_chain:
            self._commit(self._draft)
            return self._activations
        refresh = (self._since_commit - 1) % self.steps_per_image == 0
        if refresh:
            self._last_conversation = self.prompt_builder.conversation()
            chain = self.generator.generate(
                self._last_conversation, self._draft, self.write_tokens_per_step, commit=False
            )
        else:
            chain = self.generator.extend(self.write_tokens_per_step)
        self._take_draft(chain)
        if chain.finished or len(chain.tokens) >= self.generator.max_len:
            if refresh:
                # このステップで最新のフレームを読んで書き終えたので、そのまま確定する
                self.generator.commit_draft(self._last_conversation, chain.text)
                self._take_chain(chain)
                self._draft = []
                self._since_commit = 0
            else:
                self._commit(self._draft)
        return self._activations

    def _commit(self, prefix: list[int]) -> None:
        """最新のフレームの下で書きかけを積み直し、残りを書き切って確定する。"""
        self._write_chain(prefix)
        self._draft = []
        self._since_commit = 0

    def _take_draft(self, chain: Chain) -> None:
        """書き進めた書きかけを持つ。会話には確定させない。書きかけにサブタスクが
        まだなければ、確定済みのチェーンの活性を持ち続ける。"""
        self._draft = chain.tokens
        subtask_positions = self._subtask_positions(chain)
        if subtask_positions.shape[0] > 0:
            self._activations = self._read_activations(subtask_positions)
        self._output_tokens = len(chain.tokens)
        self._input_tokens = chain.prompt_tokens
        self._msec = chain.msec
