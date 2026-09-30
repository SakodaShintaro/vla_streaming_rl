# SPDX-License-Identifier: MIT
"""書きかけのチェーンを毎ステップ少しずつ書き進め、生成の費用を全ステップに
償却するチェーン。

チェーンを確定して行動を読む周期は ``CoTBatch`` と同じ ``steps_per_chain`` で、
確定の次のステップから新しいチェーンを書き始め、以降の毎ステップ
``write_tokens_per_step`` トークンずつ書き進める。``steps_per_image`` ステップ
ごとの書き進めでは、確定済みの会話の境界までキャッシュを巻き戻し、最新の
フレームのターン（履歴には確定させない）、生成プロンプトの末尾、書きかけの
返答を積み直してから続きを書く。それ以外のステップはキャッシュの続きを書く
だけで、新しいフレームは読まない。確定のステップでは最新のフレームの下で
書きかけを積み直し、残りを書き切る。

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
        steps_per_chain = high_level_config.cot_steps_per_chain
        steps_per_image = high_level_config.cot_steps_per_image
        write_tokens_per_step = high_level_config.cot_write_tokens_per_step
        assert steps_per_image >= 1, steps_per_image
        assert steps_per_chain % steps_per_image == 0, (
            f"cot_steps_per_image {steps_per_image} must divide "
            f"cot_steps_per_chain {steps_per_chain}"
        )
        assert write_tokens_per_step >= 1, write_tokens_per_step
        self.steps_per_image = steps_per_image
        self.write_tokens_per_step = write_tokens_per_step
        super().__init__(high_level_config, prompt_builder, device)

    def reset(self) -> None:
        super().reset()
        self._draft: list[int] = []

    @torch.inference_mode()
    def advance(self) -> torch.Tensor:
        """This environment step's activations: the chain is committed every
        ``steps_per_chain`` steps and advanced on every step in between, on the
        latest frame every ``steps_per_image`` steps.

        Returns:
            (tokens_per_step, layers_num, hidden_size) bfloat16.
        """
        if self._until_next == 0:
            self._write_chain(self._draft)
            self._draft = []
            self._until_next = self.steps_per_chain
        elif (self.steps_per_chain - self._until_next - 1) % self.steps_per_image == 0:
            self._last_conversation = self.prompt_builder.conversation()
            self._take_draft(
                self.generator.generate(
                    self._last_conversation, self._draft, self.write_tokens_per_step, commit=False
                )
            )
        else:
            self._take_draft(self.generator.extend(self.write_tokens_per_step))
        self._until_next -= 1
        return self._activations

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
