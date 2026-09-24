# SPDX-License-Identifier: MIT
import torch
import torch.nn.functional as F

from vla_streaming_rl.agents.prompt import SUBTASK_RE, PromptBuilder, assistant_turn

from .chain_generator import Chain, ChainGenerator


class CoTBatch:
    """A whole chain of thought written every ``steps_per_chain`` environment
    steps and held in between.

    The chain is written by :class:`ChainGenerator`, the generator the zero-shot
    controller's local backend writes its replies with, so a chain is the
    baseline's reasoning measured under RL by construction. What is added here
    is the cadence, the pooling of the chain's ``<subtask>`` activations to one
    step's read, and the reply going back into the builder's conversation.
    """

    def __init__(
        self,
        model_id: str,
        load_in_4bit: bool,
        tokens_per_step: int,
        max_len: int,
        temperature: float,
        steps_per_chain: int,
        use_cuda_graph: bool,
        prompt_budget: int,
        prompt_builder: PromptBuilder,
        device: torch.device,
    ) -> None:
        assert tokens_per_step >= 1, f"tokens_per_step must be positive; got {tokens_per_step}"
        assert steps_per_chain >= 1, f"steps_per_chain must be positive; got {steps_per_chain}"
        assert max_len >= tokens_per_step, (
            f"max_len {max_len} below {tokens_per_step}: the pool would stretch a chain "
            "shorter than one step's read"
        )
        # Thinking off: with the <think> block left open the model spends the
        # chain reasoning about the request rather than about the scene.
        self.generator = ChainGenerator(
            model_id=model_id,
            load_in_4bit=load_in_4bit,
            max_len=max_len,
            temperature=temperature,
            enable_thinking=False,
            use_cuda_graph=use_cuda_graph,
            prompt_budget=prompt_budget,
            device=device,
        )
        self.tokens_per_step = tokens_per_step
        self.steps_per_chain = steps_per_chain
        # The conversation is the agent's; a chain reads it on the steps it
        # writes and puts what it wrote back as that turn's reply.
        self.prompt_builder = prompt_builder
        self.device = device
        self.reset()

    def reset(self) -> None:
        """Drop the chain. The next advance writes a new one on the frame it is
        given; the conversation it is written into is the builder's to reset."""
        self._text = ""
        self._output_tokens = 0
        self._last_conversation = []
        # What the last chain cost. Kept between writes, since the steps that
        # hold one are not the steps that paid for it.
        self._input_tokens = 0
        self._msec = 0.0
        self._activations = torch.zeros(
            (self.tokens_per_step, self.generator.layers_num, self.generator.hidden_size),
            dtype=torch.bfloat16,
            device=self.device,
        )
        # Zero means "write one now", so the first advance of an episode always
        # reasons about that episode's own first frame.
        self._until_next = 0

    def age(self) -> int:
        """How many environment steps ago the chain now being read was written.

        0 on the step that wrote it, up to ``steps_per_chain - 1`` on the last
        step that holds it. What the encoder needs alongside the activations:
        the same chain means something different on the frame it was written
        about than it does fifteen steps later, and nothing else in the
        observation says which of the two this is -- `episode_step` carries it
        only modulo a period the encoder cannot take.
        """
        return self.steps_per_chain - 1 - self._until_next

    @torch.inference_mode()
    def advance(self) -> torch.Tensor:
        """This environment step's activations, writing a fresh chain when due.

        The builder's conversation is read on the steps that write a chain and
        not at all in between, which is what makes the chain the slow loop.

        Returns:
            (tokens_per_step, layers_num, hidden_size) bfloat16. The same tensor
            on every step until the next chain is written.
        """
        if self._until_next == 0:
            self._write_chain()
            self._until_next = self.steps_per_chain
        self._until_next -= 1
        return self._activations

    def _write_chain(self) -> None:
        self._last_conversation = self.prompt_builder.conversation()
        chain = self.generator.generate(self._last_conversation)
        subtask_positions = self._subtask_positions(chain)
        if subtask_positions.shape[0] == 0:
            self._activations = torch.zeros_like(self._activations)
        else:
            self._activations = self._read_activations(subtask_positions)
        self._text = chain.text
        self._output_tokens = len(chain.tokens)
        self._input_tokens = chain.prompt_tokens
        self._msec = chain.msec
        self.prompt_builder.add_reply(chain.text)

    def _subtask_positions(self, chain: Chain) -> torch.Tensor:
        """The rows of ``chain.positions`` whose tokens fall inside the reply's
        ``<subtask>`` section, empty when the reply wrote none: the subtask is
        the one part of a reply addressed to the policy below, so it is all the
        policy reads."""
        tokenizer = self.generator.processor.tokenizer
        text = tokenizer.decode(chain.tokens, skip_special_tokens=True)
        match = SUBTASK_RE.search(text)
        if match is None:
            return chain.positions[:0]
        rows = []
        start = 0
        for i in range(len(chain.tokens)):
            end = len(tokenizer.decode(chain.tokens[: i + 1], skip_special_tokens=True))
            if start < match.end(1) and end > match.start(1):
                rows.append(i)
            start = end
        if len(rows) == 0:
            return chain.positions[:0]
        return chain.positions[rows]

    def _read_activations(self, positions: torch.Tensor) -> torch.Tensor:
        """``positions`` (a span of the chain, (span_len, layers_num,
        hidden_size), each position's activation the state that chose its
        token), every depth kept, pooled to one step's read:
        (tokens_per_step, layers_num, hidden_size). Pooled along the span
        rather than sliced to a fixed length, so a span of any length reaches
        the fixed read without padding.
        """
        pooled = F.adaptive_avg_pool1d(
            positions.to(torch.float32).permute(1, 2, 0), self.tokens_per_step
        )
        return pooled.permute(2, 0, 1).to(torch.bfloat16)

    def stats(self) -> dict:
        """What the last chain cost: the tokens it was given, the tokens it
        wrote, and the wall time the write took."""
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "msec": self._msec,
        }

    def text(self) -> str:
        """The chain as written, for logging."""
        return self._text

    def exchange(self) -> list[dict]:
        """The last write as it happened: the conversation the model was given
        and the chain it wrote back, for the render panel."""
        return self._last_conversation + [assistant_turn(self.text())]
