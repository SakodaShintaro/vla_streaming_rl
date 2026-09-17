# SPDX-License-Identifier: MIT
import time

import torch
import torch.nn.functional as F
from transformers import StaticCache

from vla_streaming_rl.agents.prompt import PromptBuilder, assistant_turn

from .vlm_backbone import load_model
from .vlm_inputs import render_conversation


class CoTBatch:
    """A whole chain of thought written every ``steps_per_chain`` environment
    steps and held in between.

    A chain is written as one prefill of the conversation followed by one
    decode step per token, each reading the activation at every depth behind
    the token it samples. With ``use_cuda_graph`` the decode step is
    ``torch.compile``d in ``reduce-overhead`` mode, which replays it as a CUDA
    graph; the prefill is never compiled. Every buffer the step reads or writes
    keeps its address for that: the cache is a ``StaticCache`` allocated once
    for ``prompt_budget + max_len`` tokens and reset per chain, and the step's
    inputs are fixed tensors written in place. ``prompt_budget`` has to cover
    the longest conversation the run sends -- roughly 200 tokens per held
    exchange of a 256x256 frame under its text and reply -- and a prompt beyond
    it fails at the chain that sends it. The first chain of a run pays the
    compilation, under a minute.
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
        assert prompt_budget >= 1, prompt_budget
        self.model, self.processor = load_model(
            model_id, use_lora=False, load_in_4bit=load_in_4bit, device=device
        )
        self.model.eval().requires_grad_(False)
        self.prompt_budget = prompt_budget
        self._cache = StaticCache(config=self.model.config, max_cache_len=prompt_budget + max_len)
        self._token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self._cache_position = torch.zeros(1, dtype=torch.long, device=device)
        self._position_ids = torch.zeros(3, 1, 1, dtype=torch.long, device=device)
        self._step = (
            torch.compile(self._forward_step, mode="reduce-overhead", fullgraph=False)
            if use_cuda_graph
            else self._forward_step
        )
        self.eos_token_id = self.processor.tokenizer.eos_token_id
        self.tokens_per_step = tokens_per_step
        self.max_len = max_len
        # Decoded the way the zero-shot controller decodes: it reads the same
        # conversation through the same model, so a chain written any other way
        # would not be the baseline's reasoning measured under RL.
        assert temperature >= 0.0, temperature
        self.temperature = temperature
        self.steps_per_chain = steps_per_chain
        # The conversation is the agent's; a chain reads it on the steps it
        # writes and puts what it wrote back as that turn's reply.
        self.prompt_builder = prompt_builder
        self.device = device
        text_config = self.model.config.text_config
        self.hidden_size = text_config.hidden_size
        # The embedding plus every layer's output, matching `CoTStream`.
        self.layers_num = text_config.num_hidden_layers + 1
        self.reset()

    def reset(self) -> None:
        """Drop the chain. The next advance writes a new one on the frame it is
        given; the conversation it is written into is the builder's to reset."""
        self._tokens = []
        self._last_conversation = []
        # What the last chain cost. Kept between writes, since the steps that
        # hold one are not the steps that paid for it.
        self._input_tokens = 0
        self._msec = 0.0
        self._activations = torch.zeros(
            (self.tokens_per_step, self.layers_num, self.hidden_size),
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
        start = time.perf_counter()
        # Thinking off: with the <think> block left open the model spends the
        # chain reasoning about the request rather than about the scene.
        self._last_conversation = self.prompt_builder.conversation()
        text, images = render_conversation(
            self.processor, self._last_conversation, enable_thinking=False
        )
        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            do_rescale=False,
        ).to(self.device)
        prompt_len = int(inputs["input_ids"].shape[1])
        assert prompt_len <= self.prompt_budget, (
            f"prompt of {prompt_len} tokens exceeds cot_prompt_budget={self.prompt_budget}; raise it"
        )

        self._cache.reset()
        outputs = self.model(
            **inputs, past_key_values=self._cache, use_cache=True, output_hidden_states=True
        )
        positions = [self._last_position(outputs.hidden_states)]
        self._tokens = [self._sample(outputs.logits[0, -1])]
        position = prompt_len
        rope_deltas = self.model.model.rope_deltas
        while self._tokens[-1] != self.eos_token_id and len(self._tokens) < self.max_len:
            self._token.fill_(self._tokens[-1])
            self._cache_position.fill_(position)
            self._position_ids.copy_(
                (self._cache_position.view(1, 1, -1) + rope_deltas.unsqueeze(0)).expand(3, -1, -1)
            )
            torch.compiler.cudagraph_mark_step_begin()
            outputs = self._step(self._token, self._cache_position, self._position_ids)
            positions.append(self._last_position(outputs.hidden_states))
            self._tokens.append(self._sample(outputs.logits[0, -1]))
            position += 1
        self._activations = self._read_activations(torch.stack(positions))
        self._input_tokens = prompt_len
        self._msec = (time.perf_counter() - start) * 1000.0
        self.prompt_builder.add_reply(self.text())

    def _forward_step(
        self, token: torch.Tensor, cache_position: torch.Tensor, position_ids: torch.Tensor
    ):
        """One decode step over the static cache. Positions are handed in rather
        than left to the model, which would build them on the host; rope_deltas
        is the offset the image tokens introduced, left by the prefill."""
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
        hidden_size), copied out of the step's output buffers -- which a
        replayed graph overwrites on its next step."""
        return torch.stack([depth[0, -1] for depth in hidden_states]).to(torch.bfloat16)

    def _sample(self, logits: torch.Tensor) -> int:
        """The token the logits imply, decoded the way the zero-shot controller
        decodes: greedy at temperature 0, sampled otherwise."""
        if self.temperature == 0.0:
            return int(logits.argmax().item())
        probs = torch.softmax(logits.float() / self.temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def _read_activations(self, positions: torch.Tensor) -> torch.Tensor:
        """The whole chain, every depth kept, pooled to one step's read:
        (tokens_per_step, layers_num, hidden_size).

        ``positions`` is (chain_len, layers_num, hidden_size): each position's
        activation is the state that chose its token. A chain stops where the
        model stops it, so the positions are pooled along the chain rather than
        sliced to its tail: the whole chain reaches the policy, and a chain that
        ended early needs no padding to reach the fixed read.
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
            "output_tokens": len(self._tokens),
            "msec": self._msec,
        }

    def text(self) -> str:
        """The chain as written, for logging."""
        return self.processor.tokenizer.decode(self._tokens, skip_special_tokens=True).strip()

    def exchange(self) -> list[dict]:
        """The last write as it happened: the conversation the model was given
        and the chain it wrote back, for the render panel."""
        return self._last_conversation + [assistant_turn(self.text())]
