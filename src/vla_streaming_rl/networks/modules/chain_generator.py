# SPDX-License-Identifier: MIT
import time
from dataclasses import dataclass

import torch
from transformers import StaticCache

from .vlm_backbone import load_model
from .vlm_inputs import encode_conversation


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
    """The one way a local VLM writes a reply in this repo, whoever reads it:
    the zero-shot controller reads the text, the chain of thought reads the
    activations as well, so the two decode alike by construction.

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
        max_len: int,
        temperature: float,
        enable_thinking: bool,
        use_cuda_graph: bool,
        prompt_budget: int,
        device: torch.device,
    ) -> None:
        assert max_len >= 1, max_len
        assert prompt_budget >= 1, prompt_budget
        assert temperature >= 0.0, temperature
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
        self.max_len = max_len
        self.temperature = temperature
        self.top_k = self.model.generation_config.top_k
        self.top_p = self.model.generation_config.top_p
        self.enable_thinking = enable_thinking
        self.device = device
        text_config = self.model.config.text_config
        self.hidden_size = text_config.hidden_size
        # The embedding plus every layer's output, matching `CoTStream`.
        self.layers_num = text_config.num_hidden_layers + 1

    @torch.inference_mode()
    def generate(self, conversation: list[dict]) -> Chain:
        """Write a reply to ``conversation``."""
        start = time.perf_counter()
        inputs = encode_conversation(
            self.processor, conversation, self.enable_thinking, self.device
        )
        prompt_len = int(inputs["input_ids"].shape[1])
        assert prompt_len <= self.prompt_budget, (
            f"prompt of {prompt_len} tokens exceeds cot_prompt_budget={self.prompt_budget}; raise it"
        )

        self._cache.reset()
        outputs = self.model(
            **inputs, past_key_values=self._cache, use_cache=True, output_hidden_states=True
        )
        positions = [self._last_position(outputs.hidden_states)]
        tokens = [self._sample(outputs.logits[0, -1])]
        position = prompt_len
        rope_deltas = self.model.model.rope_deltas
        while tokens[-1] != self.eos_token_id and len(tokens) < self.max_len:
            self._token.fill_(tokens[-1])
            self._cache_position.fill_(position)
            self._position_ids.copy_(
                (self._cache_position.view(1, 1, -1) + rope_deltas.unsqueeze(0)).expand(3, -1, -1)
            )
            torch.compiler.cudagraph_mark_step_begin()
            outputs = self._step(self._token, self._cache_position, self._position_ids)
            positions.append(self._last_position(outputs.hidden_states))
            tokens.append(self._sample(outputs.logits[0, -1]))
            position += 1
        return Chain(
            tokens=tokens,
            text=self.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
            positions=torch.stack(positions),
            prompt_tokens=prompt_len,
            msec=(time.perf_counter() - start) * 1000.0,
            finished=tokens[-1] == self.eos_token_id,
        )

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
