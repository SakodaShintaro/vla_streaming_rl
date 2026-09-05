# SPDX-License-Identifier: MIT
"""A frozen VLM kept mid-thought across environment steps.

The chain of thought is the slow loop. It is prefilled from one frame and then
advanced by a fixed budget of tokens per environment step, so a single line of
reasoning spans many control ticks. When it ends -- EOS, or ``max_len`` tokens --
the next advance prefills again from whatever frame is current, starting a fresh
chain on a fresh image. That prefill carries the chain that just ended as the
model's own turn, so the new one picks up where the last left off instead of
opening on the scene from scratch.

What leaves this module is not text but the activation feeding the VLM's lm_head
at each generated position: the state the model was in when it chose that token,
which carries far more than the token id does. Every hidden state behind it is
kept -- the embedding and each layer's output -- leaving which depth to read to
a weighting the network trains. Nothing here trains and nothing here is
differentiable, so downstream the chain is an ordinary observation stream,
stored in the replay buffer next to the image.

Not an ``nn.Module`` on purpose: registering it would put a frozen 0.8B model
into the network's ``parameters()`` and its ``state_dict()``.
"""

import time

import torch
from transformers import StaticCache

from vla_streaming_rl.agents.prompt import PromptBuilder

from .vlm_backbone import load_model


class CoTStream:
    # The cache is allocated once at a fixed length, so it has to cover the
    # longest prompt any run will prefill: the image tokens, the chat template,
    # the instruction and the environment's task text. At 6 attention layers and
    # 2 key-value heads the buffer costs single-digit megabytes, so this is set
    # far above what any environment sends rather than tuned.
    PROMPT_BUDGET = 1024

    # The frame the throwaway chain used for recording is never looked at, only
    # resized by the processor, so its size is arbitrary.
    CAPTURE_FRAME_SIDE = 64

    def __init__(
        self,
        model_id: str,
        tokens_per_step: int,
        max_len: int,
        temperature: float,
        use_cuda_graph: bool,
        prompt_builder: PromptBuilder,
        device: torch.device,
    ) -> None:
        assert tokens_per_step >= 1, f"tokens_per_step must be positive; got {tokens_per_step}"
        assert max_len >= tokens_per_step, (
            f"max_len {max_len} below the per-step budget {tokens_per_step}: "
            "every step would restart the chain"
        )
        self.model, self.processor = load_model(model_id, use_lora=False, device=device)
        self.model.eval().requires_grad_(False)
        self.tokens_per_step = tokens_per_step
        self.max_len = max_len
        self.temperature = temperature
        # The conversation is the agent's; a chain reads it on the steps it
        # restarts and writes its own turn back when it ends.
        self.prompt_builder = prompt_builder
        self.device = device
        text_config = self.model.config.text_config
        self.hidden_size = text_config.hidden_size
        # The embedding plus every layer's output.
        self.layers_num = text_config.num_hidden_layers + 1
        self.eos_token_id = self.processor.tokenizer.eos_token_id
        # One cache for the whole run. A new chain resets it in place rather than
        # replacing it, so its buffers keep the addresses a graph records.
        # The chain that ended last is quoted back into the prompt, so the prefill
        # has to fit one more chain's worth of tokens on top of the standing one.
        self._cache_len = self.PROMPT_BUDGET + max_len + max_len
        self._cache = StaticCache(config=self.model.config, max_cache_len=self._cache_len)
        # What a recorded step reads. Their contents change every step; their
        # addresses must not, which is the whole point of recording one.
        self._graph_token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self._graph_cache_position = torch.zeros(1, dtype=torch.long, device=device)
        self._graph_position_ids = torch.zeros(3, 1, 1, dtype=torch.long, device=device)
        self._graph = None
        self.reset()
        if use_cuda_graph:
            # Record one decode step, so the steady state replays as a single
            # call instead of the ~2500 kernel launches issuing it costs.
            # Recording runs the step, which writes the cache and advances the
            # linear-attention state, so it happens on a throwaway chain over a
            # blank frame; the reset at the end leaves the real chain to start
            # clean.
            #
            # ``no_grad`` rather than ``inference_mode``: this is where the
            # cache allocates, and a tensor born in inference mode cannot be
            # written from outside it, which is what recording then does.
            with torch.no_grad():
                blank_frame = torch.zeros(3, self.CAPTURE_FRAME_SIDE, self.CAPTURE_FRAME_SIDE)
                self._prefill(
                    [
                        {"role": "system", "content": [{"type": "text", "text": ""}]},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": blank_frame},
                                {"type": "text", "text": ""},
                            ],
                        },
                    ]
                )
                self._write_step_inputs(self._position)

                # Capture must follow a few runs on a side stream, which is also
                # what settles the workspaces the kernels allocate on first use.
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self._forward_step()
                torch.cuda.current_stream().wait_stream(stream)

                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph):
                    outputs = self._forward_step()
                # Every hidden state the recorded step writes is kept, so a
                # replay can read whichever depths are wanted; they live in the
                # graph's own pool.
                self._graph_hidden = outputs.hidden_states
                self._graph_logits = outputs.logits
                # The throwaway chain may have ended and answered the blank turn
                # it opened on, so the conversation is reset with it.
                self.reset()
                self.prompt_builder.reset()

    def reset(self) -> None:
        """Drop the chain. The next advance prefills from the frame it is given."""
        self._needs_prefill = True
        self._hidden = None
        self._next_token = None
        self._tokens = []
        # The prompt the chain was prefilled on, and what this step's tokens
        # cost. Reported to the render panel, not used by the chain itself.
        self._input_tokens = 0
        self._msec = 0.0
        # Kept here rather than read back from the cache: a recorded step advances
        # the cache inside the graph, where a host-side counter is the only thing
        # that stays in step with it.
        self._position = 0
        self._cache.reset()

    @torch.inference_mode()
    def advance(self) -> torch.Tensor:
        """The ``tokens_per_step`` activations this environment step issues.

        The builder's conversation is read only where a chain restarts, and only
        its tail -- the standing task, the chain that ended last, and the turn
        being opened on -- because a prefill has to fit ``PROMPT_BUDGET``. That
        the frame is read there and nowhere else is what makes the chain the
        slow loop.

        Returns:
            (tokens_per_step, layers_num, hidden_size) bfloat16.
        """
        start = time.perf_counter()
        activations = []
        while len(activations) < self.tokens_per_step:
            if self._needs_prefill:
                self._prefill(self.prompt_builder.conversation())
            activations.append(self._hidden)
            position = self._position
            self._write_step_inputs(position)
            if self._graph is None:
                outputs = self._forward_step()
                hidden_states, logits = outputs.hidden_states, outputs.logits
            else:
                self._graph.replay()
                hidden_states, logits = self._graph_hidden, self._graph_logits
            self._position = position + 1
            self._consume(hidden_states, logits[0, -1])
        self._msec = (time.perf_counter() - start) * 1000.0
        return torch.stack(activations)

    def _prefill(self, conversation: list[dict]) -> None:
        # The standing task, the chain that ended last as the model's own turn,
        # and the turn being opened on. Everything before those is dropped rather
        # than sent: a prefill has to fit ``PROMPT_BUDGET`` however long the
        # conversation has grown, and the only frame it carries is the current.
        turn = conversation[-1]
        image = turn["content"][0]["image"]
        replies = [reply for reply in conversation if reply["role"] == "assistant"]
        current = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": turn["content"][1]["text"]},
            ],
        }
        # Thinking off: with the <think> block left open the model spends the
        # chain reasoning about the request rather than about the scene.
        text = self.processor.apply_chat_template(
            [conversation[0]] + replies[-1:] + [current],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.processor(
            text=[text],
            images=[image.detach().float().clamp(0.0, 1.0)],
            return_tensors="pt",
            do_rescale=False,
        ).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]
        assert prompt_len + self.max_len <= self._cache_len, (
            f"prompt of {prompt_len} tokens plus a {self.max_len}-token chain exceeds the "
            f"cache length {self._cache_len}; raise CoTStream.PROMPT_BUDGET"
        )
        self._tokens = []
        self._input_tokens = int(prompt_len)
        self._cache.reset()
        self._needs_prefill = False
        outputs = self.model(
            **inputs,
            past_key_values=self._cache,
            use_cache=True,
            output_hidden_states=True,
        )
        self._position = prompt_len
        self._consume(outputs.hidden_states, outputs.logits[0, -1])

    def _write_step_inputs(self, position: int) -> None:
        """Fill the buffers the step reads.

        Positions are handed in rather than left to the model, which builds them
        on the host and copies them across -- a transfer capture forbids.
        rope_deltas is the offset the image tokens introduced, left by the
        prefill; it is read here and copied, so a recorded step never sees it.
        """
        self._graph_token.copy_(self._next_token)
        self._graph_cache_position.fill_(position)
        rope_deltas = self.model.model.rope_deltas
        self._graph_position_ids.copy_(
            (self._graph_cache_position.view(1, 1, -1) + rope_deltas.unsqueeze(0)).expand(3, -1, -1)
        )

    def _forward_step(self):
        return self.model(
            input_ids=self._graph_token,
            past_key_values=self._cache,
            use_cache=True,
            output_hidden_states=True,
            cache_position=self._graph_cache_position,
            position_ids=self._graph_position_ids,
        )

    def _consume(self, hidden_states, logits: torch.Tensor) -> None:
        """Keep the position's activation at every depth behind it --
        (layers_num, hidden_size) -- sample the token it implies, and decide
        whether the chain lives on.

        Stacked into a tensor of this module's own: after a replay these are the
        graph's own output buffers, which the next replay overwrites while the
        step's earlier activations are still being collected.
        """
        self._hidden = torch.stack([state[0, -1] for state in hidden_states]).to(torch.bfloat16)
        probs = torch.softmax(logits.float() / self.temperature, dim=-1)
        self._next_token = torch.multinomial(probs, 1).view(1, 1)
        self._tokens.append(self._next_token.item())
        ended = self._tokens[-1] == self.eos_token_id or len(self._tokens) >= self.max_len
        if ended:
            self.prompt_builder.add_reply(self.text())
            self._needs_prefill = True

    def stats(self) -> dict:
        """What the chain costs: the prompt it was prefilled on, the tokens it
        has written since, and the wall time this step's tokens took."""
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": len(self._tokens),
            "msec": self._msec,
        }

    def text(self) -> str:
        """The chain as written so far, for logging."""
        return self.processor.tokenizer.decode(self._tokens, skip_special_tokens=True).strip()
