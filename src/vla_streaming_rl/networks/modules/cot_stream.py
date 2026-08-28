# SPDX-License-Identifier: MIT
"""A frozen VLM kept mid-thought across environment steps.

The chain of thought is the slow loop. It is prefilled from one frame and then
advanced by a fixed budget of tokens per environment step, so a single line of
reasoning spans many control ticks. When it ends -- EOS, or ``max_len`` tokens --
the next advance prefills again from whatever frame is current, starting a fresh
chain on a fresh image.

What leaves this module is not text but the activation feeding the VLM's lm_head
at each generated position: the state the model was in when it chose that token,
which carries far more than the token id does. ``all_layers`` widens that to
every hidden state behind it -- the embedding and each layer's output -- leaving
which depth to read to a weighting the network trains, at 25x the storage. Nothing here trains and nothing
here is differentiable, so downstream the chain is an ordinary observation
stream, stored in the replay buffer next to the image.

Not an ``nn.Module`` on purpose: registering it would put a frozen 0.8B model
into the network's ``parameters()`` and its ``state_dict()``.
"""

import torch
from torchvision.transforms.functional import to_pil_image
from transformers import StaticCache

from .vlm_backbone import load_model


class CoTStream:
    # The cache is allocated once at a fixed length, so it has to cover the
    # longest prompt any run will prefill: the image tokens, the chat template,
    # the instruction and the environment's task text. At 6 attention layers and
    # 2 key-value heads the buffer costs single-digit megabytes, so this is set
    # far above what any environment sends rather than tuned.
    PROMPT_BUDGET = 512

    # The frame the throwaway chain used for recording is never looked at, only
    # resized by the processor, so its size is arbitrary.
    CAPTURE_FRAME_SIDE = 64

    # Written out in full rather than left to the model's own thinking mode,
    # which spends its budget restating the request ("The user wants me to...")
    # instead of the scene.
    INSTRUCTION = (
        "This is what an agent sees right now. Keep up a running commentary on "
        "its situation, in short plain sentences.\n"
        "Say: where the agent is relative to whatever matters around it, which "
        "way it is heading, what is about to go wrong, and what it should be "
        "trying to do next.\n"
        "Write the commentary only. No preamble, no restating this request, no "
        "headings or lists, no numbers or control values."
    )

    def __init__(
        self,
        model_id: str,
        tokens_per_step: int,
        max_len: int,
        temperature: float,
        all_layers: bool,
        use_cuda_graph: bool,
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
        self.device = device
        text_config = self.model.config.text_config
        self.hidden_size = text_config.hidden_size
        self.all_layers = all_layers
        # The embedding plus every layer's output, or only the last one.
        self.layers_num = text_config.num_hidden_layers + 1 if all_layers else 1
        self.eos_token_id = self.processor.tokenizer.eos_token_id
        # One cache for the whole run. A new chain resets it in place rather than
        # replacing it, so its buffers keep the addresses a graph records.
        self._cache_len = self.PROMPT_BUDGET + max_len
        self._cache = StaticCache(config=self.model.config, max_cache_len=self._cache_len)
        # What a recorded step reads. Their contents change every step; their
        # addresses must not, which is the whole point of recording one.
        self._graph_token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self._graph_cache_position = torch.zeros(1, dtype=torch.long, device=device)
        self._graph_position_ids = torch.zeros(3, 1, 1, dtype=torch.long, device=device)
        self._graph = None
        self.reset()
        if use_cuda_graph:
            self._capture_decode()

    def reset(self) -> None:
        """Drop the chain. The next advance prefills from the frame it is given."""
        self._needs_prefill = True
        self._hidden = None
        self._next_token = None
        self._tokens = []
        # Kept here rather than read back from the cache: a recorded step advances
        # the cache inside the graph, where a host-side counter is the only thing
        # that stays in step with it.
        self._position = 0
        self._cache.reset()

    @torch.inference_mode()
    def advance(self, image: torch.Tensor, task_prompt: str) -> torch.Tensor:
        """The ``tokens_per_step`` activations this environment step issues.

        Args:
            image: (C, H, W) float tensor in [0, 1]; read only when the chain
                restarts, which is what makes the chain the slow loop.
            task_prompt: the env's language instruction, empty where the env sets
                none (or where ``use_prompt`` is off), leaving just the standing
                instruction.

        Returns:
            (tokens_per_step, layers_num, hidden_size) bfloat16.
        """
        activations = []
        while len(activations) < self.tokens_per_step:
            if self._needs_prefill:
                self._prefill(image, task_prompt)
            activations.append(self._hidden)
            self._decode()
        return torch.stack(activations)

    def _prefill(self, image: torch.Tensor, task_prompt: str) -> None:
        prompt = f"{task_prompt}\n{self.INSTRUCTION}".strip()
        content = [{"type": "image"}, {"type": "text", "text": prompt}]
        # Thinking off: with the <think> block left open the model spends the
        # chain reasoning about the request rather than about the scene.
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        picture = to_pil_image(image.detach().float().clamp(0.0, 1.0).cpu())
        inputs = self.processor(text=[text], images=[picture], return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]
        assert prompt_len + self.max_len <= self._cache_len, (
            f"prompt of {prompt_len} tokens plus a {self.max_len}-token chain exceeds the "
            f"cache length {self._cache_len}; raise CoTStream.PROMPT_BUDGET"
        )
        self._tokens = []
        self._cache.reset()
        self._needs_prefill = False
        outputs = self.model(
            **inputs,
            past_key_values=self._cache,
            use_cache=True,
            output_hidden_states=True,
        )
        self._position = prompt_len
        self._consume(self._depths(outputs.hidden_states), outputs.logits[0, -1])

    @torch.no_grad()
    def _capture_decode(self) -> None:
        """Record one decode step, so the steady state replays as a single call
        instead of the ~2500 kernel launches issuing it costs.

        Recording runs the step, which writes the cache and advances the
        linear-attention state, so it happens on a throwaway chain over a blank
        frame; the reset at the end leaves the real chain to start clean.

        ``no_grad`` rather than ``inference_mode``: this is where the cache
        allocates, and a tensor born in inference mode cannot be written from
        outside it, which is what recording then does.
        """
        self._prefill(torch.zeros(3, self.CAPTURE_FRAME_SIDE, self.CAPTURE_FRAME_SIDE), "")
        self._write_step_inputs(self._position)

        # Capture must follow a few runs on a side stream, which is also what
        # settles the workspaces the kernels allocate on first use.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._forward_step()
        torch.cuda.current_stream().wait_stream(stream)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            outputs = self._forward_step()
        # Every hidden state the recorded step writes is kept, so a replay can
        # read whichever depths are wanted; they live in the graph's own pool.
        self._graph_hidden = outputs.hidden_states
        self._graph_logits = outputs.logits
        self.reset()

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

    def _decode(self) -> None:
        position = self._position
        self._write_step_inputs(position)
        if self._graph is None:
            outputs = self._forward_step()
            hidden_states, logits = outputs.hidden_states, outputs.logits
        else:
            self._graph.replay()
            hidden_states, logits = self._graph_hidden, self._graph_logits
        self._position = position + 1
        self._consume(self._depths(hidden_states), logits[0, -1])

    def _depths(self, hidden_states) -> torch.Tensor:
        """This position's activations at the depths the stream keeps:
        (layers_num, hidden_size)."""
        if not self.all_layers:
            return hidden_states[-1][0, -1].unsqueeze(0)
        return torch.stack([state[0, -1] for state in hidden_states])

    def _consume(self, hidden: torch.Tensor, logits: torch.Tensor) -> None:
        """Take the position's lm_head input, sample the token it implies, and
        decide whether the chain lives on."""
        # Copied, not referenced: after a replay this is the graph's own output
        # buffer, which the next replay overwrites while the step's earlier
        # activations are still being collected.
        self._hidden = hidden.to(torch.bfloat16).clone()
        probs = torch.softmax(logits.float() / self.temperature, dim=-1)
        self._next_token = torch.multinomial(probs, 1).view(1, 1)
        self._tokens.append(self._next_token.item())
        ended = self._tokens[-1] == self.eos_token_id or len(self._tokens) >= self.max_len
        if ended:
            self._needs_prefill = True

    def text(self) -> str:
        """The chain as written so far, for logging."""
        return self.processor.tokenizer.decode(self._tokens, skip_special_tokens=True).strip()
