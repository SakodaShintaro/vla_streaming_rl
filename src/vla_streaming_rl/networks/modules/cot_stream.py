# SPDX-License-Identifier: MIT
"""A frozen VLM kept mid-thought across environment steps.

The chain of thought is one unbroken line of reasoning per episode. The standing
prompt is prefilled once, and every environment step appends that step's frame
followed by a ``plan:`` heading, then draws a fixed budget of tokens. Nothing is
ever re-prefilled: the KV cache and the linear-attention recurrent state carry
straight through, so what the model writes at step t is conditioned on every
frame and every word since the episode began.

The context is bounded without ever restarting. Of the model's 24 layers, 18 are
linear attention whose state is a fixed-size recurrent tensor and cannot
overflow; only the 6 full-attention layers hold a growing KV. Those are kept to
``CACHE_LEN`` slots by pinning the prompt in the first ``PREFIX_SLOTS`` -- the
attention sink, without which the model degenerates once the opening tokens fall
out of view -- and sliding everything after it, dropping ``COMPACT_BLOCK``
tokens at a time. Keys are cached after their rotary embedding is applied, so
attention scores depend only on the distance between two positions and dropping
old slots leaves the remaining geometry intact.

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

import torch
from transformers import StaticCache

from .vlm_backbone import load_model


class CoTStream:
    # Slots the six full-attention layers hold. Measured: decoding is no slower
    # at 8192 than at 2048, and 72% slower at 24576, so this is the whole budget
    # that comes for free.
    CACHE_LEN = 8192

    # Reserved for the standing prompt, which is written once and never slid
    # out. Sized well above what any environment's framing plus instruction
    # tokenizes to; the prefill asserts it fits.
    PREFIX_SLOTS = 512

    # How much of the stream is dropped when the cache fills. The shift is a
    # ~100MB copy over six layers, under 0.2ms, so this trades a negligible cost
    # every ~13 steps against holding a longer window.
    COMPACT_BLOCK = 1024

    # What every step appends before drawing its tokens: this step's frame, then
    # a heading, so a step is one labelled line rather than a slice of a
    # sentence the next frame interrupts. Single heading rather than a rotation:
    # with one label the eight activations mean the same thing at every step,
    # which is what the downstream network reads.
    CHUNK_TEMPLATE = "<|vision_start|><|image_pad|><|vision_end|>\nplan:"

    # The frame the throwaway chain used for recording is never looked at, only
    # resized by the processor, so its size is arbitrary.
    CAPTURE_FRAME_SIDE = 64

    # How much of the tail ``text`` reports. The chain runs for the whole
    # episode, so the whole thing is far too long for a render panel or a log
    # row; consecutive rows overlap enough to stitch back together.
    DISPLAY_TOKENS = 64

    # Written out in full rather than left to the model's own thinking mode,
    # which spends its budget restating the request ("The user wants me to...")
    # instead of the scene. First person throughout: the earlier third-person
    # phrasing produced commentary *about* an agent, and the carry text it
    # needed produced commentary about the commentary itself.
    INSTRUCTION = (
        "You are the animal in this arena, looking through your own eyes.\n"
        "A new frame of your view arrives before each line, followed by "
        "`plan:`. Write that line and nothing else: one short sentence, "
        'starting with "I", saying what you go for next and what in the frame '
        "you just saw says so -- which object or which direction is worth "
        "heading for, and what has to be kept away from.\n"
        "Each line is about the frame that just arrived. Do not repeat a line "
        "you have already written; say what has changed.\n"
        "No preamble, no restating this request, no headings of your own, no "
        "numbers or control values."
    )

    # Sampling these would either end the chain or desynchronize the cache from
    # the images actually fed into it.
    BANNED_TOKENS = (
        "<|im_end|>",
        "<|endoftext|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    )

    def __init__(
        self,
        model_id: str,
        tokens_per_step: int,
        temperature: float,
        use_cuda_graph: bool,
        device: torch.device,
    ) -> None:
        assert tokens_per_step >= 1, f"tokens_per_step must be positive; got {tokens_per_step}"
        self.model, self.processor = load_model(model_id, use_lora=False, device=device)
        self.model.eval().requires_grad_(False)
        self.tokens_per_step = tokens_per_step
        self.temperature = temperature
        self.device = device
        text_config = self.model.config.text_config
        self.hidden_size = text_config.hidden_size
        # The embedding plus every layer's output.
        self.layers_num = text_config.num_hidden_layers + 1
        self._banned = self._banned_token_ids()
        # One cache for the whole run. A new episode resets it in place rather
        # than replacing it, so its buffers keep the addresses a graph records.
        self._cache = StaticCache(config=self.model.config, max_cache_len=self.CACHE_LEN)
        # What a recorded step reads. Their contents change every step; their
        # addresses must not, which is the whole point of recording one.
        self._graph_token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self._graph_cache_position = torch.zeros(1, dtype=torch.long, device=device)
        self._graph_position_ids = torch.zeros(3, 1, 1, dtype=torch.long, device=device)
        self._graph = None
        self._chunk_shape = None
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
                self._prefill("")
                self._append_chunk(torch.zeros(3, self.CAPTURE_FRAME_SIDE, self.CAPTURE_FRAME_SIDE))
                self._write_step_inputs()

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
                self.reset()

    def _banned_token_ids(self) -> torch.Tensor:
        tokenizer = self.processor.tokenizer
        ids = {tokenizer.eos_token_id}
        for token in self.BANNED_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and tokenizer.convert_ids_to_tokens(token_id) == token:
                ids.add(token_id)
        return torch.tensor(sorted(ids), dtype=torch.long, device=self.device)

    def reset(self) -> None:
        """End the episode's chain. The next advance prefills the standing
        prompt again and starts a fresh line from that episode's first frame."""
        self._needs_prefill = True
        self._hidden = None
        self._next_token = None
        self._history = []
        self._prefill_text = ""
        # Two counters, because the cache slot a token lands in and the position
        # its rotary embedding is built from stop agreeing here. A slot is one
        # token, but an image advances the rotary position only by the side of
        # its token grid, and sliding the window moves tokens to lower slots
        # while leaving their positions alone.
        self._slot = 0
        self._rope = 0
        self._prefix_len = 0
        self._cache.reset()

    @torch.inference_mode()
    def advance(self, image: torch.Tensor, task_prompt: str) -> torch.Tensor:
        """The ``tokens_per_step`` activations this environment step issues.

        Args:
            image: (C, H, W) float tensor in [0, 1], appended to the chain as
                this step's frame.
            task_prompt: the env's language instruction, read on the episode's
                first step and prefilled once. Empty where the env sets none (or
                where ``use_prompt`` is off), leaving just the standing
                instruction.

        Returns:
            (tokens_per_step, layers_num, hidden_size) bfloat16.
        """
        if self._needs_prefill:
            self._prefill(task_prompt)
        self._ensure_chunk(image)
        self._slide_if_full()
        self._append_chunk(image)
        activations = []
        for _ in range(self.tokens_per_step):
            activations.append(self._hidden)
            self._write_step_inputs()
            if self._graph is None:
                outputs = self._forward_step()
                hidden_states, logits = outputs.hidden_states, outputs.logits
            else:
                self._graph.replay()
                hidden_states, logits = self._graph_hidden, self._graph_logits
            self._slot += 1
            self._rope += 1
            self._consume(hidden_states, logits[0, -1])
        return torch.stack(activations)

    def _prefill(self, task_prompt: str) -> None:
        """Write the standing prompt into the pinned head of the cache.

        Text only: the episode's frames all arrive as chunks, so nothing here
        occupies sink slots with a frame that will be stale a step later.
        """
        prompt = "\n".join(part for part in (task_prompt, self.INSTRUCTION) if part)
        # Thinking off: with the <think> block left open the model spends the
        # chain reasoning about the request rather than about the scene.
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]
        assert prompt_len <= self.PREFIX_SLOTS, (
            f"prompt of {prompt_len} tokens exceeds the {self.PREFIX_SLOTS} pinned slots; "
            "raise CoTStream.PREFIX_SLOTS"
        )
        self._history = []
        self._prefill_text = text
        self._cache.reset()
        self._needs_prefill = False
        position_ids = torch.arange(prompt_len, device=self.device).view(1, 1, -1).expand(3, 1, -1)
        outputs = self.model(
            **inputs,
            past_key_values=self._cache,
            use_cache=True,
            output_hidden_states=True,
            cache_position=torch.arange(prompt_len, device=self.device),
            position_ids=position_ids,
        )
        self._slot = prompt_len
        self._rope = prompt_len
        self._prefix_len = prompt_len
        self._consume(outputs.hidden_states, outputs.logits[0, -1])

    def _ensure_chunk(self, image: torch.Tensor) -> None:
        """Lay out the per-step chunk once, for frames of this shape.

        Every step appends the same token pattern -- one frame's worth of image
        tokens then the heading -- so its ids and the rotary positions they take
        relative to the chunk's start are fixed, and only the pixels change.
        """
        if self._chunk_shape == tuple(image.shape):
            return
        text = self.CHUNK_TEMPLATE
        inputs = self.processor(
            text=[text],
            images=[image.detach().float().clamp(0.0, 1.0)],
            return_tensors="pt",
            do_rescale=False,
        ).to(self.device)
        relative, _ = self.model.model.get_rope_index(
            inputs["input_ids"],
            mm_token_type_ids=inputs["mm_token_type_ids"],
            image_grid_thw=inputs["image_grid_thw"],
        )
        self._chunk_shape = tuple(image.shape)
        self._chunk_ids = inputs["input_ids"]
        self._chunk_len = inputs["input_ids"].shape[1]
        self._chunk_relative_positions = relative
        self._chunk_rope_span = int(relative.max().item()) + 1
        assert self._chunk_len + self.tokens_per_step <= self.CACHE_LEN - self.PREFIX_SLOTS, (
            f"a step's {self._chunk_len + self.tokens_per_step} tokens do not fit the "
            f"{self.CACHE_LEN - self.PREFIX_SLOTS} sliding slots"
        )

    def _append_chunk(self, image: torch.Tensor) -> None:
        """Append this step's frame and heading to the live chain."""
        self._ensure_chunk(image)
        pixels = self.processor.image_processor(
            images=[image.detach().float().clamp(0.0, 1.0)],
            return_tensors="pt",
            do_rescale=False,
        ).to(self.device)
        cache_position = torch.arange(self._slot, self._slot + self._chunk_len, device=self.device)
        outputs = self.model(
            input_ids=self._chunk_ids,
            pixel_values=pixels["pixel_values"],
            image_grid_thw=pixels["image_grid_thw"],
            past_key_values=self._cache,
            use_cache=True,
            output_hidden_states=True,
            cache_position=cache_position,
            position_ids=self._chunk_relative_positions + self._rope,
        )
        self._slot += self._chunk_len
        self._rope += self._chunk_rope_span
        self._consume(outputs.hidden_states, outputs.logits[0, -1])

    def _slide_if_full(self) -> None:
        """Drop the oldest ``COMPACT_BLOCK`` tokens of the stream when the next
        step would not fit.

        The pinned head stays where it is and every later token moves down by a
        block, which is the whole of the sliding window: the mask reaches
        exactly as far as the counter the layers keep, so lowering that counter
        alongside the copy is what hides the tail that was dropped. Rotary
        positions are untouched -- they are already baked into the cached keys,
        and only distances between them matter.
        """
        if self._slot + self._chunk_len + self.tokens_per_step <= self.CACHE_LEN:
            return
        source_start = self._prefix_len + self.COMPACT_BLOCK
        kept = self._slot - source_start
        for layer in self._cache.layers:
            if getattr(layer, "keys", None) is None:
                continue
            for cached in (layer.keys, layer.values):
                moved = cached[:, :, source_start : self._slot].clone()
                cached[:, :, self._prefix_len : self._prefix_len + kept].copy_(moved)
            layer.cumulative_length.sub_(self.COMPACT_BLOCK)
        self._slot -= self.COMPACT_BLOCK

    def _write_step_inputs(self) -> None:
        """Fill the buffers the step reads.

        The token being fed is also what the chain has actually written: a
        sample drawn at a prefill or at a chunk's last position is thrown away
        by the frame that follows it, so only the ones fed back here belong in
        the text.
        """
        self._history.append(self._next_token.item())
        del self._history[: -self.DISPLAY_TOKENS]
        self._graph_token.copy_(self._next_token)
        self._graph_cache_position.fill_(self._slot)
        self._graph_position_ids.fill_(self._rope)

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
        (layers_num, hidden_size) -- and sample the token it implies.

        Stacked into a tensor of this module's own: after a replay these are the
        graph's own output buffers, which the next replay overwrites while the
        step's earlier activations are still being collected.
        """
        self._hidden = torch.stack([state[0, -1] for state in hidden_states]).to(torch.bfloat16)
        allowed = logits.float().index_fill(0, self._banned, float("-inf"))
        probs = torch.softmax(allowed / self.temperature, dim=-1)
        self._next_token = torch.multinomial(probs, 1).view(1, 1)

    def prompt_text(self) -> str:
        """Verbatim what was prefilled, so what is logged beside the chain is
        what the model was actually given rather than a paraphrase of it."""
        return self._prefill_text

    def text(self) -> str:
        """The tail of the chain as written so far, for logging."""
        return self.processor.tokenizer.decode(
            self._history[-self.DISPLAY_TOKENS :], skip_special_tokens=True
        ).strip()
