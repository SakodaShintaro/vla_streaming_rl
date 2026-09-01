# SPDX-License-Identifier: MIT
"""A frozen VLM writing one commentary across a whole episode.

The context is one plain document, not a conversation: there are no turns and
no roles in it, because there is no one to take them. It reads

    <the env's language for the episode>
    <the record of the episode so far: one frame per step, oldest first>
    <the one turn boundary in the document>
    <the commentary, as one unbroken piece of prose>

and the chain simply carries that last piece on. Its own words are the last
thing it reads, so it continues the sentence it was in the middle of instead of
opening a new one, which is what putting the commentary at the end buys: an
observation appended between its sentences would restart it every tick, and at
a handful of tokens a tick it would never get past a restart.

The commentary is not free prose. It cycles through ``SLOTS``, one tagged run
each: what is in the frame, where that leaves things given what came before,
and what to do about it. Only the closing tags are the chain's -- every opening
tag is written for it, the moment the tag before it closes -- so the cycle
cannot be lost, only the answers are the model's, and a slot that never closes
is closed for it once it has run ``SLOT_BUDGET`` tokens.

That layout means the record grows in the middle of the document, so the
commentary cannot simply be appended to a cache. It does not have to be. Each
step rolls the cache back to the end of the record, appends its own step there,
and lays the commentary down again after it: the record -- frames included, and
they are most of what a step costs -- is written once and never re-read, and
what is re-read is only the commentary, which is held to the same window of
steps the record is.

Rolling back is exact in both halves of this model. The 6 full-attention layers
keep a KV cache whose tail is simply dropped, and the 18 linear-attention layers
keep a fixed-size recurrent state, which is snapshotted the moment the step's
own record is in and restored before the next one, so no commentary is ever fed
into it twice.

Left alone the record grows without bound, so it is held to a sliding window of
recent steps over the episode's own text, which is never evicted -- the
arrangement "Sliding-Window Beats Linear Attention" (Jolicoeur-Martineau et al.,
2026) measures, with that text in the part its four attention sinks play.
Positions are not rebased when a step is evicted. StreamingLLM rebases because
it targets models whose trained context is a few thousand tokens; here an
episode is tens of thousands against the 262k this one was trained for, so the
cached keys keep the rotation they were written with and the arithmetic stays
exact.

Nothing in this module says what to write. That is the prompt builder's, which
is where every string a policy reads as language is composed; the chain is
handed the episode's text and writes under it.

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

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformers import StaticCache
from transformers.cache_utils import LinearAttentionLayer, StaticLayer

from vla_streaming_rl.utils import PANEL_INPUT, PANEL_OUTPUT

from .vlm_backbone import load_model

# Stands in for everything the model is shown, so that its own chat template
# can be rendered once and cut at the seam between what it reads and what it
# writes.
_SHOWN_SENTINEL = "<<<<SHOWN_SENTINEL>>>>"


def _states(layer: LinearAttentionLayer) -> list[torch.Tensor]:
    """Everything a linear-attention layer carries between tokens. The layer
    keeps them keyed by state index; a snapshot only needs them in a fixed
    order, and the tensors themselves are written in place, so their addresses
    survive both the snapshot and the graph that records them."""
    return list(layer.conv_states.values()) + list(layer.recurrent_states.values())


@dataclass
class _Step:
    """One environment step's line of the record, which is its frame alone.

    Only what it occupies in the cache is worth keeping: the frame is a run of
    placeholder tokens that a reader has nothing to read in, and the render
    strip is already showing the picture.
    """

    slots: int


class CoTStream:
    # The episode's own text is written once and never evicted, so this has to
    # cover whatever language an env opens an episode with.
    PREFIX_BUDGET = 1024

    # What one step of the record is allowed to add: the frame (66 tokens at the
    # resolutions these envs render), the action and the readings. A step over
    # this is a prompt that has to be shortened, not a budget to raise, since
    # the window is counted in steps.
    STEP_BUDGET = 256

    # The three questions the commentary answers, in the order it answers them,
    # and the name each answer is tagged with. It runs back to the first once
    # the last has closed.
    SLOTS = ("view", "situation", "plan")

    # How long one answer may run before it is closed for the chain. This is a
    # valve, not the way an answer normally ends: a slot the chain never closes
    # would swallow the cycle, but a slot cut off at the budget is a sentence
    # that never finished. It has to sit well above what a sentence costs, and
    # well above ``tokens_per_step`` -- level with it, every answer is cut at
    # the end of the first step it was written in.
    SLOT_BUDGET = 96

    # How many cycles of the three the commentary keeps. What is kept is what
    # gets laid down again every step, so this is the one thing in the document
    # that is read more than once.
    CYCLES_KEPT = 2

    # What the tags are, said once at the top of the document. This is the one
    # thing the module says for itself: the prompt builder settles what the
    # chain is looking at and what it is for, and this settles only the shape
    # its answers take, which is this module's own and would mean nothing to a
    # run without one.
    # Says how to answer, never what to answer with. Any wording that puts a
    # description next to a tag's name is copied into that tag word for word --
    # it is the likeliest continuation of the tag, and a small model takes it --
    # so the names are listed and left to say what they mean.
    PROTOCOL = (
        "The pictures below are what you have been shown, oldest first. The "
        "last one is what is in front of you now; the ones above it are where "
        "you have just been.\n"
        "Keep an account of it in three tags that come round in turn -- {0}, "
        "then {1}, then {2} -- one plain sentence in each, each answering its "
        "own tag and not the tag before it.\n"
        "The account is your own, so write it in the first person: what I can "
        "see, where that leaves me, what I do next. Everything above is "
        "addressed to you, but nothing you write is addressed to anyone.\n"
        "The tag that opens an answer is written for you: write the sentence, "
        "close the tag, and write nothing else -- no other tag, no markup, no "
        "numbers, and not what you have already said."
    )

    # The length of a run of tokens the commentary may not say twice. Prose
    # carried on from itself with nothing to stop it collapses into a loop --
    # a small model given its own text back will settle into repeating a
    # sentence of it forever -- and blocking the repeat of a run this long
    # breaks the loop without touching the short phrases prose reuses.
    REPEAT_BLOCK = 6

    def __init__(
        self,
        model_id: str,
        observation_shape: Sequence[int],
        tokens_per_step: int,
        temperature: float,
        window_steps: int,
        use_cuda_graph: bool,
        device: torch.device,
    ) -> None:
        assert tokens_per_step >= 1, f"tokens_per_step must be positive; got {tokens_per_step}"
        assert window_steps >= 1, f"window_steps must be positive; got {window_steps}"
        self.model, self.processor = load_model(model_id, use_lora=False, device=device)
        self.model.eval().requires_grad_(False)
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor
        self.observation_shape = tuple(observation_shape)
        self.tokens_per_step = tokens_per_step
        self.temperature = temperature
        self.window_steps = window_steps
        self.device = device
        config = self.model.config
        text_config = config.text_config
        self.hidden_size = text_config.hidden_size
        # The embedding plus every layer's output.
        self.layers_num = text_config.num_hidden_layers + 1
        self.max_position = text_config.max_position_embeddings

        # What the chain may not choose. It writes one unbroken line of prose,
        # so the tokens that would end it or open a line of their own are taken
        # out of its vocabulary: every special token, end-of-turn among them,
        # and every token carrying a newline. Without this the chain finishes a
        # sentence and then writes the record's next `Action:` line itself,
        # which is the format it has just been shown a dozen times over.
        banned = set(self.tokenizer.get_added_vocab().values())
        banned.update(
            token_id
            for token, token_id in self.tokenizer.get_vocab().items()
            if "\n" in token or "Ċ" in token  # raw, and byte-level BPE's newline
        )
        self.banned_token_ids = torch.tensor(sorted(banned), device=device)

        # --- The frame's tokens. Observation dimensions are fixed for a run, so
        #     the grid is measured once here and the per-step path only pushes
        #     pixels through the processor.
        dummy_frame = torch.zeros(self.observation_shape)
        grid = self.image_processor(images=[dummy_frame], return_tensors="pt", do_rescale=False)
        self.image_grid_thw = grid["image_grid_thw"].to(device)
        image_tokens_num = int(
            self.image_grid_thw[0].prod().item() // self.image_processor.merge_size**2
        )
        self.image_token_id = config.image_token_id
        vision_start, image_pad, vision_end = self.tokenizer.convert_ids_to_tokens(
            [config.vision_start_token_id, config.image_token_id, config.vision_end_token_id]
        )
        self._image_text = vision_start + image_pad * image_tokens_num + vision_end

        # --- Where what the model reads ends and what it writes begins, taken
        #     from its own chat template rather than spelled out here. The
        #     template is rendered once and cut at the sentinel: what comes
        #     before opens the turn everything shown sits in, and what comes
        #     after closes it and opens the one the commentary is written in.
        #     That seam is the only one in the document -- the turn the
        #     commentary is in is opened once and never closed, so the chain
        #     carries one piece of writing across the whole episode -- and it is
        #     what makes an instruct model read the protocol as an instruction
        #     instead of as text to carry on.
        rendered = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": _SHOWN_SENTINEL}]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        self._turn_open, self._seam = rendered.split(_SHOWN_SENTINEL)

        # --- The tags. An opening tag carries the line break that puts its
        #     answer on a line of its own, which the chain cannot write itself.
        self._open_ids = [self.tokenizer(f"\n<{name}>")["input_ids"] for name in self.SLOTS]
        self._close_ids = [self.tokenizer(f"</{name}>")["input_ids"] for name in self.SLOTS]
        self._close_texts = [f"</{name}>" for name in self.SLOTS]
        cycle = sum(
            len(opening) + self.SLOT_BUDGET + len(closing)
            for opening, closing in zip(self._open_ids, self._close_ids)
        )
        self.commentary_budget = self.CYCLES_KEPT * cycle

        # --- One cache for the whole run. An episode resets it in place rather
        #     than replacing it, so its buffers keep the addresses a graph
        #     records. The commentary and the lead into it sit past the record,
        #     and a step's worth of tokens is generated past the budget before
        #     the commentary is trimmed back to it.
        self._cache_len = (
            self.PREFIX_BUDGET
            + window_steps * self.STEP_BUDGET
            + self.commentary_budget
            + self.STEP_BUDGET
        )
        self._cache = StaticCache(config=config, max_cache_len=self._cache_len)
        # The two halves eviction and rollback act on: keys and values whose
        # tail can be dropped, and recurrent states that have to be put back.
        self._kv_layers = [layer for layer in self._cache.layers if isinstance(layer, StaticLayer)]
        self._linear_layers = [
            layer for layer in self._cache.layers if isinstance(layer, LinearAttentionLayer)
        ]

        # What a recorded step reads. Their contents change every step; their
        # addresses must not, which is the whole point of recording one.
        self._graph_token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self._graph_position_ids = torch.zeros(3, 1, 1, dtype=torch.long, device=device)
        self._graph = None
        self.reset()
        if use_cuda_graph:
            # Record one decode step, so the steady state replays as a single
            # call instead of the ~2500 kernel launches issuing it costs.
            # Recording runs the step, which writes the cache and advances the
            # linear-attention state, so it happens on a throwaway document over
            # a blank frame; the reset at the end leaves the real one to start
            # clean.
            #
            # ``no_grad`` rather than ``inference_mode``: this is where the
            # cache allocates, and a tensor born in inference mode cannot be
            # written from outside it, which is what recording then does.
            with torch.no_grad():
                self._prefill_prefix("")
                self._write_record(torch.zeros(self.observation_shape, device=self.device))
                self._write_commentary()
                self._write_step_inputs(self._next_token)

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

    @torch.inference_mode()
    def reset(self) -> None:
        """End the episode's context. The next advance opens a fresh one.

        Under inference mode because the cache buffers are born there on the
        first episode, and a tensor born in inference mode cannot be written
        from outside it.
        """
        self._needs_prefix = True
        self._hidden = None
        self._next_token = 0
        self._steps: list[_Step] = []
        self._slot = 0
        # The tag that opens the first answer is written before anything is
        # asked of the chain, so what it reads always ends on a question.
        self._commentary: list[int] = list(self._open_ids[0])
        self._answer: list[int] = []
        self._forced: list[int] = []
        self._prefix_text = ""
        self._prefix_slots = 0
        self._snapshot: list[list[torch.Tensor]] = []
        # Where the record ends, which is what a step rolls back to. Kept here
        # rather than read back from the cache: a recorded step advances the
        # cache inside the graph, where host-side counters are the only things
        # that stay in step with it. Slots and positions differ because a frame
        # spends one cache slot per token but advances the position by the
        # extent of its grid.
        self._record_slots = 0
        self._record_position = 0
        self._slots = 0
        self._position = 0
        self._cache.reset()

    @torch.inference_mode()
    def advance(self, image: torch.Tensor, episode_text: str) -> torch.Tensor:
        """The ``tokens_per_step`` activations this environment step issues.

        Args:
            image: (C, H, W) float tensor in [0, 1], this tick's frame.
            episode_text: the episode's language, read only when the context is
                opened. Empty where the env sets none (or where ``use_prompt``
                is off), which leaves the chain nothing to write under and is
                the ablation that measures what it was worth.

        Returns:
            (tokens_per_step, layers_num, hidden_size) bfloat16.
        """
        if self._needs_prefix:
            self._prefill_prefix(episode_text)
        self._roll_back()
        self._evict()
        self._write_record(image)
        self._write_commentary()
        activations = []
        while len(activations) < self.tokens_per_step:
            # Whatever the rule owes the document goes down first, so the step's
            # own activations are all read at positions the chain chose.
            while self._forced:
                self._feed(self._forced.pop(0))
                self._answer = []
            # Held before feeding, which samples the one after it.
            chosen = self._next_token
            activations.append(self._hidden)
            self._feed(chosen)
            self._answer.append(chosen)
            self._close_answer()
        self._commentary = self._commentary[-self.commentary_budget :]
        return torch.stack(activations)

    # --- the document -------------------------------------------------------

    def _prefill_prefix(self, episode_text: str) -> None:
        """Open the episode's document with the env's own language for it and
        the shape the answers take, written once and never evicted."""
        protocol = self.PROTOCOL.format(*self.SLOTS)
        shown = "\n".join(part for part in (episode_text, protocol) if part)
        input_ids = self.tokenizer(f"{self._turn_open}{shown}\n", return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(self.device)
        length = input_ids.shape[1]
        assert length <= self.PREFIX_BUDGET, (
            f"episode text of {length} tokens exceeds the prefix budget "
            f"{self.PREFIX_BUDGET}; raise CoTStream.PREFIX_BUDGET"
        )
        position_ids = torch.arange(length, device=self.device).view(1, 1, -1).expand(3, 1, -1)
        self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=self._cache,
            use_cache=True,
            logits_to_keep=1,
        )
        self._needs_prefix = False
        self._prefix_text = episode_text
        self._prefix_slots = length
        self._slots = length
        self._position = length
        self._mark_record()

    def _write_record(self, image: torch.Tensor) -> None:
        """Append this step to the record, which is the last thing written to
        the cache that outlives the step."""
        text = f"{self._image_text}\n"
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
        length = input_ids.shape[1]
        assert length <= self.STEP_BUDGET, (
            f"a step of {length} tokens exceeds the step budget {self.STEP_BUDGET}; "
            "shorten the step's language"
        )
        assert self._position + length <= self.max_position, (
            f"the episode has run past the {self.max_position} positions the model was trained "
            "for; the document is appended to, so an episode this long needs positions rebased"
        )
        mm_token_type_ids = (input_ids == self.image_token_id).long()
        pixel_values = self.image_processor(images=[image], return_tensors="pt", do_rescale=False)[
            "pixel_values"
        ].to(self.device, torch.bfloat16)
        # The frame's tokens carry a grid rather than a single position, so the
        # step's own layout is computed by the model and only then placed after
        # what the document already holds.
        local_positions, _ = self.model.model.get_rope_index(
            input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=self.image_grid_thw,
        )
        self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=self.image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            position_ids=local_positions + self._position,
            past_key_values=self._cache,
            use_cache=True,
            logits_to_keep=1,
        )
        self._steps.append(_Step(length))
        self._slots += length
        self._position += int(local_positions.max()) + 1
        self._mark_record()

    def _close_answer(self) -> None:
        """Move on to the next question once the chain has closed this one, or
        once it has had its ``SLOT_BUDGET`` tokens to. Closing it for the chain
        means writing the tag it did not; either way the tag that opens the next
        answer is owed to the document."""
        closing = self._close_texts[self._slot]
        answer = self.tokenizer.decode(self._answer)
        closed = closing in answer
        # An answer that has started writing its own closing tag is given the
        # rest of it, budget or no budget: cutting in there leaves the tag
        # written twice, once half and once whole.
        mid_tag = any(answer.endswith(closing[:cut]) for cut in range(1, len(closing)))
        if not closed and (mid_tag or len(self._answer) < self.SLOT_BUDGET):
            return
        self._forced.extend([] if closed else self._close_ids[self._slot])
        self._slot = (self._slot + 1) % len(self.SLOTS)
        self._forced.extend(self._open_ids[self._slot])

    def _write_commentary(self) -> None:
        """Lay the commentary down after the record and read off the activation
        the chain writes its next token from. This is the only part of the
        document that is written more than once."""
        seam_ids = self.tokenizer(self._seam, return_tensors="pt")["input_ids"][0]
        input_ids = torch.cat(
            [
                seam_ids.to(self.device),
                torch.tensor(self._commentary, dtype=torch.long, device=self.device),
            ]
        ).view(1, -1)
        length = input_ids.shape[1]
        assert self._slots + length + self.tokens_per_step <= self._cache_len, (
            f"the commentary and the step it is written under need "
            f"{self._slots + length + self.tokens_per_step} slots of the cache's {self._cache_len}"
        )
        position_ids = (
            torch.arange(self._position, self._position + length, device=self.device)
            .view(1, 1, -1)
            .expand(3, 1, -1)
        )
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=self._cache,
            use_cache=True,
            output_hidden_states=True,
            logits_to_keep=1,
        )
        self._slots += length
        self._position += length
        self._consume(outputs.hidden_states, outputs.logits[0, -1])

    def _mark_record(self) -> None:
        """Take the document's end as the record's end: the point a step rolls
        back to, and the recurrent state that belongs to it."""
        self._record_slots = self._slots
        self._record_position = self._position
        self._snapshot = [
            [state.clone() for state in _states(layer)] for layer in self._linear_layers
        ]

    def _roll_back(self) -> None:
        """Drop the commentary the last step laid down, so that this step writes
        its own record directly after the one before it.

        The attention layers only have to forget the tail of their cache, which
        is what the length they report does. The linear layers cannot forget,
        so they are put back to the state marked when the record last grew.
        """
        for layer in self._kv_layers:
            layer.cumulative_length.fill_(self._record_slots)
        for layer, marked in zip(self._linear_layers, self._snapshot):
            for state, mark in zip(_states(layer), marked):
                state.copy_(mark)
        self._slots = self._record_slots
        self._position = self._record_position

    def _evict(self) -> None:
        """Make room for the step about to be written by dropping the oldest
        ones, which slide out from directly behind the episode's own text.

        What follows them moves down over the gap; the keys keep the rotation
        they were cached with, so what survives keeps the positions it was
        written at and only its slot moves. The mask follows the length the
        layers report, which is what is decremented here.
        """
        while len(self._steps) >= self.window_steps:
            length = self._steps.pop(0).slots
            start = self._prefix_slots
            kept = self._slots - start - length
            for layer in self._kv_layers:
                layer.keys[:, :, start : start + kept] = layer.keys[
                    :, :, start + length : self._slots
                ].clone()
                layer.values[:, :, start : start + kept] = layer.values[
                    :, :, start + length : self._slots
                ].clone()
                layer.cumulative_length.sub_(length)
            self._slots -= length
        self._record_slots = self._slots

    # --- decoding -----------------------------------------------------------

    def _write_step_inputs(self, token_id: int) -> None:
        """Fill the buffers the step reads.

        Positions are handed in rather than left to the model, which builds them
        on the host and copies them across -- a transfer capture forbids. A
        commentary token is text, so all three axes sit at the same position.
        """
        self._graph_token.fill_(token_id)
        self._graph_position_ids.fill_(self._position)

    def _forward_step(self):
        return self.model(
            input_ids=self._graph_token,
            past_key_values=self._cache,
            use_cache=True,
            output_hidden_states=True,
            position_ids=self._graph_position_ids,
        )

    def _feed(self, token_id: int) -> None:
        """Put one token into the document and read the next one off."""
        self._commentary.append(token_id)
        self._write_step_inputs(token_id)
        if self._graph is None:
            outputs = self._forward_step()
            hidden_states, logits = outputs.hidden_states, outputs.logits
        else:
            self._graph.replay()
            hidden_states, logits = self._graph_hidden, self._graph_logits
        self._slots += 1
        self._position += 1
        self._consume(hidden_states, logits[0, -1])

    def _repeats(self) -> list[int]:
        """The tokens that would say a run of ``REPEAT_BLOCK`` this answer
        already holds a second time. Looked for inside the answer rather than
        across the commentary, whose tags come round again by design."""
        tail = self._answer[1 - self.REPEAT_BLOCK :]
        return [
            self._answer[index + len(tail)]
            for index in range(len(self._answer) - len(tail))
            if self._answer[index : index + len(tail)] == tail
        ]

    def _consume(self, hidden_states, logits: torch.Tensor) -> None:
        """Keep the position's activation at every depth behind it --
        (layers_num, hidden_size) -- and sample the token it implies.

        Stacked into a tensor of this module's own: after a replay these are the
        graph's own output buffers, which the next replay overwrites while the
        step's earlier activations are still being collected.
        """
        self._hidden = torch.stack([state[0, -1] for state in hidden_states]).to(torch.bfloat16)
        # ``float`` copies, so masking never writes into a replay's own logits
        # buffer.
        logits = logits.float()
        logits[self.banned_token_ids] = float("-inf")
        logits[self._repeats()] = float("-inf")
        probs = torch.softmax(logits / self.temperature, dim=-1)
        self._next_token = int(torch.multinomial(probs, 1))

    def transcript(self) -> list[tuple[str, str]]:
        """The document as it currently stands, as ``(role, text)`` entries:
        what the model was handed, and the one piece it wrote.

        This is the whole of what the VLM is reading right now, so what the
        window has evicted is gone from here as well.
        """
        entries = [(PANEL_INPUT, self._prefix_text)]
        written = self.tokenizer.decode(self._commentary, skip_special_tokens=True)
        return entries + [(PANEL_OUTPUT, written.strip())]
