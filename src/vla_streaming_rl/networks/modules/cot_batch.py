# SPDX-License-Identifier: MIT
"""A frozen VLM run to the end of a chain, once every N environment steps.

The same job as ``CoTStream`` and the same interface, taking the opposite side
of one trade. ``CoTStream`` keeps a chain mid-thought and issues a fixed budget
of tokens per control tick, so the reasoning is spread thin over many steps and
is always partly written. Here the chain runs to completion in a single call and
the steps in between read what it produced, so what the agent sees is a finished
thought about a frame that is up to ``steps_per_chain`` ticks old.

That is why the KV cache does not appear: nothing is carried between calls, and
a call is one ordinary ``generate`` over a chat-template prompt. Measured on an
episode's frames, carrying the cache across steps saved nothing at these context
lengths -- generation is bound by the per-token launch overhead, and the prefill
it removes is worth about one token of that.

What leaves this module is what leaves ``CoTStream``: the activation feeding the
lm_head at each generated position, every depth behind it kept, as an ordinary
observation stream. Nothing here trains and nothing here is differentiable.

Not an ``nn.Module``, for the same reason as ``CoTStream``: registering it would
put a frozen VLM into the network's ``parameters()`` and ``state_dict()``.
"""

import re

import torch

from .cot_stream import CoTStream
from .vlm_backbone import load_model


class CoTBatch:
    # The prompt is `CoTStream`'s, not a variant of it. The two differ in when
    # the model is run, and a second wording would confound that with a
    # difference in what it was asked.
    INSTRUCTION = CoTStream.INSTRUCTION
    CARRY = CoTStream.CARRY
    CARRY_SENTENCES = CoTStream.CARRY_SENTENCES

    def __init__(
        self,
        model_id: str,
        tokens_per_step: int,
        max_len: int,
        temperature: float,
        carry_prev: bool,
        steps_per_chain: int,
        device: torch.device,
    ) -> None:
        assert tokens_per_step >= 1, f"tokens_per_step must be positive; got {tokens_per_step}"
        assert steps_per_chain >= 1, f"steps_per_chain must be positive; got {steps_per_chain}"
        assert max_len >= tokens_per_step, (
            f"max_len {max_len} below {tokens_per_step}: a chain could not fill one step's read"
        )
        self.model, self.processor = load_model(model_id, use_lora=False, device=device)
        self.model.eval().requires_grad_(False)
        self.tokens_per_step = tokens_per_step
        self.max_len = max_len
        self.temperature = temperature
        self.carry_prev = carry_prev
        self.steps_per_chain = steps_per_chain
        self.device = device
        text_config = self.model.config.text_config
        self.hidden_size = text_config.hidden_size
        # The embedding plus every layer's output, matching `CoTStream`.
        self.layers_num = text_config.num_hidden_layers + 1
        self.reset()

    def reset(self) -> None:
        """Drop the chain. The next advance writes a new one from the frame it is given."""
        self._tokens = []
        self._prev_text = ""
        self._activations = torch.zeros(
            (self.tokens_per_step, self.layers_num, self.hidden_size),
            dtype=torch.bfloat16,
            device=self.device,
        )
        # Zero means "write one now", so the first advance of an episode always
        # reasons about that episode's own first frame.
        self._until_next = 0

    @torch.inference_mode()
    def advance(self, image: torch.Tensor, task_prompt: str) -> torch.Tensor:
        """This environment step's activations, writing a fresh chain when due.

        Args:
            image: (C, H, W) float tensor in [0, 1]; read only on the steps that
                write a chain, which is what makes the chain the slow loop.
            task_prompt: the env's language instruction, empty where the env sets
                none, leaving just the standing instruction.

        Returns:
            (tokens_per_step, layers_num, hidden_size) bfloat16. The same tensor
            on every step until the next chain is written.
        """
        if self._until_next == 0:
            self._write_chain(image, task_prompt)
            self._until_next = self.steps_per_chain
        self._until_next -= 1
        return self._activations

    def _write_chain(self, image: torch.Tensor, task_prompt: str) -> None:
        carried = self.CARRY.format(previous=self._prev_text) if self._prev_text else ""
        prompt = "\n".join(part for part in (task_prompt, carried, self.INSTRUCTION) if part)
        content = [{"type": "image"}, {"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
            # Thinking off: with the <think> block left open the model spends the
            # chain reasoning about the request rather than about the scene.
            enable_thinking=False,
        )
        inputs = self.processor(
            text=[text],
            images=[image.to(torch.float32)],
            return_tensors="pt",
            do_rescale=False,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_len,
            do_sample=True,
            temperature=self.temperature,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        prompt_len = inputs["input_ids"].shape[1]
        self._tokens = outputs.sequences[0, prompt_len:].tolist()
        self._activations = self._read_activations(outputs.hidden_states)
        # The last `CARRY_SENTENCES` sentences are what a carried chain quotes
        # back. A chain cut at `max_len` ends mid-sentence, and continuing a
        # fragment is what is wanted.
        pieces = [piece for piece in re.split(r"(?<=[.!?])\s+", self.text()) if piece]
        self._prev_text = " ".join(pieces[-self.CARRY_SENTENCES :]) if self.carry_prev else ""

    def _read_activations(self, hidden_states) -> torch.Tensor:
        """The tail of the chain, every depth kept: (tokens_per_step, layers_num, hidden_size).

        ``generate`` reports one entry per generated position, each a tuple over
        depths; the position's own activation is the last row of each. The tail
        is taken rather than the head because that is where the chain has said
        what it concluded, and a chain shorter than one step's read is held at
        its first position rather than padded with zeros, which would read as an
        activation the model never produced.
        """
        positions = [
            torch.stack([depth[0, -1] for depth in step]).to(torch.bfloat16)
            for step in hidden_states[-self.tokens_per_step :]
        ]
        while len(positions) < self.tokens_per_step:
            positions.insert(0, positions[0])
        return torch.stack(positions)

    def text(self) -> str:
        """The chain as written, for logging."""
        return self.processor.tokenizer.decode(self._tokens, skip_special_tokens=True).strip()
