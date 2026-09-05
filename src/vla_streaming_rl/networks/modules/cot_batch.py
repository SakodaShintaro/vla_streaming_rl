# SPDX-License-Identifier: MIT
import torch

from .cot_stream import CoTStream
from .vlm_backbone import load_model


class CoTBatch:
    # The instruction is `CoTStream`'s, not a variant of it. The two differ in
    # when the model is run, and a second wording would confound that with a
    # difference in what it was asked. Its CARRY has no counterpart here: what
    # that quotes back is in the conversation already.
    INSTRUCTION = CoTStream.INSTRUCTION

    # The instruction opens the conversation and is never repeated in it. Later
    # turns are the new frame under this cue instead, which is what asks for a
    # delta rather than a fresh description: told only to continue, the model
    # paraphrases what it already said and the chain stops carrying anything new.
    # Greedy decoding writes the sentence it just wrote again, for as many
    # tokens as it is given, until this is on.
    REPETITION_PENALTY = 1.15

    CONTINUE = (
        "The current frame. Carry the commentary forward: write what is new or "
        "has changed since the last one, not what it already says."
    )

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
        # `temperature` is accepted so both modes take the same parameters, and
        # ignored: a chain written in one call is decoded greedily, which is how
        # this was measured. `CoTStream` samples because it has to keep one
        # chain alive over many steps and a greedy one there settles into a loop
        # it can never leave.
        del temperature
        self.carry_prev = carry_prev
        self.steps_per_chain = steps_per_chain
        self.device = device
        text_config = self.model.config.text_config
        self.hidden_size = text_config.hidden_size
        # The embedding plus every layer's output, matching `CoTStream`.
        self.layers_num = text_config.num_hidden_layers + 1
        self.reset()

    def reset(self) -> None:
        """Drop the conversation. The next advance opens a new one on the frame it
        is given."""
        self._tokens = []
        self._conversation = []
        self._images = []
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
        # carry_prev off is the ablation: every chain sees only its own frame,
        # so the conversation goes no further back than the turn about to open.
        self._conversation = self._conversation if self.carry_prev else []
        self._images = self._images if self.carry_prev else []
        opening = "\n".join(part for part in (task_prompt, self.INSTRUCTION) if part)
        asked = opening if not self._conversation else self.CONTINUE
        self._conversation.append(
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": asked}]}
        )
        self._images.append(image.to(torch.float32))
        text = self.processor.apply_chat_template(
            self._conversation,
            add_generation_prompt=True,
            tokenize=False,
            # Thinking off: with the <think> block left open the model spends the
            # chain reasoning about the request rather than about the scene.
            enable_thinking=False,
        )
        inputs = self.processor(
            text=[text],
            images=self._images,
            return_tensors="pt",
            do_rescale=False,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_len,
            do_sample=False,
            repetition_penalty=self.REPETITION_PENALTY,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        prompt_len = inputs["input_ids"].shape[1]
        self._tokens = outputs.sequences[0, prompt_len:].tolist()
        self._activations = self._read_activations(outputs.hidden_states)
        self._conversation.append(
            {"role": "assistant", "content": [{"type": "text", "text": self.text()}]}
        )

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
