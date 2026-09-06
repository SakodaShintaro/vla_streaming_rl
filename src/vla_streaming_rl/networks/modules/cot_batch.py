# SPDX-License-Identifier: MIT
import time

import torch
import torch.nn.functional as F

from vla_streaming_rl.agents.prompt import PromptBuilder

from .vlm_backbone import load_model, sampling_kwargs


class CoTBatch:
    def __init__(
        self,
        model_id: str,
        tokens_per_step: int,
        max_len: int,
        temperature: float,
        steps_per_chain: int,
        prompt_builder: PromptBuilder,
        device: torch.device,
    ) -> None:
        assert tokens_per_step >= 1, f"tokens_per_step must be positive; got {tokens_per_step}"
        assert steps_per_chain >= 1, f"steps_per_chain must be positive; got {steps_per_chain}"
        assert max_len >= tokens_per_step, (
            f"max_len {max_len} below {tokens_per_step}: the pool would stretch a chain "
            "shorter than one step's read"
        )
        self.model, self.processor = load_model(model_id, use_lora=False, device=device)
        self.model.eval().requires_grad_(False)
        self.tokens_per_step = tokens_per_step
        self.max_len = max_len
        # Decoded the way the zero-shot controller decodes: it reads the same
        # conversation through the same model, so a chain written any other way
        # would not be the baseline's reasoning measured under RL. 0 is greedy,
        # which `generate` spells as do_sample=False rather than a zero divisor.
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
        messages, images = self._render(self.prompt_builder.conversation())
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            # Thinking off: with the <think> block left open the model spends the
            # chain reasoning about the request rather than about the scene.
            enable_thinking=False,
        )
        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            do_rescale=False,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_len,
            **sampling_kwargs(self.temperature),
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        prompt_len = inputs["input_ids"].shape[1]
        self._tokens = outputs.sequences[0, prompt_len:].tolist()
        self._activations = self._read_activations(outputs.hidden_states)
        self._input_tokens = int(prompt_len)
        self._msec = (time.perf_counter() - start) * 1000.0
        self.prompt_builder.add_reply(self.text())

    def _render(self, turns: list[dict]) -> tuple[list[dict], list[torch.Tensor]]:
        """The turns as the processor takes them: the frames pulled out into
        their own list, since the chat template wants a placeholder where each
        one goes and the pixels handed over beside it."""
        images = [
            part["image"].to(torch.float32)
            for turn in turns
            for part in turn["content"]
            if part["type"] == "image"
        ]
        messages = [
            {
                "role": turn["role"],
                "content": [
                    {key: value for key, value in part.items() if key != "image"}
                    for part in turn["content"]
                ],
            }
            for turn in turns
        ]
        return messages, images

    def _read_activations(self, hidden_states) -> torch.Tensor:
        """The whole chain, every depth kept, pooled to one step's read:
        (tokens_per_step, layers_num, hidden_size).

        ``generate`` reports one entry per generated position, each a tuple over
        depths; the position's own activation is the last row of each. A chain
        stops where the model stops it, so the positions are pooled along the
        chain rather than sliced to its tail: the whole chain reaches the policy,
        and a chain that ended early needs no padding to reach the fixed read.
        """
        positions = torch.stack(
            [torch.stack([depth[0, -1] for depth in step]) for step in hidden_states]
        )
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
