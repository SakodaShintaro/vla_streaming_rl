# SPDX-License-Identifier: MIT
"""A frozen VLM kept mid-thought across environment steps.

The chain of thought is the slow loop. It is prefilled from one frame and then
advanced by a fixed budget of tokens per environment step, so a single line of
reasoning spans many control ticks. When it ends -- EOS, or ``max_len`` tokens --
the next advance prefills again from whatever frame is current, starting a fresh
chain on a fresh image.

What leaves this module is not text but the activation feeding the VLM's lm_head
at each generated position: the state the model was in when it chose that token,
which carries far more than the token id does. Nothing here trains and nothing
here is differentiable, so downstream the chain is an ordinary observation
stream, stored in the replay buffer next to the image.

Not an ``nn.Module`` on purpose: registering it would put a frozen 0.8B model
into the network's ``parameters()`` and its ``state_dict()``.
"""

import torch
from torchvision.transforms.functional import to_pil_image

from .vlm_backbone import load_model


class CoTStream:
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
        self.hidden_size = self.model.config.text_config.hidden_size
        self.eos_token_id = self.processor.tokenizer.eos_token_id
        self.reset()

    def reset(self) -> None:
        """Drop the chain. The next advance prefills from the frame it is given."""
        self._cache = None
        self._hidden = None
        self._next_token = None
        self._tokens = []

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
            (tokens_per_step, hidden_size) bfloat16.
        """
        activations = []
        while len(activations) < self.tokens_per_step:
            if self._cache is None:
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
        self._tokens = []
        self._consume(self.model(**inputs, use_cache=True, output_hidden_states=True))

    def _decode(self) -> None:
        # Positions are handed in rather than left to the model, which builds them
        # on the host and copies them across -- a transfer CUDA graph capture
        # forbids, and this step is the one worth capturing. rope_deltas is the
        # offset the image tokens introduced, recorded by the prefill.
        position = self._cache.get_seq_length()
        cache_position = torch.arange(position, position + 1, device=self.device)
        rope_deltas = self.model.model.rope_deltas
        position_ids = (cache_position.view(1, 1, -1) + rope_deltas.unsqueeze(0)).expand(3, -1, -1)
        self._consume(
            self.model(
                input_ids=self._next_token,
                past_key_values=self._cache,
                use_cache=True,
                output_hidden_states=True,
                cache_position=cache_position,
                position_ids=position_ids,
            )
        )

    def _consume(self, outputs) -> None:
        """Take the position's lm_head input, sample the token it implies, and
        decide whether the chain lives on."""
        self._cache = outputs.past_key_values
        # hidden_states[-1] is post-final-norm, i.e. exactly what lm_head reads.
        self._hidden = outputs.hidden_states[-1][0, -1].to(torch.bfloat16)
        probs = torch.softmax(outputs.logits[0, -1].float() / self.temperature, dim=-1)
        self._next_token = torch.multinomial(probs, 1).view(1, 1)
        self._tokens.append(self._next_token.item())
        ended = self._tokens[-1] == self.eos_token_id or len(self._tokens) >= self.max_len
        if ended:
            self._cache = None

    def text(self) -> str:
        """The chain as written so far, for logging."""
        return self.processor.tokenizer.decode(self._tokens, skip_special_tokens=True).strip()
