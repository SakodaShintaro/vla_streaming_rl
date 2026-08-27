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
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from .cot_static_cache import CoTStaticCache
from .vlm_backbone import load_model

# Qwen3.5 decides the linear-attention mask by reading a device value into a
# Python ``if``, which syncs the device to the host on every forward and so bars
# CUDA graph capture of the decode step. When no attention mask is given the
# branch cannot change the answer -- its second term is false, so both arms
# return None -- and short-circuiting that case removes the sync while leaving
# every other case, padded batches included, on the original.
assert hasattr(Qwen3_5TextModel, "_update_linear_attn_mask"), (
    "Qwen3_5TextModel._update_linear_attn_mask is gone; this transformers "
    "version no longer matches the patch in cot_stream.py"
)
_ORIGINAL_LINEAR_ATTN_MASK = Qwen3_5TextModel._update_linear_attn_mask


def _linear_attn_mask_without_sync(self, attention_mask, cache_position):
    if attention_mask is None:
        return None
    return _ORIGINAL_LINEAR_ATTN_MASK(self, attention_mask, cache_position)


Qwen3_5TextModel._update_linear_attn_mask = _linear_attn_mask_without_sync


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
        self.hidden_size = self.model.config.text_config.hidden_size
        self.eos_token_id = self.processor.tokenizer.eos_token_id
        # One cache for the whole run. A new chain resets it in place rather than
        # replacing it, so its buffers keep the addresses a graph records.
        self._cache = CoTStaticCache(self.model.config.text_config, self.PROMPT_BUDGET + max_len)
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
            (tokens_per_step, hidden_size) bfloat16.
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
        assert prompt_len + self.max_len <= self._cache.max_len, (
            f"prompt of {prompt_len} tokens plus a {self.max_len}-token chain exceeds the "
            f"cache length {self._cache.max_len}; raise CoTStream.PROMPT_BUDGET"
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
        self._cache.length = prompt_len
        self._consume(outputs.hidden_states[-1][0, -1], outputs.logits[0, -1])

    @torch.inference_mode()
    def _capture_decode(self) -> None:
        """Record one decode step, so the steady state replays as a single call
        instead of the ~2500 kernel launches issuing it costs.

        Recording runs the step, which writes the cache and advances the
        linear-attention state, so it happens on a throwaway chain over a blank
        frame; the reset at the end leaves the real chain to start clean.
        """
        self._prefill(torch.zeros(3, self.CAPTURE_FRAME_SIDE, self.CAPTURE_FRAME_SIDE), "")
        self._write_step_inputs(self._cache.length)

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
        self._graph_hidden = outputs.hidden_states[-1]
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
        position = self._cache.length
        self._write_step_inputs(position)
        if self._graph is None:
            outputs = self._forward_step()
            hidden, logits = outputs.hidden_states[-1], outputs.logits
        else:
            self._graph.replay()
            hidden, logits = self._graph_hidden, self._graph_logits
        self._cache.length = position + 1
        self._consume(hidden[0, -1], logits[0, -1])

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
