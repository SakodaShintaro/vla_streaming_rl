# SPDX-License-Identifier: MIT
"""Where the zero-shot VLM controller's text generation runs.

Both backends take the same neutral chat messages and return the same
`VLMResponse`, so `ZeroShotVLMAgent` builds one prompt and reports one set of
telemetry whichever is in use. A message's ``content`` is always a list of
``{"type": "text", "text": ...}`` / ``{"type": "image", "image": <PIL image>}``
parts -- the format transformers' chat templates require -- which
`OpenRouterBackend` converts into the OpenAI wire format's data URLs.
"""

import base64
import io
import os
from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from openai import OpenAI
from PIL import Image

from vla_streaming_rl.networks.modules.vlm_backbone import load_model

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The protocol's answer section, and the close of a chat template's own thinking
# block. The opener ends in a newline so that every action text is a single
# token after it -- straight after `>` the tokenizer merges some of them into
# the tag, which would leave those actions' logits not comparable with the rest.
ANSWER_OPEN = "<answer>\n"
ANSWER_CLOSE = "</answer>"
THINK_CLOSE = "</think>"


@dataclass(frozen=True)
class VLMResponse:
    """One generation, in the terms the agent logs and parses."""

    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _to_openai_content(content: list[dict]):
    # A text-only turn goes over the wire as a plain string: some models reject
    # a parts list on the system and assistant roles.
    if all(part["type"] == "text" for part in content):
        return "\n".join(part["text"] for part in content)
    return [
        part
        if part["type"] == "text"
        else {"type": "image_url", "image_url": {"url": _png_data_url(part["image"])}}
        for part in content
    ]


class OpenRouterBackend:
    """A model hosted on OpenRouter, reached over its OpenAI-compatible endpoint.

    Switching to a stronger model is a matter of changing ``model_id`` (e.g.
    ``anthropic/claude-opus-5``, ``google/gemini-3.1-pro-preview``,
    ``openai/gpt-5.2``).
    """

    def __init__(
        self,
        *,
        model_id: str,
        max_new_tokens: int,
        reasoning_max_tokens: int,
        temperature: float,
        api_max_retries: int,
    ) -> None:
        # One API call per env step means a single upstream hiccup (a shared-pool
        # 429, a 5xx) would otherwise abort a run that is minutes deep. The SDK
        # retries those with exponential backoff; only give it room to.
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            max_retries=api_max_retries,
        )
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        # The protocol already asks for the chain of thought in <think>, so a
        # Qwen model's own thinking is a second, hidden copy of it that eats the
        # same token budget: with no cap it routinely burns the whole budget and
        # returns an empty `content` (finish_reason=length). 0 turns it off.
        self.reasoning = (
            {"enabled": False}
            if reasoning_max_tokens == 0
            else {"max_tokens": reasoning_max_tokens}
        )
        self.temperature = temperature

    def generate(self, messages: list[dict]) -> VLMResponse:
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=_to_openai_messages(messages),
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            extra_body={"reasoning": self.reasoning},
        )
        choice = completion.choices[0]
        return VLMResponse(
            text=choice.message.content or "",
            finish_reason=str(choice.finish_reason),
            prompt_tokens=int(completion.usage.prompt_tokens),
            completion_tokens=int(completion.usage.completion_tokens),
        )


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    return [
        {"role": message["role"], "content": _to_openai_content(message["content"])}
        for message in messages
    ]


class LocalVLMBackend:
    """A Qwen3.5 checkpoint generating in this process.

    The same `load_model` the trained networks use, so a local run needs no
    server and no extra dependency -- it does share the GPU with whatever else
    the run has loaded, and it generates one response per env step, so it is
    slower per step than a hosted model.

    The action is chosen rather than read back: the ``<answer>`` opener is
    written into the model's own reply and the token after it is drawn from the
    env's action texts alone, so a response always carries a legal action. That
    only works where the actions can be enumerated, which rules out an env with
    a continuous action text.

    ``reasoning_max_tokens`` caps the chat template's own thinking block first,
    which is what the hosted backend's `reasoning` parameter does server-side.
    There is no server here to enforce it, so the block is generated on its own
    and closed at the budget.
    """

    def __init__(
        self,
        *,
        model_id: str,
        action_choices: list[str],
        reasoning_max_tokens: int,
        temperature: float,
    ) -> None:
        assert action_choices, (
            "the local backend picks among action texts; this env enumerates none"
        )
        assert temperature > 0.0, temperature
        self.device = torch.device("cuda")
        self.model, self.processor = load_model(model_id, use_lora=False, device=self.device)
        self.model.eval()
        self.action_choices = action_choices
        self.action_token_ids = [self._action_token_id(choice) for choice in action_choices]
        self.reasoning_max_tokens = reasoning_max_tokens
        # As on the hosted backend, 0 means the model does no thinking of its
        # own; here that is the chat template's block, which it then renders
        # already closed.
        self.enable_thinking = reasoning_max_tokens != 0
        self.temperature = temperature

    def _action_token_id(self, choice: str) -> int:
        """The one token ``choice`` becomes where it is generated, after the opener."""
        tokenizer = self.processor.tokenizer
        opener = tokenizer.encode(ANSWER_OPEN, add_special_tokens=False)
        whole = tokenizer.encode(ANSWER_OPEN + choice, add_special_tokens=False)
        assert whole[: len(opener)] == opener and len(whole) == len(opener) + 1, (
            f"{choice!r} is not a single token after {ANSWER_OPEN!r}: {whole}"
        )
        return whole[-1]

    def _render(self, messages: list[dict], prefill: str):
        """The conversation with ``prefill`` already written as the model's reply,
        which `generate` continues instead of starting a fresh turn."""
        conversation = messages + [
            {"role": "assistant", "content": [{"type": "text", "text": prefill}]}
        ]
        return self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=False,
            continue_final_message=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            # Forwarded to the chat template as a jinja variable, which is where
            # a Qwen3 template reads its thinking switch from. `apply_chat_template`
            # also offers it to the processor, which has no use for it and logs
            # one "not a valid argument" line about it per process.
            enable_thinking=self.enable_thinking,
        ).to(self.device)

    def _reason(self, messages: list[dict]) -> tuple[str, int]:
        """The model's own thinking, closed off so the answer can follow it."""
        if self.reasoning_max_tokens == 0:
            return "", 0
        inputs = self._render(messages, "")
        prompt_tokens = int(inputs["input_ids"].shape[1])
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.reasoning_max_tokens,
            do_sample=True,
            temperature=self.temperature,
            stop_strings=[THINK_CLOSE],
            tokenizer=self.processor.tokenizer,
        )
        # `generate` returns the prompt followed by the completion, so the
        # thinking is what sits past the prompt length.
        ids = generated[0, prompt_tokens:]
        text = self.processor.decode(ids, skip_special_tokens=True)
        # Closed here rather than trusted to close itself: the budget cuts the
        # block off mid-sentence whenever the model would have run past it.
        return f"{text.split(THINK_CLOSE)[0].strip()}\n{THINK_CLOSE}\n\n", int(ids.shape[0])

    @torch.inference_mode()
    def generate(self, messages: list[dict]) -> VLMResponse:
        reasoning, reasoning_tokens = self._reason(messages)
        prefill = f"{reasoning}{ANSWER_OPEN}"
        inputs = self._render(messages, prefill)
        # One forward pass over the prompt: the action is the softmax over just
        # the action tokens' logits, so no other token can ever be produced.
        logits = self.model(**inputs).logits[0, -1, self.action_token_ids]
        index = int(torch.multinomial(torch.softmax(logits / self.temperature, dim=-1), 1))
        return VLMResponse(
            text=f"{prefill}{self.action_choices[index]}{ANSWER_CLOSE}",
            # The action is drawn from a closed set, so there is no budget to run out of.
            finish_reason="stop",
            # The thinking block is part of this prompt but was generated, not
            # given, so it is counted once -- as completion.
            prompt_tokens=int(inputs["input_ids"].shape[1]) - reasoning_tokens,
            completion_tokens=reasoning_tokens + 1,
        )


def build_vlm_backend(args: DictConfig, action_choices: list[str]):
    assert args.vlm_backend in ("openrouter", "local"), args.vlm_backend
    if args.vlm_backend == "openrouter":
        return OpenRouterBackend(
            model_id=args.openrouter_model_id,
            max_new_tokens=args.max_new_tokens,
            reasoning_max_tokens=args.reasoning_max_tokens,
            temperature=args.temperature,
            api_max_retries=args.api_max_retries,
        )
    return LocalVLMBackend(
        model_id=args.vlm_model_id,
        action_choices=action_choices,
        reasoning_max_tokens=args.reasoning_max_tokens,
        temperature=args.temperature,
    )
