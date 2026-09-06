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

# The close of the protocol's answer section, which is where a local generation
# is stopped: the action is the last thing the protocol asks for, so nothing
# past it is worth the latency.
ANSWER_CLOSE = "</answer>"


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
        # OpenRouter reports an upstream failure in the body of a 200, which the
        # SDK's own retries never see; `choices` is then absent and the payload
        # is the only account of what went wrong.
        assert completion.choices, f"OpenRouter returned no choices: {completion}"
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

    The protocol is the hosted backend's, run locally: the model writes the
    whole reply itself and the agent reads the action out of it, so a response
    that ignores the format fails here exactly as it would there. Constraining
    the action to a legal token instead would make the two backends measure
    different things under one baseline's name.

    ``reasoning_max_tokens`` turns the chat template's own thinking block on and
    off, which is what the hosted backend's `reasoning` parameter does
    server-side. There is no server here to hold it to a budget, so the cap
    itself is the shared ``max_new_tokens``.
    """

    def __init__(
        self,
        *,
        model_id: str,
        max_new_tokens: int,
        reasoning_max_tokens: int,
        temperature: float,
    ) -> None:
        assert temperature > 0.0, temperature
        self.device = torch.device("cuda")
        self.model, self.processor = load_model(model_id, use_lora=False, device=self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        # As on the hosted backend, 0 means the model does no thinking of its
        # own; here that is the chat template's block, which it then renders
        # already closed.
        self.enable_thinking = reasoning_max_tokens != 0
        self.temperature = temperature

    def _render(self, messages: list[dict]):
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            # Forwarded to the chat template as a jinja variable, which is where
            # a Qwen3 template reads its thinking switch from. `apply_chat_template`
            # also offers it to the processor, which has no use for it and logs
            # one "not a valid argument" line about it per process.
            enable_thinking=self.enable_thinking,
        ).to(self.device)

    @torch.inference_mode()
    def generate(self, messages: list[dict]) -> VLMResponse:
        inputs = self._render(messages)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            stop_strings=[ANSWER_CLOSE],
            tokenizer=self.processor.tokenizer,
        )
        ids = generated[0, prompt_tokens:]
        text = self.processor.decode(ids, skip_special_tokens=True)
        return VLMResponse(
            text=text,
            # What the hosted backend reports: the answer closed the reply, or
            # the budget ran out before it did.
            finish_reason="stop" if ANSWER_CLOSE in text else "length",
            prompt_tokens=prompt_tokens,
            completion_tokens=int(ids.shape[0]),
        )


def build_vlm_backend(args: DictConfig):
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
        max_new_tokens=args.max_new_tokens,
        reasoning_max_tokens=args.reasoning_max_tokens,
        temperature=args.temperature,
    )
