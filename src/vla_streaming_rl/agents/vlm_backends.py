# SPDX-License-Identifier: MIT
"""Where the zero-shot VLM controller's text generation runs.

Both backends take the same neutral chat messages and return the same
`VLMResponse`, so `ZeroShotVLMAgent` builds one prompt and reports one set of
telemetry whichever is in use. A message's ``content`` is always a list of
``{"type": "text", "text": ...}`` / ``{"type": "image", "image": <PIL image>}``
parts -- the format transformers' chat templates require -- plus
``{"type": "video", "video": <path to an mp4>}``, which is how a whole episode
goes over at once (see `episode_critic`). `OpenRouterBackend` converts them
into the OpenAI wire format's data URLs; a local backend reads images only.
"""

import base64
import io
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig
from openai import BadRequestError, OpenAI
from PIL import Image

from vla_streaming_rl.networks.modules.vlm_backbone import load_model, sampling_kwargs

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


def _mp4_data_url(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{payload}"


def _to_openai_part(part: dict) -> dict:
    if part["type"] == "text":
        return part
    if part["type"] == "video":
        return {"type": "video_url", "video_url": {"url": _mp4_data_url(part["video"])}}
    return {"type": "image_url", "image_url": {"url": _png_data_url(part["image"])}}


def _to_openai_content(content: list[dict]):
    # A text-only turn goes over the wire as a plain string: some models reject
    # a parts list on the system and assistant roles.
    if all(part["type"] == "text" for part in content):
        return "\n".join(part["text"] for part in content)
    return [_to_openai_part(part) for part in content]


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    return [
        {"role": message["role"], "content": _to_openai_content(message["content"])}
        for message in messages
    ]


class OpenRouterBackend:
    # First wait before re-sending a request whose 200 carried an upstream
    # error; each further attempt doubles it.
    BODY_RETRY_BASE_SECONDS = 2.0

    def __init__(
        self,
        *,
        model_id: str,
        max_new_tokens: int,
        reasoning_max_tokens: int,
        temperature: float,
        api_max_retries: int,
        body_max_retries: int,
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
        self.body_max_retries = body_max_retries

    def _attempt(self, messages: list[dict]):
        """One request, as either the completion or what went wrong with it.

        An upstream failure reaches this client two ways -- a 400 the SDK raises
        (`Provider returned error`) and a 200 whose body carries the error with
        no `choices` -- and neither is retried by the SDK. Both are the same
        transient thing here, so both come back as a failure to retry.
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=_to_openai_messages(messages),
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                extra_body={"reasoning": self.reasoning},
            )
        except BadRequestError as error:
            return None, str(error)
        return completion, None if completion.choices else f"no choices: {completion}"

    def generate(self, messages: list[dict]) -> VLMResponse:
        # A provider hiccup (a shared-pool 400, a multimodal download that timed
        # out upstream) must not lose a run that is hours deep, so it is waited
        # out here; what the last attempt said is the only account of it.
        for attempt in range(self.body_max_retries):
            completion, failure = self._attempt(messages)
            if failure is None:
                break
            delay = self.BODY_RETRY_BASE_SECONDS * 2**attempt
            print(f"OpenRouter request failed, retrying in {delay:.1f}s: {failure}")
            time.sleep(delay)
        assert failure is None, (
            f"OpenRouter request failed {self.body_max_retries} times: {failure}"
        )
        choice = completion.choices[0]
        return VLMResponse(
            text=choice.message.content or "",
            finish_reason=str(choice.finish_reason),
            prompt_tokens=int(completion.usage.prompt_tokens),
            completion_tokens=int(completion.usage.completion_tokens),
        )


class LocalVLMBackend:
    def __init__(
        self,
        *,
        model_id: str,
        max_new_tokens: int,
        reasoning_max_tokens: int,
        temperature: float,
    ) -> None:
        assert temperature >= 0.0, temperature
        self.device = torch.device("cuda")
        self.model, self.processor = load_model(model_id, use_lora=False, device=self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        # As on the hosted backend, 0 means the model does no thinking of its
        # own; here that is the chat template's block, which it then renders
        # already closed.
        self.enable_thinking = reasoning_max_tokens != 0
        self.temperature = temperature

    @torch.inference_mode()
    def generate(self, messages: list[dict]) -> VLMResponse:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.enable_thinking,
        ).to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            **sampling_kwargs(self.temperature),
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
            body_max_retries=args.body_max_retries,
        )
    return LocalVLMBackend(
        model_id=args.vlm_model_id,
        max_new_tokens=args.max_new_tokens,
        reasoning_max_tokens=args.reasoning_max_tokens,
        temperature=args.temperature,
    )
