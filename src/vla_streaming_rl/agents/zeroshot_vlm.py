# SPDX-License-Identifier: MIT
"""Zero-shot VLM controller backed by OpenRouter.

The agent talks to any chat/vision model hosted on OpenRouter through the
OpenAI-compatible endpoint, so switching to a stronger model is a matter of
changing ``model_id`` (e.g. ``anthropic/claude-opus-5``,
``google/gemini-3.1-pro-preview``, ``openai/gpt-5.2``). It learns nothing: it is
the zero-shot baseline the trained agents are measured against, so it plugs into
the same trainer loop and reports the same telemetry.
"""

import base64
import io
import os
import re
import time
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from openai import OpenAI
from PIL import Image

from vla_streaming_rl.agents.base import Agent, StepResult
from vla_streaming_rl.utils import overlay_caption

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# Stands in for the assistant turn of a step whose response did not follow the
# format, so the history stays an honest record of what was actually executed.
NO_ACTION = "(no valid action; the previous action was repeated)"

# The render panel is text only, so it is a zero-height image whose caption band
# is the whole panel. The band reserves a fixed number of lines, so the panel
# keeps a constant size across a run (the stable-panel contract in
# ``StepResult``); only the width is chosen here. Text past the band's line
# budget is dropped, so the decoded action leads and the raw response follows.
_OUTPUT_PANEL_WIDTH = 512


def _text_panel(text: str, width: int) -> np.ndarray:
    return overlay_caption(np.zeros((0, width, 3), dtype=np.uint8), text)


def build_format_hint(action_spec: str) -> str:
    """The response protocol. Every generated token is latency (one request per
    env step), so the model is asked for a short justification and the action,
    and nothing else -- no scene description, no restating of the task."""
    return (
        "Reply with exactly two sections and no other text. "
        "First, in AT MOST two short sentences inside <think>...</think>, "
        "say what in the current image decides your next action, taking the previous "
        "reward (if shown) into account. Do not describe the scene in general, do not "
        "restate the task, and do not repeat your earlier reasoning. "
        f"Then output the action inside <answer>...</answer> using the format: {action_spec}. "
        "The text inside <answer> must contain ONLY the action -- no commentary, no labels."
    )


def encode_image(image: np.ndarray, image_side: int) -> str:
    """PNG data URL of a CHW float observation, resized to ``image_side``."""
    hwc = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
    resized = Image.fromarray(hwc).resize((image_side, image_side), Image.LANCZOS)
    buffer = io.BytesIO()
    resized.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


class ZeroShotVLMAgent(Agent):
    """Zero-shot VLM controller driven by a short structured CoT protocol.

    At each step the current observation is sent to the model as a user turn;
    the model is prompted to produce two XML-style sections in order:

      ``<think>...</think>`` -- at most two sentences on what in the current
      image decides the next action.

      ``<answer>...</answer>`` -- the textual action that the env's
      ``parse_action_text`` decodes into an action vector.

    Latency is one API round trip per env step and scales with the tokens
    generated, so the protocol buys only the justification that changes the
    action: a full scene description (the ``<perception>`` section of the
    original Odysseus protocol) tripled the output for no measured benefit.

    The most recent ``seq_len`` turns of the current episode (image, the ACTION
    taken, reward observed AFTER it) are kept in a FIFO and prepended to the chat
    as in-context history. The <think> section is deliberately not kept:
    replaying it made the model copy its own earlier thoughts instead of reading
    the frame.

    A response that does not follow the format is a failure of the model and is
    reported as one (``parse_failed`` in the metrics); nothing is recovered from
    the rest of the text, since a phrase picked out of the reasoning is not the
    action the model chose. The env keeps running on the previous action.
    """

    def __init__(
        self,
        *,
        action_space: gym.spaces.Box,
        parse_action_text,
        action_spec: str,
        model_id: str,
        seq_len: int,
        max_new_tokens: int,
        reasoning_max_tokens: int,
        image_side: int,
        temperature: float,
        api_max_retries: int,
        reset_on_episode_end: bool,
    ) -> None:
        super().__init__(horizon=1, reset_on_episode_end=reset_on_episode_end)
        # One API call per env step means a single upstream hiccup (a shared-pool
        # 429, a 5xx) would otherwise abort a run that is minutes deep. The SDK
        # retries those with exponential backoff; only give it room to.
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            max_retries=api_max_retries,
        )
        self.model_id = model_id

        self.action_space = action_space
        self.action_dim = int(np.prod(action_space.shape))
        self.parse_action_text = parse_action_text
        self.format_hint = build_format_hint(action_spec)
        self.seq_len = seq_len
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
        self.image_side = image_side
        self.temperature = temperature

        self.history: deque[tuple[str, str, float | None]] = deque(maxlen=max(seq_len, 1))
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.step_in_episode = 0

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def select_action(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult:
        del global_step, terminated, truncated, info

        # The reward shown with a past turn is the one observed after it.
        if self.step_in_episode > 0 and self.history:
            past_url, past_response, _ = self.history[-1]
            self.history[-1] = (past_url, past_response, reward)

        image_url = encode_image(obs["image"], self.image_side)
        messages = self._build_messages(
            obs["language"],
            image_url,
            current_reward=reward if self.step_in_episode > 0 else None,
        )

        request_start = time.time()
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            extra_body={"reasoning": self.reasoning},
        )
        api_msec = (time.time() - request_start) * 1000

        choice = completion.choices[0]
        response_text = choice.message.content or ""
        answer_match = ANSWER_RE.search(response_text)
        answer_text = answer_match.group(1).strip() if answer_match is not None else ""
        action, parse_ok = self._parse_action(answer_text)

        # Only the action goes into the history, not the <think> section that
        # produced it: replaying the full response made the model copy its own
        # earlier thoughts verbatim instead of reading the current frame, and it
        # was half the prompt.
        if self.seq_len > 0:
            self.history.append((image_url, answer_text if parse_ok else NO_ACTION, None))
        self.last_action = action
        self.step_in_episode += 1

        # `finish_reason` and the token counts are what distinguishes a model
        # that ignored the format from one that never got to emit an answer (a
        # reasoning model can spend the whole budget before `content` starts).
        print(
            f"  [step {self.step_in_episode:4d}] reward={reward:+.3f} "
            f"parse={'ok' if parse_ok else 'failed'} api={api_msec:.0f}ms "
            f"finish={choice.finish_reason} "
            f"tokens={completion.usage.prompt_tokens}->{completion.usage.completion_tokens} "
            f"text={response_text!r}"
        )
        metrics = {
            "vlm/parse_failed": float(not parse_ok),
            "vlm/api_msec": api_msec,
            "vlm/prompt_tokens": float(completion.usage.prompt_tokens),
            "vlm/completion_tokens": float(completion.usage.completion_tokens),
        }
        caption = (
            f"answer: {answer_text!r}  parse: {'ok' if parse_ok else 'failed'}  "
            f"finish: {choice.finish_reason}  ||  {response_text}"
        )
        panels = {"output": _text_panel(caption, _OUTPUT_PANEL_WIDTH)}
        return StepResult(action=action, metrics=metrics, panels=panels)

    def step(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult:
        return self.select_action(global_step, obs, reward, terminated, truncated, info)

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        del score, feedback_text
        if self.reset_on_episode_end:
            self.history.clear()
            self.last_action = np.zeros(self.action_dim, dtype=np.float32)
            self.step_in_episode = 0
        return {}

    def optimizer_state_dict(self) -> dict:
        # the baseline queries a hosted model; there is nothing to optimize
        return {}

    def load_optimizer_state_dict(self, state: dict) -> None:
        del state

    def _preprocess(self, obs: dict[str, Any], info: dict) -> str:
        del info
        return encode_image(obs["image"], self.image_side)

    def _to_env_action(self, net_action: np.ndarray) -> np.ndarray:
        return np.clip(net_action, self.action_space.low, self.action_space.high)

    # ------------------------------------------------------------------
    # Action decoding
    # ------------------------------------------------------------------

    def _parse_action(self, answer_text: str) -> tuple[np.ndarray, bool]:
        action_array, parse_ok = self.parse_action_text(answer_text)
        if parse_ok:
            return self._to_env_action(action_array[0].astype(np.float32)), True
        return self.last_action, False

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _build_messages(
        self, task_prompt: str, image_url: str, current_reward: float | None
    ) -> list[dict]:
        system_prompt = f"{task_prompt}\n\n{self.format_hint}"
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        # The reward shown before user turn i is the one stored on history[i-1],
        # i.e. the reward observed AFTER history[i-1]'s action. The oldest entry
        # in the FIFO has no predecessor left, so its reward prefix is dropped.
        for i in range(len(self.history)):
            past_url, past_response, _ = self.history[i]
            reward_prefix = self.history[i - 1][2] if i > 0 else None
            messages.append(
                {"role": "user", "content": self._build_user_content(past_url, reward_prefix)}
            )
            messages.append({"role": "assistant", "content": past_response})
        messages.append(
            {"role": "user", "content": self._build_user_content(image_url, current_reward)}
        )
        return messages

    @staticmethod
    def _build_user_content(image_url: str, reward: float | None) -> list[dict]:
        content: list[dict] = []
        if reward is not None:
            content.append({"type": "text", "text": f"Previous reward: {reward:+.3f}"})
        content.append({"type": "image_url", "image_url": {"url": image_url}})
        return content
