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

# Fixed size so the render strip keeps a constant frame size across a run
# (the stable-panel contract in ``StepResult``).
_ANSWER_PANEL_SHAPE = (96, 384, 3)


def build_format_hint(action_spec: str) -> str:
    return (
        "Use the following structured response format. "
        "First, describe what you see in the current image inside <perception>...</perception>. "
        "Then, lay out your strategy step by step inside <reasoning>...</reasoning>, "
        "justifying the action you intend to take based on the current image and the "
        "previous reward (if shown). "
        f"Finally, output the action inside <answer>...</answer> using the format: {action_spec}. "
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
    """Zero-shot VLM controller following the Odysseus structured CoT protocol.

    At each step the current observation is sent to the model as a user turn;
    the model is prompted to produce three XML-style sections in order:

      ``<perception>...</perception>`` -- describe the visual state of the scene,
      grounding nearby obstacles, agent location, and interactive elements.

      ``<reasoning>...</reasoning>`` -- lay out a step-by-step strategy that
      justifies the next action.

      ``<answer>...</answer>`` -- emit the textual action that the env's
      ``parse_action_text`` decodes into an action vector.

    The most recent ``seq_len`` turns of the current episode (image, assistant
    response, reward observed AFTER that response) are kept in a FIFO and
    prepended to the chat as in-context history.

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
    ) -> None:
        super().__init__(learning_mode="streaming", horizon=1)
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
        self.model_id = model_id

        self.action_space = action_space
        self.action_dim = int(np.prod(action_space.shape))
        self.parse_action_text = parse_action_text
        self.format_hint = build_format_hint(action_spec)
        self.seq_len = seq_len
        self.max_new_tokens = max_new_tokens
        # The protocol already asks for the chain of thought in <reasoning>, so a
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

        if self.seq_len > 0:
            self.history.append((image_url, response_text, None))
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
        panels = {"answer": self._answer_panel(answer_text, parse_ok, choice.finish_reason)}
        return StepResult(action=action, metrics=metrics, panels=panels)

    def _step_streaming(
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
        self.history.clear()
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.step_in_episode = 0
        return {}

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

    @staticmethod
    def _answer_panel(answer_text: str, parse_ok: bool, finish_reason: str) -> np.ndarray:
        """What the model answered, next to the frame it answered it for."""
        canvas = np.zeros(_ANSWER_PANEL_SHAPE, dtype=np.uint8)
        caption = (
            f"answer: {answer_text!r}  parse: {'ok' if parse_ok else 'failed'}  "
            f"finish: {finish_reason}"
        )
        return overlay_caption(canvas, caption)
