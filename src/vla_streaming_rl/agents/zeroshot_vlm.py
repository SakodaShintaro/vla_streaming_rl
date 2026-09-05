# SPDX-License-Identifier: MIT
"""Zero-shot VLM controller.

The agent builds one chat prompt per env step and hands it to a `VLMBackend`,
so the same protocol runs against a model hosted on OpenRouter or a Qwen3.5
checkpoint generating locally (see `vlm_backends`). It learns nothing: it is
the zero-shot baseline the trained agents are measured against, so it plugs
into the same trainer loop and reports the same telemetry.
"""

import re
import time
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image

from vla_streaming_rl.agents.base import Agent, StepResult
from vla_streaming_rl.agents.prompt import PromptBuilder
from vla_streaming_rl.utils import render_text_panel

# The LAST <answer> is the one that counts: a model's reasoning sometimes quotes
# the tag before writing the real section, and reading the first one then takes
# the whole reasoning as the action.
ANSWER_RE = re.compile(r"<answer>(?!.*<answer>)(.*?)</answer>", re.DOTALL)

# Stands in for the assistant turn of a step whose response did not follow the
# format, so the history stays an honest record of what was actually executed.
NO_ACTION = "(no valid action; the previous action was repeated)"

# The render panel is text only. Its size is fixed here rather than by the text,
# so the panel keeps a constant size across a run (the stable-panel contract in
# ``StepResult``). Text past the last line that fits is dropped, so the decoded
# action leads and the raw response follows.
_OUTPUT_PANEL_WIDTH = 512
_OUTPUT_PANEL_HEIGHT = 406


def preprocess_image(image: np.ndarray, image_side: int) -> Image.Image:
    """A CHW float observation as a square RGB image of side ``image_side``."""
    hwc = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(hwc).resize((image_side, image_side), Image.LANCZOS)


class ZeroShotVLMAgent(Agent):
    def __init__(
        self,
        *,
        action_space: gym.spaces.Box,
        parse_action_text,
        backend,
        image_side: int,
        reset_on_episode_end: bool,
        prompt_builder: PromptBuilder,
    ) -> None:
        super().__init__(
            horizon=1,
            reset_on_episode_end=reset_on_episode_end,
            prompt_builder=prompt_builder,
        )
        self.backend = backend

        self.action_space = action_space
        self.action_dim = int(np.prod(action_space.shape))
        self.parse_action_text = parse_action_text
        self.image_side = image_side

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
        del global_step, terminated, truncated

        image = preprocess_image(obs["image"], self.image_side)
        self.prompt_builder.observe(obs, reward, info, image)
        prompt = self.prompt_builder.task_text()

        request_start = time.time()
        response = self.backend.generate(self.prompt_builder.conversation())
        api_msec = (time.time() - request_start) * 1000

        response_text = response.text
        answer_match = ANSWER_RE.search(response_text)
        answer_text = answer_match.group(1).strip() if answer_match is not None else ""
        action, parse_ok = self._parse_action(answer_text)

        # Only the action is handed back as this turn's reply, not the <think>
        # section that produced it: replaying the full response made the model
        # copy its own earlier thoughts verbatim instead of reading the current
        # frame, and it was half the prompt.
        self.prompt_builder.add_reply(answer_text if parse_ok else NO_ACTION)
        self.last_action = action
        self.step_in_episode += 1

        # `finish_reason` and the token counts are what distinguishes a model
        # that ignored the format from one that never got to emit an answer (a
        # reasoning model can spend the whole budget before `content` starts).
        print(
            f"  [step {self.step_in_episode:4d}] reward={reward:+.3f} "
            f"parse={'ok' if parse_ok else 'failed'} api={api_msec:.0f}ms "
            f"finish={response.finish_reason} "
            f"tokens={response.prompt_tokens}->{response.completion_tokens} "
            f"text={response_text!r}"
        )
        metrics = {
            "vlm/parse_failed": float(not parse_ok),
            "vlm/api_msec": api_msec,
            "vlm/prompt_tokens": float(response.prompt_tokens),
            "vlm/completion_tokens": float(response.completion_tokens),
        }
        caption = (
            f"answer: {answer_text!r}  parse: {'ok' if parse_ok else 'failed'}  "
            f"finish: {response.finish_reason}  ||  {response_text}"
        )
        panels = {"output": render_text_panel(caption, _OUTPUT_PANEL_WIDTH, _OUTPUT_PANEL_HEIGHT)}
        return StepResult(
            action=action,
            metrics=metrics,
            panels=panels,
            texts={"prompt": prompt},
        )

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

    def on_episode_end(self, score: float) -> dict:
        del score
        if self.reset_on_episode_end:
            self.prompt_builder.reset()
            self.last_action = np.zeros(self.action_dim, dtype=np.float32)
            self.step_in_episode = 0
        return {}

    def optimizer_state_dict(self) -> dict:
        # the baseline queries a hosted model; there is nothing to optimize
        return {}

    def load_optimizer_state_dict(self, state: dict) -> None:
        del state

    def _preprocess(self, obs: dict[str, Any], info: dict) -> Image.Image:
        del info
        return preprocess_image(obs["image"], self.image_side)

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
