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
from vla_streaming_rl.utils import render_conversation_panel

# The LAST <answer> is the one that counts: a model's reasoning sometimes quotes
# the tag before writing the real section, and reading the first one then takes
# the whole reasoning as the action.
ANSWER_RE = re.compile(r"<answer>(?!.*<answer>)(.*?)</answer>", re.DOTALL)

# Stands in for the assistant turn of a step whose response did not follow the
# format, so the history stays an honest record of what was actually executed.
NO_ACTION = "(no valid action; the previous action was repeated)"


def preprocess_image(image: np.ndarray) -> Image.Image:
    """A CHW float observation as an RGB image, which is what a backend takes.

    The resolution is left alone: both backends resize for themselves -- the
    local processor to a multiple of its patch size, the hosted one server-side
    -- so scaling here only moves bytes without changing what the model sees.
    """
    return Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8))


class ZeroShotVLMAgent(Agent):
    # Wide and tall enough for several turns of the conversation at once, as on
    # the trained side: the panel is the only place a run shows what the model
    # was actually asked. Fixed rather than grown from the text, so the strip
    # keeps a constant size across a run (the stable-panel contract in
    # ``StepResult``).
    PANEL_WIDTH = 680
    PANEL_HEIGHT = 560

    def __init__(
        self,
        *,
        action_space: gym.spaces.Box,
        parse_action_text,
        backend,
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

        image = preprocess_image(obs["image"])
        self.prompt_builder.observe(obs, reward, info, image)
        prompt = self.prompt_builder.task_text()

        request_start = time.time()
        response = self.backend.generate(self.prompt_builder.conversation())
        api_msec = (time.time() - request_start) * 1000

        response_text = response.text
        answer_match = ANSWER_RE.search(response_text)
        answer_text = answer_match.group(1).strip() if answer_match is not None else ""
        action_array, parse_ok = self.parse_action_text(answer_text)
        # A response that did not follow the format leaves the env running on the
        # previous action; nothing is recovered from the rest of the text.
        action = (
            self._to_env_action(action_array[0].astype(np.float32))
            if parse_ok
            else self.last_action
        )

        # The reply is handed back as written, <think> section and all, so the
        # conversation is the whole record of what the model said -- what the
        # render panel draws is then what the model itself reads. A response that
        # did not parse says so, since the env carried on without it.
        self.prompt_builder.add_reply(response_text if parse_ok else f"{response_text} {NO_ACTION}")
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
        status = (
            f"in {response.prompt_tokens} tok   out {response.completion_tokens} tok   "
            f"{api_msec:.0f} ms   parse {'ok' if parse_ok else 'failed'}   "
            f"{response.finish_reason}"
        )
        panels = {
            "conversation": render_conversation_panel(
                self.prompt_builder.conversation(), status, self.PANEL_WIDTH, self.PANEL_HEIGHT
            )
        }
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
        return preprocess_image(obs["image"])

    def _to_env_action(self, net_action: np.ndarray) -> np.ndarray:
        return np.clip(net_action, self.action_space.low, self.action_space.high)
