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
# Standing still rather than repeating the last action, because repeating turns a
# malformed reply into a committed one: a run that answers badly while walking
# keeps walking into whatever it could not describe.
NO_ACTION = "(no valid action; the agent stood still)"


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
        steps_per_action: int,
    ) -> None:
        assert steps_per_action >= 1, steps_per_action
        super().__init__(
            horizon=1,
            reset_on_episode_end=reset_on_episode_end,
            prompt_builder=prompt_builder,
        )
        self.backend = backend
        # One generation every `steps_per_action` steps, the action held in
        # between, which is the cadence `CoTBatch` writes a chain at. The
        # conversation then advances once per that many steps at both ends, so a
        # turn covers the same stretch of an episode either way and the two are
        # comparable without the baseline paying for a generation per tick.
        self.steps_per_action = steps_per_action

        self.action_space = action_space
        self.action_dim = int(np.prod(action_space.shape))
        self.parse_action_text = parse_action_text

        self.held_action = np.zeros(self.action_dim, dtype=np.float32)
        self.held_status = ""
        self.held_metrics = {}
        self.steps_until_next = 0
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

        if self.steps_until_next == 0:
            self._write_action()
            self.steps_until_next = self.steps_per_action
        self.steps_until_next -= 1
        self.step_in_episode += 1

        panels = {
            "conversation": render_conversation_panel(
                self.prompt_builder.conversation(),
                self.held_status,
                self.PANEL_WIDTH,
                self.PANEL_HEIGHT,
            )
        }
        return StepResult(
            action=self.held_action,
            metrics=self.held_metrics,
            panels=panels,
            texts={"prompt": prompt},
        )

    def _write_action(self) -> None:
        """Generate on this step's conversation and hold what it decided.

        What the generation cost is held with it, since the steps that run on an
        action are not the steps that paid for it.
        """
        request_start = time.time()
        response = self.backend.generate(self.prompt_builder.conversation())
        api_msec = (time.time() - request_start) * 1000

        response_text = response.text
        answer_match = ANSWER_RE.search(response_text)
        answer_text = answer_match.group(1).strip() if answer_match is not None else ""
        action_array, parse_ok = self.parse_action_text(answer_text)
        # A response that did not follow the format stands the agent still for
        # the steps it would have driven; nothing is recovered from the rest of
        # the text.
        self.held_action = (
            self._to_env_action(action_array[0].astype(np.float32))
            if parse_ok
            else np.zeros(self.action_dim, dtype=np.float32)
        )

        # The reply is handed back as written, <think> section and all, so the
        # conversation is the whole record of what the model said -- what the
        # render panel draws is then what the model itself reads. A response that
        # did not parse says so, since the env carried on without it.
        self.prompt_builder.add_reply(response_text if parse_ok else f"{response_text} {NO_ACTION}")

        self.held_metrics = {
            "vlm/parse_failed": float(not parse_ok),
            "vlm/api_msec": api_msec,
            "vlm/prompt_tokens": float(response.prompt_tokens),
            "vlm/completion_tokens": float(response.completion_tokens),
        }
        self.held_status = (
            f"in {response.prompt_tokens} tok   out {response.completion_tokens} tok   "
            f"{api_msec:.0f} ms   parse {'ok' if parse_ok else 'failed'}   "
            f"{response.finish_reason}"
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
            self.held_action = np.zeros(self.action_dim, dtype=np.float32)
            self.held_status = ""
            self.held_metrics = {}
            # Zero means "generate now", so the first step of an episode decides
            # on that episode's own first frame.
            self.steps_until_next = 0
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
