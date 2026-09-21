# SPDX-License-Identifier: MIT
"""Zero-shot VLM controller.

The same model is asked three times every ``steps_per_action`` steps, each in a
conversation of its own. As the judge it reads every frame of the steps the last
subtask stood for and says how far that subtask got, which is only logged here
and is what a trained low-level policy would be paid. As the planner it reads
what the env asks and names a subtask; as the actor it reads that subtask, never
the env's own instruction, and names the action that is then held. All go
through a `VLMBackend`, so the
same protocol runs against a model hosted on OpenRouter or a Qwen3.5 checkpoint
generating locally (see `vlm_backends`). It learns nothing: it is the zero-shot
baseline the trained agents are measured against, so it plugs into the same
trainer loop and reports the same telemetry.
"""

import time
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image

from vla_streaming_rl.agents.base import Agent, StepResult
from vla_streaming_rl.agents.prompt import (
    ACHIEVED_RE,
    ANSWER_RE,
    SUBTASK_RE,
    PromptBuilder,
    assistant_turn,
)
from vla_streaming_rl.utils import render_conversation_panel


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
        actor_builder: PromptBuilder,
        steps_per_action: int,
    ) -> None:
        assert steps_per_action >= 1, steps_per_action
        super().__init__(
            horizon=1,
            reset_on_episode_end=reset_on_episode_end,
            prompt_builder=prompt_builder,
        )
        self.backend = backend
        self.actor_builder = actor_builder
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
        self.hold_steps = 0
        self.held_exchange = []
        self.held_status = ""
        self.held_metrics = {}
        self.planner_exchange = []
        self.planner_status = ""
        self.planner_metrics = {}
        self.judge_metrics = {"subtask/achieved": 0.0, "subtask/judge_failed": 0.0}
        self.subtask = ""
        self.answer = ""
        self.achieved = ""
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
        del global_step

        image = preprocess_image(obs["image"])
        self.prompt_builder.observe(obs, reward, info, image)
        prompt = self.prompt_builder.task_text()
        episode_done = terminated or truncated
        steps_since_subtask = self.steps_per_action - self.steps_until_next
        subtask_over = self.steps_until_next == 0 or episode_done
        judged = subtask_over and self.step_in_episode > 0
        if judged:
            self._judge_subtask(steps_since_subtask)
        next_due = self.steps_until_next == 0 and not episode_done
        if next_due:
            self._write_subtask()
        self.actor_builder.observe(obs, reward, info, image)

        if next_due:
            self._write_action()
            self.steps_until_next = self.steps_per_action
        steps_since_write = self.steps_per_action - self.steps_until_next
        action = (
            self.held_action
            if steps_since_write < self.hold_steps
            else np.zeros(self.action_dim, dtype=np.float32)
        )
        self.steps_until_next = 0 if episode_done else self.steps_until_next - 1
        self.step_in_episode = 0 if episode_done else self.step_in_episode + 1

        texts = {}
        if judged or next_due:
            texts = {
                "prompt": prompt,
                "achieved": self.achieved if judged else "",
                "subtask": self.subtask if next_due else "",
                "answer": self.answer if next_due else "",
            }

        panels = {
            "planner": render_conversation_panel(
                self.planner_exchange,
                self.planner_status,
                self.PANEL_WIDTH,
                self.PANEL_HEIGHT,
            ),
            "conversation": render_conversation_panel(
                self.held_exchange,
                self.held_status,
                self.PANEL_WIDTH,
                self.PANEL_HEIGHT,
            ),
        }
        return StepResult(
            action=action,
            metrics=self.held_metrics,
            panels=panels,
            texts=texts,
        )

    def _judge_subtask(self, steps: int) -> None:
        """Ask how far the subtask that just ran for ``steps`` steps got, off
        every frame of those steps: all it stood for, or what the episode's end
        left it. A reply without a number counts as nothing achieved."""
        request_start = time.time()
        response = self.backend.generate(
            self.prompt_builder.judge_conversation(self.subtask, steps)
        )
        achieved_match = ACHIEVED_RE.search(response.text)
        self.achieved = response.text
        self.judge_metrics = {
            "subtask/achieved": (
                float(np.clip(float(achieved_match.group(1)), 0.0, 1.0))
                if achieved_match is not None
                else 0.0
            ),
            "subtask/judge_failed": float(achieved_match is None),
            "vlm/judge_msec": (time.time() - request_start) * 1000,
            "vlm/judge_prompt_tokens": float(response.prompt_tokens),
        }

    def _write_subtask(self) -> None:
        """Generate on the planner's conversation and hand what it named to the
        actor, whose turn this step then carries it. A reply without the section
        is the subtask as it stands."""
        request_start = time.time()
        conversation = self.prompt_builder.conversation()
        response = self.backend.generate(conversation)
        planner_msec = (time.time() - request_start) * 1000

        self.planner_exchange = conversation + [assistant_turn(response.text)]
        subtask_match = SUBTASK_RE.search(response.text)
        self.subtask = (
            subtask_match.group(1).strip() if subtask_match is not None else response.text
        )
        self.prompt_builder.add_reply(response.text)
        self.actor_builder.set_subtask(self.subtask)
        self.planner_status = (
            f"in {response.prompt_tokens} tok   out {response.completion_tokens} tok   "
            f"{planner_msec:.0f} ms   {response.finish_reason}"
        )
        self.planner_metrics = {
            "vlm/planner_msec": planner_msec,
            "vlm/planner_prompt_tokens": float(response.prompt_tokens),
            "vlm/planner_completion_tokens": float(response.completion_tokens),
        }

    def _write_action(self) -> None:
        """Generate on this step's conversation and hold what it decided.

        What the generation cost is held with it, since the steps that run on an
        action are not the steps that paid for it.
        """
        request_start = time.time()
        conversation = self.actor_builder.conversation()
        response = self.backend.generate(conversation)
        api_msec = (time.time() - request_start) * 1000

        response_text = response.text
        self.held_exchange = conversation + [assistant_turn(response_text)]
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
        # One row per step the reply asked for, cut at the next generation.
        self.hold_steps = min(len(action_array), self.steps_per_action)
        self.answer = answer_text

        # The reply is handed back as written, <think> section and all, so the
        # conversation is the whole record of what the model said -- what the
        # render panel draws is then what the model itself reads. A reply that
        # named no runnable action is answered by the env in its own turn.
        self.actor_builder.add_reply(response_text)
        if not parse_ok:
            self.actor_builder.reject(answer_text)

        self.held_metrics = {
            **self.judge_metrics,
            **self.planner_metrics,
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
            self.actor_builder.reset()
            self.held_action = np.zeros(self.action_dim, dtype=np.float32)
            self.hold_steps = 0
            self.held_exchange = []
            self.held_status = ""
            self.held_metrics = {}
            self.planner_exchange = []
            self.planner_status = ""
            self.subtask = ""
            self.answer = ""
            self.achieved = ""
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
