# SPDX-License-Identifier: MIT
"""Zero-shot VLM controller.

The agent builds one chat prompt per env step and hands it to a `VLMBackend`,
so the same protocol runs against a model hosted on OpenRouter or a Qwen3.5
checkpoint generating locally (see `vlm_backends`). It learns nothing: it is
the zero-shot baseline the trained agents are measured against, so it plugs
into the same trainer loop and reports the same telemetry.
"""

import time
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image

from vla_streaming_rl.agents.base import Agent, StepResult
from vla_streaming_rl.agents.prompt import PromptBuilder, assistant_turn, read_action_reply
from vla_streaming_rl.networks.modules.cot_stream import CoTStream
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
        self.hold_steps = 0
        self.held_exchange = []
        self.held_status = ""
        self.held_metrics = {}
        self.steps_until_next = 0
        self.replies_in_episode = 0
        self.fresh_metrics = {}

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
        episode_done = terminated or truncated

        image = preprocess_image(obs["image"])
        self.prompt_builder.observe(obs, reward, info, image)
        prompt = self.prompt_builder.task_text()

        if isinstance(self.backend, CoTStream):
            # stream: チェーンは毎ステップ書き進められ、確定したステップで行動を読む
            self.backend.advance()
            steps_since_write = self.backend.age()
            if steps_since_write == 0:
                self._hold_reply(
                    self.backend.text(),
                    self.backend.exchange(),
                    self.backend.stats(),
                    "stop",
                    self.backend.achieved(),
                )
        else:
            if self.steps_until_next == 0:
                self._write_action()
                self.steps_until_next = self.steps_per_action
            steps_since_write = self.steps_per_action - self.steps_until_next
            self.steps_until_next -= 1
        if episode_done and steps_since_write != 0 and self.replies_in_episode > 0:
            # エピソードが終わると実行中のサブタスクには次の返答が来ないので、終端の
            # フレームで判定する
            achieved = (
                self.backend.judge_current()
                if isinstance(self.backend, CoTStream)
                else self.backend.judge_current(self.prompt_builder.conversation())
            )
            if achieved is not None:
                self.fresh_metrics = {"vlm/achieved": achieved}
        action = (
            self.held_action
            if steps_since_write < self.hold_steps
            else np.zeros(self.action_dim, dtype=np.float32)
        )

        panels = {
            "conversation": render_conversation_panel(
                self.held_exchange,
                self.held_status,
                self.PANEL_WIDTH,
                self.PANEL_HEIGHT,
            )
        }
        # 達成度は返答を確定したステップでだけ記録する
        metrics = {**self.held_metrics, **self.fresh_metrics}
        self.fresh_metrics = {}
        return StepResult(
            action=action,
            metrics=metrics,
            panels=panels,
            texts={"prompt": prompt},
        )

    def _write_action(self) -> None:
        """Generate on this step's conversation and hold what it decided.

        What the generation cost is held with it, since the steps that run on an
        action are not the steps that paid for it.
        """
        request_start = time.time()
        conversation = self.prompt_builder.conversation()
        response = self.backend.generate(conversation)
        api_msec = (time.time() - request_start) * 1000

        # The reply is handed back as written, <think> section and all, so the
        # conversation is the whole record of what the model said -- what the
        # render panel draws is then what the model itself reads.
        self.prompt_builder.add_reply(response.text)
        self._hold_reply(
            response.text,
            conversation + [assistant_turn(response.text)],
            {
                "input_tokens": response.prompt_tokens,
                "output_tokens": response.completion_tokens,
                "msec": api_msec,
            },
            response.finish_reason,
            # エピソードの最初の返答には判定すべき前のサブタスクがない
            response.achieved if self.replies_in_episode > 0 else None,
        )

    def _hold_reply(
        self,
        text: str,
        exchange: list[dict],
        stats: dict,
        finish_reason: str,
        achieved: float | None,
    ) -> None:
        """返答の <action> を読み、その行動と続けるステップ数を保持する。読めなかった
        ときは、行動として実行できず止まっていたことを伝える user の発言を会話に足す。"""
        self.held_exchange = exchange
        answer_text, self.held_action, self.hold_steps, parse_ok = read_action_reply(
            text,
            self.parse_action_text,
            self.action_space.low,
            self.action_space.high,
            self.steps_per_action,
        )
        if not parse_ok:
            self.prompt_builder.reject(answer_text)
        self.held_metrics = {
            "vlm/parse_failed": float(not parse_ok),
            "vlm/api_msec": stats["msec"],
            "vlm/prompt_tokens": float(stats["input_tokens"]),
            "vlm/completion_tokens": float(stats["output_tokens"]),
        }
        self.held_status = (
            f"in {stats['input_tokens']} tok   out {stats['output_tokens']} tok   "
            f"{stats['msec']:.0f} ms   parse {'ok' if parse_ok else 'failed'}   "
            f"{finish_reason}"
        )
        self.replies_in_episode += 1
        if achieved is not None:
            self.fresh_metrics = {"vlm/achieved": achieved}

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
            if isinstance(self.backend, CoTStream):
                self.backend.reset()
            else:
                self.backend.reset_cache()
            self.held_action = np.zeros(self.action_dim, dtype=np.float32)
            self.hold_steps = 0
            self.held_exchange = []
            self.held_status = ""
            self.held_metrics = {}
            # Zero means "generate now", so the first step of an episode decides
            # on that episode's own first frame.
            self.steps_until_next = 0
            self.replies_in_episode = 0
            self.fresh_metrics = {}
        return {}

    def optimizer_state_dict(self) -> dict:
        # the baseline queries a hosted model; there is nothing to optimize
        return {}

    def load_optimizer_state_dict(self, state: dict) -> None:
        del state
