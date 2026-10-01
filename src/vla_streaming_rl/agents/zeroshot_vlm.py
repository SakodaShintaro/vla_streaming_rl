# SPDX-License-Identifier: MIT
"""Zero-shot VLM controller.

The high-level policy alone: the same chain module the trained agents carry
(`HighLevelPolicy`) writes the replies on the same conversation, and the action each reply names is held for as many
steps as it asks. It learns nothing: it is the zero-shot baseline the trained
agents are measured against, so it plugs into the same trainer loop and reports
the same telemetry.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from vla_streaming_rl.agents.base import (
    CONVERSATION_PANEL_HEIGHT,
    CONVERSATION_PANEL_WIDTH,
    Agent,
    StepResult,
)
from vla_streaming_rl.agents.prompt import PromptBuilder, read_action_reply
from vla_streaming_rl.networks.modules.high_level_policy import HighLevelPolicy
from vla_streaming_rl.utils import render_conversation_panel


class ZeroShotVLMAgent(Agent):
    def __init__(
        self,
        *,
        action_space: gym.spaces.Box,
        parse_action_text,
        chain: HighLevelPolicy,
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
        self.chain = chain
        # 返答が求めうる保持の長さの上限
        self.steps_per_action = steps_per_action

        self.action_space = action_space
        self.action_dim = int(np.prod(action_space.shape))
        self.parse_action_text = parse_action_text

        self.held_action = np.zeros(self.action_dim, dtype=np.float32)
        self.hold_steps = 0
        # 持っている返答の <action> を読めたか
        self.parse_ok = True

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

        self.prompt_builder.observe(obs, reward, info, torch.from_numpy(obs["image"]))
        prompt = self.prompt_builder.task_text()

        # 高レベル方策を進め、返答を確定したステップでその行動を読む
        reply = self.chain.advance()
        # 達成度は判定したステップでだけ記録する
        achieved_metrics = {}
        if reply.age == 0:
            # 確定した返答の <action> を読み、その行動と続けるステップ数を保持する。読め
            # なかったときは、行動として実行できず止まっていたことを伝える user の発言を
            # 会話に足す
            answer_text, self.held_action, self.hold_steps, self.parse_ok = read_action_reply(
                reply.text,
                self.parse_action_text,
                self.action_space.low,
                self.action_space.high,
                self.steps_per_action,
            )
            if not self.parse_ok:
                self.prompt_builder.reject(answer_text)
            if reply.achieved is not None:
                achieved_metrics = {"vlm/achieved": reply.achieved}
        elif episode_done:
            # エピソードが終わると実行中のサブタスクには次の返答が来ないので、終端の
            # フレームで判定する
            achieved_metrics = {"vlm/achieved": self.chain.judge_current()}
        action = (
            self.held_action
            if reply.age < self.hold_steps
            else np.zeros(self.action_dim, dtype=np.float32)
        )

        status = (
            f"in {reply.input_tokens} tok   out {reply.output_tokens} tok   "
            f"{reply.msec:.0f} ms   parse {'ok' if self.parse_ok else 'failed'}"
        )
        panels = {
            "conversation": render_conversation_panel(
                reply.exchange,
                status,
                CONVERSATION_PANEL_WIDTH,
                CONVERSATION_PANEL_HEIGHT,
            )
        }
        metrics = {
            "vlm/parse_failed": float(not self.parse_ok),
            "vlm/msec": reply.msec,
            "vlm/prompt_tokens": float(reply.input_tokens),
            "vlm/completion_tokens": float(reply.output_tokens),
            **achieved_metrics,
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
            # 次の advance がそのエピソードの最初のフレームで返答を書く
            self.chain.reset()
            self.held_action = np.zeros(self.action_dim, dtype=np.float32)
            self.hold_steps = 0
            self.held_exchange = []
            self.held_status = ""
            self.held_metrics = {}
            self.fresh_metrics = {}
        return {}

    def optimizer_state_dict(self) -> dict:
        # the baseline learns nothing; there is nothing to optimize
        return {}

    def load_optimizer_state_dict(self, state: dict) -> None:
        del state
