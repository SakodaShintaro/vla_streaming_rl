# SPDX-License-Identifier: MIT
import time

import numpy as np

from vla_streaming_rl.agents.prompt import ACHIEVED_RE, SUBTASK_RE, PromptBuilder, assistant_turn


class SubtaskCycle:
    """The slow loop every agent that reads a VLM shares: what is judged, what is
    planned, and on which steps.

    Every ``steps_per_subtask`` steps the model is asked twice. As the judge it
    reads every frame of the steps the last subtask stood for and says how far
    that subtask got; as the planner it reads what the env asks and names the
    next subtask. An episode's end judges the subtask it cut short, off the
    frames it did run for, and plans nothing on the terminal frame. Who then
    acts on the subtask -- the same model asked a third time, or a trained
    policy -- is the agent's own business, which is the one thing the agents
    differ in.

    ``generate`` takes a conversation and returns whatever the caller's
    generator returns, read here only for its ``text``; the planner's is kept
    whole as ``plan``, so a caller that reads more than text off it can.
    """

    def __init__(self, generate, prompt_builder: PromptBuilder, steps_per_subtask: int) -> None:
        assert steps_per_subtask >= 1, steps_per_subtask
        self.generate = generate
        self.prompt_builder = prompt_builder
        self.steps_per_subtask = steps_per_subtask
        self.reset()

    def reset(self) -> None:
        # Zero means "plan now", so the first step of an episode plans on that
        # episode's own first frame.
        self._until_next = 0
        self._step_in_episode = 0
        self.judged = False
        self.planned = False
        self.subtask = ""
        self.achieved = 0.0
        self.judge_failed = False
        self.judge_text = ""
        self.judge_msec = 0.0
        self.plan = None
        self.plan_msec = 0.0
        self.exchange = []

    def age(self) -> int:
        """How many steps ago the subtask now standing was written: 0 on the
        step that wrote it."""
        return self.steps_per_subtask - 1 - self._until_next

    def advance(self, episode_done: bool) -> None:
        """One environment step, after the builder has observed it. Sets
        ``judged`` and ``planned`` to what this step did."""
        steps_since_subtask = self.steps_per_subtask - self._until_next
        subtask_over = self._until_next == 0 or episode_done
        self.judged = subtask_over and self._step_in_episode > 0
        if self.judged:
            self._judge(steps_since_subtask)
        self.planned = self._until_next == 0 and not episode_done
        if self.planned:
            self._plan()
            self._until_next = self.steps_per_subtask
        self._until_next = 0 if episode_done else self._until_next - 1
        self._step_in_episode = 0 if episode_done else self._step_in_episode + 1

    def _judge(self, steps: int) -> None:
        """A reply without a number counts as nothing achieved."""
        start = time.perf_counter()
        reply = self.generate(self.prompt_builder.judge_conversation(self.subtask, steps))
        self.judge_msec = (time.perf_counter() - start) * 1000.0
        match = ACHIEVED_RE.search(reply.text)
        self.judge_text = reply.text
        self.judge_failed = match is None
        self.achieved = float(np.clip(float(match.group(1)), 0.0, 1.0)) if match else 0.0

    def _plan(self) -> None:
        """A reply without the section is the subtask as it stands."""
        start = time.perf_counter()
        conversation = self.prompt_builder.conversation()
        self.plan = self.generate(conversation)
        self.plan_msec = (time.perf_counter() - start) * 1000.0
        match = SUBTASK_RE.search(self.plan.text)
        self.subtask = match.group(1).strip() if match else self.plan.text
        self.exchange = conversation + [assistant_turn(self.plan.text)]
        self.prompt_builder.add_reply(self.plan.text)
