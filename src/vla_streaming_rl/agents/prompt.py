# SPDX-License-Identifier: MIT
"""Language input composed on the agent side.

The environment reports state; the agent decides what to say about it. Every
string a policy reads as language is built here out of the structured
observation the env already publishes, so the prompt belongs to the run's agent
config rather than to the simulator: two agents can drive the same env with
different framing, and the env carries no text of its own.

There is one builder per environment, and it always writes the prompt of the
high-level policy: what the env asks, how often it is asked, and the sections
the reply is read out of. The reply names no action; the low-level policy acts
on every step, reading the reply's subtask.

A builder is called once per environment step with the observation, the reward
and the info the agent itself received, and returns the conversation as it then
stands: the standing task, every turn a chain has already answered, and this
tick's own turn. Holding the conversation here is what keeps the VLM modules
free of any environment's vocabulary -- they run a model over what they are
handed and compose no text of their own beyond how a chain is continued.
"""

import re
from abc import ABC, abstractmethod
from typing import Any

from gymnasium import Env
from omegaconf import DictConfig

# 返答の先頭の採点の区間。前の返答からの行動が、そのサブタスクの実現に向けてどれだけ
# 適切だったかを 0〜9 の1桁で書かせ、その位置で各数字を選ぶ確率の期待値から点数を読む
SCORE_TAG = "<score>"

REPLY_PROTOCOL = (
    "Before anything else, compare the frames since your previous reply with the subtask of "
    "that reply and judge how far the agent's actions moved toward it. "
    f"Reply with {SCORE_TAG}an integer from 0 to 9 for that judgment: 0 if the actions "
    "worked against the subtask, 9 if they got it done</score> "
    "then <subtask>one short sentence on what the agent should get done by your next "
    "reply</subtask>."
)

# 最後の <subtask> を読む。思考の中でタグを引用することがあり、最初のものを読むと
# 思考全体をサブタスクとして取ってしまう
SUBTASK_RE = re.compile(r"<subtask>(?!.*<subtask>)(.*?)</subtask>", re.DOTALL)


def assistant_turn(text: str) -> dict:
    """A reply as the message the conversation holds it as."""
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def user_turn(seconds: float, image, text: str) -> dict:
    """A turn the env puts to the agent: the frame under its timestamp, then the
    turn's text. The timestamp comes first and is written the way the model's
    video frames carry theirs, which is the order it learned them in."""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": f"<{seconds:.1f} seconds>"},
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ],
    }


class PromptBuilder(ABC):
    """The conversation this run's VLMs read, carried across environment steps.

    Every tick ``observe`` records what the agent is looking at. A chain reads
    that through ``conversation`` on the steps it actually writes -- one step in
    ``steps_per_reply`` -- and hands back what it wrote through
    ``add_reply``, which is what puts the turn it answered into the conversation
    for good. Between replies, every ``steps_per_observation`` ticks the turn is
    kept as a user turn with no reply (``add_observation``); the other ticks are
    overwritten rather than accumulated, so the conversation holds the turns a
    chain saw and not every step of the run.

    ``history_turns`` is how many of those exchanges it keeps. Every turn still
    held is re-read on every step that follows, so a conversation left to grow
    charges the whole episode for its own beginning; the oldest exchange is
    dropped instead.
    """

    def __init__(
        self, env: Env, history_turns: int, steps_per_reply: int, steps_per_observation: int
    ) -> None:
        assert history_turns >= 0, history_turns
        assert steps_per_reply >= 1, steps_per_reply
        assert steps_per_observation >= 1, steps_per_observation
        self.history_turns = history_turns
        self.steps_per_reply = steps_per_reply
        # 返答を書かないステップでも、この間隔で観測を会話に残す
        self.steps_per_observation = steps_per_observation
        self.decision_fps = env.metadata["decision_fps"]
        self._turns = []
        self._current = {}
        self._task_text = ""
        self._tick = 0

    def reset(self) -> None:
        """A finished episode ends the conversation; the next one opens its own."""
        self._turns = []
        self._current = {}
        self._tick = 0

    def observe(self, obs: dict[str, Any], reward: float, info: dict, image) -> None:
        """What the agent is looking at this tick: the turn a chain would read if
        it wrote one now. Overwritten every step until one does. Its timestamp
        is the episode's clock at the rate the env takes decisions."""
        self._task_text = self._task(obs, info)
        self._current = user_turn(
            self._tick / self.decision_fps, image, self._turn(obs, reward, info)
        )
        self._tick += 1

    def conversation(self) -> list[dict]:
        """What a chain about to write reads: the standing task, the turns it has
        already answered, and the turn it is being asked about now."""
        opening = {"role": "system", "content": [{"type": "text", "text": self._task_text}]}
        return [opening] + self._turns + [self._current]

    def turn_text(self) -> str:
        """This tick's own text, the live numbers under the frame. What a learner
        stores beside the frame, so the turn can be rebuilt from its row later."""
        return self._current["content"][2]["text"]

    def add_reply(self, text: str) -> None:
        """What the chain wrote about that turn, which settles the pair into the
        conversation, dropping the oldest exchange once ``history_turns`` are
        held."""
        turns = self._turns + [self._current, assistant_turn(text)]
        self._turns = turns[max(0, len(turns) - 2 * self.history_turns) :]

    def add_observation(self) -> None:
        """返答を書かないステップの観測を、返答のない user のターンとして会話に残す。
        次に返答を書くとき、その間の観測をまとめて読めるようにするため。"""
        turns = self._turns + [self._current]
        self._turns = turns[max(0, len(turns) - 2 * self.history_turns) :]

    def task_text(self) -> str:
        """The standing task: what the policy reads, and what it tokenizes.

        The same string on every tick of an episode, so a network that tokenizes
        it gets the same token ids throughout. The live numbers are not in it;
        they reach the policy through the scalar branch and the chain through
        the turns.
        """
        return self._task_text

    def close_episode(self, text: str) -> None:
        """End the episode inside the conversation instead of dropping it: the
        turns stay and ``text`` says how it went, so the attempt that follows
        reads what the ones before it did and what came of them."""
        self._turns = self._turns + [{"role": "user", "content": [{"type": "text", "text": text}]}]
        self._current = {}

    @abstractmethod
    def _task(self, obs: dict[str, Any], info: dict) -> str:
        """The standing task: what the env asks, unchanged through an episode."""

    @abstractmethod
    def _turn(self, obs: dict[str, Any], reward: float, info: dict) -> str:
        """What this tick alone says: the live numbers under the frame."""


# --- CarRacing ---------------------------------------------------------------

CAR_RACING_FRAMING = (
    "You control the red car in CarRacing-v3 (top-down). Stay on the gray road "
    "and avoid going onto the green grass; hug the road center when possible."
)


class CarRacingPromptBuilder(PromptBuilder):
    """CarRacing."""

    def _task(self, obs: dict[str, Any], info: dict) -> str:
        del obs, info
        return f"{CAR_RACING_FRAMING}"

    def _turn(self, obs: dict[str, Any], reward: float, info: dict) -> str:
        del obs, reward, info
        return ""


# --- Animal-AI ---------------------------------------------------------------

ANIMALAI_FRAMING = (
    "You control the agent in Animal-AI (first-person view). "
    "Health drains every step and the episode fails when it reaches 0. "
    "Green and yellow spheres are goals that give reward: green ones look "
    "yellow-green and touching one gives its reward and ends the episode; yellow "
    "ones are bright yellow and the episode goes on after touching them. Red "
    "spheres and red zones are penalties that end the episode as a failure, never "
    "touch them. "
    "A sphere's reward equals its size: between goals the bigger one is worth "
    "more, and a bigger red sphere costs more. "
    "Walls are never the goal, whatever their color. "
    "Turn until the target is at the center of the view, then move forward in "
    "bursts short enough to keep it centered, turning to re-center it whenever "
    "it drifts. A target that slips out of view is usually right beside or "
    "behind you: turn back to that same target rather than switching to "
    "another. Move to a new spot only if a full turn shows nothing. "
)


def _animalai_turn(obs: dict[str, Any], reward: float, average_forward_speed: float) -> str:
    """Animal-AI's live scalars: the numbers the frame cannot show.

    The forward speed is the average over the steps since the last answer
    rather than this tick's own, so it reports what came of that answer's
    subtask rather than the motion of one moment.

    Read off the observation the network's scalar branch is fed, cut down to
    what a reply can act on: of the three velocity components only the forward
    one, since a model handed all three reads the vector as motion the animal
    cannot have -- it reported flying and ascending off a standing still frame.
    The lateral and vertical components still reach the policy through the
    scalar branch. These are what the env reports, not how the run frames it.
    """
    return (
        f"Average forward speed since your last answer: {average_forward_speed:+.2f}. "
        f"Reward: {reward:+.3f}. "
        f"Return so far: {obs['episode_return'][0]:+.3f}. "
        f"Pass mark: {obs['pass_mark'][0]:+.3f}. "
        f"Return needed: {obs['remaining_return'][0]:+.3f}. "
        f"Health: {obs['health'][0]:.2f}. "
        f"Global step: {int(obs['global_step'][0])}. "
        f"Episode step: {int(obs['episode_step'][0])}."
    )


class AnimalAIPromptBuilder(PromptBuilder):
    """Animal-AI: the framing and the live scalars, the same words on every
    task -- the prompt carries no per-task knowledge."""

    def __init__(
        self, env: Env, history_turns: int, steps_per_reply: int, steps_per_observation: int
    ) -> None:
        super().__init__(env, history_turns, steps_per_reply, steps_per_observation)
        self._forward_speed_sum = 0.0
        self._forward_speed_steps = 0

    def reset(self) -> None:
        super().reset()
        self._forward_speed_sum = 0.0
        self._forward_speed_steps = 0

    def add_reply(self, text: str) -> None:
        """Settle the reply and start the forward speed's average afresh, so the
        next turn reports what came of this answer."""
        super().add_reply(text)
        self._forward_speed_sum = 0.0
        self._forward_speed_steps = 0

    def _cadence(self) -> str:
        return (
            f"You are shown a new frame every {self.steps_per_observation} steps and asked "
            f"for a new reply every {self.steps_per_reply} steps; in between, the agent "
            f"acts on its own toward the subtask of your latest reply."
        )

    def _task(self, obs: dict[str, Any], info: dict) -> str:
        del obs, info
        return f"{ANIMALAI_FRAMING} {self._cadence()} {REPLY_PROTOCOL}"

    def _turn(self, obs: dict[str, Any], reward: float, info: dict) -> str:
        del info
        self._forward_speed_sum += float(obs["velocity"][2])
        self._forward_speed_steps += 1
        return _animalai_turn(obs, reward, self._forward_speed_sum / self._forward_speed_steps)


# --- CARLA -------------------------------------------------------------------

CARLA_FRAMING = (
    "Drive a car along a route in CARLA. Follow the planned route, "
    "obey traffic rules, and avoid collisions."
)
# CARLA RoadOption (agents.navigation.local_planner.RoadOption) -> the upcoming
# maneuver sentence, so navigation intent reaches the policy as language. The
# env reports the raw command through ``info["maneuver_command"]``; VOID (-1) is
# what it reports when there is no route to read a maneuver off.
CARLA_MANEUVER = {
    -1: "The ego vehicle is following the lane straight ahead.",  # VOID
    1: "The ego vehicle is turning left at the upcoming intersection.",  # LEFT
    2: "The ego vehicle is turning right at the upcoming intersection.",  # RIGHT
    3: "The ego vehicle is going straight through the upcoming intersection.",  # STRAIGHT
    4: "The ego vehicle is following the lane straight ahead.",  # LANEFOLLOW
    5: "The ego vehicle is changing to the left lane.",  # CHANGELANELEFT
    6: "The ego vehicle is changing to the right lane.",  # CHANGELANERIGHT
}


class CarlaPromptBuilder(PromptBuilder):
    """CARLA."""

    def _task(self, obs: dict[str, Any], info: dict) -> str:
        del obs, info
        return f"{CARLA_FRAMING} {REPLY_PROTOCOL}"

    def _turn(self, obs: dict[str, Any], reward: float, info: dict) -> str:
        del obs, reward
        return CARLA_MANEUVER[info["maneuver_command"]]


# One builder per environment, whoever ends up acting on what it says.
PROMPT_BUILDERS = {
    "CarRacing-v3": CarRacingPromptBuilder,
    "AnimalAI-v0": AnimalAIPromptBuilder,
    "CARLA-Leaderboard-v0": CarlaPromptBuilder,
}


def build_prompt_builder(env: Env, args: DictConfig) -> PromptBuilder:
    assert args.env_id in PROMPT_BUILDERS, f"No prompt builder for {args.env_id}"
    # 高レベル方策の窓 high_level.seq_len に、チェーンの周期で収まるやり取りの数
    history_turns = (args.high_level.seq_len - 1) // args.high_level.steps_per_reply
    return PROMPT_BUILDERS[args.env_id](
        env, history_turns, args.high_level.steps_per_reply, args.high_level.steps_per_observation
    )
