# SPDX-License-Identifier: MIT
import re
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from gymnasium import Env
from omegaconf import DictConfig

TEXT_ACTION_PROTOCOL = "Reply with the action only."


class PromptBuilder(ABC):
    """環境ごとの、VLM に読ませる言葉と、返答に書かせる行動の決まり。

    毎ステップ ``observe`` で、エピソードを通して変わらないタスクの文と、そのステップの
    フレームに添える数値の文を作る。会話そのものは VLM の KV キャッシュが持つので、ここには
    残さない。返答に書かせる行動の名前と、返答の文から行動を読む関数もここが持つ。
    """

    def __init__(self, env: Env, thinking_interval: int) -> None:
        # env から読むものは環境ごとに違う（Animal-AI はアリーナの一覧）
        del env
        assert thinking_interval >= 0, thinking_interval
        # 思考を書かせる間隔（ステップ数）。思考を書かせない設定では 0
        self.thinking_interval = thinking_interval
        self._task_text = ""
        self._turn_text = ""
        # エピソードの中で何ステップ目か
        self._tick = 0

    def reset(self) -> None:
        """A finished episode ends the conversation; the next one opens its own."""
        self._tick = 0

    def observe(
        self, obs: dict[str, Any], reward: float, info: dict, previous_action: np.ndarray
    ) -> None:
        """このステップのタスクの文と数値の文を作る。``previous_action`` は前のステップに
        実際に環境へ渡された行動（誰が選んだかによらない）。"""
        self._task_text = self._task(obs, info)
        self._turn_text = self._turn(obs, reward, info, previous_action)
        self._tick += 1

    def turn_text(self) -> str:
        """This tick's own text, the live numbers under the frame."""
        return self._turn_text

    def on_reply(self) -> None:
        """ネットワークがこのステップに返答を書いたことを知らせる。返答をまたいで数値を
        まとめる環境だけが使う。"""

    def task_text(self) -> str:
        """The standing task: what the policy reads, and what it tokenizes.

        The same string on every tick of an episode, so a network that tokenizes
        it gets the same token ids throughout. The live numbers are not in it;
        they reach the policy through the scalar branch and the chain through
        the turns.
        """
        return self._task_text

    def _protocol(self) -> str:
        """返答の書式。思考を書かせるときは、その間隔と、思考を何に使うかを先に伝える。
        思考のあとに書くのは1ステップの行動だけなので、それまでの振り返りと、次に考える
        までの方針は、ここで求めておく。"""
        if self.thinking_interval == 0:
            return TEXT_ACTION_PROTOCOL
        return (
            f"On the first step and every {self.thinking_interval} steps after it you think "
            f"before you reply: use that to review the frames and the actions taken since you "
            f"last thought, and to decide what to do until you next think. On the other steps "
            f"you reply at once, carrying on with what you last decided. {TEXT_ACTION_PROTOCOL}"
        )

    def reflect(self, score: float, answer) -> str | None:
        """Rewrite the ended episode's task instruction off a reflection the
        model writes, when the builder holds a per-task instruction to rewrite,
        and return the new instruction (None when nothing was rewritten).
        ``answer`` puts a request to the model on top of the episode's
        conversation and returns its reply text. The base builder holds none,
        so this is a no-op."""
        del score, answer
        return None

    @abstractmethod
    def text_action_names(self) -> list[str]:
        """返答に書かせる行動の名前。数値で書かせる環境では空で、返答を名前に絞らない。"""

    @abstractmethod
    def parse_action_text(self, action_text: str) -> tuple[np.ndarray, bool]:
        """返答の行動の文を環境の行動にしたものと、読めたかどうか。"""

    @abstractmethod
    def _task(self, obs: dict[str, Any], info: dict) -> str:
        """The standing task: what the env asks, unchanged through an episode."""

    @abstractmethod
    def _turn(
        self, obs: dict[str, Any], reward: float, info: dict, previous_action: np.ndarray
    ) -> str:
        """What this tick alone says: the live numbers under the frame."""


# --- CarRacing / CARLA -------------------------------------------------------

# CarRacing と CARLA の行動の書式。どちらも行動は (steer, gas_or_brake) で、
# SteerAccelPromptBuilder.parse_action_text がこれを読む
STEER_ACCEL_ACTION_FORMAT = (
    "Write the action as `steer=<value>, accel=<value>`, where each <value> is a float in [-1, 1]."
)


class SteerAccelPromptBuilder(PromptBuilder):
    """行動を ``steer=<value>, accel=<value>`` の数値で書かせる環境（CarRacing と CARLA）。"""

    def text_action_names(self) -> list[str]:
        # 行動は数値で書かせるので、決まった候補はない
        return []

    def parse_action_text(self, action_text: str) -> tuple[np.ndarray, bool]:
        """Decode `steer=<value>, accel=<value>` into the one action it names."""
        match = re.search(r"steer=([+-]?\d*\.?\d+),\s*accel=([+-]?\d*\.?\d+)", action_text)
        if match is None:
            return np.zeros(2, dtype=np.float32), False
        action = np.array([float(match.group(1)), float(match.group(2))], dtype=np.float32)
        return np.clip(action, -1.0, 1.0), True


# --- CarRacing ---------------------------------------------------------------

CAR_RACING_TEXT_ACTION_PROMPT = (
    "You control the red car in CarRacing-v3 (top-down). Stay on the gray road "
    "and avoid going onto the green grass; hug the road center when possible. "
    f"{STEER_ACCEL_ACTION_FORMAT}"
)


class CarRacingPromptBuilder(SteerAccelPromptBuilder):
    """CarRacing."""

    def _task(self, obs: dict[str, Any], info: dict) -> str:
        del obs, info
        return f"{CAR_RACING_TEXT_ACTION_PROMPT}"

    def _turn(
        self, obs: dict[str, Any], reward: float, info: dict, previous_action: np.ndarray
    ) -> str:
        del obs, reward, info, previous_action
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
    "A sphere's reward equals its size: between goals the bigger one is worth more. "
    "Turn until the target is at the center of the view, then move forward "
    "with keeping it centered."
)
# Animal-AI's native action is MultiDiscrete([3, 3]): one move (noop / forward
# / back) and one rotation (noop / right / left) per tick. The env exposes it as
# Box(-1, 1, shape=(2,)) and discretizes back with a +/-1/3 dead-zone, so each
# named action maps onto the extreme Box value that survives that dead-zone.
# An action is written as one of four names, each a move or a rotation alone,
# taken for one step.
ANIMALAI_ACTIONS = {
    "move_forward": (1.0, 0.0),
    "move_backward": (-1.0, 0.0),
    "turn_right": (0.0, 1.0),
    "turn_left": (0.0, -1.0),
}

ANIMALAI_REFLECTION_REQUEST = (
    "The episode just ended in {outcome}: return {score:+.3f} against pass mark "
    '{pass_mark:+.3f}. Your standing instruction for this task so far: "{task}" '
    "Write the instruction to follow on the next attempt at this task, keeping "
    "what worked and fixing what did not. Reply with the "
    "instruction only, at most three sentences."
)
# これより長い指示は、常設のプロンプトを膨らませるので採らない
ANIMALAI_INSTRUCTION_CHARS_MAX = 500


def _animalai_turn(obs: dict[str, Any], reward: float, average_forward_speed: float) -> str:
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
    def __init__(self, env: Env, thinking_interval: int) -> None:
        super().__init__(env, thinking_interval)
        self.tasks = {arena.task: "" for arena in env.unwrapped.selector.arenas}
        self._task_key = ""
        self._pass_mark = 0.0
        self._forward_speed_sum = 0.0
        self._forward_speed_steps = 0

    def reset(self) -> None:
        super().reset()
        self._forward_speed_sum = 0.0
        self._forward_speed_steps = 0

    def on_reply(self) -> None:
        """Start the forward speed's average afresh, so the next turn reports
        what came of this answer."""
        self._forward_speed_sum = 0.0
        self._forward_speed_steps = 0

    def text_action_names(self) -> list[str]:
        return list(ANIMALAI_ACTIONS)

    def parse_action_text(self, action_text: str) -> tuple[np.ndarray, bool]:
        """Decode the action's name into the Box action that the env discretizes
        back into Animal-AI's native MultiDiscrete([3, 3]) pair. Anything that is
        not exactly one of the names stays a format violation, reported as such
        rather than repaired here."""
        name = action_text.strip().lower()
        if name not in ANIMALAI_ACTIONS:
            return np.zeros(2, dtype=np.float32), False
        return np.array(ANIMALAI_ACTIONS[name], dtype=np.float32), True

    def _action_format(self) -> str:
        return (
            f"Write the action as one of {' / '.join(ANIMALAI_ACTIONS)}. It is taken for one "
            f"step, and you are then shown the next frame and asked again. One turn step "
            f"rotates 10 degrees; one forward step moves about 1.5 units, the arena being "
            f"40 units across. Each frame comes with the action that was actually taken on "
            f"the step before it, as two numbers: move (+1 forward, -1 backward) and turn "
            f"(+1 right, -1 left). It can differ from the action you named."
        )

    def _task(self, obs: dict[str, Any], info: dict) -> str:
        del obs
        self._task_key = info["arena_name"].rsplit("-", 1)[0]
        task = self.tasks[self._task_key]
        task_sentence = f"Task: {task} " if task != "" else ""
        return f"{ANIMALAI_FRAMING} {self._action_format()} {task_sentence}{self._protocol()}"

    def reflect(self, score: float, answer) -> str | None:
        if self._task_key == "":
            return None
        request = ANIMALAI_REFLECTION_REQUEST.format(
            outcome="success" if score >= self._pass_mark else "failure",
            score=score,
            pass_mark=self._pass_mark,
            task=self.tasks[self._task_key],
        )
        reply = answer(request)
        if reply == "" or len(reply) > ANIMALAI_INSTRUCTION_CHARS_MAX:
            return None
        self.tasks[self._task_key] = reply
        return reply

    def _turn(
        self, obs: dict[str, Any], reward: float, info: dict, previous_action: np.ndarray
    ) -> str:
        del info
        self._pass_mark = float(obs["pass_mark"][0])
        self._forward_speed_sum += float(obs["velocity"][2])
        self._forward_speed_steps += 1
        scalars = _animalai_turn(obs, reward, self._forward_speed_sum / self._forward_speed_steps)
        if self._tick == 0:
            # エピソードの最初のステップには、その前に取られた行動がない
            return scalars
        return (
            f"Action taken: move {previous_action[0]:+.2f}, turn {previous_action[1]:+.2f}. "
            f"{scalars}"
        )


# --- CARLA -------------------------------------------------------------------

CARLA_TEXT_ACTION_FRAMING = (
    "Drive a car along a route in CARLA. Follow the planned route, "
    "obey traffic rules, and avoid collisions."
)
# CARLA RoadOption (agents.navigation.local_planner.RoadOption) -> the upcoming
# maneuver sentence, so navigation intent reaches the policy as language. The
# env reports the raw command through ``info["maneuver_command"]``; VOID (-1) is
# what it reports when there is no route to read a maneuver off.
CARLA_TEXT_ACTION_MANEUVER = {
    -1: "The ego vehicle is following the lane straight ahead.",  # VOID
    1: "The ego vehicle is turning left at the upcoming intersection.",  # LEFT
    2: "The ego vehicle is turning right at the upcoming intersection.",  # RIGHT
    3: "The ego vehicle is going straight through the upcoming intersection.",  # STRAIGHT
    4: "The ego vehicle is following the lane straight ahead.",  # LANEFOLLOW
    5: "The ego vehicle is changing to the left lane.",  # CHANGELANELEFT
    6: "The ego vehicle is changing to the right lane.",  # CHANGELANERIGHT
}


class CarlaPromptBuilder(SteerAccelPromptBuilder):
    """CARLA."""

    def _task(self, obs: dict[str, Any], info: dict) -> str:
        del obs, info
        return f"{CARLA_TEXT_ACTION_FRAMING} {STEER_ACCEL_ACTION_FORMAT} {self._protocol()}"

    def _turn(
        self, obs: dict[str, Any], reward: float, info: dict, previous_action: np.ndarray
    ) -> str:
        del obs, reward, previous_action
        return CARLA_TEXT_ACTION_MANEUVER[info["maneuver_command"]]


# One builder per environment, whoever ends up acting on what it says.
PROMPT_BUILDERS = {
    "CarRacing-v3": CarRacingPromptBuilder,
    "AnimalAI-v0": AnimalAIPromptBuilder,
    "CARLA-Leaderboard-v0": CarlaPromptBuilder,
}


def build_prompt_builder(env: Env, args: DictConfig) -> PromptBuilder:
    assert args.env_id in PROMPT_BUILDERS, f"No prompt builder for {args.env_id}"
    # 思考を書かせない設定では、思考の間隔もプロンプトに書かない
    thinking_interval = 0 if args.vla.reasoning_effort == "none" else args.vla.thinking_interval
    return PROMPT_BUILDERS[args.env_id](env, thinking_interval)
