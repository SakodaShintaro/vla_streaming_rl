# SPDX-License-Identifier: MIT
"""Language input composed on the agent side.

The environment reports state; the agent decides what to say about it. Every
string a policy reads as language is built here out of the structured
observation the env already publishes, so the prompt belongs to the run's agent
config rather than to the simulator: two agents can drive the same env with
different framing, and the env carries no text of its own.

There is one builder per environment, and it always writes the prompt of an
agent about to act: what the env asks, the action vocabulary it is asked in, the
arena's own instruction, and the three sections the answer is read out of. Whether
the action then comes from the reply or from a policy head is the reader's
business, not the prompt's -- a run that reads the language as conditioning is
reading the same words a run that acts on it would, so the two are comparable
without a second wording to keep in step.

A builder is called once per environment step with the observation, the reward
and the info the agent itself received, and returns the conversation as it then
stands: the standing task, every turn a chain has already answered, and this
tick's own turn. Holding the conversation here is what keeps the VLM modules
free of any environment's vocabulary -- they run a model over what they are
handed and compose no text of their own beyond how a chain is continued.
"""

import csv
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from gymnasium import Env
from omegaconf import DictConfig

ARENA_TASK_CSV = Path("./external/animal-ai/configs/AnimalAI_prompt.csv")

TEXT_ACTION_PROTOCOL = (
    "Reply with <memory>at most two short sentences on what you have tried so far in "
    "this episode and what came of it, rewritten from the memory of your last reply</memory> "
    "then <reason>one short sentence on what decides the next action</reason> "
    "then <answer>the action only</answer>."
)

# The LAST <answer> is the one that counts: a model's reasoning sometimes quotes
# the tag before writing the real section, and reading the first one then takes
# the whole reasoning as the action.
ANSWER_RE = re.compile(r"<answer>(?!.*<answer>)(.*?)</answer>", re.DOTALL)


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


def _load_arena_tasks(env: Env) -> dict[str, str]:
    """The per-task instruction, keyed by the "XX-YY" prefix of an arena label.

    One row per Olympics task, its columns being category (= level), scenario
    (= task within that level), a description, and the instruction in Japanese
    then English. Every arena the selector can serve has to have a row, which is
    checked here so a missing instruction fails at startup rather than at the
    episode that draws that arena.
    """
    with ARENA_TASK_CSV.open(encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file))[1:]
    tasks = {f"{int(row[0]):02d}-{int(row[1]):02d}": row[4] for row in rows}
    assert len(tasks) == len(rows), f"{ARENA_TASK_CSV} repeats a category/scenario pair"
    assert all(arena.name.rsplit("-", 1)[0] in tasks for arena in env.unwrapped.selector.arenas), (
        f"{ARENA_TASK_CSV} is missing a row for an arena of the selector"
    )
    return tasks


class PromptBuilder(ABC):
    """The conversation this run's VLMs read, carried across environment steps.

    Every tick ``observe`` records what the agent is looking at. A chain reads
    that through ``conversation`` on the steps it actually writes -- one step in
    ``cot_steps_per_chain`` -- and hands back what it wrote through
    ``add_reply``, which is what puts the turn it answered into the conversation
    for good. The ticks in between are overwritten rather than accumulated, so
    the conversation holds the turns a chain saw and not every step of the run.

    ``history_turns`` is how many of those exchanges it keeps. Every turn still
    held is re-read on every step that follows, so a conversation left to grow
    charges the whole episode for its own beginning; the oldest exchange is
    dropped instead.
    """

    def __init__(self, env: Env, history_turns: int, steps_per_action: int) -> None:
        assert history_turns >= 0, history_turns
        assert steps_per_action >= 1, steps_per_action
        self.history_turns = history_turns
        self.steps_per_action = steps_per_action
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

    def reject(self, answer: str) -> None:
        """Say, as the env and not as the agent, that the last reply named no
        action it could run. A complaint folded into the assistant's own turn
        reads back as something the agent chose to say; this is what it was
        told."""
        self._turns = self._turns + [
            {"role": "user", "content": [{"type": "text", "text": self._rejection_text(answer)}]}
        ]

    def _rejection_text(self, answer: str) -> str:
        return f"({answer!r} is not an action -- nothing ran.)"

    @abstractmethod
    def _task(self, obs: dict[str, Any], info: dict) -> str:
        """The standing task: what the env asks, unchanged through an episode."""

    @abstractmethod
    def _turn(self, obs: dict[str, Any], reward: float, info: dict) -> str:
        """What this tick alone says: the live numbers under the frame."""


# --- CarRacing ---------------------------------------------------------------

CAR_RACING_TEXT_ACTION_PROMPT = (
    "You control the red car in CarRacing-v3 (top-down). Stay on the gray road "
    "and avoid going onto the green grass; hug the road center when possible. "
    "Write the action as `steer=<value>, accel=<value>`, where each <value> is a "
    "float in [-1, 1]."
)


class CarRacingPromptBuilder(PromptBuilder):
    """CarRacing."""

    def _task(self, obs: dict[str, Any], info: dict) -> str:
        del obs, info
        return f"{CAR_RACING_TEXT_ACTION_PROMPT}"

    def _turn(self, obs: dict[str, Any], reward: float, info: dict) -> str:
        del obs, reward, info
        return ""


# --- Animal-AI ---------------------------------------------------------------

ANIMALAI_FRAMING = (
    "You control the agent in Animal-AI (first-person view). "
    "Health drains every step and the episode fails when it reaches 0. "
    "Green spheres look yellow-green and touching one ends the episode; yellow "
    "spheres are bright yellow and the episode goes on after touching them; red "
    "spheres and red zones end the episode as a failure, never touch them. "
    "Walls are never the goal, whatever their color. "
    "Turn until the target is at the center of the view, then move forward; "
    "move to a new spot if a full turn shows nothing. "
)
ANIMALAI_ACTION_NAMES = "move_forward / move_backward / turn_right / turn_left"


def _animalai_turn(obs: dict[str, Any], reward: float, average_forward_speed: float) -> str:
    """Animal-AI's live scalars: the numbers the frame cannot show.

    The forward speed is the average over the steps since the last answer
    rather than this tick's own: an action of fewer steps than the interval has
    run out by the time the next answer is asked for, so the speed at that
    moment reads 0 whether the move got somewhere or not.

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
    """Animal-AI: the framing, this arena's own instruction, and the live
    scalars.

    The instruction is looked up from the episode's arena name rather than
    handed over as text by the env, so the env carries no vocabulary of its own.
    """

    def __init__(self, env: Env, history_turns: int, steps_per_action: int) -> None:
        super().__init__(env, history_turns, steps_per_action)
        self.tasks = _load_arena_tasks(env)
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

    def _action_format(self) -> str:
        return (
            f"Write the action as `<name>(<n>)`, for example `move_forward(4)`: "
            f"<name> is one of {ANIMALAI_ACTION_NAMES}, and <n> is how many steps "
            f"in a row it is taken, an integer from 1 to {self.steps_per_action}."
        )

    def _rejection_text(self, answer: str) -> str:
        return f"(`{answer}` is not an action -- the agent stood still. {self._action_format()})"

    def _task(self, obs: dict[str, Any], info: dict) -> str:
        del obs
        return (
            f"{ANIMALAI_FRAMING} "
            f"{self._action_format()} "
            f"Task: {self.tasks[info['arena_name'].rsplit('-', 1)[0]]}. "
            f"{TEXT_ACTION_PROTOCOL}"
        )

    def _turn(self, obs: dict[str, Any], reward: float, info: dict) -> str:
        del info
        self._forward_speed_sum += float(obs["velocity"][2])
        self._forward_speed_steps += 1
        return _animalai_turn(obs, reward, self._forward_speed_sum / self._forward_speed_steps)


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


class CarlaPromptBuilder(PromptBuilder):
    """CARLA."""

    def _task(self, obs: dict[str, Any], info: dict) -> str:
        del obs, info
        return f"{CARLA_TEXT_ACTION_FRAMING} {TEXT_ACTION_PROTOCOL}"

    def _turn(self, obs: dict[str, Any], reward: float, info: dict) -> str:
        del obs, reward
        return CARLA_TEXT_ACTION_MANEUVER[info["maneuver_command"]]


# One builder per environment, whoever ends up acting on what it says.
PROMPT_BUILDERS = {
    "CarRacing-v3": CarRacingPromptBuilder,
    "AnimalAI-v0": AnimalAIPromptBuilder,
    "CARLA-Leaderboard-v0": CarlaPromptBuilder,
}


def build_prompt_builder(env: Env, args: DictConfig) -> PromptBuilder:
    assert args.env_id in PROMPT_BUILDERS, f"No prompt builder for {args.env_id}"
    # The exchanges a window of ``seq_len`` ticks holds at the chain's cadence:
    # the same count the trained network reads off its replay buffer, so the
    # two see the same history for the same config.
    history_turns = (args.seq_len - 1) // args.cot_steps_per_chain
    return PROMPT_BUILDERS[args.env_id](env, history_turns, args.cot_steps_per_chain)
