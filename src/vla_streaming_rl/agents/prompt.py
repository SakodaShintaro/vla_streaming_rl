# SPDX-License-Identifier: MIT
"""Language input composed on the agent side.

The environment reports state; the agent decides what to say about it. Every
string a policy reads as language is built here out of the structured
observation the env already publishes, so the prompt belongs to the run's agent
config rather than to the simulator: two agents can drive the same env with
different framing, the env carries no text of its own, and ``use_prompt`` is a
choice of builder instead of a blanking step in the trainer.

There is one builder per environment, and it always writes the prompt of an
agent about to act: what the env asks, the action vocabulary it is asked in, the
arena's own instruction, and the two sections the answer is read out of. Whether
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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from gymnasium import Env
from omegaconf import DictConfig

ARENA_TASK_CSV = Path("./external/animal-ai/configs/AnimalAI_prompt.csv")

# How an answer is to be written where the VLM writes the action itself: a short
# justification and then the action alone, which the reader takes out of
# <answer>. The same in every environment, so it is stated once here; what stays
# per-environment is the encoding of the action, which each prompt below spells
# out beside everything else that env asks for.
#
# Every generated token is latency (one generation per env step), so this buys
# only the justification that changes the action: a full scene description (the
# <perception> section of the original Odysseus protocol) tripled the output for
# no measured benefit.
TEXT_ACTION_PROTOCOL = (
    "Reply with exactly two sections and no other text. "
    "First, in AT MOST two short sentences inside <think>...</think>, say what in "
    "the current image decides your next action, taking the previous reward (if "
    "shown) into account. Do not describe the scene in general, do not restate "
    "the task, and do not repeat your earlier reasoning. "
    "Then output the action inside <answer>...</answer>, which must contain ONLY "
    "the action -- no commentary, no labels."
)


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

    def __init__(self, env: Env, history_turns: int) -> None:
        del env
        assert history_turns >= 0, history_turns
        self.history_turns = history_turns
        self._turns = []
        self._current = {}
        self._task_text = ""

    def reset(self) -> None:
        """A finished episode ends the conversation; the next one opens its own."""
        self._turns = []
        self._current = {}

    def observe(self, obs: dict[str, Any], reward: float, info: dict, image) -> None:
        """What the agent is looking at this tick: the turn a chain would read if
        it wrote one now. Overwritten every step until one does."""
        self._task_text = self._task(obs, info)
        self._current = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self._turn(obs, reward, info)},
            ],
        }

    def conversation(self) -> list[dict]:
        """What a chain about to write reads: the standing task, the turns it has
        already answered, and the turn it is being asked about now."""
        opening = {"role": "system", "content": [{"type": "text", "text": self._task_text}]}
        return [opening] + self._turns + [self._current]

    def add_reply(self, text: str) -> None:
        """What the chain wrote about that turn, which settles the pair into the
        conversation, dropping the oldest exchange once ``history_turns`` are
        held."""
        turns = self._turns + [
            self._current,
            {"role": "assistant", "content": [{"type": "text", "text": text}]},
        ]
        self._turns = turns[max(0, len(turns) - 2 * self.history_turns) :]

    def task_text(self) -> str:
        """The standing task: what the policy reads, and what it tokenizes.

        The same string on every tick of an episode, so a network that tokenizes
        it gets the same token ids throughout. The live numbers are not in it;
        they reach the policy through the scalar branch and the chain through
        the turns.
        """
        return self._task_text

    @abstractmethod
    def _task(self, obs: dict[str, Any], info: dict) -> str:
        """The standing task: what the env asks, unchanged through an episode."""

    @abstractmethod
    def _turn(self, obs: dict[str, Any], reward: float, info: dict) -> str:
        """What this tick alone says: the live numbers under the frame."""


class EmptyPromptBuilder(PromptBuilder):
    """No language at all: what ``use_prompt: 0`` selects, and the ablation a
    language-conditioned run is measured against."""

    def _task(self, obs: dict[str, Any], info: dict) -> str:
        del obs, info
        return ""

    def _turn(self, obs: dict[str, Any], reward: float, info: dict) -> str:
        del obs, reward, info
        return ""


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
        return f"{CAR_RACING_TEXT_ACTION_PROMPT} {TEXT_ACTION_PROTOCOL}"

    def _turn(self, obs: dict[str, Any], reward: float, info: dict) -> str:
        del obs, reward, info
        return ""


# --- Animal-AI ---------------------------------------------------------------

# Grown from the wording the 2026-08-16 baseline ran (cleared 33 of its 72
# arenas): the task in one sentence and the action vocabulary spelled out. What
# that wording left out and this adds is the step between seeing and acting.
# Reading a level-02 episode, the baseline wrote "the goal sphere is visible to
# the right, which is on the opposite side of the scene from the current forward
# direction" and answered `walk backward, no turn`: it had the direction and no
# rule that turns it into a turn. `turn right` was not chosen once in 250 steps.
ANIMALAI_FRAMING = (
    "You control the agent in Animal-AI (first-person view). "
    "Find and reach the green or yellow goal sphere; avoid red zones. "
    "Action space: one move and one rotation, applied on the same tick, "
    "written as `<move>, <rotation>`. "
    "The move is one of: stand still, walk forward, walk backward. "
    "The rotation is one of: no turn, turn right, turn left. "
    "For example `walk forward, no turn` goes straight ahead, "
    "`stand still, turn left` turns on the spot, and "
    "`walk forward, turn right` walks while turning. "
    "Face what you are heading for before you close on it: while it sits off to "
    "one side of the view, turn on the spot towards it, and walk forward once "
    "it is centered. "
    "You see only what is in front of you while the arena extends all around "
    "you, so what you are looking for is out of view more often than not, and "
    "turning on the spot is how it is found. "
    "What this arena asks of you follows as `Task:`. "
)


def _animalai_turn(obs: dict[str, Any], reward: float) -> str:
    """Animal-AI's live scalars: the numbers the frame cannot show.

    Read off the observation the network's scalar branch is fed, cut down to
    what a reply can act on: of the three velocity components only the forward
    one, since a model handed all three reads the vector as motion the animal
    cannot have -- it reported flying and ascending off a standing still frame.
    The lateral and vertical components still reach the policy through the
    scalar branch. These are what the env reports, not how the run frames it.
    """
    forward_speed = obs["velocity"][2]
    return (
        f"Forward speed: {forward_speed:+.2f}. "
        f"Reward: {reward:+.2f}. "
        f"Return so far: {obs['episode_return'][0]:+.2f}. "
        f"Pass mark: {obs['pass_mark'][0]:+.2f}. "
        f"Return needed: {obs['remaining_return'][0]:+.2f}. "
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

    def __init__(self, env: Env, history_turns: int) -> None:
        super().__init__(env, history_turns)
        self.tasks = _load_arena_tasks(env)

    def _task(self, obs: dict[str, Any], info: dict) -> str:
        del obs
        return (
            f"{ANIMALAI_FRAMING} "
            f"Task: {self.tasks[info['arena_name'].rsplit('-', 1)[0]]}. "
            f"{TEXT_ACTION_PROTOCOL}"
        )

    def _turn(self, obs: dict[str, Any], reward: float, info: dict) -> str:
        del info
        return _animalai_turn(obs, reward)


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
    if not args.use_prompt:
        return EmptyPromptBuilder(env, args.prompt_history_turns)

    assert args.env_id in PROMPT_BUILDERS, f"No prompt builder for {args.env_id}"
    return PROMPT_BUILDERS[args.env_id](env, args.prompt_history_turns)
