# SPDX-License-Identifier: MIT
"""Language input composed on the agent side.

The environment reports state; the agent decides what to say about it. Every
string a policy reads as language is built here out of the structured
observation the env already publishes, so the prompt belongs to the run's agent
config rather than to the simulator: two agents can drive the same env with
different framing, the env carries no text of its own, and ``use_prompt`` is a
choice of builder instead of a blanking step in the trainer.

A builder yields one string, ``episode_text``: what holds for the whole
episode, the framing and the arena's own instruction. The chain of thought
prefills it once and never reads it again, and what a tick adds to its context
is the frame alone.

It carries no number, names no action, and reports nothing that changes tick to
tick. The scalars an env reports reach the policy through its own branch and
the action through its own; spelling either out here only gave the chain of
thought a machine-readable format to imitate instead of writing about the
scene.

``__call__`` composes the first two into the single string the networks that
read the prompt as one blob still expect.

There is one builder per (environment, regime), and the two regimes are
deliberately kept apart rather than composed out of shared pieces:

- ``text_action``: the VLM writes the action itself, which
  ``parse_action_text`` decodes. The prompt has to teach the action encoding.
- everything else (``none``, ``high_level``): the action comes from the policy
  head and the language is either read as conditioning or generated as a
  high-level intent. Naming the action encoding here would only mislead.

Each class therefore holds its own complete text. The duplication is the point:
rewording one regime must not move the other, since the two are measured
against each other.

A builder is called once per environment step with the observation and info the
agent itself received, and returns the whole prompt for that tick.
"""

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from gymnasium import Env
from omegaconf import DictConfig

ARENA_TASK_CSV = Path("./external/animal-ai/configs/AnimalAI_prompt.csv")


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
    """The language input for one tick, from that tick's observation and info."""

    def __init__(self, env: Env) -> None:
        del env

    @abstractmethod
    def episode_text(self, obs: dict[str, Any], info: dict) -> str:
        """What holds for the whole episode."""


class EmptyPromptBuilder(PromptBuilder):
    """No language at all: what ``use_prompt: 0`` selects, and the ablation a
    language-conditioned run is measured against."""

    def episode_text(self, obs: dict[str, Any], info: dict) -> str:
        del obs, info
        return ""


# --- CarRacing ---------------------------------------------------------------

CAR_RACING_TEXT_ACTION_PROMPT = "You control the red car in CarRacing-v3 (top-down). Stay on the gray road and avoid going onto the green grass; hug the road center when possible."

CAR_RACING_HIGH_LEVEL_PROMPT = (
    "You are driving the red car in CarRacing-v3, seen from above. The road is "
    "gray and the grass beside it is green, and the car is at the middle of the "
    "picture pointing up it. "
    "What matters is which way the road bends ahead, whether you are on the "
    "center of it or drifting off toward the grass, and whether the corner "
    "ahead has to be taken slower than you are going. "
    "The steering and the throttle are not yours to write."
)


class CarRacingTextActionPromptBuilder(PromptBuilder):
    """CarRacing, for the regime where the VLM writes the action."""

    def episode_text(self, obs: dict[str, Any], info: dict) -> str:
        del obs, info
        return CAR_RACING_TEXT_ACTION_PROMPT


class CarRacingHighLevelPromptBuilder(PromptBuilder):
    """CarRacing, for the regime where the policy head writes the action."""

    def episode_text(self, obs: dict[str, Any], info: dict) -> str:
        del obs, info
        return CAR_RACING_HIGH_LEVEL_PROMPT


# --- Animal-AI ---------------------------------------------------------------

ANIMALAI_TEXT_ACTION_FRAMING = (
    "You control an animal in a 3D arena, seen from its own point of view. "
    "Touching a green sphere scores points and ends the episode, and a larger "
    "sphere scores more. Touching a yellow sphere scores points and the episode "
    "continues. Touching a red sphere loses points and ends the episode. "
    "Entering a red zone ends the episode immediately. An orange zone drains "
    "your health, and your health also drains as time passes. "
    "What this arena asks of you follows as `Task:`. "
    "Action space: two letters, a move and a rotation applied on the same tick. "
    "The move is F (walk forward), B (walk backward) or N (stand still). "
    "The rotation is R (turn right), L (turn left) or N (no turn). "
    "So FN walks straight ahead, NL turns left on the spot, FR walks while "
    "turning right, and NN does nothing. "
    "Bring whatever you are heading for to the center of your view before you "
    "walk forward: turn on the spot (NR or NL) until it is centered, and only "
    "then move. "
    "Keep your speed down: stand still (NN) for a tick whenever you are "
    "carrying too much of it. "
)

ANIMALAI_HIGH_LEVEL_FRAMING = (
    "You control an animal in a 3D arena, seen from its own point of view. "
    "Touching a green sphere scores points and ends the episode, and a larger "
    "sphere scores more. Touching a yellow sphere scores points and the episode "
    "continues. Touching a red sphere loses points and ends the episode. "
    "Entering a red zone ends the episode immediately. An orange zone drains "
    "your health, and your health also drains as time passes."
)


class AnimalAITextActionPromptBuilder(PromptBuilder):
    """Animal-AI, for the regime where the VLM writes the action.

    Framing text and this arena's own instruction stand for the episode, and a
    tick reports nothing beyond its frame and the letters of the move that led
    to it. The arena instruction is looked up from the episode's arena name
    rather than handed over as text by the env.
    """

    def __init__(self, env: Env) -> None:
        self.tasks = _load_arena_tasks(env)

    def episode_text(self, obs: dict[str, Any], info: dict) -> str:
        del obs
        return (
            f"{ANIMALAI_TEXT_ACTION_FRAMING} "
            f"Task: {self.tasks[info['arena_name'].rsplit('-', 1)[0]]}."
        )


class AnimalAIHighLevelPromptBuilder(PromptBuilder):
    """Animal-AI, for the regime where the policy head writes the action.

    The same arena instruction as the text-action regime, framed as a decision
    about where to go rather than as an action to spell out.
    """

    def __init__(self, env: Env) -> None:
        self.tasks = _load_arena_tasks(env)

    def episode_text(self, obs: dict[str, Any], info: dict) -> str:
        del obs
        return (
            f"{ANIMALAI_HIGH_LEVEL_FRAMING} "
            f"Task: {self.tasks[info['arena_name'].rsplit('-', 1)[0]]}."
        )


# --- CARLA -------------------------------------------------------------------

CARLA_TEXT_ACTION_FRAMING = (
    "Drive a car along a route in CARLA. Follow the planned route, "
    "obey traffic rules, and avoid collisions."
)

CARLA_HIGH_LEVEL_FRAMING = (
    "You are driving a car along a route in CARLA. Follow the planned route, "
    "obey traffic rules, and avoid collisions. "
    "What matters is what the road ahead asks of the car right now, what has "
    "to be given way to, and whether the speed has to come down before it. "
    "The steering, the throttle and the brake are not yours to write."
)


class CarlaTextActionPromptBuilder(PromptBuilder):
    """CARLA, for the regime where the VLM writes the action."""

    def episode_text(self, obs: dict[str, Any], info: dict) -> str:
        del obs, info
        return CARLA_TEXT_ACTION_FRAMING


class CarlaHighLevelPromptBuilder(PromptBuilder):
    """CARLA, for the regime where the policy head writes the action."""

    def episode_text(self, obs: dict[str, Any], info: dict) -> str:
        del obs, info
        return CARLA_HIGH_LEVEL_FRAMING


# One builder per (env_id, regime). "text_action" is the regime in which the VLM
# writes the action itself; "high_level" covers the modes that leave the action
# to the policy head.
PROMPT_BUILDERS = {
    ("CarRacing-v3", "text_action"): CarRacingTextActionPromptBuilder,
    ("CarRacing-v3", "high_level"): CarRacingHighLevelPromptBuilder,
    ("AnimalAI-v0", "text_action"): AnimalAITextActionPromptBuilder,
    ("AnimalAI-v0", "high_level"): AnimalAIHighLevelPromptBuilder,
    ("CARLA-Leaderboard-v0", "text_action"): CarlaTextActionPromptBuilder,
    ("CARLA-Leaderboard-v0", "high_level"): CarlaHighLevelPromptBuilder,
}


def build_prompt_builder(env: Env, args: DictConfig) -> PromptBuilder:
    if not args.use_prompt:
        return EmptyPromptBuilder(env)

    assert args.text_action_mode in ("none", "high_level", "text_action"), (
        f"Unknown text_action_mode: {args.text_action_mode!r}"
    )
    # ``none`` generates no text at all and ``high_level`` generates an intent;
    # both leave the action to the policy head, so both read the same prompt.
    regime = "text_action" if args.text_action_mode == "text_action" else "high_level"

    key = (args.env_id, regime)
    assert key in PROMPT_BUILDERS, f"No prompt builder for {key}"
    return PROMPT_BUILDERS[key](env)
