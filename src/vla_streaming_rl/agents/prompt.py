# SPDX-License-Identifier: MIT
"""Language input composed on the agent side.

The environment reports state; the agent decides what to say about it. Every
string a policy reads as language is built here out of the structured
observation the env already publishes, so the prompt belongs to the run's agent
config rather than to the simulator: two agents can drive the same env with
different framing, the env carries no text of its own, and ``use_prompt`` is a
choice of builder instead of a blanking step in the trainer.

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
    """

    # How a chain is to be written, the same in every environment. Written out
    # in full rather than left to the model's own thinking mode, which spends
    # its budget restating the request ("The user wants me to...") instead of
    # the scene.
    #
    # First person, because the task that follows is addressed to the agent
    # ("You control an animal...") and the chain is the agent's own thought.
    #
    # Whether asking plainly is enough turns out to be a question about the
    # model, not the wording. Measured over a recorded episode with
    # local/probe_cot_prompt.py, this text holds at 2B and above -- no "you", a
    # few "I" a chain -- while 0.8B ignores it and writes advice to the agent
    # instead ("Your health is critically low", "You need to find a safe path").
    # 0.8B can be made to comply, but only by an identity line, a rule per
    # sentence and forbidding the word outright, which is a prompt bent around
    # one model. ``cot_model_id`` carries the cost of that choice instead.
    #
    # The delta line is stated as a difference because a chain told only to
    # continue paraphrases what it already said and stops carrying new
    # information. What it already said is in the conversation as its own turns,
    # so nothing has to quote it back.
    BASE = (
        "This is what you can see right now. Think aloud as you act, in the "
        "first person.\n"
        "Say where I am relative to whatever matters around me, which way I am "
        "heading, what is about to go wrong, and what I should do next.\n"
        "Write only what has changed since my last thought, not what it already "
        "says.\n"
        "Two or three short sentences. No preamble, no lists, no numbers."
    )

    def __init__(self, env: Env) -> None:
        del env
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
        self._task_text = f"{self.BASE}\n{self._task(obs, info)}"
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
        conversation."""
        self._turns = self._turns + [
            self._current,
            {"role": "assistant", "content": [{"type": "text", "text": text}]},
        ]

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

CAR_RACING_TEXT_ACTION_PROMPT = "You control the red car in CarRacing-v3 (top-down). Stay on the gray road and avoid going onto the green grass; hug the road center when possible."

CAR_RACING_HIGH_LEVEL_PROMPT = (
    "You control the red car in CarRacing-v3 (top-down). The road is gray and "
    "the grass beside it is green. "
    "The steering and the throttle are not yours to write: say in one short "
    "sentence where the car should be going next -- which way the road bends "
    "ahead, whether it is on the center of the road or has to come back to it, "
    "and whether the corner ahead has to be taken slower."
)


class CarRacingTextActionPromptBuilder(PromptBuilder):
    """CarRacing, for the regime where the VLM writes the action."""

    def _task(self, obs: dict[str, Any], info: dict) -> str:
        del obs, info
        return CAR_RACING_TEXT_ACTION_PROMPT

    def _turn(self, obs: dict[str, Any], reward: float, info: dict) -> str:
        del obs, reward, info
        return ""


class CarRacingHighLevelPromptBuilder(PromptBuilder):
    """CarRacing, for the regime where the policy head writes the action."""

    def _task(self, obs: dict[str, Any], info: dict) -> str:
        del obs, info
        return CAR_RACING_HIGH_LEVEL_PROMPT

    def _turn(self, obs: dict[str, Any], reward: float, info: dict) -> str:
        del obs, reward, info
        return ""


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
    "Keep your speed down -- the third velocity component reported below should "
    "stay at about 10 or less, so stand still (NN) for a tick whenever it climbs "
    "past that. "
)

ANIMALAI_HIGH_LEVEL_FRAMING = (
    "You control an animal in a 3D arena, seen from its own point of view. "
    "Touching a green sphere scores points and ends the episode, and a larger "
    "sphere scores more. Touching a yellow sphere scores points and the episode "
    "continues. Touching a red sphere loses points and ends the episode. "
    "Entering a red zone ends the episode immediately. An orange zone drains "
    "your health, and your health also drains as time passes. "
    "What this arena asks of you follows as `Task:`. "
    "The movement itself is not yours to write: say in one short sentence what "
    "to go for next and what in the current view says so -- which object or "
    "which direction is worth heading for, and what has to be kept away from. "
    "Face what you are heading for before closing on it, and stay slow enough "
    "that the third velocity component reported below stays at about 10 or less. "
)


def _animalai_turn(obs: dict[str, Any], reward: float) -> str:
    """Animal-AI's live scalars: the numbers the frame cannot show.

    The same values the network's scalar branch is fed, so the sentence states
    exactly what the policy reads. Shared by both regimes -- these are what the
    env reports, not how the run frames it.
    """
    velocity_x, velocity_y, velocity_z = obs["velocity"]
    return (
        f"Velocity: ({velocity_x:+.2f}, {velocity_y:+.2f}, {velocity_z:+.2f}). "
        f"Reward: {reward:+.2f}. "
        f"Return so far: {obs['episode_return'][0]:+.2f}. "
        f"Pass mark: {obs['pass_mark'][0]:+.2f}. "
        f"Return needed: {obs['remaining_return'][0]:+.2f}. "
        f"Health: {obs['health'][0]:.2f}. "
        f"Global step: {int(obs['global_step'][0])}. "
        f"Episode step: {int(obs['episode_step'][0])}."
    )


class AnimalAITextActionPromptBuilder(PromptBuilder):
    """Animal-AI, for the regime where the VLM writes the action.

    Framing text, this arena's own instruction, and the live scalars. The
    scalars are read off the observation the agent already holds, so the
    sentence states exactly the numbers the network's scalar branch is fed, and
    the arena instruction is looked up from the episode's arena name rather than
    handed over as text by the env.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.tasks = _load_arena_tasks(env)

    def _task(self, obs: dict[str, Any], info: dict) -> str:
        del obs
        return (
            f"{ANIMALAI_TEXT_ACTION_FRAMING} "
            f"Task: {self.tasks[info['arena_name'].rsplit('-', 1)[0]]}."
        )

    def _turn(self, obs: dict[str, Any], reward: float, info: dict) -> str:
        del info
        return _animalai_turn(obs, reward)


class AnimalAIHighLevelPromptBuilder(PromptBuilder):
    """Animal-AI, for the regime where the policy head writes the action.

    The same arena instruction and the same live scalars as the text-action
    regime, framed as a decision about where to go rather than as an action to
    spell out.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.tasks = _load_arena_tasks(env)

    def _task(self, obs: dict[str, Any], info: dict) -> str:
        del obs
        return (
            f"{ANIMALAI_HIGH_LEVEL_FRAMING} "
            f"Task: {self.tasks[info['arena_name'].rsplit('-', 1)[0]]}."
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

CARLA_HIGH_LEVEL_FRAMING = (
    "Drive a car along a route in CARLA. Follow the planned route, "
    "obey traffic rules, and avoid collisions. "
    "The steering, the throttle and the brake are not yours to write: say in "
    "one short sentence what the next maneuver asks of the car right now -- "
    "what has to be given way to, and whether the speed has to come down "
    "before it."
)
CARLA_HIGH_LEVEL_MANEUVER = {
    -1: "The ego vehicle is following the lane straight ahead.",  # VOID
    1: "The ego vehicle is turning left at the upcoming intersection.",  # LEFT
    2: "The ego vehicle is turning right at the upcoming intersection.",  # RIGHT
    3: "The ego vehicle is going straight through the upcoming intersection.",  # STRAIGHT
    4: "The ego vehicle is following the lane straight ahead.",  # LANEFOLLOW
    5: "The ego vehicle is changing to the left lane.",  # CHANGELANELEFT
    6: "The ego vehicle is changing to the right lane.",  # CHANGELANERIGHT
}


class CarlaTextActionPromptBuilder(PromptBuilder):
    """CARLA, for the regime where the VLM writes the action."""

    def _task(self, obs: dict[str, Any], info: dict) -> str:
        del obs, info
        return CARLA_TEXT_ACTION_FRAMING

    def _turn(self, obs: dict[str, Any], reward: float, info: dict) -> str:
        del obs, reward
        return CARLA_TEXT_ACTION_MANEUVER[info["maneuver_command"]]


class CarlaHighLevelPromptBuilder(PromptBuilder):
    """CARLA, for the regime where the policy head writes the action."""

    def _task(self, obs: dict[str, Any], info: dict) -> str:
        del obs, info
        return CARLA_HIGH_LEVEL_FRAMING

    def _turn(self, obs: dict[str, Any], reward: float, info: dict) -> str:
        del obs, reward
        return CARLA_HIGH_LEVEL_MANEUVER[info["maneuver_command"]]


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
