# SPDX-License-Identifier: MIT
"""Animal-AI Gymnasium environment.

Wraps Animal-AI v5 (which exposes a Unity ML-Agents BehaviorSpec interface)
as a single-agent gym.Env. The action is Box(-1, 1, shape=(2,)) for parity with
the project's other environments (CARLA, GUI games):
    dim 0: forward (+1) / back (-1)
    dim 1: rotate right (+1) / left (-1)
How Unity consumes it depends on `continuous_action`:
  - False: the official MultiDiscrete([3, 3]) branches, reached by
           discretizing with a +/-1/3 dead-zone (0=noop, 1=forward/right,
           2=back/left). Works with any Animal-AI binary.
  - True:  the value is passed through as a throttle / turn rate. This needs a
           binary rebuilt from animal-ai-unity with the hybrid action spec
           (2 continuous actions alongside the branches); +/-1 there reproduces
           the discrete branches exactly, so it is a strict superset.

Everything about talking to Unity lives in `AnimalAIEnv`. The only thing that
differs between the three ways we run Animal-AI is *which arena each episode
loads*, so that is factored out into an `ArenaSelector`, picked by the
`mode` config field (see `build_selector`):

  - "staged"  training with the paper's cumulative 11-stage curriculum, one
              stage per competition level, sampled uniformly within the stage
              (`StagedSelector`).
  - "success" one stage per level, advanced by clearing a round of it: the
              stage's whole pool (30 arenas at stage 1, 60 at stage 2, up to
              300) drawn without replacement, then the next stage if the
              round's success rate reached `advance_success_rate`
              (`SuccessDrivenSelector`).
  - "sequential" no curriculum: every training arena in label order, wrapping
              around forever (`SequentialSelector`, cycling).
  - "random"  no curriculum: every episode draws uniformly at random from the
              whole training set (`RandomSelector`).
  - "eval"    every configs/competition/ arena once, in order -- the
              900-arena Testbed sweep (the same `SequentialSelector`, not
              cycling, so it ends).

Every arena comes from configs/competition/, whose files are named
XX-YY-ZZ.yaml for level, task and variant. Each of the 300 tasks ships three
variants; the training modes serve one variant of each (`train_variant`: 300
arenas, 30 per level) while eval sweeps all 900. Training and eval therefore
overlap by construction -- the trained variant is scored again, and within a
task the variants are often identical files -- so `seen_in_training` is what
separates the held-out part of a Testbed score from the rest.

`end_at_pass_mark` ends an episode as soon as a collected reward carries the
return to the arena's pass mark, which Unity itself never does (see `step`).
"""

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import yaml
from animalai import AnimalAIEnvironment
from gymnasium import spaces
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)

COMPETITION_DIR = Path("./external/animal-ai/configs/competition")
ARENA_PROMPT_CSV = Path("./external/animal-ai/configs/AnimalAI_prompt.csv")


def _load_arena_prompts() -> dict[str, str]:
    """The per-task instruction, keyed by the "XX-YY" prefix of an arena label.

    One row per Olympics task, its columns being category (= level), scenario
    (= task within that level), a description, and the instruction in Japanese
    then English.
    """
    with ARENA_PROMPT_CSV.open(encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file))[1:]
    prompts = {f"{int(row[0]):02d}-{int(row[1]):02d}": row[4] for row in rows}
    assert len(prompts) == len(rows), f"{ARENA_PROMPT_CSV} repeats a category/scenario pair"
    return prompts


# Environment parameter the rebuilt binary reads to pick continuous over
# discrete actions (see TrainingAgent.ReadAction in animal-ai-unity).
CONTINUOUS_ACTIONS_KEY = "continuousActions"


def _aai_environment_class(extra_args: list[str]) -> type:
    """`AnimalAIEnvironment` subclass that actually passes `extra_args` to Unity.

    `AnimalAIEnvironment.__init__` rebuilds the player's command line from
    `executable_args` and discards whatever the caller handed it as
    `additional_args`, so the switches the rebuilt binary reads at startup
    (`--topDownCamera`, `--topDownResolution`) never arrive. `executable_args`
    is a staticmethod, so the arguments are bound to a class rather than an
    instance.
    """

    class _AnimalAIEnvironmentWithArgs(AnimalAIEnvironment):
        @staticmethod
        def executable_args(*args) -> list[str]:
            return AnimalAIEnvironment.executable_args(*args) + extra_args

    return _AnimalAIEnvironmentWithArgs


# Shaping coefficients.
GOAL_BONUS = 0.5
RAMPS_COEF = 0.01
BACK_MOVE_COEF = 0.001
PASS_MARK_BONUS = 1.0


def _to_discrete(value: float) -> int:
    """Map a continuous control in [-1, 1] to AAI's {0=noop, 1, 2}."""
    if value >= 1.0 / 3.0:
        return 1
    elif value <= -1.0 / 3.0:
        return 2
    return 0


# Animal-AI yaml uses custom !ArenaConfig/!Item/!Vector3/!RGB tags. We register
# them as plain mappings so yaml.load can parse the full structure.
class _AAILoader(yaml.SafeLoader):
    pass


def _aai_tag_constructor(loader, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


for _tag in ("!ArenaConfig", "!Arena", "!Item", "!Vector3", "!RGB"):
    _AAILoader.add_constructor(_tag, _aai_tag_constructor)


def _parse_arena(yaml_path: Path) -> tuple[float, list[dict]]:
    """Return (pass_mark, items) from an Animal-AI arena yaml.

    The arena's ``t`` is deliberately not read: it is not a step cap but the
    decay rate of the agent's health, and collecting a reward refills that
    health, so an episode routinely runs well past ``t`` steps. What is left of
    the episode is the health value in the AAI vector observation, not a step
    count, so that is what the agent is given (see `_read_observation`).

    Each item dict carries one (position, size, rotation) triple in arena
    coordinates: {name, x, z, size_x, size_z, rotation}. yaml entries with
    multiple positions are expanded into multiple item dicts; if `sizes` is
    shorter than `positions` the last given size is reused (AAI's convention).
    """
    cfg = yaml.load(yaml_path.read_text(), Loader=_AAILoader)
    arena = cfg["arenas"][0]
    pass_mark = float(arena.get("pass_mark", 0))
    items_out: list[dict] = []
    for item in arena.get("items", []) or []:
        name = item["name"]
        positions = item.get("positions") or []
        sizes = item.get("sizes") or []
        rotations = item.get("rotations") or []
        for i, pos in enumerate(positions):
            size = sizes[i] if i < len(sizes) else (sizes[-1] if sizes else None)
            rot = rotations[i] if i < len(rotations) else (rotations[-1] if rotations else 0)
            items_out.append(
                {
                    "name": name,
                    "x": float(pos["x"]),
                    "z": float(pos["z"]),
                    "size_x": float(size["x"]) if size else 1.0,
                    "size_z": float(size["z"]) if size else 1.0,
                    "rotation": float(rot),
                }
            )
    return pass_mark, items_out


@dataclass(frozen=True)
class Arena:
    """One arena yaml plus the label it is reported under (info, logs, render)."""

    path: Path
    name: str


def _competition_arenas() -> list[Arena]:
    """Every arena of the Olympics set, labeled by its "XX-YY-ZZ" stem.

    The three fields are level, task and variant, so the label carries the
    curriculum stage ("01") and the task family ("01-24") an arena belongs to.
    It is also what names per-arena files (train.py writes a video per arena)
    and wandb metrics, so training and eval report the same arena under the
    same name.
    """
    paths = sorted(COMPETITION_DIR.glob("*.yaml"))
    assert paths, f"no arena yamls under {COMPETITION_DIR}"
    return [Arena(path, path.stem) for path in paths]


def _training_arenas(variant: str) -> list[Arena]:
    """The training set: the `variant` numbered copy of every Olympics task.

    Each task family XX-YY ships three variants, so one variant is one arena
    per family -- 300 of the 900, which is the arena set the paper's
    curriculum trains on. The other variants are what eval scores.
    """
    arenas = [arena for arena in _competition_arenas() if arena.name.rsplit("-", 1)[1] == variant]
    assert arenas, f"no competition arena has variant {variant!r}"
    return arenas


def _training_stages(variant: str) -> list[list[Arena]]:
    """The training arenas grouped into one stage per curriculum level.

    The Olympics levels are the paper's curriculum stages in order (01 food
    retrieval ... 10 causal reasoning), and every level holds 30 tasks, so
    each stage is 30 arenas. `StagedSelector` accumulates them; the rounds of
    `SuccessDrivenSelector` are one stage each.
    """
    arenas = _training_arenas(variant)
    return [
        [arena for arena in arenas if arena.name.split("-")[0] == level]
        for level in sorted({arena.name.split("-")[0] for arena in arenas})
    ]


def _cumulative_stages(variant: str) -> list[list[Arena]]:
    """Each stage's arena pool: its own level plus every level before it.

    30 arenas at stage 1, 60 at stage 2, up to all 300 at stage 10, so a stage
    keeps drawing on everything the curriculum has unlocked so far.
    """
    pools: list[list[Arena]] = []
    for level_arenas in _training_stages(variant):
        pools.append((pools[-1] if pools else []) + level_arenas)
    return pools


def _arena_signature(path: Path) -> str:
    """Digest of what makes two arena yamls the same episode.

    Item order and mapping key order are how a file happens to be written,
    not what it runs, so both are canonicalized away; `pass_mark` and `t`
    decide the episode alongside the items, so both are in. Used to find the
    eval arenas that are repeats of a training arena -- within a task family
    the variants are often the same arena written out three times, and a score
    over those is not a held-out score.
    """
    arena = yaml.load(path.read_text(), Loader=_AAILoader)["arenas"][0]
    pass_mark = float(arena["pass_mark"]) if "pass_mark" in arena else 0.0
    items = arena["items"] if "items" in arena else []
    payload = json.dumps(
        {
            "pass_mark": pass_mark,
            "t": arena["t"],
            "items": sorted(json.dumps(item, sort_keys=True) for item in items),
        },
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()


def seen_in_training(variant: str) -> set[str]:
    """Labels of the eval arenas a run trained on `variant` has already seen.

    The training variant itself, plus every other variant whose arena is
    identical to the one its family contributed to training.
    """
    trained = {_arena_signature(arena.path) for arena in _training_arenas(variant)}
    return {
        arena.name for arena in _competition_arenas() if _arena_signature(arena.path) in trained
    }


class ArenaSelector:
    """Picks the arena each episode runs, and counts how each one has gone.

    `arenas` is every arena this selector can serve: `AnimalAIEnv` boots Unity
    on the first of them and resolves `options={"arena_stem": ...}` against
    their names. Per-arena attempt/success counts are kept for every mode, so
    the curriculum-progress panel and the resume snapshot do not depend on
    which selector is running.
    """

    def __init__(self, arenas: list[Arena]):
        self.arenas = arenas
        self._arena_by_name = {arena.name: arena for arena in arenas}
        names = [arena.name for arena in arenas]
        self._attempts = dict.fromkeys(names, 0)
        self._successes = dict.fromkeys(names, 0)

    @property
    def is_exhausted(self) -> bool:
        """True once there is nothing left to run (finite sweeps only)."""
        return False

    def next_arena(self, global_step: int) -> Arena:
        raise NotImplementedError

    def on_episode_end(self, arena: Arena, success: bool) -> None:
        """Record the attempt. Curricula extend this to advance themselves."""
        self._attempts[arena.name] += 1
        self._successes[arena.name] += int(success)

    def arena_record(self, arena: Arena) -> tuple[int, int]:
        """(attempts, successes) so far for one arena."""
        return self._attempts[arena.name], self._successes[arena.name]

    def is_cleared(self, arena: Arena) -> bool:
        """Whether one arena counts as solved: passed at least once, unless a
        curriculum wants a stricter bar (see `SuccessDrivenSelector`)."""
        return self._successes[arena.name] > 0

    def progress_by_group(self) -> list[tuple[str, int, int, int]]:
        """(group, cleared, failed, untried) per arena group, in group order.

        `cleared` counts arenas `is_cleared` accepts, `failed` ones attempted
        without reaching that bar, `untried` ones never run. The group is the
        arena label's first "-" separated token, which is the competition
        level ("01-01-01" -> "01") and so also the curriculum stage.
        """
        counts: dict[str, list[int]] = {}
        for arena in self.arenas:
            group = counts.setdefault(arena.name.split("-")[0], [0, 0, 0])
            attempts = self._attempts[arena.name]
            group[0 if self.is_cleared(arena) else (1 if attempts else 2)] += 1
        return [(group, *counts[group]) for group in sorted(counts)]

    def info(self, global_step: int) -> dict:
        """Selector-specific fields merged into each reset/step info dict."""
        return {}

    def status(self, global_step: int) -> str:
        """One-line progress summary drawn on the top-down render."""
        raise NotImplementedError

    def state(self) -> dict:
        """Resume snapshot: attempted arenas only, so the file stays small.

        `arena_cleared` is this selector's own `is_cleared` verdict, so what
        train.py reports as cleared is what the running curriculum acts on."""
        attempts = {name: n for name, n in self._attempts.items() if n > 0}
        arena_by_name = {arena.name: arena for arena in self.arenas}
        return {
            "arena_attempts": attempts,
            "arena_successes": {name: self._successes[name] for name in attempts},
            "arena_cleared": {name: self.is_cleared(arena_by_name[name]) for name in attempts},
            "progress": self.progress_state(),
        }

    def progress_state(self) -> dict:
        """Whatever else this selector needs to resume, beyond the per-arena
        counts. Empty for the selectors whose position follows from those."""
        return {}

    def load_state(self, arena_attempts: dict, arena_successes: dict, progress: dict) -> None:
        """Restore a `state()` snapshot, dropping arenas this set does not have."""
        known = set(self._attempts)
        self._attempts.update({k: int(v) for k, v in arena_attempts.items() if k in known})
        self._successes.update({k: int(v) for k, v in arena_successes.items() if k in known})


class StagedSelector(ArenaSelector):
    """The paper's cumulative curriculum (Animal-AI paper, Section 4.1.3).

    One stage per Olympics level, plus a final stage repeating the last one,
    so the 10 levels give 11. Stage i samples uniformly from every arena of
    stages 1..i, and runs for `steps_per_stage` steps; stage 11 runs until
    train.py's step_limit ends the run -- the paper's "further five million
    steps on the last stage".
    """

    def __init__(self, variant: str, steps_per_stage: int, seed: int):
        stages = _cumulative_stages(variant)
        super().__init__(stages[-1])
        self._stages = stages + [stages[-1]]
        self.steps_per_stage = steps_per_stage
        self._rng = np.random.default_rng(seed)

    def _stage_index(self, global_step: int) -> int:
        return min(global_step // self.steps_per_stage, len(self._stages) - 1)

    def _stage_arenas(self, global_step: int) -> list[Arena]:
        return self._stages[self._stage_index(global_step)]

    def next_arena(self, global_step: int) -> Arena:
        stage_arenas = self._stage_arenas(global_step)
        return stage_arenas[int(self._rng.integers(len(stage_arenas)))]

    def info(self, global_step: int) -> dict:
        return {"stage": self._stage_index(global_step) + 1}

    def status(self, global_step: int) -> str:
        return f"stage:{self._stage_index(global_step) + 1}/{len(self._stages)}  step:{global_step}"


class SuccessDrivenSelector(ArenaSelector):
    """One stage per Olympics level, advanced by clearing a round of it.

    A round is the stage's whole pool drawn without replacement, so every arena
    it has unlocked is played exactly once before any is played again. When the
    round ends its success rate is compared with `advance_success_rate`: at or
    above it the next stage opens, below it the same stage runs another round
    with a fresh draw order.

    The pools are cumulative, so a round is 30 arenas at stage 1, 60 at stage
    2 and so on up to 300, and nothing already learned is dropped. At the
    default rate of 1.0 stage 2 therefore asks for 60 straight passes.
    """

    def __init__(self, variant: str, advance_success_rate: float, seed: int):
        stages = _cumulative_stages(variant)
        super().__init__(stages[-1])
        self._stages = stages
        self.advance_success_rate = advance_success_rate
        self._rng = np.random.default_rng(seed)
        self._stage = 0
        self._queue: list[str] = []
        self._round_attempts = 0
        self._round_successes = 0
        self._last_round_rate = 0.0
        self._advanced = False

    def _refill(self) -> None:
        names = [arena.name for arena in self._stages[self._stage]]
        order = self._rng.permutation(len(names))
        self._queue = [names[int(i)] for i in order]

    def _round_rate(self) -> float:
        assert self._round_attempts > 0
        return self._round_successes / self._round_attempts

    def next_arena(self, global_step: int) -> Arena:
        """Take the next draw of the round. Drawing is what consumes it, so an
        episode the caller abandons still moves the round along."""
        if not self._queue:
            self._refill()
        return self._arena_by_name[self._queue.pop(0)]

    def on_episode_end(self, arena: Arena, success: bool) -> None:
        super().on_episode_end(arena, success)
        self._round_attempts += 1
        self._round_successes += int(success)
        self._advanced = False
        if self._round_attempts < len(self._stages[self._stage]):
            return
        # The counters reset here but info() is read after this call, so the
        # rate the decision was made on is kept rather than reported as 0/0.
        self._last_round_rate = self._round_rate()
        if self._last_round_rate >= self.advance_success_rate and self._stage + 1 < len(
            self._stages
        ):
            self._stage += 1
            self._advanced = True
        self._round_attempts = 0
        self._round_successes = 0
        self._queue = []

    def info(self, global_step: int) -> dict:
        return {
            "stage": self._stage + 1,
            "round_index": self._round_attempts,
            "round_success_rate": (self._round_rate() if self._round_attempts > 0 else 0.0),
            "last_round_success_rate": self._last_round_rate,
            "advanced": self._advanced,
            "cleared_count": sum(self.is_cleared(arena) for arena in self.arenas),
        }

    def status(self, global_step: int) -> str:
        size = len(self._stages[self._stage])
        rate = self._round_rate() if self._round_attempts > 0 else 0.0
        return (
            f"stage:{self._stage + 1}/{len(self._stages)}"
            f"  round:{self._round_attempts}/{size} rate:{rate:.2f}"
            f"  step:{global_step}"
        )

    def progress_state(self) -> dict:
        return {
            "stage": self._stage,
            "queue": list(self._queue),
            "round_attempts": self._round_attempts,
            "round_successes": self._round_successes,
            "last_round_rate": self._last_round_rate,
        }

    def load_state(self, arena_attempts: dict, arena_successes: dict, progress: dict) -> None:
        """Resume mid-round: the stage, what is left of its draw order and the
        round's tally so far all have to come back, since none of them can be
        recovered from the per-arena totals."""
        super().load_state(arena_attempts, arena_successes, progress)
        if not progress:
            return
        self._stage = int(progress["stage"])
        self._round_attempts = int(progress["round_attempts"])
        self._round_successes = int(progress["round_successes"])
        self._last_round_rate = float(progress["last_round_rate"])
        known = set(self._attempts)
        self._queue = [name for name in progress["queue"] if name in known]
        if not self._queue:
            self._refill()


class SequentialSelector(ArenaSelector):
    """Every arena of `arenas` in label order.

    `cycle` is what separates the two ways this is used:

      - False: one pass, then `is_exhausted`. Over the whole competition set
        this is the paper's Testbed protocol -- one episode per arena, pass/fail
        by that arena's pass mark (scripts/test_trained_agent.py loops until
        exhausted).
      - True: wrap back to the first arena instead of ending, so the set is
        replayed until train.py's step_limit stops the run. This is the
        curriculum-free training mode, running the training arenas in level
        order rather than by stage.
    """

    def __init__(self, arenas: list[Arena], cycle: bool):
        super().__init__(arenas)
        self.cycle = cycle
        self._next_index = 0

    @property
    def is_exhausted(self) -> bool:
        return not self.cycle and self._next_index >= len(self.arenas)

    def next_arena(self, global_step: int) -> Arena:
        arena = self.arenas[self._next_index % len(self.arenas)]
        self._next_index += 1
        return arena

    def info(self, global_step: int) -> dict:
        return {
            "arena_index": (self._next_index - 1) % len(self.arenas),
            "arena_total": len(self.arenas),
            "lap": (self._next_index - 1) // len(self.arenas),
        }

    def status(self, global_step: int) -> str:
        cleared = sum(self.is_cleared(arena) for arena in self.arenas)
        return (
            f"{(self._next_index - 1) % len(self.arenas) + 1}/{len(self.arenas)}"
            f"  lap:{(self._next_index - 1) // len(self.arenas) + 1}"
            f"  cleared:{cleared}/{len(self.arenas)}  step:{global_step}"
        )

    def load_state(self, arena_attempts: dict, arena_successes: dict, progress: dict) -> None:
        """Resume where the run left off: one episode per arena visit, so the
        total attempt count is exactly how far into the order we are."""
        super().load_state(arena_attempts, arena_successes, progress)
        self._next_index = sum(self._attempts.values())


class RandomSelector(ArenaSelector):
    """Every episode drawn uniformly at random from `arenas`.

    No curriculum and no ordering: unlike `SequentialSelector(cycle=True)` an
    arena can repeat before the whole set has been seen, and unlike
    `StagedSelector` nothing is withheld -- the full set is available from the
    first episode. The curriculum-free baseline the staged modes are compared
    against. Runs forever, so train.py's step_limit ends the run.
    """

    def __init__(self, arenas: list[Arena], seed: int):
        super().__init__(arenas)
        self._rng = np.random.default_rng(seed)

    def next_arena(self, global_step: int) -> Arena:
        return self.arenas[int(self._rng.integers(len(self.arenas)))]

    def info(self, global_step: int) -> dict:
        cleared = sum(self.is_cleared(arena) for arena in self.arenas)
        return {"arena_total": len(self.arenas), "cleared_count": cleared}

    def status(self, global_step: int) -> str:
        cleared = sum(self.is_cleared(arena) for arena in self.arenas)
        untried = sum(attempts == 0 for attempts in self._attempts.values())
        return (
            f"random  cleared:{cleared}/{len(self.arenas)}  untried:{untried}  step:{global_step}"
        )


def build_selector(
    mode: str, train_variant: str, steps_per_stage: int, advance_success_rate: float, seed: int
) -> ArenaSelector:
    """Build the arena selector named by `mode` (see the module docstring).

    The training modes serve the `train_variant` copy of every task; "eval"
    serves the whole competition set, the paper's fixed 900-arena Testbed.
    Which of those the run has already trained on follows from
    `train_variant` -- `seen_in_training` is what reports it.

    Every mode's parameters are always supplied; a mode ignores the ones that
    do not apply to it.
    """
    builders = {
        "staged": lambda: StagedSelector(
            variant=train_variant, steps_per_stage=steps_per_stage, seed=seed
        ),
        "success": lambda: SuccessDrivenSelector(
            variant=train_variant, advance_success_rate=advance_success_rate, seed=seed
        ),
        "sequential": lambda: SequentialSelector(
            arenas=_training_arenas(train_variant), cycle=True
        ),
        "random": lambda: RandomSelector(arenas=_training_arenas(train_variant), seed=seed),
        "eval": lambda: SequentialSelector(arenas=_competition_arenas(), cycle=False),
    }
    assert mode in builders, f"unknown Animal-AI mode {mode!r}; expected one of {sorted(builders)}"
    return builders[mode]()


# RGB colors (matplotlib-style) for each AAI item type when drawn top-down.
_ITEM_COLORS: dict[str, tuple[int, int, int]] = {
    "GoodGoal": (40, 200, 40),
    "GoodGoalMulti": (40, 230, 80),
    "GoodGoalBounce": (40, 200, 120),
    "GoodGoalMultiBounce": (40, 230, 160),
    "BadGoal": (220, 40, 40),
    "BadGoalBounce": (220, 100, 40),
    "DeathZone": (140, 20, 20),
    "HotZone": (220, 120, 40),
    "Wall": (110, 110, 110),
    "WallTransparent": (200, 200, 220),
    "Ramp": (140, 100, 60),
    "Cardbox1": (210, 160, 80),
    "Cardbox2": (180, 130, 70),
    "LObject": (220, 200, 50),
    "LObject2": (200, 180, 50),
    "UObject": (220, 220, 50),
    "CylinderTunnel": (160, 190, 210),
    "CylinderTunnelTransparent": (200, 220, 230),
}
_DEFAULT_ITEM_COLOR: tuple[int, int, int] = (180, 180, 180)
_AGENT_COLOR: tuple[int, int, int] = (40, 80, 220)
_ARENA_SIZE_M = 40.0  # standard Animal-AI arena is 40 m square
_RENDER_SIZE_PX = 256
_HEADER_LINE_PX = 20


def _draw_header(lines: list[str], width_px: int) -> np.ndarray:
    """Caption strip, one row per line. Splitting a long status over two lines
    keeps the type readable where fitting it on one would shrink it away."""
    header = np.full((_HEADER_LINE_PX * len(lines), width_px, 3), 215, dtype=np.uint8)
    max_width = width_px - 8
    for row, text in enumerate(lines):
        font_scale = 0.45
        (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        # Shrink rather than let a long line run off the right edge.
        font_scale *= min(1.0, max_width / width)
        cv2.putText(
            header,
            text,
            (4, _HEADER_LINE_PX * row + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return header


_PROGRESS_ROW_PX = 22
_PROGRESS_LABEL_PX = 76
_PROGRESS_LEGEND_PX = 22
# cleared / attempted-but-never-passed / never attempted.
_PROGRESS_COLORS = ((60, 170, 60), (205, 75, 75), (205, 205, 205))
_PROGRESS_LEGEND = ("pass", "fail", "n/a")


def _render_progress(groups: list[tuple[str, int, int, int]]) -> np.ndarray:
    """One stacked horizontal bar per arena group (see
    `ArenaSelector.progress_by_group`): what share of it the agent has ever
    passed, tried without passing, and not yet been given."""
    passed = sum(group[1] for group in groups)
    total = sum(sum(group[1:]) for group in groups)
    header = _draw_header([f"passed:{passed}/{total}"], _RENDER_SIZE_PX)
    header_px = header.shape[0]
    height = header_px + _PROGRESS_ROW_PX * len(groups) + _PROGRESS_LEGEND_PX
    canvas = np.full((height, _RENDER_SIZE_PX, 3), 240, dtype=np.uint8)
    canvas[:header_px] = header

    bar_x = _PROGRESS_LABEL_PX
    bar_width = _RENDER_SIZE_PX - bar_x - 6
    for row, (name, *counts) in enumerate(groups):
        top = header_px + row * _PROGRESS_ROW_PX
        cv2.putText(
            canvas,
            name,
            (4, top + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        # Walk cumulative counts so the segments always tile the bar exactly.
        left = bar_x
        filled = 0
        for count, color in zip(counts, _PROGRESS_COLORS):
            filled += count
            right = bar_x + round(bar_width * filled / sum(counts))
            if right > left:
                cv2.rectangle(canvas, (left, top + 3), (right, top + 18), color, cv2.FILLED)
            left = right

    legend_top = height - _PROGRESS_LEGEND_PX
    for i, (label, color) in enumerate(zip(_PROGRESS_LEGEND, _PROGRESS_COLORS)):
        swatch_x = 6 + i * 84
        cv2.rectangle(
            canvas, (swatch_x, legend_top + 6), (swatch_x + 14, legend_top + 16), color, cv2.FILLED
        )
        cv2.putText(
            canvas,
            label,
            (swatch_x + 18, legend_top + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _fit_square(image: np.ndarray, size_px: int) -> np.ndarray:
    """Scale a camera frame to the pane the schematic would have filled."""
    return cv2.resize(image, (size_px, size_px), interpolation=cv2.INTER_NEAREST)


def _render_topdown(items: list[dict], agent_xyz: tuple[float, float, float] | None) -> np.ndarray:
    scale = _RENDER_SIZE_PX / _ARENA_SIZE_M
    canvas = np.full((_RENDER_SIZE_PX, _RENDER_SIZE_PX, 3), 240, dtype=np.uint8)

    # Flip z so the +z arena axis points up in the image (Unity-editor-like).
    def to_px(x: float, z: float) -> tuple[int, int]:
        return int(round(x * scale)), int(round((_ARENA_SIZE_M - z) * scale))

    cv2.rectangle(canvas, (0, 0), (_RENDER_SIZE_PX - 1, _RENDER_SIZE_PX - 1), (60, 60, 60), 2)

    for item in items:
        if item["name"] == "Agent":
            continue
        # x or z = -1 in the yaml means Unity randomizes the position at
        # reset; we don't know the actual location, so skip drawing.
        if item["x"] < 0 or item["z"] < 0:
            continue
        cx, cy = to_px(item["x"], item["z"])
        color = _ITEM_COLORS.get(item["name"], _DEFAULT_ITEM_COLOR)
        # Goals are spherical in the Unity scene; draw as circles so they
        # are visually distinct from rectangular zones/walls/boxes.
        if "Goal" in item["name"]:
            radius = max(int(0.5 * item["size_x"] * scale), 3)
            cv2.circle(canvas, (cx, cy), radius, color, cv2.FILLED)
            cv2.circle(canvas, (cx, cy), radius, (20, 20, 20), 1)
            continue
        sx_px = max(item["size_x"] * scale, 3.0)
        sz_px = max(item["size_z"] * scale, 3.0)
        # Unity Y-axis rotation is CW from above (in left-handed world);
        # our z-flipped image preserves "north up" so we pass the raw angle
        # to cv2 (positive cv2 angle is CW in image after the z flip).
        rect = ((float(cx), float(cy)), (sx_px, sz_px), item["rotation"])
        box = np.intp(cv2.boxPoints(rect))
        transparent = "Transparent" in item["name"]
        cv2.drawContours(canvas, [box], 0, color, 1 if transparent else cv2.FILLED)

    if agent_xyz is not None:
        ax, _ay, az = agent_xyz
        apx, apy = to_px(ax, az)
        radius = max(int(0.6 * scale), 4)
        cv2.circle(canvas, (apx, apy), radius, _AGENT_COLOR, cv2.FILLED)
        cv2.circle(canvas, (apx, apy), radius, (20, 20, 20), 1)

    return canvas


class AnimalAIEnv(gym.Env):
    """Animal-AI environment with continuous Box action space.

    `selector` decides which arena each episode loads; see the module
    docstring for the three of them.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        resolution: int,
        seed: int,
        base_port: int,
        binary_path: str,
        continuous_action: bool,
        topdown_camera: bool,
        topdown_resolution: int,
        end_at_pass_mark: bool,
        selector: ArenaSelector,
    ):
        super().__init__()
        self.continuous_action = continuous_action
        self.topdown_camera = topdown_camera
        self.topdown_resolution = topdown_resolution
        self.end_at_pass_mark = end_at_pass_mark
        self.selector = selector
        self._arena_by_name = {arena.name: arena for arena in selector.arenas}
        # Framing text shared by every arena; the arena's own instruction comes
        # from the prompt csv. `make_env` replaces this with the action encoding.
        self.prompt = "You control the agent in Animal-AI (first-person view)."
        self._arena_prompts = _load_arena_prompts()
        assert all(
            arena.name.rsplit("-", 1)[0] in self._arena_prompts for arena in selector.arenas
        ), f"{ARENA_PROMPT_CSV} is missing a row for an arena of the selector"

        self.binary_path = str(Path(binary_path).expanduser())
        self.resolution = resolution
        self.seed_value = seed
        # Add jitter so parallel envs don't fight over the same socket.
        self.base_port = base_port + random.randint(0, 1000)
        self.render_mode = "rgb_array"

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(resolution, resolution, 3), dtype=np.uint8
        )

        self._aai: AnimalAIEnvironment | None = None
        self._behavior_name: str | None = None
        # How many continuous actions the binary accepts: 0 for an official
        # build, 2 for one rebuilt with the hybrid action spec. Read off the
        # behavior spec at launch.
        self._continuous_size = 0
        self._latest_image: np.ndarray | None = None
        # The overhead camera's view, on the binaries that have one. It is not
        # part of `observation_space`: `render` draws it, the policy never sees it.
        self._latest_topdown_image: np.ndarray | None = None
        self.global_step = 0
        self.episode_step = 0
        self.arena_name: str = ""
        self.arena_prompt: str = ""
        self.pass_mark: float = 0.0
        self._arena: Arena | None = None
        self._arena_items: list[dict] = []
        self._episode_return = 0.0
        # (x, y, z) agent position in arena coords; populated each step from
        # the AAI vector observation. None before the first reset.
        self._agent_xyz: tuple[float, float, float] | None = None
        # (vx, vy, vz) agent velocity from the AAI vector observation.
        self._agent_velocity: np.ndarray = np.zeros(3, dtype=np.float32)
        # Agent health from the AAI vector observation: it decays at a rate set
        # by the arena's `t`, refills whenever a reward is collected, and the
        # episode ends when it hits 0 -- the real "time left" of the episode.
        self._agent_health: float = 0.0

    def set_global_step(self, global_step: int) -> None:
        """Resume hook: restore the step counter the curriculum schedules on."""
        self.global_step = int(global_step)

    def get_curriculum_state(self) -> dict:
        return self.selector.state()

    def set_curriculum_state(
        self, arena_attempts: dict, arena_successes: dict, progress: dict
    ) -> None:
        self.selector.load_state(arena_attempts, arena_successes, progress)

    def _ensure_started(self):
        if self._aai is not None:
            return
        # Queued before the constructor's first reset, which is when the
        # channel's messages reach Unity.
        parameters_channel = EnvironmentParametersChannel()
        parameters_channel.set_float_parameter(
            CONTINUOUS_ACTIONS_KEY, float(self.continuous_action)
        )
        environment_class = _aai_environment_class(
            [
                "--topDownCamera",
                str(int(self.topdown_camera)),
                "--topDownResolution",
                str(self.topdown_resolution),
            ]
        )
        self._aai = environment_class(
            side_channels=[parameters_channel],
            file_name=self.binary_path,
            arenas_configurations=str(self.selector.arenas[0].path),
            seed=self.seed_value,
            play=False,
            useCamera=True,
            resolution=self.resolution,
            useRayCasts=False,
            # One decision per 5 Academy steps (= 5 physics steps of 0.02 s),
            # so an arena's `t` costs t*5 physics steps and an episode advances
            # in 0.1 s of simulated time. This is the only one of these knobs
            # that changes what the agent experiences.
            decisionPeriod=5,
            # Run the simulation as fast as it will go: `timescale` matches the
            # paper's training scripts, and no frame rate cap. Neither changes
            # the agent's experience -- the physics step is fixed at 0.02 s and
            # decisions are counted in Academy steps, not frames -- so these
            # only buy wall-clock speed (~9x over the default timescale=1).
            timescale=300,
            targetFrameRate=-1,
            # Left at the AAI default (real time drives frame duration) as the
            # paper's scripts had it. Setting it would only change how the same
            # physics steps are split across frames.
            captureFrameRate=0,
            # `no_graphics=True` (the alternative for headless) disables the
            # renderer entirely and produces a solid-color image, which is
            # unusable for vision policies.
            no_graphics=False,
            base_port=self.base_port,
            inference=False,
            use_YAML=True,
        )
        self._behavior_name = next(iter(self._aai.behavior_specs.keys()))
        spec = self._aai.behavior_specs[self._behavior_name]
        # ML-Agents sorts an agent's sensors by name, so the observations arrive
        # as ["CameraSensor", "TopDownCameraSensor", "VectorSensor"] -- first
        # person, overhead, scalars. The overhead camera is there only on a
        # rebuilt binary running with topdown_camera, which leaves the scalars
        # last either way.
        self._observation_count = len(spec.observation_specs)
        assert self._observation_count in (2, 3), self._observation_count
        self._continuous_size = spec.action_spec.continuous_size
        assert self._continuous_size == 2 or not self.continuous_action, (
            f"{self.binary_path} exposes {self._continuous_size} continuous actions; "
            "continuous_action=True needs a binary rebuilt from animal-ai-unity."
        )

    def _decode_obs(self, obs_chw_float: np.ndarray) -> np.ndarray:
        # AAI emits float32 in [0, 1] with shape (3, H, W).
        return (obs_chw_float.transpose(1, 2, 0) * 255.0).astype(np.uint8)

    def _read_observation(self, steps) -> None:
        # Vector obs layout (useCamera=True, useRayCasts=False): the last
        # observation is [health, vx, vy, vz, x, y, z].
        vec = steps.obs[self._observation_count - 1][0]
        self._latest_image = self._decode_obs(steps.obs[0][0])
        # Watched in `render`, never handed to the policy.
        self._latest_topdown_image = (
            self._decode_obs(steps.obs[1][0]) if self._observation_count == 3 else None
        )
        self._agent_health = float(vec[0])
        self._agent_xyz = (float(vec[4]), float(vec[5]), float(vec[6]))
        self._agent_velocity = np.array(
            [float(vec[1]), float(vec[2]), float(vec[3])], dtype=np.float32
        )

    def _build_prompt(self) -> str:
        """Task text plus the current scalar observations, so a language-conditioned
        policy reads the same values the scalar observation vector carries."""
        vx, vy, vz = self._agent_velocity
        return (
            f"{self.prompt} "
            f"Task: {self.arena_prompt}. "
            f"Velocity: ({vx:+.2f}, {vy:+.2f}, {vz:+.2f}). "
            f"Return so far: {self._episode_return:+.2f}. "
            f"Pass mark: {self.pass_mark:+.2f}. "
            f"Return needed: {self.pass_mark - self._episode_return:+.2f}. "
            f"Health: {self._agent_health:.2f}. "
            f"Global step: {self.global_step}. "
            f"Episode step: {self.episode_step}."
        )

    def _shape_reward(self, reward: float, episode_done: bool) -> float:
        """Reaching a goal is worth more than the arena says, climbing is encouraged,
        walking backwards is discouraged, and an episode ends with a bonus for having
        cleared the arena's pass mark or a penalty for not."""
        velocity = self._agent_velocity
        if reward > 0.1:
            reward += GOAL_BONUS
        if velocity[1] > 0.01:
            reward += float(velocity[1]) * RAMPS_COEF
        if velocity[2] < 0:
            reward += float(velocity[2]) * BACK_MOVE_COEF
        if episode_done:
            cleared = self._episode_return >= self.pass_mark
            reward += PASS_MARK_BONUS if cleared else -PASS_MARK_BONUS
        return reward

    def _build_info(self, shaped_reward: float) -> dict:
        info = {
            "task_prompt": self._build_prompt(),
            "shaped_reward": shaped_reward,
            "arena_name": self.arena_name,
            "arena_yaml": str(self._arena.path),
            "pass_mark": self.pass_mark,
            "global_step": self.global_step,
            "episode_step": self.episode_step,
            "health": self._agent_health,
            "velocity": self._agent_velocity,
            "agent_xyz": self._agent_xyz,
        }
        info.update(self.selector.info(self.global_step))
        return info

    def reset(self, seed: int | None, options: dict | None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._ensure_started()

        # `options={"arena_stem": name}` pins one arena instead of asking the
        # selector (scripts/collect_probe_data.py repeats a single arena).
        forced_name = options["arena_stem"] if options is not None else None
        self._arena = (
            self._arena_by_name[forced_name]
            if forced_name is not None
            else self.selector.next_arena(self.global_step)
        )

        self.arena_name = self._arena.name
        self.arena_prompt = self._arena_prompts[self.arena_name.rsplit("-", 1)[0]]
        self.pass_mark, self._arena_items = _parse_arena(self._arena.path)
        self._aai.reset(arenas_configurations=str(self._arena.path))
        self.episode_step = 0
        self._episode_return = 0.0

        decision_steps, _ = self._aai.get_steps(self._behavior_name)
        self._read_observation(decision_steps)
        return self._latest_image, self._build_info(0.0)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.global_step += 1
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0).reshape(1, 2)
        # Both halves go out every step; which one Unity obeys was fixed at
        # launch by CONTINUOUS_ACTIONS_KEY. The slice is empty for an official
        # binary, which has no continuous actions to fill.
        action_tuple = ActionTuple(
            continuous=a[:, : self._continuous_size],
            discrete=np.array(
                [[_to_discrete(float(a[0, 0])), _to_discrete(float(a[0, 1]))]], dtype=np.int32
            ),
        )
        self._aai.set_actions(self._behavior_name, action_tuple)
        self._aai.step()

        decision_steps, terminal_steps = self._aai.get_steps(self._behavior_name)
        episode_over = len(terminal_steps) > 0
        # `interrupted` is AAI running the agent's health down to 0, i.e. a time
        # limit rather than a real terminal (goal reached / death zone).
        interrupted = episode_over and bool(terminal_steps.interrupted[0])
        steps = terminal_steps if episode_over else decision_steps
        reward = float(steps.reward[0])
        self._read_observation(steps)

        self.episode_step += 1
        self._episode_return += reward
        terminated = episode_over and not interrupted
        truncated = interrupted

        # Unity never ends an arena on the pass mark, so without this an agent
        # that has earned it keeps bleeding 1/t per step. The reward gate keeps
        # a negative pass mark from being cleared by the first step's -1/t.
        if self.end_at_pass_mark and not episode_over and reward > 0.0:
            terminated = terminated or self._episode_return >= self.pass_mark

        shaped_reward = self._shape_reward(reward, terminated or truncated)

        if not (episode_over or terminated):
            return (
                self._latest_image,
                reward,
                terminated,
                truncated,
                self._build_info(shaped_reward),
            )

        success = self._episode_return >= self.pass_mark
        # Before _build_info, so the selector reports post-episode progress.
        self.selector.on_episode_end(self._arena, success)
        info = self._build_info(shaped_reward)
        info["success"] = bool(success)
        return self._latest_image, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """This episode's arena from above, beside the arena set's coverage.

        The arena pane is the overhead camera when the binary renders one, and
        otherwise a schematic drawn from the arena yaml -- the camera shows where
        everything is, the schematic only where everything started.
        """
        if self.render_mode != "rgb_array":
            return None
        attempts, successes = self.selector.arena_record(self._arena)
        header_lines = [
            f"{self.arena_name}  {successes}/{attempts}",
            self.selector.status(self.global_step),
        ]
        arena = (
            _fit_square(self._latest_topdown_image, _RENDER_SIZE_PX)
            if self._latest_topdown_image is not None
            else _render_topdown(self._arena_items, self._agent_xyz)
        )
        topdown = np.vstack([_draw_header(header_lines, _RENDER_SIZE_PX), arena])
        progress = _render_progress(self.selector.progress_by_group())
        padding = np.full(
            (topdown.shape[0] - progress.shape[0], progress.shape[1], 3), 240, dtype=np.uint8
        )
        return np.hstack([topdown, np.vstack([progress, padding])])

    def close(self):
        if self._aai is not None:
            self._aai.close()
            self._aai = None


if __name__ == "__main__":
    env = AnimalAIEnv(
        resolution=96,
        seed=0,
        base_port=5005,
        binary_path="~/animalai_env/Linux/animalAI.x86_64",
        continuous_action=False,
        topdown_camera=False,
        topdown_resolution=96,
        end_at_pass_mark=True,
        selector=StagedSelector(variant="01", steps_per_stage=2_000_000, seed=0),
    )
    for episode in range(8):
        obs, info = env.reset(seed=episode, options=None)
        print(f"ep={episode} arena={info['arena_name']} stage={info['stage']}")
        total_reward = 0.0
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total_reward += reward
            if terminated or truncated:
                break
        print(f"  return={total_reward:.4f} success={info.get('success')}")
    env.close()
