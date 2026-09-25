# SPDX-License-Identifier: MIT
"""Animal-AI arena catalog and per-episode arena selection.

Pure logic with no Unity dependency: which arenas exist, which of them a run
trains on, and which one the next episode loads. `AnimalAIEnv` (see
`animalai_env.py`) drives a selector built here, picked by the `mode` config
field (see `build_selector`):

  - "staged"  training with the paper's cumulative 11-stage curriculum, one
              stage per competition level, sampled uniformly within the stage
              (`StagedSelector`).
  - "success" one stage per level, advanced by clearing a round of it: the
              stage's whole pool (30 arenas at stage 1, 60 at stage 2, up to
              300) drawn without replacement, then the next stage if the
              round's success rate reached `advance_success_rate`
              (`SuccessDrivenSelector`).
  - "sequential" no curriculum: every training arena once, in label order,
              then the run ends (`SequentialSelector`, not cycling).
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
"""

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

COMPETITION_DIR = Path("./external/animal-ai/configs/competition")


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


@lru_cache(maxsize=None)
def _load_arena_yaml(yaml_path: Path) -> dict:
    """The first arena mapping of an arena yaml, parsed once per file: both
    `parse_arena` and `_arena_signature` read from here, and `seen_in_training`
    hashes the whole 900-arena set, so the cache is what keeps that a single
    parse per file. The mapping is shared -- callers read, never mutate."""
    return yaml.load(yaml_path.read_text(), Loader=_AAILoader)["arenas"][0]


def parse_arena(yaml_path: Path) -> tuple[float, list[dict]]:
    """Return (pass_mark, items) from an Animal-AI arena yaml.

    The arena's ``t`` is deliberately not read: it is not a step cap but the
    decay rate of the agent's health, and collecting a reward refills that
    health, so an episode routinely runs well past ``t`` steps. What is left of
    the episode is the health value in the AAI vector observation, not a step
    count, so that is what the agent is given (see `AnimalAIEnv._read_observation`).

    Each item dict carries one (position, size, rotation) triple in arena
    coordinates: {name, x, z, size_x, size_z, rotation}. yaml entries with
    multiple positions are expanded into multiple item dicts; if `sizes` is
    shorter than `positions` the last given size is reused (AAI's convention).
    """
    arena = _load_arena_yaml(yaml_path)
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


def _variant_arenas(variant: str) -> list[Arena]:
    """The `variant` numbered copy of every Olympics task, at every level.

    Each task family XX-YY ships three variants, so one variant is one arena
    per family -- 300 of the 900, which is the arena set the paper's
    curriculum trains on. The other variants are what eval scores.
    """
    arenas = [arena for arena in _competition_arenas() if arena.name.rsplit("-", 1)[1] == variant]
    assert arenas, f"no competition arena has variant {variant!r}"
    return arenas


def _all_levels(variant: str) -> list[str]:
    """Every Olympics level the `variant` copy has an arena at, in order."""
    return sorted({arena.name.split("-")[0] for arena in _variant_arenas(variant)})


def training_levels(mode: str, variant: str, train_levels: list[str]) -> list[str]:
    """The levels a run in `mode` trains on: `train_levels` for the sweep modes,
    every level for the curriculum modes, whose stages are the levels."""
    if mode in ("sequential", "random"):
        return list(train_levels)
    return _all_levels(variant)


def _training_arenas(variant: str, levels: list[str]) -> list[Arena]:
    """The training set: the `variant` copy of every task of each of `levels`,
    one block of arenas per entry, in the order the entries are given.

    `levels` narrowed below every level is for looking at some levels on their
    own -- what a run scores on levels 01 and 02 without waiting for a
    curriculum to reach them. A repeated entry repeats its block, so
    `["01", "01"]` sweeps level 01 twice in the sequential mode and doubles
    its draw weight in the random mode. The curriculum modes ignore it: their
    stages are the levels, and hiding levels from them would leave the stage
    numbers meaning something else.
    """
    assert len(levels) > 0, "levels must name at least one level"
    available = _all_levels(variant)
    assert all(level in available for level in levels), (
        f"levels {list(levels)} names a level with no arena at variant {variant!r}; "
        f"available: {available}"
    )
    arenas = _variant_arenas(variant)
    return [arena for level in levels for arena in arenas if arena.name.split("-")[0] == level]


def _training_stages(variant: str) -> list[list[Arena]]:
    """The training arenas grouped into one stage per curriculum level.

    The Olympics levels are the paper's curriculum stages in order (01 food
    retrieval ... 10 causal reasoning), and every level holds 30 tasks, so
    each stage is 30 arenas. `StagedSelector` accumulates them; the rounds of
    `SuccessDrivenSelector` are one stage each.
    """
    arenas = _variant_arenas(variant)
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


@lru_cache(maxsize=None)
def _arena_signature(path: Path) -> str:
    """Digest of what makes two arena yamls the same episode.

    Item order and mapping key order are how a file happens to be written,
    not what it runs, so both are canonicalized away; `pass_mark` and `t`
    decide the episode alongside the items, so both are in. Used to find the
    eval arenas that are repeats of a training arena -- within a task family
    the variants are often the same arena written out three times, and a score
    over those is not a held-out score.
    """
    arena = _load_arena_yaml(path)
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


def seen_in_training(variant: str, levels: list[str]) -> set[str]:
    """Labels of the eval arenas a run trained on `variant` has already seen.

    The training variant itself, plus every other variant whose arena is
    identical to the one its family contributed to training. `levels` is the
    run's `train_levels`, so a run held to some levels is not credited with the
    rest.
    """
    trained = {_arena_signature(arena.path) for arena in _training_arenas(variant, levels)}
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

    def arena_by_name(self, name: str) -> Arena:
        return self._arena_by_name[name]

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

    def cleared_count(self) -> int:
        return sum(self.is_cleared(arena) for arena in self.arenas)

    def progress_by_group(self) -> list[tuple[str, int, int]]:
        """(group, successes, failures) per arena group over every episode run
        so far, in group order.

        Episodes rather than arenas, so the bar reads as the group's success
        rate over the whole run and not as whether each arena was ever passed
        once. The group is the arena label's first "-" separated token, which
        is the competition level ("01-01-01" -> "01") and so also the
        curriculum stage.
        """
        counts: dict[str, list[int]] = {}
        for arena in self.arenas:
            group = counts.setdefault(arena.name.split("-")[0], [0, 0])
            attempts, successes = self._attempts[arena.name], self._successes[arena.name]
            group[0] += successes
            group[1] += attempts - successes
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
        return {
            "arena_attempts": attempts,
            "arena_successes": {name: self._successes[name] for name in attempts},
            "arena_cleared": {
                name: self.is_cleared(self._arena_by_name[name]) for name in attempts
            },
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
        return f"stage:{self._stage_index(global_step) + 1}/{len(self._stages)}"


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
        self._round_by_level: dict[str, list[int]] = {}
        self._last_round_rate = 0.0
        self._last_round_level_rate: dict[str, float] = {}
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
        level = self._round_by_level.setdefault(arena.name.split("-")[0], [0, 0])
        level[0] += int(success)
        level[1] += 1
        self._advanced = False
        if self._round_attempts < len(self._stages[self._stage]):
            return
        # The counters reset here but info() is read after this call, so the
        # rate the decision was made on is kept rather than reported as 0/0.
        self._last_round_rate = self._round_rate()
        self._last_round_level_rate = {
            level: successes / attempts
            for level, (successes, attempts) in sorted(self._round_by_level.items())
        }
        self._round_by_level = {}
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
            "last_round_level_success_rate": dict(self._last_round_level_rate),
            "advanced": self._advanced,
            "cleared_count": self.cleared_count(),
        }

    def status(self, global_step: int) -> str:
        del global_step
        size = len(self._stages[self._stage])
        rate = self._round_rate() if self._round_attempts > 0 else 0.0
        return (
            f"stage:{self._stage + 1}/{len(self._stages)}"
            f"  round:{self._round_attempts}/{size} rate:{rate:.2f}"
        )

    def progress_state(self) -> dict:
        return {
            "stage": self._stage,
            "queue": list(self._queue),
            "round_attempts": self._round_attempts,
            "round_successes": self._round_successes,
            "round_by_level": {
                level: list(counts) for level, counts in self._round_by_level.items()
            },
            "last_round_rate": self._last_round_rate,
            "last_round_level_rate": dict(self._last_round_level_rate),
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
        self._round_by_level = {
            level: [int(counts[0]), int(counts[1])]
            for level, counts in progress["round_by_level"].items()
        }
        self._last_round_level_rate = {
            level: float(rate) for level, rate in progress["last_round_level_rate"].items()
        }
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
        replayed until train.py's step_limit stops the run.
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
        del global_step
        return (
            f"{(self._next_index - 1) % len(self.arenas) + 1}/{len(self.arenas)}"
            f"  lap:{(self._next_index - 1) // len(self.arenas) + 1}"
            f"  cleared:{self.cleared_count()}/{len(self.arenas)}"
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
        return {"arena_total": len(self.arenas), "cleared_count": self.cleared_count()}

    def status(self, global_step: int) -> str:
        del global_step
        untried = sum(attempts == 0 for attempts in self._attempts.values())
        return f"random  cleared:{self.cleared_count()}/{len(self.arenas)}  untried:{untried}"


def build_selector(
    mode: str,
    train_variant: str,
    train_levels: list[str],
    steps_per_stage: int,
    advance_success_rate: float,
    seed: int,
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
        # One pass and done: the curriculum-free sweep is a measurement of the
        # training set, so a second lap over the same arenas would only mix
        # repeats into the score.
        "sequential": lambda: SequentialSelector(
            arenas=_training_arenas(train_variant, train_levels), cycle=False
        ),
        "random": lambda: RandomSelector(
            arenas=_training_arenas(train_variant, train_levels), seed=seed
        ),
        "eval": lambda: SequentialSelector(arenas=_competition_arenas(), cycle=False),
    }
    assert mode in builders, f"unknown Animal-AI mode {mode!r}; expected one of {sorted(builders)}"
    return builders[mode]()
