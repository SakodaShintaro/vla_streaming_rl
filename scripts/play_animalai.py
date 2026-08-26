# SPDX-License-Identifier: MIT
"""Play the Animal-AI curriculum arenas by hand.

Merges every arena YAML under ``--arena-root`` into one multi-arena
ArenaConfig and opens it in the Unity player, where W/A/S/D drives the agent
and R (or collecting a reward) moves on to the next arena in the file.

The arenas are sorted by path and then reversed, so play starts at the last
one (``stage09/arena029.yaml`` for the paper curriculum) and works backwards
towards ``stage00/arena000.yaml``.

Every arena's time limit is multiplied by ``--time-scale`` (100 by default) so
a human has time to look around, and the ``blackouts`` frames of the 42 dark
arenas are scaled with it so the lights still go out at the same point of the
episode. ``--no-time-limit`` instead writes ``t: 0``, which in Animal-AI means
the episode only ends when a reward is collected; the environment rejects a
blackout list longer than the episode, so that mode drops the ``blackouts``
lines and those arenas simply stay lit.

``--manual`` drives the agent from Python instead of handing the window to
Unity's own play mode. That is the only way to see the score: in play mode the
Unity player owns the simulation and reports nothing back, so nothing can read
the reward. In manual mode the terminal shows a live line with the arena's
pass mark and the return so far, and the arena ends as soon as the return
reaches the pass mark. The cost is that Unity then renders its training camera
(third person, no C-key switching) and the keys have to be typed into the
terminal, not the Unity window.

The rewritten configs go to a temporary file or directory -- the submodule's
arena YAMLs are never modified.

Run it in an environment that has the animalai package, e.g.

    uv run --no-project --python 3.10 --with animalai==5.0.1 \
        python scripts/play_animalai.py

Controls (play mode): W/S move, A/D turn, C switches camera, R moves to the
next arena, Q quits. In ``--manual`` mode every key is typed into the terminal
instead: W/A/S/D drive, N skips to the next arena, R restarts the current one
and Q quits.
"""

import argparse
import os
import random
import re
import select
import sys
import tempfile
import termios
import time
import tty
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARENA_ROOT = REPO_ROOT / "external/animal-ai/configs/paper_curriculum_split"
DEFAULT_BINARY = Path.home() / "animalai_env/Linux/animalAI.x86_64"

# "    t: 500" / "    timeLimit: 500" -- the key name differs by AAI version,
# so rewrite whichever one the file uses and keep it.
TIME_LIMIT_RE = re.compile(r"^(\s*)(t|timeLimit):\s*([0-9]+)\s*$")
# "    blackouts: [15, 60]" (frames the lights switch at) or "[-20]" (switch
# every 20 frames). Both are counted in the same units as the time limit.
BLACKOUTS_RE = re.compile(r"^(\s*)blackouts:\s*\[([-0-9,\s]*)\]\s*$")
PASS_MARK_RE = re.compile(r"^\s*(pass_mark|passMark):\s*(-?[0-9.]+)\s*$")

# How long a key press keeps driving the agent. Terminal auto-repeat is far
# slower than the decision rate, so without this the agent would move in
# single-step twitches.
KEY_HOLD_SECONDS = 0.25
# AAI's discrete branches: [forward/back, rotate]. Rotation 1 turns clockwise
# (measured by walking the agent before and after a turn).
NOOP, FORWARD, BACKWARD = 0, 1, 2
TURN_RIGHT, TURN_LEFT = 1, 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena-root", type=Path, default=DEFAULT_ARENA_ROOT)
    parser.add_argument(
        "--time-scale",
        type=int,
        default=100,
        help="multiply every arena's time limit by this (default: 100)",
    )
    parser.add_argument(
        "--no-time-limit",
        action="store_true",
        help="write t: 0 instead (episode ends only on a reward) and drop every blackout",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="drive from Python: show the arena name, pass mark and return, and end on a pass",
    )
    parser.add_argument(
        "--forward",
        action="store_true",
        help="play in sorted order (stage00 first) instead of reversed",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="play only the first N arenas of the order (0: all)"
    )
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument(
        "--out", type=Path, default=None, help="write the merged config here instead of /tmp"
    )
    return parser.parse_args()


def show(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)


def rescale_time(line: str, time_scale: int, no_time_limit: bool) -> str | None:
    """Rewrite one time-limit or blackout line; None means drop the line."""
    time_limit_match = TIME_LIMIT_RE.match(line)
    if time_limit_match is not None:
        indent, key, value = time_limit_match.groups()
        new_value = 0 if no_time_limit else int(value) * time_scale
        return f"{indent}{key}: {new_value}\n"

    blackouts_match = BLACKOUTS_RE.match(line)
    if blackouts_match is not None:
        if no_time_limit:
            return None
        indent, frames = blackouts_match.groups()
        scaled = [str(int(frame) * time_scale) for frame in frames.split(",") if frame.strip()]
        return f"{indent}blackouts: [{', '.join(scaled)}]\n"

    return line


def single_arena_config(path: Path, time_scale: int, no_time_limit: bool) -> str:
    """One arena's own config, time-rescaled, header and all."""
    out = []
    for line in path.read_text().splitlines(keepends=True):
        rewritten = rescale_time(line, time_scale, no_time_limit)
        if rewritten is not None:
            out.append(rewritten)
    return "".join(out)


def arena_pass_mark(text: str) -> float:
    marks = []
    for line in text.splitlines():
        match = PASS_MARK_RE.match(line)
        if match is not None:
            marks.append(float(match.group(2)))
    assert len(marks) == 1, f"expected exactly one pass mark, found {marks}"
    return marks[0]


def merge_arenas(paths: list[Path], time_scale: int, no_time_limit: bool) -> str:
    """Concatenate the single-arena configs into one, renumbering the arena keys."""
    out = ["!ArenaConfig\n", "randomizeArenas: false\n", "arenas:\n"]
    for index, path in enumerate(paths):
        lines = path.read_text().splitlines(keepends=True)
        # Every file is "!ArenaConfig\narenas:\n  0: !Arena\n    ...", so drop
        # the header lines and renumber the arena key.
        arena_key_lines = [i for i, line in enumerate(lines) if re.match(r"^\s+\d+: !Arena", line)]
        assert len(arena_key_lines) == 1, f"{path} does not hold exactly one arena"
        out.append(f"  # {show(path)}\n")
        out.append(f"  {index}: !Arena\n")
        for line in lines[arena_key_lines[0] + 1 :]:
            rewritten = rescale_time(line, time_scale, no_time_limit)
            if rewritten is not None:
                out.append(rewritten)
        if not out[-1].endswith("\n"):
            out.append("\n")
    return "".join(out)


def write_config(merged: str, out: Path | None) -> Path:
    if out is not None:
        out.write_text(merged)
        return out
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="aai_play_", delete=False)
    handle.write(merged)
    handle.close()
    return Path(handle.name)


def prepare_configs(paths: list[Path], time_scale: int, no_time_limit: bool) -> list[Path]:
    """Write one time-rescaled config per arena, in play order."""
    config_dir = Path(tempfile.mkdtemp(prefix="aai_play_"))
    configs = []
    for index, path in enumerate(paths):
        config = config_dir / f"{index:04d}_{path.parent.name}_{path.name}"
        config.write_text(single_arena_config(path, time_scale, no_time_limit))
        configs.append(config)
    print(f"  configs: {config_dir}")
    return configs


def arena_line(index: int, total: int, path: Path, pass_mark: float) -> str:
    return f"[{index + 1}/{total}] {path.parent.name}/{path.stem}  pass mark {pass_mark:+.2f}"


class TerminalKeys:
    """Read single key presses without waiting for Enter."""

    def __init__(self, file_descriptor: int):
        self.file_descriptor = file_descriptor
        self.saved_attributes = termios.tcgetattr(file_descriptor)

    def __enter__(self) -> "TerminalKeys":
        tty.setcbreak(self.file_descriptor)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        termios.tcsetattr(self.file_descriptor, termios.TCSADRAIN, self.saved_attributes)

    def pressed(self) -> str:
        """Every key typed since the last call, oldest first."""
        typed = ""
        while select.select([self.file_descriptor], [], [], 0)[0]:
            typed += os.read(self.file_descriptor, 32).decode("utf-8", "ignore")
        return typed.lower()


def action_from_keys(
    typed: str, held: tuple[int, int], held_at: float
) -> tuple[tuple[int, int], float]:
    """Fold the keys typed since the last step into (forward, turn) and its timestamp."""
    forward, turn = held
    now = time.monotonic()
    pressed_at = held_at
    for key in typed:
        if key == "w":
            forward, pressed_at = FORWARD, now
        elif key == "s":
            forward, pressed_at = BACKWARD, now
        elif key == "d":
            turn, pressed_at = TURN_RIGHT, now
        elif key == "a":
            turn, pressed_at = TURN_LEFT, now
    if now - pressed_at > KEY_HOLD_SECONDS:
        return (NOOP, NOOP), pressed_at
    return (forward, turn), pressed_at


def run_manual(paths: list[Path], configs: list[Path], binary: Path) -> None:
    """Play the arenas one by one, scoring them from Python."""
    assert sys.stdin.isatty(), "--manual reads keys from the terminal, so run it in one"

    from animalai import AnimalAIEnvironment
    from mlagents_envs.base_env import ActionTuple

    environment = AnimalAIEnvironment(
        file_name=str(binary),
        arenas_configurations=str(configs[0]),
        base_port=5005 + random.randint(0, 1000),
        play=False,
        inference=True,
        useCamera=True,
        resolution=64,
        useRayCasts=False,
        timescale=1,
        targetFrameRate=60,
    )
    behavior_name = next(iter(environment.behavior_specs.keys()))

    try:
        with TerminalKeys(sys.stdin.fileno()) as keys:
            for index, config in enumerate(configs):
                pass_mark = arena_pass_mark(config.read_text())
                replay = True
                while replay:
                    replay = False
                    environment.reset(arenas_configurations=str(config))
                    print(arena_line(index, len(configs), paths[index], pass_mark), flush=True)
                    episode_return = 0.0
                    held, held_at = (NOOP, NOOP), 0.0
                    while True:
                        decision, terminal = environment.get_steps(behavior_name)
                        if len(terminal) > 0:
                            episode_return += float(terminal.reward[0])
                            break
                        episode_return += float(decision.reward[0])
                        health = float(decision.obs[1][0][0])
                        print(
                            f"\r[{index + 1}/{len(configs)}] {paths[index].stem} "
                            f"({paths[index].parent.name})  "
                            f"return {episode_return:+.2f} / pass {pass_mark:+.2f}  "
                            f"health {health:5.1f}   ",
                            end="",
                            flush=True,
                        )
                        # A pass mark of 0 or less is met by simply staying
                        # alive, so only a positive score ends an arena early.
                        if episode_return > 0.0 and episode_return >= pass_mark:
                            break

                        typed = keys.pressed()
                        if "q" in typed:
                            print("\nquit")
                            return
                        if "n" in typed:
                            break
                        if "r" in typed:
                            replay = True
                            break

                        held, held_at = action_from_keys(typed, held, held_at)
                        actions = ActionTuple(
                            continuous=np.zeros((1, 0), dtype=np.float32),
                            discrete=np.array([list(held)], dtype=np.int32),
                        )
                        environment.set_actions(behavior_name, actions)
                        environment.step()

                    verdict = "PASS" if episode_return >= pass_mark else "fail"
                    if replay:
                        verdict = "restart"
                    print(
                        f"\r[{index + 1}/{len(configs)}] {show(paths[index])}  "
                        f"return {episode_return:+.2f} / pass {pass_mark:+.2f}  {verdict}"
                        + " "
                        * 20
                    )
    finally:
        environment.close()


def run_play(config_path: Path, binary: Path) -> None:
    """Hand the window to Unity's play mode and wait for the player to close it."""
    from animalai import AnimalAIEnvironment

    environment = AnimalAIEnvironment(
        file_name=str(binary),
        arenas_configurations=str(config_path),
        base_port=5005 + random.randint(0, 1000),
        play=True,
    )
    # The constructor returns as soon as Unity is up, so block on the player
    # process itself; Q in the window (or Ctrl+C here) ends the session.
    try:
        environment._process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        environment.close()


def main() -> None:
    args = parse_args()
    assert args.arena_root.is_dir(), f"no such arena directory: {args.arena_root}"
    assert args.binary.exists(), f"Animal-AI binary not found: {args.binary}"

    paths = sorted(args.arena_root.rglob("*.yaml"))
    assert paths, f"no arena YAML under {args.arena_root}"
    if not args.forward:
        paths = paths[::-1]
    if args.limit > 0:
        paths = paths[: args.limit]

    limit_desc = "no time limit" if args.no_time_limit else f"time limit x{args.time_scale}"
    print(f"{len(paths)} arenas, {limit_desc}")
    print(f"  first: {show(paths[0])}")
    print(f"  last:  {show(paths[-1])}")

    if args.manual:
        configs = prepare_configs(paths, args.time_scale, args.no_time_limit)
        print("Controls (type here): W/S move, A/D turn, N next, R restart, Q quit", flush=True)
        run_manual(paths, configs, args.binary)
        return

    config_path = write_config(merge_arenas(paths, args.time_scale, args.no_time_limit), args.out)
    print(f"  merged config: {config_path}")
    print("Controls: W/S move, A/D turn, C camera, R next arena, Q quit", flush=True)
    run_play(config_path, args.binary)


if __name__ == "__main__":
    main()
