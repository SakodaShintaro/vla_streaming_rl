# SPDX-License-Identifier: MIT
"""Screenshot every Animal-AI arena from the player's overhead camera.

Opens the arenas in Unity's play mode the same way scripts/play_animalai.py
does, switches the camera with C, then walks the whole set: let the arena
settle, grab the player window, press R for the next arena. The result is one
PNG per arena, plus contact sheets with --montage so the whole curriculum can
be looked over at once.

--arena-root defaults to the whole 900-arena competition set; point it at a
directory holding a subset to sweep only those:

    python scripts/capture_animalai.py \
        --arena-root /tmp/some_arenas \
        --out-dir /tmp/aai_subset --montage

C cycles first person -> third person -> overhead, and the player starts on
the first of those, so --camera-presses says how many times to press it before
the sweep starts (2 by default). The first PNG shows which camera you got.

Keys reach the player through the X11 XTEST extension, so this needs an X
session and python-xlib:

    uv pip install python-xlib

Input focus is handed to the player window before every key press, so leave
the machine alone while it runs -- and keep the window unobstructed, because
a covered window captures whatever covers it.
"""

import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image
from Xlib import XK, X, display
from Xlib.ext import xtest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from play_animalai import (  # noqa: E402
    DEFAULT_ARENA_ROOT,
    DEFAULT_BINARY,
    merge_arenas,
    show,
    write_config,
)

PLAYER_WINDOW_NAME = "Animal-AI"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena-root", type=Path, default=DEFAULT_ARENA_ROOT)
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.yaml",
        help="glob picking the arenas under --arena-root, e.g. '04-22-*.yaml' for one task",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--camera-presses",
        type=int,
        default=0,
        help="how many times to press C before capturing (default: 2, the overhead camera)",
    )
    parser.add_argument("--settle", type=float, default=0.5)
    parser.add_argument(
        "--limit", type=int, default=0, help="capture only the first N arenas (0: all)"
    )
    parser.add_argument(
        "--time-scale",
        type=int,
        default=100,
        help="multiply every arena's time limit by this so nothing expires mid-sweep",
    )
    parser.add_argument(
        "--montage",
        action="store_true",
        help="also write contact sheets of the screenshots with ImageMagick",
    )
    parser.add_argument("--montage-per-sheet", type=int, default=90)
    parser.add_argument(
        "--montage-width", type=int, default=320, help="thumbnail width on the contact sheet"
    )
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    return parser.parse_args()


def matching_windows(screen_display: display.Display, window, name: str) -> list:
    """Every window in the tree whose title is `name`, deepest first."""
    found = []
    for child in window.query_tree().children:
        found.extend(matching_windows(screen_display, child, name))
    if window.get_wm_name() == name:
        found.append(window)
    return found


def window_pid(screen_display: display.Display, window) -> int:
    """The window's _NET_WM_PID, or 0 when the property is missing."""
    atom = screen_display.intern_atom("_NET_WM_PID")
    prop = window.get_full_property(atom, X.AnyPropertyType)
    if prop is None:
        return 0
    return int(prop.value[0])


def find_player_window(screen_display: display.Display, pid: int, timeout: float):
    """Wait for the player window this process opened."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        candidates = matching_windows(
            screen_display, screen_display.screen().root, PLAYER_WINDOW_NAME
        )
        own = [window for window in candidates if window_pid(screen_display, window) == pid]
        if len(own) > 0:
            return own[0]
        if len(candidates) == 1:
            # Unity did not advertise its pid; a single match is unambiguous.
            return candidates[0]
        time.sleep(0.5)
    raise AssertionError(f"no {PLAYER_WINDOW_NAME} window appeared within {timeout:.0f}s")


def focus_window(screen_display: display.Display, window) -> None:
    window.configure(stack_mode=X.Above)
    screen_display.set_input_focus(window, X.RevertToParent, X.CurrentTime)
    screen_display.sync()


def press_key(screen_display: display.Display, window, key: str) -> None:
    """Send one key press to the player window through XTEST."""
    focus_window(screen_display, window)
    keycode = screen_display.keysym_to_keycode(XK.string_to_keysym(key))
    assert keycode != 0, f"no keycode for {key}"
    xtest.fake_input(screen_display, X.KeyPress, keycode)
    screen_display.sync()
    time.sleep(0.05)
    xtest.fake_input(screen_display, X.KeyRelease, keycode)
    screen_display.sync()


def capture_window(window) -> Image.Image:
    geometry = window.get_geometry()
    raw = window.get_image(0, 0, geometry.width, geometry.height, X.ZPixmap, 0xFFFFFFFF)
    return Image.frombytes("RGB", (geometry.width, geometry.height), raw.data, "raw", "BGRX")


def write_montages(shots: list[Path], out_dir: Path, per_sheet: int, width: int) -> None:
    """Tile the screenshots into labeled contact sheets."""
    for sheet_index in range(0, len(shots), per_sheet):
        page = shots[sheet_index : sheet_index + per_sheet]
        sheet = out_dir / f"montage_{sheet_index // per_sheet:03d}.png"
        command = ["montage", "-label", "%t", "-tile", "6x", "-geometry", f"{width}x+4+4"]
        command.extend(str(shot) for shot in page)
        command.append(str(sheet))
        subprocess.run(command, check=True)
        print(f"montage: {sheet} ({len(page)} arenas)", flush=True)


def main() -> None:
    args = parse_args()
    assert "DISPLAY" in os.environ, "capturing needs an X session (DISPLAY is unset)"
    assert args.arena_root.is_dir(), f"no such arena directory: {args.arena_root}"
    assert args.binary.exists(), f"Animal-AI binary not found: {args.binary}"

    # Sorted order, so a contact sheet reads level 01 first.
    paths = sorted(args.arena_root.rglob(args.pattern))
    assert paths, f"no arena matching {args.pattern} under {args.arena_root}"
    if args.limit > 0:
        paths = paths[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_config(merge_arenas(paths, args.time_scale, False), None)
    print(f"{len(paths)} arenas -> {args.out_dir}")
    print(f"  first: {show(paths[0])}")
    print(f"  last:  {show(paths[-1])}")
    print(f"  merged config: {config_path}", flush=True)

    from animalai import AnimalAIEnvironment

    environment = AnimalAIEnvironment(
        file_name=str(args.binary),
        arenas_configurations=str(config_path),
        base_port=5005 + random.randint(0, 1000),
        play=True,
    )
    screen_display = display.Display()
    shots = []
    try:
        window = find_player_window(screen_display, environment._process.pid, 60.0)
        time.sleep(args.settle)
        for _ in range(args.camera_presses):
            press_key(screen_display, window, "c")
            time.sleep(0.2)

        for index, path in enumerate(paths):
            time.sleep(args.settle)
            shot = args.out_dir / f"{index:04d}_{path.parent.name}_{path.stem}.png"
            capture_window(window).save(shot)
            shots.append(shot)
            print(f"[{index + 1}/{len(paths)}] {show(path)} -> {shot.name}", flush=True)
            # R loads the next arena of the merged config, and wraps around
            # after the last one -- harmless, the sweep is over by then.
            press_key(screen_display, window, "r")
    finally:
        environment.close()
        screen_display.close()

    if args.montage:
        write_montages(shots, args.out_dir, args.montage_per_sheet, args.montage_width)


if __name__ == "__main__":
    main()
