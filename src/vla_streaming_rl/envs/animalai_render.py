# SPDX-License-Identifier: MIT
"""Drawing helpers for `AnimalAIEnv.render`: the top-down arena schematic,
the caption strip above it, and the curriculum-progress bars beside it."""

import cv2
import numpy as np

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
RENDER_SIZE_PX = 256
_HEADER_LINE_PX = 20


def draw_header(lines: list[str], width_px: int) -> np.ndarray:
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
# passed episodes / failed episodes / no episode run yet.
_PROGRESS_COLORS = ((60, 170, 60), (205, 75, 75), (205, 205, 205))
_PROGRESS_LEGEND = ("pass", "fail", "n/a")


def render_progress(groups: list[tuple[str, int, int]]) -> np.ndarray:
    """One horizontal bar per arena group (see
    `ArenaSelector.progress_by_group`): the group's success rate over every
    episode run so far, passed in green and failed in red, grey until the
    group has been run at all."""
    passed = sum(group[1] for group in groups)
    total = sum(sum(group[1:]) for group in groups)
    header = draw_header([f"passed:{passed}/{total}"], RENDER_SIZE_PX)
    header_px = header.shape[0]
    height = header_px + _PROGRESS_ROW_PX * len(groups) + _PROGRESS_LEGEND_PX
    canvas = np.full((height, RENDER_SIZE_PX, 3), 240, dtype=np.uint8)
    canvas[:header_px] = header

    bar_x = _PROGRESS_LABEL_PX
    bar_width = RENDER_SIZE_PX - bar_x - 6
    for row, (name, passed, failed) in enumerate(groups):
        top = header_px + row * _PROGRESS_ROW_PX
        attempts = passed + failed
        label = f"{name} {passed / attempts:.0%}" if attempts else name
        cv2.putText(
            canvas,
            label,
            (4, top + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        # A group never run is one grey bar; otherwise the passed and failed
        # episodes tile the bar exactly.
        counts = (passed, failed, 0) if attempts else (0, 0, 1)
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


def fit_square(image: np.ndarray, size_px: int) -> np.ndarray:
    """Scale a camera frame to the pane the schematic would have filled."""
    return cv2.resize(image, (size_px, size_px), interpolation=cv2.INTER_NEAREST)


def render_topdown(items: list[dict], agent_xyz: tuple[float, float, float] | None) -> np.ndarray:
    scale = RENDER_SIZE_PX / _ARENA_SIZE_M
    canvas = np.full((RENDER_SIZE_PX, RENDER_SIZE_PX, 3), 240, dtype=np.uint8)

    # Flip z so the +z arena axis points up in the image (Unity-editor-like).
    def to_px(x: float, z: float) -> tuple[int, int]:
        return int(round(x * scale)), int(round((_ARENA_SIZE_M - z) * scale))

    cv2.rectangle(canvas, (0, 0), (RENDER_SIZE_PX - 1, RENDER_SIZE_PX - 1), (60, 60, 60), 2)

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
