# SPDX-License-Identifier: MIT
"""Drawing helpers for `AnimalAIEnv.render`: the caption strip, the overhead
camera's pane under it, and the curriculum-progress bars beside them."""

import cv2
import numpy as np

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
    """Scale a camera frame to the arena pane."""
    return cv2.resize(image, (size_px, size_px), interpolation=cv2.INTER_NEAREST)
