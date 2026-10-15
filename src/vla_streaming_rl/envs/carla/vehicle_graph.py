# SPDX-License-Identifier: MIT
from dataclasses import dataclass

import cv2
import numpy as np

# Comfort thresholds (from Alpamayo comfort_reward.py)
COMFORT_MAX_ABS_MAG_JERK = 8.37  # [m/s^3]
COMFORT_MAX_ABS_LAT_ACCEL = 4.89  # [m/s^2]
COMFORT_MAX_LON_ACCEL = 2.40  # [m/s^2]
COMFORT_MIN_LON_ACCEL = -4.05  # [m/s^2]


@dataclass
class GraphConfig:
    label: str
    color: str
    ymin: float
    ymax: float
    thresholds: list[float]


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


GRAPH_CONFIGS = [
    GraphConfig(label="throttle", color="#00ff00", ymin=0.0, ymax=1.0, thresholds=[]),
    GraphConfig(label="brake", color="#0080ff", ymin=0.0, ymax=1.0, thresholds=[]),
    GraphConfig(label="steer", color="#ffff00", ymin=-1.0, ymax=1.0, thresholds=[]),
    GraphConfig(label="vel", color="#ff8000", ymin=0.0, ymax=80.0, thresholds=[]),
    GraphConfig(
        label="lat G",
        color="#ff00ff",
        ymin=-1.0,
        ymax=1.0,
        thresholds=[
            -COMFORT_MAX_ABS_LAT_ACCEL / 9.81,
            COMFORT_MAX_ABS_LAT_ACCEL / 9.81,
        ],
    ),
    GraphConfig(
        label="lon G",
        color="#00ffff",
        ymin=-1.0,
        ymax=1.0,
        thresholds=[
            COMFORT_MIN_LON_ACCEL / 9.81,
            COMFORT_MAX_LON_ACCEL / 9.81,
        ],
    ),
    GraphConfig(
        label="jerk",
        color="#ff6464",
        ymin=0.0,
        ymax=20.0,
        thresholds=[COMFORT_MAX_ABS_MAG_JERK],
    ),
]


def render_vehicle_graphs(history, panel_w: int, panel_h: int) -> np.ndarray:
    """Render time-series graphs of vehicle metrics using cv2 only.

    Returns an (panel_h, panel_w, 3) uint8 BGR image.
    """
    img = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    num = len(GRAPH_CONFIGS)
    label_w = 38
    margin_top = 3
    margin_bottom = 3
    gap = 2
    total_graph_h = panel_h - margin_top - margin_bottom - gap * (num - 1)
    graph_h = total_graph_h // num

    series_map = {
        "throttle": [p.throttle for p in history],
        "brake": [p.brake for p in history],
        "steer": [p.steering for p in history],
        "vel": [p.velocity_kph for p in history],
        "lat G": [p.lat_acceleration / 9.81 for p in history],
        "lon G": [p.lon_acceleration / 9.81 for p in history],
        "jerk": [p.jerk for p in history],
    }

    for i, cfg in enumerate(GRAPH_CONFIGS):
        y0 = margin_top + i * (graph_h + gap)
        y1 = y0 + graph_h
        x0 = label_w
        x1 = panel_w - 1

        bgr = _hex_to_bgr(cfg.color)
        gray = (80, 80, 80)

        # Graph background
        cv2.rectangle(img, (x0, y0), (x1, y1), (20, 20, 20), -1)

        # Border
        cv2.rectangle(img, (x0, y0), (x1, y1), gray, 1)

        # Label
        font_scale = 0.28
        cv2.putText(
            img,
            cfg.label,
            (2, y0 + graph_h // 2 + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            bgr,
            1,
            cv2.LINE_AA,
        )

        gh = y1 - y0

        def val_to_y(v: float) -> int:
            ratio = (v - cfg.ymin) / (cfg.ymax - cfg.ymin)
            return int(y1 - ratio * gh)

        # Threshold lines
        for thresh in cfg.thresholds:
            ty = val_to_y(thresh)
            if y0 <= ty <= y1:
                cv2.line(img, (x0, ty), (x1, ty), gray, 1)

        # Zero line (if ymin < 0 < ymax)
        if cfg.ymin < 0 < cfg.ymax:
            zy = val_to_y(0.0)
            cv2.line(img, (x0, zy), (x1, zy), (50, 50, 50), 1)

        # Plot data
        data = series_map[cfg.label]
        n = len(data)
        if n >= 2:
            xs = np.linspace(x0, x1, n).astype(np.int32)
            ys = np.array(
                [val_to_y(float(np.clip(v, cfg.ymin, cfg.ymax))) for v in data], dtype=np.int32
            )
            pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], isClosed=False, color=bgr, thickness=1, lineType=cv2.LINE_AA)

    return img


def overlay_vehicle_graphs(img: np.ndarray, history) -> None:
    """Overlay time-series graphs on the bottom-right of img (mutates img in-place)."""
    if len(history) < 2:
        return
    img_h, img_w = img.shape[:2]
    panel_w = int(img_w * 0.40)
    panel_h = int(img_h * 0.45)
    margin = 4

    graph_img = render_vehicle_graphs(history, panel_w, panel_h)
    x0 = img_w - panel_w - margin
    y0 = img_h - panel_h - margin

    # Alpha blend with background
    roi = img[y0 : y0 + panel_h, x0 : x0 + panel_w]
    blended = cv2.addWeighted(roi, 0.3, graph_img, 0.7, 0)
    img[y0 : y0 + panel_h, x0 : x0 + panel_w] = blended
