# SPDX-License-Identifier: MIT
"""Pure observation / action helpers for the CARLA leaderboard setup.

This module is the single source of truth for the observation and action
contracts used by both the training environment (``carla_leaderboard_env.py``)
and evaluation agents (e.g. the Bench2Drive team-code agent). Nothing here
touches ``carla.Client`` — only ``carla.Location`` / ``carla.Transform`` are
referenced as value types, so these helpers stay testable in isolation.
"""
from dataclasses import dataclass

import carla
import cv2
import gymnasium as gym
import numpy as np
import scipy.interpolate


@dataclass(frozen=True)
class CARLAObsConfig:
    # Matches simlingo's camera_width_0 / camera_height_0 in
    # config_simlingo.py — the SimLingo VLM was trained at this resolution
    # and crops/resamples internally, so we feed it the same.
    image_size: tuple[int, int] = (1024, 512)  # (width, height)
    fov: float = 110.0
    # Mount position copied from simlingo/team_code/config_simlingo.py
    # (camera_pos_0 = [-1.5, 0.0, 2.0]). Training/eval observation must
    # match the agent's calibration or the policy sees a different view
    # than what simlingo was trained on.
    camera_x: float = -1.5
    camera_z: float = 2.0
    map_size: int = 512
    scale: float = 0.5  # meters/pixel on the overlay
    num_interp_points: int = 1000
    route_thickness: int = 20
    triangle_size: int = 15


def make_obs_space(cfg: CARLAObsConfig) -> gym.spaces.Box:
    return gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(3, cfg.image_size[1], cfg.image_size[0]),
        dtype=np.float32,
    )


def make_action_space() -> gym.spaces.Box:
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def camera_sensor_spec(cfg: CARLAObsConfig, sensor_id: str = "Center") -> dict:
    """Sensor dict for Bench2Drive's ``sensors()`` method, matching training."""
    return {
        "type": "sensor.camera.rgb",
        "id": sensor_id,
        "x": cfg.camera_x,
        "y": 0.0,
        "z": cfg.camera_z,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "width": cfg.image_size[0],
        "height": cfg.image_size[1],
        "fov": cfg.fov,
    }


def world_to_vehicle_coords(
    world_loc: carla.Location,
    vehicle_loc: carla.Location,
    vehicle_yaw_rad: float,
    map_size: int,
    scale: float,
) -> tuple[int, int]:
    dx = world_loc.x - vehicle_loc.x
    dy = world_loc.y - vehicle_loc.y
    adjusted = -vehicle_yaw_rad + np.pi / 2
    c, s = np.cos(adjusted), np.sin(adjusted)
    rx = dx * c - dy * s
    ry = dx * s + dy * c
    return (
        int(-rx / scale + map_size // 2),
        int(-ry / scale + map_size // 2),
    )


class RouteTracker:
    """Holds the interpolated route, tracks the closest waypoint, renders overlay."""

    def __init__(self, waypoints: list[carla.Location], cfg: CARLAObsConfig) -> None:
        self.waypoints = waypoints
        self.cfg = cfg
        self.current_index = 0
        self._min_distance = 0.0
        self._segment_t = 0.0

    @classmethod
    def from_raw_waypoints(
        cls, raw: list[carla.Location], cfg: CARLAObsConfig
    ) -> "RouteTracker":
        """Interpolate to ``cfg.num_interp_points`` and wrap in a tracker."""
        arr = np.array([[w.x, w.y, w.z] for w in raw])
        t0 = np.linspace(0, 1, len(arr))
        t1 = np.linspace(0, 1, cfg.num_interp_points)
        interp = scipy.interpolate.interp1d(t0, arr, axis=0, kind="linear")
        coords = interp(t1)
        waypoints = [carla.Location(*c) for c in coords]
        return cls(waypoints, cfg)

    def update(self, vehicle_loc: carla.Location) -> None:
        """Forward-only nearest-waypoint search over a 20-point window."""
        min_dist = float("inf")
        closest_idx = self.current_index
        search_end = min(self.current_index + 20, len(self.waypoints))
        for i in range(self.current_index, search_end):
            d = vehicle_loc.distance(self.waypoints[i])
            if d < min_dist:
                min_dist = d
                closest_idx = i
        self.current_index = closest_idx
        self._min_distance = min_dist

        # Continuous sub-segment progress: project the vehicle onto the line
        # from waypoints[current_index] to waypoints[current_index+1] and
        # keep the clamped parametric position. Without this, route_completion
        # only ticks when crossing a discrete waypoint — on coarse spacings
        # (e.g. 10 m on long Town12 routes) the reward is 0 for many steps of
        # actual forward motion.
        if closest_idx + 1 < len(self.waypoints):
            a = self.waypoints[closest_idx]
            b = self.waypoints[closest_idx + 1]
            ex, ey = b.x - a.x, b.y - a.y
            vx, vy = vehicle_loc.x - a.x, vehicle_loc.y - a.y
            seg_len_sq = ex * ex + ey * ey
            t = (vx * ex + vy * ey) / max(seg_len_sq, 1e-9)
            self._segment_t = max(0.0, min(1.0, t))
        else:
            self._segment_t = 0.0

    @property
    def route_completion(self) -> float:
        n = max(1, len(self.waypoints) - 1)
        return min(1.0, (self.current_index + self._segment_t) / n)

    @property
    def min_distance_to_route(self) -> float:
        return self._min_distance

    def render_overlay(
        self, vehicle_loc: carla.Location, vehicle_yaw_rad: float
    ) -> np.ndarray:
        """Vehicle-centric top-down BGR image, ``(map_size, map_size, 3)`` uint8."""
        cfg = self.cfg
        size = cfg.map_size
        img = np.ones((size, size, 3), dtype=np.uint8) * 200
        for i in range(len(self.waypoints) - 1):
            x1, y1 = world_to_vehicle_coords(
                self.waypoints[i], vehicle_loc, vehicle_yaw_rad, size, cfg.scale
            )
            x2, y2 = world_to_vehicle_coords(
                self.waypoints[i + 1], vehicle_loc, vehicle_yaw_rad, size, cfg.scale
            )
            color = (255, 0, 0) if i < self.current_index else (100, 100, 255)
            cv2.line(img, (x1, y1), (x2, y2), color, cfg.route_thickness)
        cx, cy = size // 2, size // 2
        ts = cfg.triangle_size
        triangle = np.array(
            [
                [cx, cy - ts],
                [int(cx - ts * 0.5), int(cy + ts * 0.5)],
                [int(cx + ts * 0.5), int(cy + ts * 0.5)],
            ],
            np.int32,
        )
        cv2.fillPoly(img, [triangle], (0, 255, 0))
        return img


def compose_obs(
    camera_rgb_hwc_uint8: np.ndarray,
    overlay_bgr_hw3_uint8: np.ndarray,
    cfg: CARLAObsConfig,
) -> np.ndarray:
    """Return CHW float32 [0, 1] with overlay pasted in the bottom-right quarter."""
    w, h = cfg.image_size
    route_h = h // 4
    route_w = w // 4
    overlay_resized = cv2.resize(overlay_bgr_hw3_uint8, (route_w, route_h))
    img = camera_rgb_hwc_uint8.copy()
    img[-route_h:, -route_w:] = overlay_resized
    return img.transpose(2, 0, 1).astype(np.float32) / 255.0


def action_to_vehicle_control(action: np.ndarray) -> tuple[float, float, float]:
    """Map 2-D policy action to ``(steer, throttle, brake)``.

    ``action[1] > 0`` → throttle, ``action[1] < 0`` → brake. No floor or
    clamp beyond ``[-1, 1]``: external policies (e.g. SimLingo) round-trip
    their ``carla.VehicleControl`` through this mapping and must see the
    raw throttle/brake they emitted.
    """
    steer = float(np.clip(action[0], -1.0, 1.0))
    gas_or_brake = float(np.clip(action[1], -1.0, 1.0))
    if gas_or_brake >= 0.0:
        throttle = gas_or_brake
        brake = 0.0
    else:
        throttle = 0.0
        brake = -gas_or_brake
    return steer, throttle, brake
