"""Predicted-waypoint → CARLA control conversion.

Extracted from ``LingoAgent._control_pid`` / ``_interpolate_waypoints``.
The class owns the longitudinal (speed) and lateral (steering) PID
controllers — both keep an internal sliding window, so the conversion
is stateful across calls.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

from vla_streaming_rl.simlingo.team_code.pid_controller import (
    LateralPIDController,
    PIDController,
)


class TrajectoryToControl:
    """Convert predicted route + speed waypoints to ``(steer, throttle, brake)``.

    The longitudinal PID derives a target speed from two samples of the
    predicted speed waypoints (``half_second`` and ``one_second`` indices,
    times 2.0 to recover m/s from per-half-second displacement). The
    lateral PID consumes a densely-resampled route.
    """

    def __init__(self, config):
        self.speed_controller = PIDController(
            k_p=config.speed_kp,
            k_i=config.speed_ki,
            k_d=config.speed_kd,
            n=config.speed_n,
        )
        self.turn_controller = LateralPIDController(inference_mode=False)
        self._brake_speed = config.brake_speed
        self._brake_ratio = config.brake_ratio
        self._clip_delta = config.clip_delta
        self._clip_throttle = config.clip_throttle
        self._one_second = int(config.carla_fps // (config.wp_dilation * config.data_save_freq))

    def __call__(self, route_waypoints, velocity, speed_waypoints):
        """One PID step.

        Args:
            route_waypoints: torch tensor of shape ``(1, N_route, 2)`` —
                predicted ego-frame route. Consumed only by the lateral PID.
            velocity: 0-D / 1-element tensor of current speed (m/s).
            speed_waypoints: torch tensor of shape ``(1, N_speed, 2)`` —
                predicted ego-frame speed waypoints. Consumed only by the
                longitudinal PID, which indexes ``half_second-2`` and
                ``one_second-2`` (so ``N_speed >= one_second - 1``).
                ``N_route`` and ``N_speed`` are independent.

        Returns:
            ``(steer, throttle, brake)`` — ``steer``/``throttle`` are floats
            in ``[-1, 1]`` / ``[0, clip_throttle]``; ``brake`` is a bool.
        """
        assert route_waypoints.size(0) == 1
        route_waypoints = route_waypoints[0].data.cpu().numpy()
        speed = velocity.item()
        speed_waypoints = speed_waypoints[0].data.cpu().numpy()

        half_second = self._one_second // 2
        desired_speed = (
            np.linalg.norm(
                speed_waypoints[half_second - 2] - speed_waypoints[self._one_second - 2]
            )
            * 2.0
        )

        brake = (desired_speed < self._brake_speed) or (
            (speed / desired_speed) > self._brake_ratio
        )

        delta = np.clip(desired_speed - speed, 0.0, self._clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self._clip_throttle)
        throttle = throttle if not brake else 0.0

        route_interp = self._interpolate_waypoints(route_waypoints.squeeze())
        steer = self.turn_controller.step(route_interp, speed)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        return steer, throttle, brake

    @staticmethod
    def _interpolate_waypoints(waypoints):
        """Resample ``waypoints`` (N, D) to ~0.1-unit spacing along arc length."""
        waypoints = waypoints.copy()
        waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints))
        shift = np.roll(waypoints, 1, axis=0)
        shift[0] = shift[1]

        dists = np.linalg.norm(waypoints - shift, axis=1)
        dists = np.cumsum(dists)
        # nudge to keep dists strictly increasing for PchipInterpolator
        dists += np.arange(0, len(dists)) * 1e-4

        interp = PchipInterpolator(dists, waypoints, axis=0)
        x = np.arange(0.1, dists[-1], 0.1)
        interp_points = interp(x)

        # all points collapsed to the origin → fall back to the furthest wp
        if interp_points.shape[0] == 0:
            interp_points = waypoints[None, -1]

        return interp_points
