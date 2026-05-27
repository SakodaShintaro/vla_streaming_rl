"""Unscented Kalman filter for the ego (x, y, yaw, speed) state.

Extracted from ``LingoAgent``. Wraps ``filterpy``'s UKF with a kinematic
bicycle ``fx`` and an identity ``hx`` (measurement state == filter state).
The first ``update`` initializes the filter from the measurement, so the
caller doesn't track an init flag.
"""

import math
from collections import deque

import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

from vla_streaming_rl.simlingo.team_code.simlingo_utils import normalize_angle


class EgoStateFilter:
    """Tracks ``(x, y, yaw, speed)`` from noisy GPS + IMU compass + speed
    measurements, propagated with a kinematic bicycle model driven by the
    last applied control.
    """

    def __init__(self, dt: float, state_log_maxlen: int):
        self._points = MerweScaledSigmaPoints(
            n=4,
            alpha=0.00001,
            beta=2,
            kappa=0,
            subtract=_residual_state_x,
        )
        self._ukf = UKF(
            dim_x=4,
            dim_z=4,
            fx=_bicycle_model_forward,
            hx=_measurement_function_hx,
            dt=dt,
            points=self._points,
            x_mean_fn=_state_mean,
            z_mean_fn=_measurement_mean,
            residual_x=_residual_state_x,
            residual_z=_residual_measurement_h,
        )
        # State covariance — small along yaw / speed because the first
        # update overwrites x with the measurement anyway.
        self._ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
        # Measurement covariance — basically trust IMU compass + speedometer.
        self._ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
        # Process covariance.
        self._ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])

        self._initialized = False
        self.state_log = deque(maxlen=state_log_maxlen)

    def update(
        self,
        gps_x: float,
        gps_y: float,
        compass_rad: float,
        speed: float,
        steer: float,
        throttle: float,
        brake: float,
    ) -> np.ndarray:
        """One predict + update cycle.

        Args:
            gps_x, gps_y: ego position from GPS (CARLA world coords).
            compass_rad: IMU compass in radians, already preprocessed
                via ``preprocess_compass``.
            speed: speedometer reading in m/s.
            steer, throttle, brake: last applied ``carla.VehicleControl``
                values, used to drive the bicycle model in ``predict``.

        Returns:
            ``(4,)`` ndarray ``[x, y, yaw_rad, speed_m_s]``.
        """
        measurement = np.array([gps_x, gps_y, normalize_angle(compass_rad), speed])
        if not self._initialized:
            self._ukf.x = measurement
            self._initialized = True

        self._ukf.predict(steer=steer, throttle=throttle, brake=brake)
        self._ukf.update(measurement)

        filtered_state = self._ukf.x
        self.state_log.append(filtered_state)
        return filtered_state


# --- filter callbacks (module-private) --------------------------------------


def _bicycle_model_forward(x, dt, steer, throttle, brake):
    """Kinematic bicycle model — parameters tuned by World-on-Rails."""
    front_wb = -0.090769015
    rear_wb = 1.4178275

    steer_gain = 0.36848336
    brake_accel = -4.952399
    throt_accel = 0.5633837

    locs_0 = x[0]
    locs_1 = x[1]
    yaw = x[2]
    speed = x[3]

    if brake:
        accel = brake_accel
    else:
        accel = throt_accel * throttle

    wheel = steer_gain * steer

    beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
    next_locs_0 = locs_0.item() + speed * math.cos(yaw + beta) * dt
    next_locs_1 = locs_1.item() + speed * math.sin(yaw + beta) * dt
    next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
    next_speed = speed + accel * dt
    next_speed = next_speed * (next_speed > 0.0)  # Fast ReLU

    return np.array([next_locs_0, next_locs_1, next_yaws, next_speed])


def _measurement_function_hx(vehicle_state):
    """Identity hx — measurement is the same shape as the filter state."""
    return vehicle_state


def _state_mean(state, wm):
    """Sigma-point mean. Yaw uses circular (atan2 of sin/cos) mean."""
    x = np.zeros(4)
    sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
    sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(state[:, 3], wm))
    return x


def _measurement_mean(state, wm):
    """Sigma-point mean for measurements — same form as state mean (hx is identity)."""
    x = np.zeros(4)
    sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
    sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(state[:, 3], wm))
    return x


def _residual_state_x(a, b):
    y = a - b
    y[2] = normalize_angle(y[2])
    return y


def _residual_measurement_h(a, b):
    y = a - b
    y[2] = normalize_angle(y[2])
    return y
