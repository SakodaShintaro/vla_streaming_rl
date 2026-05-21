from collections import deque

import numpy as np


class LateralPIDController(object):
    """
    PID controller
    """

    def __init__(
        self,
        k_p=3.118357247806046,
        k_d=1.3782508892109167,
        k_i=0.6406067986034124,
        speed_scale=0.9755321901954155,
        speed_offset=1.9152884533402488,
        default_lookahead=24,
        speed_threshold=23.150102938235136,
        n=6,
        inference_mode=False,
    ):
        self.k_p = k_p
        self.k_d = k_d
        self.k_i = k_i
        self.speed_scale = speed_scale
        self.speed_offset = speed_offset
        self.default_lookahead = default_lookahead
        self.speed_threshold = speed_threshold
        self.n = n
        self.inference_mode = (
            inference_mode  # False when used in the expert, True when used with a trained model
        )

        self._saved_window = []
        self._window = []

    def step(self, route_np, current_speed):
        # numpy 2.x rejects ``int(np.array([x]))`` further down — the
        # agent passes ``speed = velocity[0].cpu().numpy()`` whose shape
        # is ``(1,)`` from a ``(1, 1)`` tensor. Coerce to a 0-d Python
        # float at the boundary so the downstream ``int(min(np.clip(...)))``
        # sees a scalar.
        current_speed = float(np.asarray(current_speed).reshape(-1)[0])
        # current_speed = current_speed*3.6

        # org not sure where this comes from
        # if self.inference_mode:  # Transfuser predicts checkpoints 1m apart, whereas in the expert the route points have distance 10cm.
        #     # n_lookahead = np.clip(self.speed_scale * current_speed + self.speed_offset, 24, 105) / 10 # range [2.4, 10.5]
        #     # n_lookahead = n_lookahead - 2
        #     # n_lookahead = int(min(n_lookahead, route_np.shape[0] - 1))  # range [2, 9] - but 0 and 1 are never used because n_lookahead is overwritten below
        #     n_lookahead = np.clip(self.speed_scale * current_speed + self.speed_offset, 24, 190) / 10 # range [2.4, 10.5]
        #     if current_speed < 10*3.6:
        #         n_lookahead = n_lookahead - 2
        #     elif current_speed > 18*3.6:
        #         n_lookahead = n_lookahead + 3
        #     n_lookahead = int(min(n_lookahead, route_np.shape[0] - 1))  # range [2, 9] - but 0 and 1 are never used because n_lookahead is overwritten below

        # else:
        #     n_lookahead = int(min(np.clip(self.speed_scale * current_speed + self.speed_offset, 24, 105), route_np.shape[0] - 1))

        # used at leaderboard
        current_speed = current_speed * 3.6
        if self.inference_mode:  # Transfuser predicts checkpoints 1m apart, whereas in the expert the route points have distance 10cm.
            n_lookahead = (
                np.clip(self.speed_scale * current_speed + self.speed_offset, 24, 105) / 10
            )  # range [2.4, 10.5]
            n_lookahead = n_lookahead - 2
            n_lookahead = int(
                min(n_lookahead, route_np.shape[0] - 1)
            )  # range [2, 9] - but 0 and 1 are never used because n_lookahead is overwritten below
        else:
            n_lookahead = int(
                min(
                    np.clip(self.speed_scale * current_speed + self.speed_offset, 24, 105),
                    route_np.shape[0] - 1,
                )
            )

        n_lookahead = min(n_lookahead, len(route_np) - 1)
        desired_heading_vec = route_np[n_lookahead]

        yaw_path = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
        heading_error = (yaw_path) % (2 * np.pi)
        heading_error = heading_error if heading_error < np.pi else heading_error - 2 * np.pi

        # the scaling doesn't deserve any specific purpose but is a leftover from a previous less efficient implementation,
        # on which we optimized the parameters
        heading_error = heading_error * 180.0 / np.pi / 90.0

        self._window.append(heading_error)
        self._window = self._window[-self.n :]

        derivative = 0.0 if len(self._window) == 1 else self._window[-1] - self._window[-2]
        integral = np.mean(self._window)

        steering = np.clip(
            self.k_p * heading_error + self.k_d * derivative + self.k_i * integral, -1.0, 1.0
        ).item()

        return steering

    def save(self):
        self._saved_window = self._window.copy()

    def load(self):
        self._window = self._saved_window.copy()


class PIDController(object):
    """
    PID controller that converts waypoints to steer, brake and throttle commands
    """

    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0, n=20):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self.window = deque([0 for _ in range(n)], maxlen=n)

    def step(self, error):
        self.window.append(error)

        if len(self.window) >= 2:
            integral = sum(self.window) / len(self.window)
            derivative = self.window[-1] - self.window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self.k_p * error + self.k_i * integral + self.k_d * derivative
