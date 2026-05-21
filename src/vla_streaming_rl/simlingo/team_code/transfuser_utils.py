"""
Some utility functions e.g. for normalizing angles
"""

import math
from collections import deque

import numpy as np


def normalize_angle(x):
    x = x % (2 * np.pi)  # force in range [0, 2 pi)
    if x > np.pi:  # move to [-pi, pi)
        x -= 2 * np.pi
    return x


def inverse_conversion_2d(point, translation, yaw):
    """
    Performs a forward coordinate conversion on a 2D point
    :param point: Point to be converted
    :param translation: 2D translation vector of the new coordinate system
    :param yaw: yaw in radian of the new coordinate system
    :return: Converted point
    """
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

    converted_point = rotation_matrix.T @ (point - translation)
    return converted_point


def preprocess_compass(compass):
    """
    Checks the compass for Nans and rotates it into the default CARLA coordinate
    system with range [-pi,pi].
    :param compass: compass value provided by the IMU, in radian
    :return: yaw of the car in radian in the CARLA coordinate system.
    """
    if math.isnan(compass):  # simulation bug
        compass = 0.0
    # The minus 90.0 degree is because the compass sensor uses a different
    # coordinate system then CARLA. Check the coordinate_sytems.txt file
    compass = normalize_angle(compass - np.deg2rad(90.0))

    return compass


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


def command_to_one_hot(command):
    if command < 0:
        command = 4
    command -= 1
    if command not in [0, 1, 2, 3, 4, 5]:
        command = 3
    cmd_one_hot = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cmd_one_hot[command] = 1.0

    return np.array(cmd_one_hot)
