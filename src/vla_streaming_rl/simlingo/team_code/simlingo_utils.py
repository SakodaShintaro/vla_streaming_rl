import math

import numpy as np
import torch


def get_camera_intrinsics(w, h, fov):
    """
    Get camera intrinsics matrix from width, height and fov.
    Returns:
      K: A float32 tensor of shape ``[3, 3]`` containing the intrinsic calibration matrices for
        the carla camera.
    """
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0

    K = torch.tensor(K, dtype=torch.float32)
    return K


def get_camera_extrinsics():
    """
    Get camera extrinsics matrix for the carla camera.
    extrinsics: A float32 tensor of shape ``[4, 4]`` containing the extrinsic calibration matrix for
      the carla camera. The extrinsics are specified as homogeneous matrices of the form ``[R t; 0 1]``
    """
    extrinsics = np.zeros((4, 4), dtype=np.float32)
    extrinsics[3, 3] = 1.0
    extrinsics[:3, :3] = np.eye(3)
    extrinsics[:3, 3] = [-1.5, 0.0, 2.0]
    extrinsics = torch.tensor(extrinsics, dtype=torch.float32)

    return extrinsics


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
