"""
Some helpful classes for planning and control for the privileged autopilot
"""

import math
import warnings
from collections import deque
from copy import deepcopy

import carla
import numpy as np


class RoutePlanner(object):
    """
    Gets the next waypoint along a path
    """

    def __init__(self, min_distance, max_distance, lat_ref=0.0, lon_ref=0.0):
        self.saved_route = deque()
        self.route = deque()
        self.saved_route_distances = deque()
        self.route_distances = deque()

        self.lat_ref = lat_ref
        self.lon_ref = lon_ref

        self.min_distance = min_distance
        self.max_distance = max_distance
        self.is_last = False

        # self.mean = np.array([0.0, 0.0, 0.0])
        # self.scale = np.array([111319.49082349832, 111319.49079327358, 1.0]) # previously: [111324.60662786, 111319.490945]

    def convert_gps_to_carla(self, gps):
        """
        Converts GPS signal into the CARLA coordinate frame
        :param gps: gps from gnss sensor
        :return: gps as numpy array in CARLA coordinates
        """
        #   gps = (gps - self.mean) * self.scale
        #   # GPS uses a different coordinate system than CARLA.
        #   # This converts from GPS -> CARLA (90° rotation)
        #   gps = np.array([gps[1], -gps[0], gps[2]])

        EARTH_RADIUS_EQUA = 6378137.0  # Constant from CARLA leaderboard GPS simulation
        lat, lon, _ = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat + 90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = (
            scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0))
            - my
        )
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        gps = np.array([x, y, gps[2]])

        return gps

    def set_route(self, global_plan, gps=False, carla_map=None):
        self.route.clear()

        for pos, cmd in global_plan:
            if gps:
                warnings.warn("deprecated", DeprecationWarning)
                pos = np.array([pos["lat"], pos["lon"], pos["z"]])
                pos = self.convert_gps_to_carla(pos)
            else:
                # important to use the z variable, otherwise there are some rare bugs at carla.map.get_waypoint(carla.Location)
                pos = np.array([pos.location.x, pos.location.y, pos.location.z])
                # pos -= self.mean

            self.route.append((pos, cmd))

        if carla_map is not None:
            for _ in range(50):
                loc = carla.Location(
                    x=self.route[-1][0][0], y=self.route[-1][0][1], z=self.route[-1][0][2]
                )
                next_loc = carla_map.get_waypoint(loc).next(1)[0].transform.location
                next_loc = np.array([next_loc.x, next_loc.y, next_loc.z])
                self.route.append((next_loc, self.route[-1][1]))

        # We do the calculations in the beginning once so that we don't have
        # to do them every time in run_step
        self.route_distances.append(0.0)
        for i in range(1, len(self.route)):
            diff = self.route[i][0] - self.route[i - 1][0]
            distance = (diff[0] ** 2 + diff[1] ** 2) ** 0.5
            self.route_distances.append(distance)

    def run_step(self, gps):
        if len(self.route) <= 2:
            self.is_last = True
            return self.route

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0
        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += self.route_distances[i]

            diff = self.route[i][0] - gps
            distance = (diff[0] ** 2 + diff[1] ** 2) ** 0.5

            if farthest_in_range < distance <= self.min_distance:
                farthest_in_range = distance
                to_pop = i

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()
                self.route_distances.popleft()

        return self.route

    def save(self):
        # because self.route saves objects of traffic lights and traffic signs a deep copy is not possible
        self.saved_route = []  # deepcopy(self.route)
        for (
            loc,
            cmd,
            d_traffic,
            traffic,
            d_stop,
            stop,
            speed_limit,
            corrected_speed_limit,
        ) in self.route:
            self.saved_route.append(
                (
                    np.copy(loc),
                    cmd,
                    d_traffic,
                    traffic,
                    d_stop,
                    stop,
                    speed_limit,
                    corrected_speed_limit,
                )
            )

        self.saved_route = deque(self.saved_route)
        self.saved_route_distances = deepcopy(self.route_distances)

    def load(self):
        self.route = self.saved_route
        self.route_distances = self.saved_route_distances
        self.is_last = False
        self.route = self.saved_route
        self.route_distances = self.saved_route_distances
        self.is_last = False
