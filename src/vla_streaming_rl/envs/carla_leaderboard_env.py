# SPDX-License-Identifier: MIT
import copy
import time
from dataclasses import dataclass
from pathlib import Path

import carla
import gymnasium as gym
import numpy as np
import py_trees
from leaderboard.utils.statistics_manager import PENALTY_PERC_DICT, PENALTY_VALUE_DICT
from srunner.scenariomanager.traffic_events import TrafficEventType

from vla_streaming_rl.envs.bench2drive_scenario_runtime import Bench2DriveRuntime
from vla_streaming_rl.envs.carla_bootstrap import get_client
from vla_streaming_rl.envs.carla_obs import (
    CARLAObsConfig,
    RouteTracker,
    action_to_vehicle_control,
    make_action_space,
    make_obs_space,
    paste_route_overlay,
)
from vla_streaming_rl.envs.eval_writer import Bench2DriveEvalWriter
from vla_streaming_rl.envs.vehicle_graph import overlay_vehicle_graphs

DT = 0.05  # [s] (20 FPS)

BASE_PROMPT = (
    "Drive a car along a route in CARLA. Follow the planned route, "
    "obey traffic rules, and avoid collisions."
)
# CARLA RoadOption (agents.navigation.local_planner.RoadOption) -> the upcoming
# maneuver sentence appended to the task prompt, so navigation intent is delivered
# to VLA agents via the existing obs["language"] channel (and shown on-screen).
COMMAND_TO_MANEUVER = {
    -1: "The ego vehicle is following the lane straight ahead.",  # VOID
    1: "The ego vehicle is turning left at the upcoming intersection.",  # LEFT
    2: "The ego vehicle is turning right at the upcoming intersection.",  # RIGHT
    3: "The ego vehicle is going straight through the upcoming intersection.",  # STRAIGHT
    4: "The ego vehicle is following the lane straight ahead.",  # LANEFOLLOW
    5: "The ego vehicle is changing to the left lane.",  # CHANGELANELEFT
    6: "The ego vehicle is changing to the right lane.",  # CHANGELANERIGHT
}
_MANEUVER_LOOKAHEAD_PTS = 10  # route points ahead to read the upcoming command from


@dataclass
class VehiclePhysics:
    """Vehicle physics and action state for a single timestep."""

    acceleration_vector: np.ndarray
    angular_velocity_vector: np.ndarray
    velocity: np.ndarray
    velocity_kph: float
    acceleration: float
    lon_acceleration: float
    lat_acceleration: float
    jerk: float
    lon_jerk: float
    prev_lon_acceleration: float
    angular_velocity: float
    angular_acceleration: float
    throttle: float
    brake: float
    steering: float

    def update(self, vehicle, throttle: float, brake: float, steering: float):
        """Update vehicle physics information and action state."""
        self.throttle = throttle
        self.brake = brake
        self.steering = steering

        # Calculate velocity
        vel = vehicle.get_velocity()
        self.velocity = np.array([vel.x, vel.y, vel.z])
        self.velocity_kph = 3.6 * np.linalg.norm(self.velocity)

        # Vehicle forward/right vectors from yaw
        yaw = np.radians(vehicle.get_transform().rotation.yaw)
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])

        # Calculate acceleration
        acceleration = vehicle.get_acceleration()
        acceleration_vec = np.array([acceleration.x, acceleration.y, acceleration.z])
        self.acceleration = np.linalg.norm(acceleration_vec)
        self.lon_acceleration = float(np.dot(acceleration_vec, forward))
        self.lat_acceleration = float(np.dot(acceleration_vec, right))

        # Calculate jerk
        jerk_vec = (acceleration_vec - self.acceleration_vector) / DT
        self.jerk = np.linalg.norm(jerk_vec)
        self.lon_jerk = (self.lon_acceleration - self.prev_lon_acceleration) / DT
        self.prev_lon_acceleration = self.lon_acceleration
        self.acceleration_vector = acceleration_vec

        # Calculate angular velocity
        angular_velocity = vehicle.get_angular_velocity()
        angular_velocity_vec = np.array(
            [angular_velocity.x, angular_velocity.y, angular_velocity.z]
        )
        self.angular_velocity = np.linalg.norm(angular_velocity_vec)

        # Calculate angular acceleration
        angular_accel_vec = (angular_velocity_vec - self.angular_velocity_vector) / DT
        self.angular_acceleration = np.linalg.norm(angular_accel_vec)
        self.angular_velocity_vector = angular_velocity_vec

    @classmethod
    def create(cls) -> "VehiclePhysics":
        return cls(
            acceleration_vector=np.zeros(3),
            angular_velocity_vector=np.zeros(3),
            velocity=np.zeros(3),
            velocity_kph=0.0,
            acceleration=0.0,
            lon_acceleration=0.0,
            lat_acceleration=0.0,
            jerk=0.0,
            lon_jerk=0.0,
            prev_lon_acceleration=0.0,
            angular_velocity=0.0,
            angular_acceleration=0.0,
            throttle=0.0,
            brake=0.0,
            steering=0.0,
        )


class CARLALeaderboardEnv(gym.Env):
    """
    CARLA Leaderboard-compliant Gymnasium environment

    - Route generation and route tracking
    - Leaderboard-compliant reward (Route Completion + Infractions)
    - Visualization of map + route + vehicle position
    """

    def __init__(
        self,
        route_xml: str | None,
        route_id: str | None,
        sequence_mode: str,
        start_index: int,
        loop: bool,
        early_off_route_m: float | None,
        early_blocked_steps: int | None,
        eval_output_dir: str | None,
    ):
        """Create the env.

        Args:
            route_xml: Path to a Bench2Drive route XML, or ``None`` to
                sample a random 200 m route in Town01 each episode.
            route_id: Specific route id to pin to (only meaningful with
                ``route_xml`` set and ``sequence_mode='random'``). ``None``
                means "pick a random route from the XML each episode".
            sequence_mode: ``"random"`` (legacy: pick a random route each
                reset; requires single-town XML) or ``"sequential"`` (walk
                the XML in **town-grouped** order starting at
                ``start_index``; town flips happen only at block
                boundaries, minimizing ``load_world`` overhead).
            start_index: Starting cursor for ``sequence_mode='sequential'``
                (0-indexed into the town-sorted config list). Ignored
                otherwise.
            loop: When ``True`` and ``sequence_mode='sequential'``, wrap
                the cursor back to 0 after the last config so training
                runs forever. When ``False``, sequential mode raises once
                exhausted (the Bench2Drive220 fixed-sweep usage). In
                ``sequence_mode='random'`` the flag is a no-op (random
                never exhausts) and either value is accepted.
            early_off_route_m: Tighter analogue of Bench2Drive's InRouteTest
                (which terminates at 30 m). When set, the env additionally
                terminates as soon as ``route_tracker.min_distance_to_route``
                exceeds this value. ``None`` disables the early check (only
                Bench2Drive's 30 m criterion applies). Useful in single-
                scenario training loops to cut wasted steps after the agent
                wanders off.
            early_blocked_steps: Tighter analogue of Bench2Drive's
                AgentBlockedTest (which terminates after 60 s = 1200 steps
                @ 20 FPS under 0.1 m/s). When set, the env additionally
                terminates after this many *consecutive* steps under
                0.1 m/s. ``None`` disables the early check.
            eval_output_dir: If set, the env writes Bench2Drive-eval-
                compatible artifacts (``eval_res/{idx:03d}_res.json``,
                ``eval_viz/{save_name}/metric_info.json``) under this
                directory after every episode — same files Bench2Drive's
                standard eval scripts produce, so the
                downstream tools (merge_route_json.py,
                efficiency_smoothness_benchmark.py) work unmodified. Only
                used when a Bench2DriveRuntime is attached. The matching
                ``weather.xml`` is auto-derived from ``route_xml``'s
                parent directory (Bench2Drive's standard layout).
        """
        super().__init__()
        self.prompt = BASE_PROMPT

        # If a Bench2Drive route XML is given, the env runs the eval-equivalent
        # RouteScenario (ego spawn, BackgroundActivity NPC traffic, scripted
        # scenarios, parked vehicles) via Bench2DriveRuntime.
        self._route_xml = route_xml
        self._route_id = route_id
        self._tm_port = 8000
        self.runtime: Bench2DriveRuntime | None = None

        # Optional early-termination thresholds (Bench2Drive's
        # InRouteTest=30m / AgentBlockedTest=60s are also active; these
        # just trigger sooner). ``_blocked_step_counter`` accumulates
        # consecutive sub-0.1m/s steps and is reset on every faster step
        # and on ``reset()``.
        self._early_off_route_m = early_off_route_m
        self._early_blocked_steps = early_blocked_steps
        self._blocked_step_counter = 0
        # Running collision-infraction total at the previous step; a jump
        # means a fresh collision (→ immediate episode termination).
        self._prev_collision_total = 0

        # Observation / action contracts come from carla_obs (shared with the
        # Bench2Drive eval agent so training and eval stay bit-aligned).
        self.obs_cfg = CARLAObsConfig()
        # Safety net only. Bench2Drive's scenario_tree (RouteCompletionTest /
        # InRouteTest / AgentBlockedTest) is what actually ends the episode;
        # this cap exists so a stuck CarlaDataProvider can't pin the env
        # forever. 20 FPS × 600 s = 12000 covers the longest Town12 routes.
        self.max_episode_steps = 12000
        self.render_mode = "rgb_array"
        self.fps = 20  # Simulation FPS

        self.observation_space = gym.spaces.Dict({"image": make_obs_space(self.obs_cfg)})
        self.action_space = make_action_space()

        # Process-wide singleton client (created once, reused by every env
        # instance). A 2nd carla.Client in one process keeps the first's
        # streaming threads alive and corrupts the session, so constructing one
        # per env would break ``trial_num>1``; get_client() shares one client
        # (and one server) across all trials. See envs/carla_bootstrap.py.
        self.client = get_client()

        # Build the runtime first (parses the XML eagerly) so we can read the
        # initial town off the upcoming config before deciding which world
        # to load. Sequential mode tolerates mixed-town XMLs (e.g. the
        # bench2drive220 driver); the env reloads the world on town change.
        if route_xml is not None:
            self.runtime = Bench2DriveRuntime(
                client=self.client,
                traffic_manager_port=self._tm_port,
                route_xml=route_xml,
                route_id=route_id,
                sequence_mode=sequence_mode,
                start_index=start_index,
                loop=loop,
            )
            initial_town = self.runtime.peek_next_town()
        else:
            initial_town = "Town12"

        # Eval-artifact writer. Only attaches when running against a
        # bench2drive XML; random-route mode has no comparable "scenario"
        # concept and no StatisticsManager would apply. weather.xml lives
        # next to the route XML in every Bench2Drive layout we ship.
        self.eval_writer: Bench2DriveEvalWriter | None = None
        if eval_output_dir is not None and self.runtime is not None:
            weather_xml = Path(route_xml).parent / "weather.xml"
            if not weather_xml.exists():
                raise FileNotFoundError(
                    f"weather_xml not found at {weather_xml} (expected next to route_xml)"
                )
            self.eval_writer = Bench2DriveEvalWriter(
                output_root=eval_output_dir,
                weather_xml=weather_xml,
            )

        # reset_settings=False keeps the sync settings we apply right after.
        self.world = self.client.load_world(initial_town, reset_settings=False)
        self.current_town: str = initial_town
        self._apply_world_settings()

        # Traffic Manager also needs to be in sync mode; otherwise world.tick
        # can deadlock on maps with background traffic.
        self.traffic_manager = self.client.get_trafficmanager(self._tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_hybrid_physics_mode(True)

        # Map and spawn points (refreshed when the world is reloaded for a
        # different town during sequential iteration).
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()

        # Sensors and vehicle
        self.vehicle = None
        self.camera_sensor = None
        self.third_person_camera = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None

        # Sensor data
        self.current_image = None
        self.third_person_image = None
        self.lane_invasion_history = []
        # Latest raw sensor readings exposed via info["sensors"] for VLA agents
        # that need leaderboard-compatible GNSS / IMU / camera frames. Tuple
        # shape mirrors the Bench2Drive AutonomousAgent sensors() contract:
        # (frame_idx, payload). None until the first post-reset tick fills them.
        self._last_camera: tuple[int, np.ndarray] | None = None
        self._last_gnss: tuple[int, np.ndarray] | None = None
        self._last_imu: tuple[int, np.ndarray] | None = None

        # Route tracker (initialized in reset)
        self.route_tracker: RouteTracker | None = None

        # Leaderboard evaluation metrics
        self.infractions = {
            "collision_pedestrian": 0,
            "collision_vehicle": 0,
            "collision_static": 0,
            "red_light": 0,
            "lane_invasion": 0,
        }
        self.prev_driving_score = 0.0
        self._latest_score_route = 0.0
        self._latest_score_penalty = 1.0
        self._latest_driving_score = 0.0
        self._latest_off_route_pct = 0.0

        # Episode information
        self.episode_step = 0
        # Steps since the env was created, never reset: it dates an experience
        # within the run, so a sequence model can order what it reads.
        self.global_step = 0

        # Vehicle physics information
        self.vehicle_physics = VehiclePhysics.create()
        self.current_action = np.zeros(2, dtype=np.float32)

        # History buffer for render graphs
        self.history_length = 200
        self.physics_history: list[VehiclePhysics] = []

    def _apply_world_settings(self) -> None:
        """Apply the sync-mode + large-map streaming settings to ``self.world``.

        Called once at construction and again after every ``load_world``
        (sequential mode crosses towns and needs to re-apply these).
        """
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        # Large Map streaming + actor activation distances are reset by
        # load_world; mirror eval (leaderboard_evaluator._load_and_wait_for_world).
        # No-op for small towns.
        settings.tile_stream_distance = 650
        settings.actor_active_distance = 650
        self.world.apply_settings(settings)
        self.world.reset_all_traffic_lights()

    def _switch_town_if_needed(self) -> None:
        """If the runtime's next scenario lives on a different town, reload it.

        Town12/13 reloads take ~30 s, so bench2drive220 mixes single-town
        runs with abrupt town flips — only reload when the town actually
        differs from what's currently in the world.
        """
        if self.runtime is None:
            return
        next_town = self.runtime.peek_next_town()
        if next_town == self.current_town:
            return

        # The previous scenario's actors/sensors have already been torn down
        # by the caller (cleanup + sensor.destroy loop). Drop the world here
        # so load_world doesn't have stale lambdas firing on the new world.
        self.world = self.client.load_world(next_town, reset_settings=False)
        self.current_town = next_town
        self._apply_world_settings()
        # TM is keyed by port and survives world swaps, but must be told the
        # new world is synchronous before the first tick.
        self.traffic_manager.set_synchronous_mode(True)
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()

    def reset(
        self,
        seed: int | None,
        options: dict[str, any] | None,
    ) -> tuple[dict[str, any], dict[str, any]]:
        super().reset(seed=seed)

        # Destroy our sensors before clearing the scenario — once their parent
        # actor is gone, sensor.destroy() can throw on already-dead handles.
        # stop() first is required: without it the streaming socket on port
        # 2001 and its listener thread are not released, and the lambda
        # closure (which holds self) is never GC'd — every reset would leak.
        for sensor in (
            self.camera_sensor,
            self.third_person_camera,
            self.collision_sensor,
            self.lane_invasion_sensor,
            self.gnss_sensor,
            self.imu_sensor,
        ):
            if sensor is not None:
                sensor.stop()
                sensor.destroy()

        if self.runtime is not None:
            # The runtime owns the ego (RouteScenario spawns it) plus all NPCs
            # and parked vehicles. Tear them down before any potential
            # load_world so the previous-world actor handles don't leak.
            self.runtime.cleanup()
            # Now safe to switch towns (sequential mode walks the XML in
            # order and may cross towns between consecutive resets).
            self._switch_town_if_needed()
            self.vehicle, route_locations = self.runtime.reset(self.world)
            # Open a fresh Bench2Drive eval record now that the scenario
            # is built. Doing it here (not in begin_episode-on-step-1)
            # means StatisticsManager.set_scenario sees the live
            # RouteScenario before the criteria sensors start firing.
            if self.eval_writer is not None:
                self.eval_writer.begin_episode(
                    route_scenario=self.runtime.route_scenario,
                    config=self.runtime.configs[self.runtime.current_index],
                    scenario_index=self.runtime.current_index,
                )
        else:
            if self.vehicle is not None:
                self.vehicle.destroy()
            route_locations, start_pose = self._sample_random_route()
            blueprint_library = self.world.get_blueprint_library()
            # role_name='hero' is required on large maps (Town13 etc.); without
            # it, attaching a camera to the vehicle crashes UE4 with SIGSEGV.
            vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
            vehicle_bp.set_attribute("role_name", "hero")
            spawn_transform = carla.Transform(
                carla.Location(
                    x=start_pose.location.x,
                    y=start_pose.location.y,
                    z=start_pose.location.z + 0.5,
                ),
                start_pose.rotation,
            )
            while True:
                try:
                    self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_transform)
                    break
                except RuntimeError:
                    time.sleep(0.1)
                    continue

        self.vehicle.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=1))

        self.route_tracker = RouteTracker.from_raw_waypoints(route_locations, self.obs_cfg)

        blueprint_library = self.world.get_blueprint_library()

        # Camera sensor
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.obs_cfg.image_size[0]))
        camera_bp.set_attribute("image_size_y", str(self.obs_cfg.image_size[1]))
        camera_bp.set_attribute("fov", str(self.obs_cfg.fov))
        camera_transform = carla.Transform(
            carla.Location(x=self.obs_cfg.camera_x, z=self.obs_cfg.camera_z)
        )
        self.camera_sensor = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        self.camera_sensor.listen(lambda image: self._process_image(image))

        # Third-person camera (for human monitoring)
        tp_bp = blueprint_library.find("sensor.camera.rgb")
        tp_bp.set_attribute("image_size_x", "800")
        tp_bp.set_attribute("image_size_y", "600")
        tp_bp.set_attribute("fov", "90")
        tp_transform = carla.Transform(carla.Location(x=-8.0, z=5.0), carla.Rotation(pitch=-20.0))
        self.third_person_camera = self.world.spawn_actor(
            tp_bp, tp_transform, attach_to=self.vehicle
        )
        self.third_person_camera.listen(lambda image: self._process_third_person_image(image))

        # Collision sensor
        collision_bp = blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        # Lane invasion sensor
        lane_invasion_bp = blueprint_library.find("sensor.other.lane_invasion")
        self.lane_invasion_sensor = self.world.spawn_actor(
            lane_invasion_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.lane_invasion_sensor.listen(lambda event: self._on_lane_invasion(event))

        # GNSS / IMU — exposed via info["sensors"] for VLA agents that need
        # leaderboard-compatible pose / inertial readings. Default attributes
        # match Bench2Drive's AutonomousAgent sensor specs.
        gnss_bp = blueprint_library.find("sensor.other.gnss")
        self.gnss_sensor = self.world.spawn_actor(
            gnss_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.gnss_sensor.listen(lambda data: self._on_gnss(data))

        imu_bp = blueprint_library.find("sensor.other.imu")
        self.imu_sensor = self.world.spawn_actor(imu_bp, carla.Transform(), attach_to=self.vehicle)
        self.imu_sensor.listen(lambda data: self._on_imu(data))

        # Initialization
        self.episode_step = 0
        self._blocked_step_counter = 0
        self._prev_collision_total = 0
        self.lane_invasion_history = []
        self.infractions = {k: 0 for k in self.infractions}
        self.prev_driving_score = 0.0
        self._latest_score_route = 0.0
        self._latest_score_penalty = 1.0
        self._latest_driving_score = 0.0
        self._latest_off_route_pct = 0.0
        self.current_image = None
        self._last_camera = None
        self._last_gnss = None
        self._last_imu = None
        self.vehicle_physics = VehiclePhysics.create()
        self.physics_history = []

        # Wait until camera, GNSS and IMU have all delivered at least one
        # frame so info["sensors"] handed back from reset is fully populated.
        while True:
            self.world.tick()
            if (
                self.current_image is not None
                and self._last_gnss is not None
                and self._last_imu is not None
            ):
                break
            time.sleep(0.1)

        # The lane invasion sensor can fire spuriously on the spawn tick when
        # the ego is placed near a marking; clear here so only agent-driven
        # crossings count.
        self.lane_invasion_history = []
        self.infractions["lane_invasion"] = 0

        self._update_spectator()

        observation = {"image": self.current_image.copy(), "sensors": self._build_sensors_dict()}
        # The (gps, world-coord) global plan for VLA agents that build their own
        # route planner. Only the Bench2Drive runtime has a
        # RouteScenario; the reset observation carries it once per episode so the
        # agent never reaches into the env to fetch it.
        if self.runtime is not None and self.runtime.route_scenario is not None:
            observation["route_plan"] = (
                self.runtime.route_scenario.gps_route,
                self.runtime.route_scenario.route,
            )
        return observation, self._build_scenario_info(
            {
                "task_prompt": self._navigation_prompt(),
                "shaped_reward": 0.0,
                **self._step_count_info(),
            }
        )

    # ``final_eval_summary`` is populated by ``close()`` (auto-merge of the
    # 220-route sweep). Trainer reads it after env.close() for wandb
    # summary; ``None`` when no eval writer is attached.
    final_eval_summary: dict[str, float] | None = None

    def _navigation_prompt(self) -> str:
        """Task prompt augmented with the upcoming route maneuver.

        The maneuver command lives on the Bench2Drive route (``route_scenario.route``
        as ``(transform, RoadOption)`` pairs); the route tracker's completion fraction
        indexes the current position, and a small look-ahead surfaces the next turn.
        Delivered via ``info["task_prompt"]`` (-> obs["language"]) so VLA agents receive
        navigation intent through the standard prompt channel."""
        if self.runtime is None or self.runtime.route_scenario is None:
            return self.prompt
        route = self.runtime.route_scenario.route
        if not route:
            return self.prompt
        idx = int(self.route_tracker.route_completion * (len(route) - 1))
        look = min(idx + _MANEUVER_LOOKAHEAD_PTS, len(route) - 1)
        cmd = route[look][1]
        command = int(cmd.value) if hasattr(cmd, "value") else int(cmd)
        maneuver = COMMAND_TO_MANEUVER.get(command, COMMAND_TO_MANEUVER[4])
        return f"{self.prompt} {maneuver}"

    def _build_scenario_info(self, base: dict[str, any]) -> dict[str, any]:
        """Merge bench2drive scenario metadata into an info dict.

        Only populated when a runtime is attached; for random-route mode
        these keys are absent (trainer side guards with ``in info``).
        """
        if self.runtime is None:
            return base
        base["scenario_index"] = self.runtime.current_index
        base["route_id"] = self.runtime.current_route_id
        base["town"] = self.runtime.current_town
        base["scenarios_total"] = self.runtime.total_scenarios
        return base

    def _step_count_info(self) -> dict[str, int]:
        return {
            "global_step": self.global_step,
            "episode_step": self.episode_step,
        }

    def step(self, action: np.ndarray) -> tuple[dict[str, any], float, bool, bool, dict[str, any]]:
        # Apply action: action[0]=steer, action[1]=gas_or_brake (same as CarRacing)
        self.current_action = np.array(action, dtype=np.float32)
        steer, throttle, brake = action_to_vehicle_control(self.current_action)
        brake *= 0.0

        control = carla.VehicleControl(
            steer=steer, throttle=throttle, brake=brake, hand_brake=False, manual_gear_shift=False
        )
        self.vehicle.apply_control(control)
        # Publish to the py_trees Blackboard the leaderboard scenarios
        # read for ControlLoss-style noise injection. Without this,
        # ``NoiseControl.update`` prints
        # ``WARNING: Couldn't add noise to the ego because the control
        # couldn't be found`` every tick and skips its noise pass.
        if self.runtime is not None:
            py_trees.blackboard.Blackboard().set("AV_control", control, overwrite=True)

        self.world.tick()

        # Tick the Bench2Drive scenario tree after the world advances so that
        # NPC commands issued this tick land on the next world.tick (matches
        # ScenarioManager._tick_scenario ordering).
        if self.runtime is not None:
            self.runtime.tick(self.world)

        # Capture ego physics for metric_info.json. Done after both
        # world.tick + scenario tick so the recorded acceleration/yaw
        # rate already reflect this step's control.
        if self.eval_writer is not None:
            self.eval_writer.record_step(self.vehicle)

        self._update_spectator()

        # Update route tracking
        self.route_tracker.update(self.vehicle.get_location())

        # Termination conditions — mirror Bench2Drive's ScenarioManager.
        # Collision / off-route / red light / stop / min-speed are recorded
        # as infractions by the scenario_tree's criteria and feed the
        # StatisticsManager Driving Score, but they do NOT stop the route.
        # The scenario_tree ends on:
        #   - RouteCompletionTest = SUCCESS  (route_completion >= 1.0)
        #   - InRouteTest = FAILURE          (30 m off-route)
        #   - AgentBlockedTest = FAILURE     (60 s under 0.1 m/s)
        # Reward = step delta of Bench2Drive score_composed; criteria
        # update happens inside the scenario tick above.
        reward = self._compute_reward()

        # A collision (vehicle / pedestrian / static) ends the episode
        # immediately. The collision sensor's callback bumps the infraction
        # counters during the world.tick above, so a jump in the running
        # total since the previous step means we hit something this step.
        collision_total = (
            self.infractions["collision_vehicle"]
            + self.infractions["collision_pedestrian"]
            + self.infractions["collision_static"]
        )
        collided = collision_total > self._prev_collision_total
        self._prev_collision_total = collision_total

        if self.runtime is not None and self.runtime.route_scenario is not None:
            scenario_running = (
                self.runtime.route_scenario.scenario_tree.status == py_trees.common.Status.RUNNING
            )
            terminated = not scenario_running
        else:
            # Random-route mode (no Bench2DriveRuntime): no scenario_tree,
            # so route completion is the only natural end. The
            # max_episode_steps cap below catches everything else.
            terminated = self.route_tracker.route_completion >= 1.0

        # Collision is a hard terminal failure — stop the episode at once.
        if collided:
            terminated = True

        # Wandering too far off-route is treated as a truncation (not a
        # natural terminal): cut the episode and hand back a fixed -1 shaped
        # reward so the agent learns the boundary, while the leaderboard
        # reward keeps reporting the score delta. Looser analogue of
        # Bench2Drive's InRouteTest, which fails at 30 m.
        off_route_truncated = (
            self._early_off_route_m is not None
            and self.route_tracker.min_distance_to_route > self._early_off_route_m
        )

        # Refresh physics (velocity etc.) from the post-tick vehicle state
        # before any consumer reads it; the blocked check below needs the
        # current-step speed, not the previous step's stale value.
        self.vehicle_physics.update(self.vehicle, throttle, brake, steer)

        # AgentBlockedTest analogue keeps terminated (FAILURE) semantics.
        if self._early_blocked_steps is not None:
            speed = float(np.linalg.norm(self.vehicle_physics.velocity))
            if speed < 0.1:
                self._blocked_step_counter += 1
                if self._blocked_step_counter >= self._early_blocked_steps:
                    terminated = True
            else:
                self._blocked_step_counter = 0

        truncated = self.episode_step >= self.max_episode_steps or off_route_truncated
        shaped_reward = -1.0 if off_route_truncated else reward

        self.episode_step += 1
        self.global_step += 1

        assert self.current_image is not None

        # Observation is the raw camera RGB; the route is given to the agent as
        # text, so no route overlay is baked into the policy input.
        observation = {"image": self.current_image.copy(), "sensors": self._build_sensors_dict()}

        # Record history
        self.physics_history.append(copy.deepcopy(self.vehicle_physics))
        if len(self.physics_history) > self.history_length:
            self.physics_history = self.physics_history[-self.history_length :]

        info = self._build_scenario_info(
            {
                "task_prompt": self._navigation_prompt(),
                "shaped_reward": shaped_reward,
                **self._step_count_info(),
                "route_completion": self.route_tracker.route_completion,
                "driving_score": self._latest_driving_score,
                "score_route": self._latest_score_route,
                "score_penalty": self._latest_score_penalty,
                "infractions": self.infractions.copy(),
                "velocity": self.vehicle_physics.velocity,
                "velocity_kph": self.vehicle_physics.velocity_kph,
                "acceleration": self.vehicle_physics.acceleration,
                "lon_acceleration": self.vehicle_physics.lon_acceleration,
                "lat_acceleration": self.vehicle_physics.lat_acceleration,
                "jerk": self.vehicle_physics.jerk,
                "lon_jerk": self.vehicle_physics.lon_jerk,
                "angular_velocity": self.vehicle_physics.angular_velocity,
                "angular_acceleration": self.vehicle_physics.angular_acceleration,
            }
        )

        # Flush per-route Bench2Drive eval artifacts at the episode
        # boundary while the RouteScenario (and its criteria) are still
        # alive — next reset would tear them down. Summary keys land on
        # the final step's info dict as ``eval_summary``.
        if (terminated or truncated) and self.eval_writer is not None:
            info["eval_summary"] = self.eval_writer.end_episode()

        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Return third-person camera image with time-series graphs overlay."""
        if self.render_mode != "rgb_array":
            return None
        if self.third_person_image is None:
            return None

        img = self.third_person_image.copy()
        overlay_vehicle_graphs(img, self.physics_history)
        vehicle_yaw_rad = np.radians(self.vehicle.get_transform().rotation.yaw)
        overlay = self.route_tracker.render_overlay(self.vehicle.get_location(), vehicle_yaw_rad)
        paste_route_overlay(img, overlay, self.obs_cfg)
        return img

    def close(self):
        for sensor in (
            self.camera_sensor,
            self.third_person_camera,
            self.collision_sensor,
            self.lane_invasion_sensor,
            self.gnss_sensor,
            self.imu_sensor,
        ):
            if sensor is not None:
                sensor.stop()
                sensor.destroy()

        if self.runtime is not None:
            # Tears down the ego, NPCs, scripted scenario actors, parked vehicles.
            self.runtime.cleanup()
        elif self.vehicle is not None:
            self.vehicle.destroy()

        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)

        # Auto-merge the 220-route eval sweep on close so the trainer
        # gets the final Driving Score / Success Rate / Efficiency /
        # Comfort summary via ``final_eval_summary`` without having to
        # know about the Bench2Drive eval mechanics.
        if self.eval_writer is not None:
            self.final_eval_summary = self.eval_writer.finalize_all()

    def _sample_random_route(self) -> tuple[list[carla.Location], carla.Transform]:
        """Sample a random ~200 m route in the current map.

        Used only when no Bench2Drive XML is configured; the XML path goes
        through ``Bench2DriveRuntime`` which spawns the ego itself.
        """
        start_wp = self.map.get_waypoint(
            np.random.choice(self.spawn_points).location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        route_waypoints = [start_wp]
        current_wp = start_wp
        distance = 0.0

        while distance < 200.0:
            next_wps = current_wp.next(2.0)
            if not next_wps:
                break

            # Randomly select next waypoint (branch at intersections)
            current_wp = np.random.choice(next_wps)
            route_waypoints.append(current_wp)

            if len(route_waypoints) > 1:
                prev_loc = route_waypoints[-2].transform.location
                curr_loc = current_wp.transform.location
                distance += prev_loc.distance(curr_loc)

        start_pose = route_waypoints[0].transform
        raw_locations = [wp.transform.location for wp in route_waypoints]
        return raw_locations, start_pose

    def _compute_reward(self) -> float:
        """
        Reward = step delta of Bench2Drive ``score_composed`` (range ~[0, 100]).

        ``score_composed = score_route(0-100) * Π(penalty_factor in (0,1])`` —
        any infraction (collision, off-route, red light, stop, ...) drops the
        score multiplicatively, so the delta is naturally negative on the step
        the infraction fires. Matches what eval reports in ``eval.json``.

        Bench2Drive runtime reads the same ``RouteScenario`` criteria that
        ``statistics_manager.compute_route_statistics`` reads at end-of-
        episode; random-route mode falls back to a coarse, sensor-based
        approximation (collisions only — no waypoint-aware off-route check).
        """
        if self.runtime is not None and self.runtime.route_scenario is not None:
            score_route, score_penalty = self._score_from_criteria()
        else:
            score_route, score_penalty = self._score_from_sensors()
        new_score = max(score_route * score_penalty, 0.0)

        # Cache for info dict.
        self._latest_score_route = score_route
        self._latest_score_penalty = score_penalty
        self._latest_driving_score = new_score

        reward = new_score - self.prev_driving_score
        self.prev_driving_score = new_score
        return reward

    def _score_from_criteria(self) -> tuple[float, float]:
        score_penalty = 1.0
        off_route_pct = 0.0
        for criterion in self.runtime.route_scenario.get_criteria():
            for event in criterion.events:
                event_type = event.get_type()
                if event_type in PENALTY_VALUE_DICT:
                    score_penalty *= PENALTY_VALUE_DICT[event_type]
                elif event_type in PENALTY_PERC_DICT:
                    pv, pt = PENALTY_PERC_DICT[event_type]
                    if pt == "increases":
                        pct = event.get_dict()["percentage"]
                        score_penalty *= 1 - (1 - pv) * pct / 100.0
                    elif pt == "decreases":
                        pct = event.get_dict()["percentage"]
                        score_penalty *= 1 - (1 - pv) * (1 - pct / 100.0)
                    # "unused" → no-op (matches statistics_manager).
                    if event_type == TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION:
                        off_route_pct = event.get_dict()["percentage"]
        self._latest_off_route_pct = off_route_pct
        # RouteCompletionTest only advances on discrete waypoint crossings
        # (external/Bench2Drive .../atomic_criteria.py); route_tracker gives
        # continuous sub-segment progress instead, for dense reward shaping.
        score_route = self.route_tracker.route_completion * 100.0
        return score_route, score_penalty

    def _score_from_sensors(self) -> tuple[float, float]:
        """Coarse fallback for random-route mode (no RouteScenario criteria)."""
        score_route = self.route_tracker.route_completion * 100.0
        score_penalty = (
            0.5 ** self.infractions["collision_pedestrian"]
            * 0.6 ** self.infractions["collision_vehicle"]
            * 0.65 ** self.infractions["collision_static"]
        )
        # No criterion-equivalent off-route check in random-route mode.
        self._latest_off_route_pct = 0.0
        return score_route, score_penalty

    def _update_spectator(self):
        """Move spectator camera to follow the vehicle from behind and above."""
        vehicle_transform = self.vehicle.get_transform()
        forward = vehicle_transform.get_forward_vector()
        spectator_location = vehicle_transform.location + carla.Location(
            x=-8.0 * forward.x, y=-8.0 * forward.y, z=5.0
        )
        spectator_transform = carla.Transform(
            spectator_location,
            carla.Rotation(pitch=-20.0, yaw=vehicle_transform.rotation.yaw),
        )
        self.world.get_spectator().set_transform(spectator_transform)

    def _process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4)
        )
        # Keep BGR HWC uint8 alongside the RGB CHW float32 obs so VLA agents
        # that expect the leaderboard sensor format can read it from
        # info["sensors"]["rgb"] without a second decode pass.
        bgr = array[:, :, :3].copy()
        self._last_camera = (image.frame, bgr)
        self.current_image = bgr[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0

    def _on_gnss(self, data):
        self._last_gnss = (
            data.frame,
            np.array([data.latitude, data.longitude, data.altitude], dtype=np.float64),
        )

    def _on_imu(self, data):
        acc = data.accelerometer
        gyr = data.gyroscope
        self._last_imu = (
            data.frame,
            np.array(
                [acc.x, acc.y, acc.z, gyr.x, gyr.y, gyr.z, float(data.compass)],
                dtype=np.float32,
            ),
        )

    def _build_sensors_dict(self) -> dict[str, tuple[int, object]]:
        """Latest sensor snapshot in Bench2Drive AutonomousAgent format.

        Speed comes from ``vehicle.get_velocity()`` rather than a dedicated
        sensor; we tag it with the camera frame so all four entries share a
        consistent monotonic index for VLA agents that want to skip stale
        readings.
        """
        vel = self.vehicle.get_velocity()
        speed_mps = float(np.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z))
        return {
            "rgb": self._last_camera,
            "gps": self._last_gnss,
            "imu": self._last_imu,
            "speed": (self._last_camera[0], {"speed": speed_mps}),
        }

    def _process_third_person_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, [2, 1, 0]]
        self.third_person_image = array

    def _on_collision(self, event):
        other_actor = event.other_actor
        if "vehicle" in other_actor.type_id:
            self.infractions["collision_vehicle"] += 1
        elif "walker" in other_actor.type_id:
            self.infractions["collision_pedestrian"] += 1
        else:
            self.infractions["collision_static"] += 1

    def _on_lane_invasion(self, event):
        self.lane_invasion_history.append(event)
        self.infractions["lane_invasion"] += 1
