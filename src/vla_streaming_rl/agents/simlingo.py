# SPDX-License-Identifier: MIT
import carla
import cv2
import gymnasium as gym
import numpy as np
import torch
from leaderboard.utils.route_manipulation import downsample_route  # type: ignore
from PIL import Image
from torch import nn, optim

from vla_streaming_rl.agents.step_result import StepResult
from vla_streaming_rl.networks.interface import InferInput
from vla_streaming_rl.networks.simlingo_network import (
    ACTION_DIM,
    NUM_WP_QUERIES,
    ROUTE_LEN,
    SPEED_WPS_LEN,
    WP_DIM,
    SimLingoNetwork,
)
from vla_streaming_rl.optimizers.adam_et import AdamET
from vla_streaming_rl.replay_buffer import ReplayBuffer
from vla_streaming_rl.reward_processor import RewardProcessor
from vla_streaming_rl.simlingo.team_code.config_simlingo import GlobalConfig
from vla_streaming_rl.simlingo.team_code.ego_state_filter import EgoStateFilter
from vla_streaming_rl.simlingo.team_code.prompt_builder import PromptBuilder
from vla_streaming_rl.simlingo.team_code.route_planner import RoutePlanner
from vla_streaming_rl.simlingo.team_code.simlingo_utils import (
    get_camera_extrinsics,
    get_camera_intrinsics,
    inverse_conversion_2d,
    preprocess_compass,
)
from vla_streaming_rl.simlingo.team_code.trajectory_to_control import TrajectoryToControl
from vla_streaming_rl.simlingo.utils.custom_types import DrivingInput
from vla_streaming_rl.simlingo.utils.internvl2_utils import build_transform, dynamic_preprocess
from vla_streaming_rl.utils import create_reward_image

# Configure pytorch for maximum performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

# Waypoint shape constants (ROUTE_LEN, SPEED_WPS_LEN, WP_DIM, NUM_WP_QUERIES,
# ACTION_DIM) are imported from ``networks.simlingo_network`` — they describe
# the SimLingo waypoint output that the network produces.

# Single-frame VLM input (seq_len=1) with a one-step bootstrap horizon, so the
# replay buffer needs ``_SEQ_LEN + _HORIZON`` contiguous indices per sample to
# form an off-policy transition (s_t, s_{t+1}) -- same convention as
# ``off_policy.py`` (``seq_len=self.seq_len + self.horizon``).
_SEQ_LEN = 1
_HORIZON = 1


def _action_vec_to_waypoints(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    route_flat = a[: ROUTE_LEN * WP_DIM]
    speed_flat = a[ROUTE_LEN * WP_DIM :]
    return (
        route_flat.view(1, ROUTE_LEN, WP_DIM),
        speed_flat.view(1, SPEED_WPS_LEN, WP_DIM),
    )


class SimLingoAgent:
    def __init__(
        self,
        *,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        network: SimLingoNetwork,
        gamma: float,
        buffer_size: int,
        batch_size: int,
        learning_starts: int,
        exploration_noise: float,
        actor_lr: float,
        critic_lr: float,
        max_grad_norm: float,
        learning_mode: str,
        et_lambda: float,
    ) -> None:
        self.observation_space = observation_space
        del action_space

        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.exploration_noise = exploration_noise
        self.max_grad_norm = max_grad_norm

        torch.cuda.empty_cache()
        self._frame_step = -1
        self.initialized = False
        self.device = torch.device("cuda")
        self.config = GlobalConfig()
        self.bias = {
            "speed_scale": 1.0,
            "speed_offset": 0.0,
            "gps_x": 0.0,
            "gps_y": 0.0,
            "compass_rad": 0.0,
        }

        self.trajectory_to_control = TrajectoryToControl(self.config)

        self.route_planner_max_distance = 50.0
        self.route_planner_min_distance = 7.5

        # The learnable network (SimLingo VLM + waypoint-head policy μ + Q
        # critic) is built outside and passed in. ``cfg`` / tokenizer /
        # num_image_token are byproducts of its construction that the inference
        # pipeline below needs.
        self.network = network
        self.cfg = network.cfg
        self.actor_heads = network.actor_heads
        self.critic = network.critic

        self.T = 1
        self.stuck_detector = 0
        self.force_move = 0

        self.ego_state_filter = EgoStateFilter(dt=1.0 / 20.0, state_log_maxlen=5)
        self.prompt_builder = PromptBuilder(
            config=self.config,
            tokenizer=network.tokenizer,
            num_image_token=network.num_image_token,
            device=self.device,
        )

        # --- RL setup ---------------------------------------------------

        feature_dim = network.feature_dim

        # ``off_policy`` trains on random replay batches (single combined loss via
        # ``compute_loss``); ``streaming`` trains online on the latest transition
        # with TD(λ) eligibility traces on the critic (AdamET) via
        # ``infer_and_compute_loss``. The actor uses AdamW in both modes.
        self.learning_mode = learning_mode
        self.actor_optimizer = optim.AdamW(
            self.actor_heads.parameters(), lr=actor_lr, weight_decay=0.0
        )
        self.critic_optimizer = (
            AdamET(self.critic.parameters(), lr=critic_lr, gamma=gamma, et_lambda=et_lambda)
            if learning_mode == "streaming"
            else optim.AdamW(self.critic.parameters(), lr=critic_lr, weight_decay=0.0)
        )

        # ``run_step`` caches each tick's per-query VLM features and the
        # action it took; ``step`` stores them in the obs / action slots so
        # ``network.compute_loss`` reads (features, action, reward, done)
        # straight from the buffer and only re-applies the waypoint heads. The obs_z
        # / rnn_state / log_prob / value / token slots stay unused
        # (shape-(1,) / 0 / empty) so the buffer machinery still type-checks.
        self.rb = ReplayBuffer(
            size=buffer_size,
            seq_len=_SEQ_LEN + _HORIZON,
            obs_shape=(NUM_WP_QUERIES, feature_dim),
            obs_z_shape=(1,),
            rnn_state_shape=(1,),
            action_shape=(ACTION_DIM,),
            output_device=self.device,
            storage_device=torch.device("cpu"),
            max_new_tokens=0,
            max_prompt_tokens=0,
            pad_token_id=0,
        )
        self._dummy_rnn_state = torch.zeros(1, device=self.device)
        self._dummy_obs_z = torch.zeros(1, device=self.device)

        # ``run_step`` writes the just-computed features / action into these
        # so the next ``step`` can store them at the buffer index for the
        # current timestep (the project's convention puts the action that
        # produced state t at ``actions[t]``).
        self._current_state: torch.Tensor = torch.zeros(
            NUM_WP_QUERIES, feature_dim, device=self.device
        )
        self._current_action_taken: torch.Tensor = torch.zeros(ACTION_DIM, device=self.device)

        # Re-run ``_init`` (rebuild the RoutePlanner from the new episode's
        # route plan) on the next tick. Set on construction and by
        # ``on_episode_end``; consumed by ``_maybe_handover_episode``.
        self._need_handover = True
        self._prev_action = torch.zeros(ACTION_DIM, device=self.device)

        # Latest executed trajectory + its critic value, cached by ``run_step``
        # for the bird's-eye visualization panel built in ``_build_info``.
        self._viz_route = np.zeros((ROUTE_LEN, WP_DIM), dtype=np.float32)
        self._viz_speed = np.zeros((SPEED_WPS_LEN, WP_DIM), dtype=np.float32)
        self._viz_q_value = 0.0
        # Critic value diagnostics for the executed action (incl. per-bin probs),
        # cached by ``run_step`` and logged verbatim in ``_build_info``.
        self._value_report: dict[str, float] = {"value": 0.0}

        # "carla" shapes the per-step Bench2Drive score delta: exact-zero
        # rewards (stuck) get a small negative push and collision spikes
        # are hard-clipped to keep the critic stable. See RewardProcessor.
        self.reward_processor = RewardProcessor("carla", 1.0)

    # --- RL-agent protocol surface used by scripts/train.py -----------------

    @torch.no_grad()
    def select_action(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
        info: dict,
    ) -> StepResult:
        self._maybe_handover_episode(info)
        env_action = self._act(info["sensors"])
        metrics, panels = self._build_info(env_action, reward)
        return StepResult(action=env_action, metrics=metrics, panels=panels)

    def step(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
        info: dict,
    ) -> StepResult:
        self._maybe_handover_episode(info)
        episode_done = terminated or truncated
        env_action = self._act(info["sensors"])
        metrics, panels = self._build_info(env_action, reward)

        # Store (features_t, action_{t-1}, reward, done). The per-query VLM
        # features go into the obs slot (obs_z is unused);
        # ``action_{t-1}`` mirrors off_policy.OffPolicyAgent.select_action's
        # add semantics so later sampling and indexing match the project
        # convention.
        self.rb.add(
            self._current_state,
            self._dummy_obs_z,
            reward,
            episode_done,
            self._dummy_rnn_state,
            self._prev_action,
            0.0,
            0.0,
            [],
            [],
        )

        metrics.update(self._train(global_step, episode_done))

        # Advance: the action just selected becomes the prev for the
        # next add. (off_policy carries it across episode boundaries
        # too — the buffer's done flag handles bootstrap correctness.)
        self._prev_action = self._current_action_taken

        return StepResult(action=env_action, metrics=metrics, panels=panels)

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        # Force handover (RoutePlanner rebuild) on the next select_action, which
        # receives the new episode's reset info.
        self._need_handover = True
        return {}

    def _build_info(self, env_action: np.ndarray, reward: float) -> tuple[dict, dict]:
        """Return ``(metrics, panels)`` for this tick.

        The waypoint policy predicts neither a goal nor a next frame. It does
        not predict the reward either, so the reward panel shows the actual
        reward only (``pred=None``). ``action_norm`` is a scalar telemetry hook.
        ``processed_reward`` is the exact reward the critic trains on — the
        "carla" transform is stateless, so logging it here matches the value
        used at train time.
        """
        metrics = {
            "action_norm": float(np.linalg.norm(env_action)),
            # ``value`` (+ any per-bin probs) is the value head's read-out of the
            # executed action's Q(s, a), matching off_policy / streaming telemetry
            # so --calibration etc. stay agent-agnostic.
            **self._value_report,
            "processed_reward": self.reward_processor.normalize(torch.tensor(reward)).item(),
        }
        panels = {
            "reward": create_reward_image(None, reward),
            "bev_value": self._render_bev_panel(),
        }
        return metrics, panels

    def _render_bev_panel(self) -> np.ndarray:
        """Top-down (ego-frame) view of SimLingo's two predicted trajectories,
        annotated with the critic's value estimate Q(s, a).

        SimLingo outputs two waypoint sets, both drawn here: the ``route``
        (geometric path, cyan) and the ``speed`` waypoints (future positions
        used for speed control, orange). Ego is the green marker near the
        bottom; waypoint index 0 (``+x``) is forward → up, ``+y`` → right (the
        PID's heading convention). Returns an RGB uint8 image for the strip.
        """
        size = 256
        scale = 6.0  # pixels per meter
        img = np.full((size, size, 3), 30, dtype=np.uint8)
        ego_px = (size // 2, int(size * 0.8))

        def to_px(forward: float, lateral: float) -> tuple[int, int]:
            return (int(ego_px[0] + lateral * scale), int(ego_px[1] - forward * scale))

        def draw_trajectory(waypoints: np.ndarray, color: tuple[int, int, int]) -> None:
            pts = [to_px(float(wp[0]), float(wp[1])) for wp in waypoints]
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], color, 2)
            for p in pts:
                cv2.circle(img, p, 2, color, -1)

        # range rings every 10 m for scale reference
        for r_m in (10, 20, 30):
            cv2.circle(img, ego_px, int(r_m * scale), (60, 60, 60), 1)

        # the two SimLingo outputs
        route_color = (0, 200, 255)
        speed_color = (255, 165, 0)
        draw_trajectory(self._viz_route, route_color)
        draw_trajectory(self._viz_speed, speed_color)

        # ego marker pointing forward (up)
        cv2.drawMarker(img, ego_px, (0, 255, 0), cv2.MARKER_TRIANGLE_UP, markerSize=12, thickness=2)

        # value annotation (green if non-negative, red otherwise) + legend
        q = self._viz_q_value
        q_color = (0, 255, 0) if q >= 0.0 else (255, 80, 80)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"Q: {q:.3f}", (8, 22), font, 0.6, q_color, 2)
        cv2.putText(img, "route", (8, size - 24), font, 0.45, route_color, 1)
        cv2.putText(img, "speed", (8, size - 8), font, 0.45, speed_color, 1)
        return img

    # --- Episode handover --------------------------------------------------

    def _maybe_handover_episode(self, info: dict) -> None:
        """On the first tick of a new episode, take the route plan from the
        env's reset ``info`` and force ``_init`` to rebuild the RoutePlanner.

        ``_need_handover`` is set on construction and by ``on_episode_end``; the
        next ``select_action`` carries the new episode's reset info (which holds
        ``route_plan``), so no reach into the env is needed.
        """
        if not self._need_handover:
            return
        # ``route_plan`` is the standard leaderboard handover — RouteScenario
        # builds both the gps and world-coord route (see CARLALeaderboardEnv).
        gps_route, world_route = info["route_plan"]
        ds_ids = downsample_route(world_route, 50)
        self._global_plan_world_coord = [(world_route[x][0], world_route[x][1]) for x in ds_ids]
        self._global_plan = [gps_route[x] for x in ds_ids]
        self.initialized = False
        self._need_handover = False

    # --- Per-tick inference ------------------------------------------------

    def _act(self, sensors: dict) -> np.ndarray:
        """One inference tick + 2-D env-action conversion.

        ``sensors`` is the env's ``info["sensors"]`` snapshot (the leaderboard
        ``{id: (frame, payload)}`` shape from
        :meth:`CARLALeaderboardEnv._build_sensors_dict`); we remap ``rgb`` →
        ``rgb_<N>`` per SimLingo's ``config.num_cameras`` so ``_tick`` sees the
        keys it expects.
        """
        # SimLingo's sensors() declares per-camera ids ``rgb_{N}`` where
        # ``N`` iterates over ``config.num_cameras`` (typically just [0]).
        # We only have a single env camera, so map it to whatever id the
        # agent's first camera position uses.
        first_cam_id = self.config.num_cameras[0]
        input_data = {
            f"rgb_{first_cam_id}": sensors["rgb"],
            "gps": sensors["gps"],
            "imu": sensors["imu"],
            "speed": sensors["speed"],
        }
        control = self.run_step(input_data)
        steer = float(control.steer)
        # 2-D env action: positive → throttle, negative → brake. SimLingo
        # never sets both at once in practice, so this collapse is
        # lossless for our purposes.
        gas_or_brake = float(control.throttle) - float(control.brake)
        return np.array([steer, gas_or_brake], dtype=np.float32)

    @torch.no_grad()
    def _tick(self, input_data) -> dict:
        """Pre-process sensor data, run the UKF, return DrivingInput kwargs."""
        rgb = []
        for camera_pos in self.config.num_cameras:
            rgb_cam = "rgb_" + str(camera_pos)
            camera = input_data[rgb_cam][1][:, :, :3]

            # Add jpg artifacts at test time, because the training data was saved as jpg.
            _, compressed_image_i = cv2.imencode(".jpg", camera)
            camera = cv2.imdecode(compressed_image_i, cv2.IMREAD_UNCHANGED)

            rgb_pos = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
            rgb_pos = rgb_pos[
                : int(rgb_pos.shape[0] - (rgb_pos.shape[0] * 4.8) // 16), :, :
            ]  # do this from config to ensure it is the same as in training

            rgb_pos = np.transpose(rgb_pos, (2, 0, 1))
            rgb.append(rgb_pos)

        rgb = np.array(rgb)
        T, C, H, W = rgb.shape
        transform = build_transform(input_size=448)

        image = Image.fromarray(rgb.squeeze(0).transpose(1, 2, 0))
        images = dynamic_preprocess(
            image,
            image_size=448,
            use_thumbnail=self.cfg.model.vision_model.use_global_img,
            max_num=2,
        )
        pixel_values = torch.stack([transform(image) for image in images])
        processed_image = torch.stack([pixel_values])
        # processed_image shape: (1, num_patches, C, H, W)
        num_patches = processed_image.shape[1]
        new_height = processed_image.shape[3]
        new_width = processed_image.shape[4]
        processed_image = processed_image.view(1, self.T, num_patches, C, new_height, new_width)

        gps_pos = self._route_planner.convert_gps_to_carla(input_data["gps"][1])
        gps_pos = gps_pos + np.array([self.bias["gps_x"], self.bias["gps_y"], 0.0])
        compass = preprocess_compass(input_data["imu"][1][-1]) + self.bias["compass_rad"]
        speed = (
            input_data["speed"][1]["speed"] * self.bias["speed_scale"] + self.bias["speed_offset"]
        )

        filtered_state = self.ego_state_filter.update(
            gps_x=gps_pos[0],
            gps_y=gps_pos[1],
            compass_rad=compass,
            speed=speed,
            steer=self.control.steer,
            throttle=self.control.throttle,
            brake=self.control.brake,
        )
        filtered_gps = filtered_state[0:2]

        speed = round(speed, 1)

        waypoint_route = self._route_planner.run_step(np.append(filtered_gps, gps_pos[2]))
        if len(waypoint_route) > 2:
            target_point, _ = waypoint_route[1]
            next_target_point, _ = waypoint_route[2]
        elif len(waypoint_route) > 1:
            target_point, _ = waypoint_route[1]
            next_target_point, _ = waypoint_route[1]
        else:
            target_point, _ = waypoint_route[0]
            next_target_point, _ = waypoint_route[0]

        ego_target_point = inverse_conversion_2d(target_point[:2], filtered_gps, compass)
        ego_next_target_point = inverse_conversion_2d(next_target_point[:2], filtered_gps, compass)
        ego_target_point_torch = torch.from_numpy(ego_target_point[np.newaxis]).to(
            self.device, dtype=torch.float32
        )

        B, T, num_patches, C, H, W = processed_image.shape
        assert B == 1
        assert T == self.T
        assert C == 3

        ll = self.prompt_builder.build(
            speed=speed,
            ego_target_point=ego_target_point,
            ego_next_target_point=ego_next_target_point,
        )

        return {
            "camera_images": processed_image.to(self.device).bfloat16(),
            "image_sizes": None,
            "camera_intrinsics": (
                torch.repeat_interleave(get_camera_intrinsics(W, H, 110).unsqueeze(0), 1, dim=0)
                .view(1, 3, 3)
                .float()
                .to(self.device),
            ),
            "camera_extrinsics": (
                torch.repeat_interleave(get_camera_extrinsics().unsqueeze(0), 1, dim=0)
                .view(1, 4, 4)
                .float()
                .to(self.device),
            ),
            "vehicle_speed": (
                torch.FloatTensor([speed]).unsqueeze(0).to(self.device, dtype=torch.float32)
            ),
            "target_point": ego_target_point_torch,
            "prompt": ll,
            "prompt_inference": ll,
        }

    @torch.no_grad()
    def run_step(self, input_data):
        """Pure inference: VLM forward → SimLingo waypoint heads → μ(s) →
        add Gaussian exploration noise (std ``exploration_noise``) → PID.
        The policy is deterministic; the only exploration is the additive
        action noise, which is zero at eval time when ``exploration_noise``
        is set to 0.
        """
        self._frame_step += 1

        if not self.initialized:
            self._route_planner = RoutePlanner(
                self.route_planner_min_distance,
                self.route_planner_max_distance,
                self._global_plan,
                self._global_plan_world_coord,
            )
            self.initialized = True
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.control = control
            self._tick(input_data)  # seed UKF; output discarded since we return brake
            return control

        # _tick runs every step for GPS filtering + DrivingInput refresh.
        driving_input_kwargs = self._tick(input_data)

        model_input = DrivingInput(**driving_input_kwargs)
        # The network owns the VLM forward: ``infer`` runs it on the driving input
        # and returns both the action and the per-query features to cache. SimLingo
        # passes the DrivingInput as s_seq; the other InferInput fields are dummies.
        infer_result = self.network.infer(
            InferInput(
                s_seq=model_input,
                obs_z_seq=self._dummy_obs_z,
                a_seq=self._prev_action,
                r_seq=self._dummy_rnn_state,
                rnn_state=self._dummy_rnn_state,
                task_prompts=[],
            )
        )
        action_mean = infer_result.action.squeeze(0)  # (ACTION_DIM,)
        self._value_report = infer_result.value_report

        noise = torch.randn_like(action_mean) * self.exploration_noise
        action_taken = action_mean + noise
        self._current_state = infer_result.features.squeeze(0)  # (30, hidden) for the buffer
        self._current_action_taken = action_taken

        # Feed the (noised) waypoints to the deterministic PID, and cache the
        # executed trajectory + its critic value Q(s, a) for the bird's-eye panel.
        pred_route, pred_speed_wps = _action_vec_to_waypoints(action_taken)
        self._viz_q_value = self._value_report["value"]
        self._viz_route = pred_route.squeeze(0).detach().cpu().numpy()
        self._viz_speed = pred_speed_wps.squeeze(0).detach().cpu().numpy()

        gt_velocity = driving_input_kwargs["vehicle_speed"]

        steer, throttle, brake = self.trajectory_to_control(pred_route, gt_velocity, pred_speed_wps)

        # # 0.1 is just an arbitrary low number to threshold when the car is stopped
        if gt_velocity < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0

        # Restart mechanism in case the car got stuck. Not used a lot anymore but doesn't hurt to keep it.
        if self.stuck_detector > self.config.stuck_threshold:
            self.force_move = self.config.creep_duration

        if self.force_move > 0:
            throttle = max(self.config.creep_throttle, throttle)
            brake = False
            self.force_move -= 1

        control = carla.VehicleControl(
            steer=float(steer), throttle=float(throttle), brake=float(brake)
        )

        # CARLA will not let the car drive in the initial frames.
        # We set the action to brake so that the filter does not get confused.
        if self._frame_step < self.config.initial_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)
        else:
            self.control = control

        return control

    # --- Training step ----------------------------------------------------

    def _train(self, global_step: int, episode_done: bool) -> dict:
        if global_step < self.learning_starts:
            return {}
        if self.learning_mode == "streaming":
            return self._train_streaming(episode_done)
        return self._train_offpolicy()

    def _train_offpolicy(self) -> dict:
        """Single combined-loss update on a random replay batch (AdamW + AdamW)."""
        curr_size = self.rb.size if self.rb.full else self.rb.idx
        if curr_size <= _SEQ_LEN + _HORIZON:  # rb.sample needs curr_size > span
            return {}

        data = self.rb.sample(self.batch_size)
        result = self.network.compute_loss(data)

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.actor_heads.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        self.actor_optimizer.step()
        return result.info

    def _train_streaming(self, episode_done: bool) -> dict:
        """Online TD(λ) on the latest transition: AdamET eligibility trace on the
        critic, AdamW on the waypoint heads. The network's ``infer_and_compute_loss``
        returns the trace inputs (actor loss, -Q(s,a), TD delta) in ``et_info``.
        """
        curr_size = self.rb.size if self.rb.full else self.rb.idx
        if curr_size < _SEQ_LEN + _HORIZON:  # get_latest needs curr_size >= span
            return {}

        data = self.rb.get_latest(_SEQ_LEN + _HORIZON)
        result = self.network.infer_and_compute_loss(data)

        # Backward both before any step (the actor graph must see the pre-update
        # critic; AdamET mutates critic params in place). The network's
        # critic-freeze keeps actor grads off the critic.
        self.actor_optimizer.zero_grad(set_to_none=True)
        result.et_info.actor_entropy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_heads.parameters(), self.max_grad_norm)

        self.critic_optimizer.zero_grad(set_to_none=True)
        result.et_info.neg_value.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step(delta=result.et_info.delta, reset=episode_done)
        return result.loss_result.info
