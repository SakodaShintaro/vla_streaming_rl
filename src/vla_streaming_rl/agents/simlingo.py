# SPDX-License-Identifier: MIT
from typing import Any

import carla
import cv2
import gymnasium as gym
import numpy as np
import torch
from leaderboard.utils.route_manipulation import downsample_route  # type: ignore
from PIL import Image
from torch import nn, optim

from vla_streaming_rl.agents.base import Agent, StepResult
from vla_streaming_rl.networks.interface import InferInput
from vla_streaming_rl.networks.simlingo_network import (
    ACTION_DIM,
    NUM_WP_QUERIES,
    ROUTE_WPS_LEN,
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
# form a transition (s_t, s_{t+1}) -- same convention as the standard agent
# (``seq_len = self.seq_len + self.horizon``).
_SEQ_LEN = 1
_HORIZON = 1


def _action_vec_to_waypoints(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    route_flat = a[: ROUTE_WPS_LEN * WP_DIM]
    speed_flat = a[ROUTE_WPS_LEN * WP_DIM :]
    return (
        route_flat.view(1, ROUTE_WPS_LEN, WP_DIM),
        speed_flat.view(1, SPEED_WPS_LEN, WP_DIM),
    )


class SimLingoAgent(Agent):
    def __init__(
        self,
        *,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Box,
        network: SimLingoNetwork,
        gamma: float,
        buffer_size: int,
        batch_size: int,
        learning_starts: int,
        exploration_noise: float,
        actor_lr: float,
        critic_lr: float,
        weight_decay: float,
        max_grad_norm: float,
        learning_mode: str,
        horizon: int,
        et_lambda: float,
    ) -> None:
        super().__init__(learning_mode=learning_mode, horizon=horizon)
        self.observation_space = observation_space
        del action_space

        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.exploration_noise = exploration_noise
        self.max_grad_norm = max_grad_norm

        torch.cuda.empty_cache()
        self._frame_step = -1
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

        self.prompt_builder = PromptBuilder(
            config=self.config,
            tokenizer=network.tokenizer,
            num_image_token=network.num_image_token,
            device=self.device,
        )

        # --- RL setup ---------------------------------------------------

        feature_dim = network.feature_dim

        # Actor / critic optimizer split (the VLM backbone is frozen). The critic
        # uses AdamET (eligibility traces) in streaming mode, AdamW for off-policy;
        # the actor heads always use AdamW.
        self.actor_optimizer = optim.AdamW(
            self.actor_heads.parameters(), lr=actor_lr, weight_decay=weight_decay
        )
        self.critic_optimizer = (
            AdamET(self.critic.parameters(), lr=critic_lr, gamma=gamma, et_lambda=et_lambda)
            if learning_mode == "streaming"
            else optim.AdamW(self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay)
        )

        # ``select_action`` writes each tick's per-query VLM features and the action it took
        # straight into the buffer so ``compute_loss`` / ``infer_and_compute_loss``
        # read (features, action, reward, done) back and only re-apply the waypoint
        # heads. The obs_z / rnn_state / token slots stay unused
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
            max_prompt_tokens=0,
            pad_token_id=0,
        )
        self._dummy_rnn_state = torch.zeros(1, device=self.device)
        self._dummy_obs_z = torch.zeros(1, device=self.device)

        # ``action_{t-1}`` stored with state t (the project's convention puts the
        # action that produced state t at ``actions[t]``); the only genuine
        # cross-tick state the per-tick pipeline carries.
        self._need_handover = True
        self._prev_action = torch.zeros(ACTION_DIM, device=self.device)

        # "carla" shapes the per-step Bench2Drive score delta: exact-zero
        # rewards (stuck) get a small negative push and collision spikes
        # are hard-clipped to keep the critic stable. See RewardProcessor.
        self.reward_processor = RewardProcessor("carla", 1.0)

    # --- agent surface -----------------------------------------------------

    def _step_streaming(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult:
        del global_step
        infer_input, gt_velocity = self._preprocess(obs, info)
        with torch.no_grad():
            live = self.network.infer(infer_input)
        state = live.features.squeeze(0)
        self.rb.add(
            state,
            self._dummy_obs_z,
            reward,
            terminated or truncated,
            self._dummy_rnn_state,
            self._prev_action,
            [],
        )
        metrics = {"processed_reward": self.reward_processor.normalize(torch.tensor(reward)).item()}

        curr_size = self.rb.size if self.rb.full else self.rb.idx
        if curr_size < _SEQ_LEN + _HORIZON:
            action_mean = live.action.squeeze(0)
            value_report = live.value_report
        else:
            data = self.rb.get_latest(_SEQ_LEN + _HORIZON)
            data.rewards = self.reward_processor.normalize(data.rewards)
            result = self.network.infer_and_compute_loss(data)
            action_mean = result.infer_result.action.squeeze(0)
            value_report = result.infer_result.value_report
            metrics.update(
                {f"losses/{key}": value for key, value in result.loss_result.info.items()}
            )
            # Backward both before any step (the actor graph must see the pre-update
            # critic; AdamET mutates critic params in place). The network's
            # critic-freeze keeps actor grads off the critic.
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            result.et_info.actor_entropy_loss.backward(retain_graph=True)
            result.et_info.neg_value.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step(delta=result.et_info.delta, reset=terminated or truncated)

        action_taken = action_mean + torch.randn_like(action_mean) * self.exploration_noise
        self._prev_action = action_taken
        env_action, viz = self._to_env_action(action_taken, gt_velocity, value_report["value"])
        metrics["action_norm"] = float(np.linalg.norm(env_action))
        metrics.update(value_report)
        return StepResult(action=env_action, metrics=metrics, panels=self._panels(reward, viz))

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        del score, feedback_text
        # Force handover (RoutePlanner rebuild) on the next tick, whose observation
        # carries the new episode's reset ``route_plan``.
        self._need_handover = True
        return {}

    # --- per-tick machinery ------------------------------------------------

    @torch.no_grad()
    def select_action(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult:
        del global_step
        infer_input, gt_velocity = self._preprocess(obs, info)
        infer_result = self.network.infer(infer_input)
        state = infer_result.features.squeeze(0)
        value_report = infer_result.value_report
        self.rb.add(
            state,
            self._dummy_obs_z,
            reward,
            terminated or truncated,
            self._dummy_rnn_state,
            self._prev_action,
            [],
        )
        action_mean = infer_result.action.squeeze(0)
        action_taken = action_mean + torch.randn_like(action_mean) * self.exploration_noise
        self._prev_action = action_taken
        env_action, viz = self._to_env_action(action_taken, gt_velocity, value_report["value"])
        metrics = {
            "action_norm": float(np.linalg.norm(env_action)),
            "processed_reward": self.reward_processor.normalize(torch.tensor(reward)).item(),
            **value_report,
        }
        return StepResult(action=env_action, metrics=metrics, panels=self._panels(reward, viz))

    @torch.no_grad()
    def _preprocess(self, obs: dict[str, Any], info: dict) -> tuple[InferInput, torch.Tensor]:
        """Build the network input (``InferInput`` wrapping a ``DrivingInput``) from
        the env's carla ``obs["sensors"]`` snapshot, rebuilding the RoutePlanner on
        the first tick of a new episode. Returns the input together with this tick's
        ego velocity, which ``_to_env_action`` feeds to the PID."""
        del info

        # Episode handover: take the new episode's route plan from the reset
        # observation (the standard leaderboard handover — RouteScenario builds both
        # the gps and world-coord route, see CARLALeaderboardEnv).
        if self._need_handover:
            gps_route, world_route = obs["route_plan"]
            ds_ids = downsample_route(world_route, 50)
            global_plan_world_coord = [(world_route[x][0], world_route[x][1]) for x in ds_ids]
            global_plan = [gps_route[x] for x in ds_ids]
            self._route_planner = RoutePlanner(
                self.route_planner_min_distance,
                self.route_planner_max_distance,
                global_plan,
                global_plan_world_coord,
            )
            self.ego_state_filter = EgoStateFilter(dt=1.0 / 20.0, state_log_maxlen=5)
            self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self._need_handover = False

        # SimLingo's sensors() declares per-camera ids ``rgb_{N}``; we have a single
        # env camera, so map it to the agent's first camera position.
        first_cam_id = self.config.num_cameras[0]
        input_data = {
            f"rgb_{first_cam_id}": obs["sensors"]["rgb"],
            "gps": obs["sensors"]["gps"],
            "imu": obs["sensors"]["imu"],
            "speed": obs["sensors"]["speed"],
        }
        self._frame_step += 1

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

        driving_input_kwargs = {
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
        gt_velocity = driving_input_kwargs["vehicle_speed"]

        infer_input = InferInput(
            s_seq=DrivingInput(**driving_input_kwargs),
            obs_z_seq=self._dummy_obs_z,
            a_seq=self._prev_action,
            r_seq=self._dummy_rnn_state,
            rnn_state=self._dummy_rnn_state,
            task_prompts=[],
        )
        return infer_input, gt_velocity

    def _to_env_action(
        self, net_action: torch.Tensor, gt_velocity: torch.Tensor, q_value: float
    ) -> tuple[np.ndarray, dict]:
        """Convert the (noised) waypoint action into the 2-D env action via the
        deterministic PID, running the stuck / creep recovery logic. Returns the env
        action together with the bird's-eye ``viz`` bundle (executed trajectory + its
        critic value) consumed by ``_panels``."""
        pred_route, pred_speed = _action_vec_to_waypoints(net_action)
        viz = {
            "q": q_value,
            "route": pred_route.squeeze(0).detach().cpu().numpy(),
            "speed": pred_speed.squeeze(0).detach().cpu().numpy(),
        }

        steer, throttle, brake = self.trajectory_to_control(pred_route, gt_velocity, pred_speed)

        # 0.1 is an arbitrary low threshold for "stopped".
        if gt_velocity < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0

        # Restart mechanism in case the car got stuck. Not used a lot anymore but
        # doesn't hurt to keep it.
        if self.stuck_detector > self.config.stuck_threshold:
            self.force_move = self.config.creep_duration

        if self.force_move > 0:
            throttle = max(self.config.creep_throttle, throttle)
            brake = False
            self.force_move -= 1

        control = carla.VehicleControl(
            steer=float(steer), throttle=float(throttle), brake=float(brake)
        )

        # CARLA will not let the car drive in the initial frames. We brake so the
        # filter does not get confused.
        if self._frame_step < self.config.initial_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)
        else:
            self.control = control

        steer = float(control.steer)
        # 2-D env action: positive → throttle, negative → brake. SimLingo never sets
        # both at once in practice, so this collapse is lossless for our purposes.
        gas_or_brake = float(control.throttle) - float(control.brake)
        return np.array([steer, gas_or_brake], dtype=np.float32), viz

    def _panels(self, reward: float, viz: dict) -> dict:
        """Reward panel (actual only — the waypoint policy predicts no reward) plus
        the top-down (ego-frame) view of SimLingo's two predicted trajectories,
        annotated with the critic's value estimate Q(s, a).

        SimLingo outputs two waypoint sets: the ``route`` (geometric path, cyan) and
        the ``speed`` waypoints (orange). Ego is the green marker near the bottom;
        waypoint index 0 (``+x``) is forward → up, ``+y`` → right (the PID's heading
        convention)."""
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

        route_color = (0, 200, 255)
        speed_color = (255, 165, 0)
        draw_trajectory(viz["route"], route_color)
        draw_trajectory(viz["speed"], speed_color)

        # ego marker pointing forward (up)
        cv2.drawMarker(img, ego_px, (0, 255, 0), cv2.MARKER_TRIANGLE_UP, markerSize=12, thickness=2)

        # value annotation (green if non-negative, red otherwise) + legend
        q = viz["q"]
        q_color = (0, 255, 0) if q >= 0.0 else (255, 80, 80)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"Q: {q:.3f}", (8, 22), font, 0.6, q_color, 2)
        cv2.putText(img, "route", (8, size - 24), font, 0.45, route_color, 1)
        cv2.putText(img, "speed", (8, size - 8), font, 0.45, speed_color, 1)

        return {"reward": create_reward_image(None, reward), "bev_value": img}
