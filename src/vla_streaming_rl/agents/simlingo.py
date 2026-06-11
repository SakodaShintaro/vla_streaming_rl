# SPDX-License-Identifier: MIT
"""DDPG-style RL agent built on top of SimLingo.

The action is SimLingo's full waypoint output: ``pred_route`` (20×2)
concatenated with ``pred_speed_wps`` (10×2), flattened to a 60-D
continuous vector. The deterministic PID controller that turns
waypoints into ``carla.VehicleControl`` is treated as part of the
environment — the policy emits waypoints, PID emits 2-D control, env
sees ``[steer, throttle - brake]``.

  μ(s) = waypoint_heads(features(s))   ← SimLingo's own route / speed
                                         MLP heads, trained directly
  Q(s, a) = ActionValueHead(pool(features(s)), a)

Rather than learning a residual on top of frozen waypoints, this is a
DDPG-like setup: the SimLingo waypoint heads (``DrivingAdaptor.route_head``
and ``speed_wps_head``) *are* the deterministic policy μ and are updated
to ascend the critic's Q-gradient. The rest of the VLM stays frozen.

Critic update (off-policy, batch from replay):
  a' = μ(s')
  y = r + γ (1 − done) Q(s', a')
  L_critic = MSE(Q(s, a), y)

Actor update (off-policy, batch from replay):
  L_actor = − Q(s, μ(s))      (deterministic policy gradient)
  Gradients flow through the waypoint heads only; the pooled state fed
  to Q is detached so the actor loss cannot move the frozen backbone.

The VLM backbone (everything except the two waypoint heads) is frozen.
``run_step`` caches the per-query VLM features and the action it took
for each tick directly into the replay buffer, so training reads
``(features, action, reward, done)`` straight from the buffer and only
re-applies the (cheap) waypoint heads — no VLM re-forward. Only the
waypoint heads and Q learn.

Three learning modes (``learning_mode``):
  - ``off_policy``: random replay batches (the critic/actor updates above).
  - ``streaming``: online TD(λ) on the latest transition every step, with the
    critic updated by eligibility traces (``AdamET``) and the actor by the
    same deterministic policy gradient + AdamW — mirroring ``StreamingAgent``.
  - ``awr``: same random-replay TD critic, but the actor is exp-weighted
    supervised regression (advantage-weighted regression). For each state we
    sample N actions around μ(s), score them with Q(s, a_i), normalize the
    scores per state and turn them into softmax weights
    w_i = softmax((Q_i − mean)/std / temperature), then regress μ(s) toward
    the candidates weighted by w_i. The critic is not differentiated through —
    only the supervised regression trains the waypoint heads.

Because the env owns the sensor lifecycle, SimLingoAgent does **not**
spawn its own multi-camera stack or wire a leaderboard
``SensorInterface``. New-episode handover (set_global_plan, hero_actor,
re-init) is detected automatically via ``env.unwrapped.vehicle.id``
changing between ticks.
"""

import json
from pathlib import Path

import carla
import cv2
import gymnasium as gym
import numpy as np
import torch
from leaderboard.utils.route_manipulation import downsample_route  # type: ignore
from PIL import Image
from torch import nn, optim

from vla_streaming_rl.agents.step_result import StepResult
from vla_streaming_rl.networks.simlingo_network import (
    _ACTION_DIM,
    _NUM_WP_QUERIES,
    _ROUTE_LEN,
    _SPEED_WPS_LEN,
    _WP_DIM,
)
from vla_streaming_rl.optimizers.adam_et import AdamET
from vla_streaming_rl.replay_buffer import ReplayBuffer
from vla_streaming_rl.reward_processor import RewardProcessor
from vla_streaming_rl.simlingo.simlingo_training.utils.custom_types import DrivingInput
from vla_streaming_rl.simlingo.simlingo_training.utils.internvl2_utils import (
    build_transform,
    dynamic_preprocess,
)
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
from vla_streaming_rl.utils import create_reward_image

# Configure pytorch for maximum performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

# Waypoint shape constants (_ROUTE_LEN, _SPEED_WPS_LEN, _WP_DIM, _NUM_WP_QUERIES,
# _ACTION_DIM) are imported from ``networks.simlingo_network`` — they describe
# the SimLingo waypoint output that the network produces.

# Single-frame VLM input (seq_len=1) with a one-step bootstrap horizon, so the
# replay buffer needs ``_SEQ_LEN + _HORIZON`` contiguous indices per sample to
# form an off-policy transition (s_t, s_{t+1}) -- same convention as
# ``off_policy.py`` (``seq_len=self.seq_len + self.horizon``).
_SEQ_LEN = 1
_HORIZON = 1


def _waypoints_to_action_vec(route_wps: torch.Tensor, speed_wps: torch.Tensor) -> torch.Tensor:
    """Flatten one sample's (1, R, 2) + (1, S, 2) waypoints to a 1-D
    action vector of length (R+S)*2 in float32.
    """
    return torch.cat([route_wps.reshape(-1), speed_wps.reshape(-1)], dim=0).to(torch.float32)


def _action_vec_to_waypoints(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    route_flat = a[: _ROUTE_LEN * _WP_DIM]
    speed_flat = a[_ROUTE_LEN * _WP_DIM :]
    return (
        route_flat.view(1, _ROUTE_LEN, _WP_DIM),
        speed_flat.view(1, _SPEED_WPS_LEN, _WP_DIM),
    )


class SimLingoAgent:
    """DDPG-style off-policy agent built on SimLingo's waypoint output.

    ``DrivingModel.forward`` returns ``(speed_wps, route, language,
    driving_features)``. The fourth element is the (1, 30, hidden) slice
    of per-query VLM features; the waypoint heads map it to the action
    and its mean over the 30 queries is the critic's state vector.

    ``run_step`` is pure inference: VLM forward → waypoint heads → action
    (+ Gaussian exploration noise) → PID. Training is off-policy:
    ``step`` writes (features, action, reward, done) into the replay
    buffer — the per-query VLM features cached by ``run_step`` go into the
    obs slot — so ``_maybe_train`` re-applies the waypoint heads to those
    cached features (no VLM re-forward) and trains them to maximize Q.
    """

    def __init__(
        self,
        *,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        env: gym.Env,
        network: nn.Module,
        scratch_dir: Path,
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
        awr_num_samples: int,
        awr_temperature: float,
        awr_sample_noise: float,
    ) -> None:
        self.observation_space = observation_space
        del action_space
        self._env_unwrapped = env.unwrapped

        scratch_dir = Path(scratch_dir)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        self.save_path_metric = str(scratch_dir) + "/metric"
        Path(self.save_path_metric).mkdir(parents=True, exist_ok=True)

        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.exploration_noise = exploration_noise
        self.max_grad_norm = max_grad_norm

        # Advantage-weighted-regression (``learning_mode == "awr"``) actor knobs:
        # how many actions to sample per state, the softmax temperature on the
        # per-state-normalized Q advantages, and the Gaussian std of the
        # candidate actions around μ(s). Unused in off_policy / streaming.
        self.awr_num_samples = int(awr_num_samples)
        self.awr_temperature = float(awr_temperature)
        self.awr_sample_noise = float(awr_sample_noise)

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
        self._driving_adaptor = network.driving_adaptor
        self.actor_heads = network.actor_heads
        self.critic = network.critic
        self.num_bins = network.num_bins

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

        # ``off_policy`` trains on random replay batches; ``streaming`` trains
        # online on the latest transition every step with TD(λ) eligibility
        # traces on the critic (AdamET), mirroring ``StreamingAgent``. The
        # actor (deterministic policy gradient on the waypoint heads) uses
        # AdamW in both modes.
        self.learning_mode = learning_mode

        # The actor optimizer owns the SimLingo waypoint heads alone; the
        # rest of the VLM stays frozen.
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
        # ``_maybe_train`` reads (features, action, reward, done) straight
        # from the buffer and only re-applies the waypoint heads. The obs_z
        # / rnn_state / log_prob / value / token slots stay unused
        # (shape-(1,) / 0 / empty) so the buffer machinery still type-checks.
        self.rb = ReplayBuffer(
            size=buffer_size,
            seq_len=_SEQ_LEN + _HORIZON,
            obs_shape=(_NUM_WP_QUERIES, feature_dim),
            obs_z_shape=(1,),
            rnn_state_shape=(1,),
            action_shape=(_ACTION_DIM,),
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
            _NUM_WP_QUERIES, feature_dim, device=self.device
        )
        self._current_action_taken: torch.Tensor = torch.zeros(_ACTION_DIM, device=self.device)

        self._attached_ego_id: int | None = None
        self._prev_action = torch.zeros(_ACTION_DIM, device=self.device)

        # Latest executed trajectory + its critic value, cached by ``run_step``
        # for the bird's-eye visualization panel built in ``_build_info``.
        self._viz_route = np.zeros((_ROUTE_LEN, _WP_DIM), dtype=np.float32)
        self._viz_speed = np.zeros((_SPEED_WPS_LEN, _WP_DIM), dtype=np.float32)
        self._viz_q_value = 0.0

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
    ) -> StepResult:
        self._maybe_handover_episode()
        env_action = self._act()
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
    ) -> StepResult:
        self._maybe_handover_episode()
        episode_done = terminated or truncated
        env_action = self._act()
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
        # Force re-init on the next select_action.
        self._attached_ego_id = None
        return {}

    def _build_info(self, env_action: np.ndarray, reward: float) -> tuple[dict, dict]:
        """Return ``(metrics, panels)`` for this tick.

        The waypoint policy predicts neither a goal nor a next frame. It does
        not predict the reward either, so the reward panel shows the actual
        reward only (``pred=None``). ``action_norm`` is a scalar telemetry hook.
        """
        metrics = {
            "action_norm": float(np.linalg.norm(env_action)),
            "q_value": self._viz_q_value,
        }
        panels = {
            "reward": create_reward_image(None, reward),
            "bev_value": self._render_bev_panel(),
        }
        return metrics, panels

    def _render_bev_panel(self) -> np.ndarray:
        """Top-down (ego-frame) view of the executed trajectory, annotated
        with the critic's value estimate Q(s, a).

        Ego is the green marker near the bottom; route index 0 (``+x``) is
        forward → up, ``+y`` → right (the PID's heading convention). Returns
        an RGB uint8 image for the render strip.
        """
        size = 256
        scale = 6.0  # pixels per meter
        img = np.full((size, size, 3), 30, dtype=np.uint8)
        ego_px = (size // 2, int(size * 0.8))

        def to_px(forward: float, lateral: float) -> tuple[int, int]:
            return (int(ego_px[0] + lateral * scale), int(ego_px[1] - forward * scale))

        # range rings every 10 m for scale reference
        for r_m in (10, 20, 30):
            cv2.circle(img, ego_px, int(r_m * scale), (60, 60, 60), 1)

        # executed route: polyline + waypoint dots
        pts = [to_px(float(wp[0]), float(wp[1])) for wp in self._viz_route]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (0, 200, 255), 2)
        for p in pts:
            cv2.circle(img, p, 2, (255, 255, 0), -1)

        # ego marker pointing forward (up)
        cv2.drawMarker(
            img, ego_px, (0, 255, 0), cv2.MARKER_TRIANGLE_UP, markerSize=12, thickness=2
        )

        # value annotation (green if non-negative, red otherwise)
        q = self._viz_q_value
        q_color = (0, 255, 0) if q >= 0.0 else (255, 80, 80)
        cv2.putText(img, f"Q: {q:.3f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, q_color, 2)
        return img

    # --- Episode handover --------------------------------------------------

    def _maybe_handover_episode(self) -> None:
        """When the env reset to a new scenario, hand the agent the new
        ego + route plan and force ``_init`` to re-run on the next tick.
        """
        ego = self._env_unwrapped.vehicle
        if ego is None:
            raise RuntimeError("SimLingoAgent: env has no live ego — was env.reset() called?")
        if ego.id == self._attached_ego_id:
            return

        runtime = self._env_unwrapped.runtime
        if runtime is None or runtime.route_scenario is None:
            raise RuntimeError("SimLingoAgent requires Bench2DriveRuntime with an active scenario")
        # ``set_global_plan`` is the standard leaderboard handover —
        # RouteScenario builds both gps_route and world-coord route.
        self._set_global_plan(runtime.route_scenario.gps_route, runtime.route_scenario.route)
        self.hero_actor = ego
        self.initialized = False
        self._attached_ego_id = ego.id

    def _set_global_plan(self, global_plan_gps, global_plan_world_coord) -> None:
        """Downsample the route (matches leaderboard ``AutonomousAgent.set_global_plan``)
        and populate the plan attributes so ``_init`` constructs the
        ``RoutePlanner`` on the first tick.
        """
        ds_ids = downsample_route(global_plan_world_coord, 50)
        self._global_plan_world_coord = [
            (global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids
        ]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]

    # --- Setup helpers -----------------------------------------------------

    def _init(self) -> None:
        """First-tick lazy init: build the RoutePlanner once the global
        plan has been set, and clear the per-episode metric log.
        """
        self._route_planner = RoutePlanner(
            self.route_planner_min_distance,
            self.route_planner_max_distance,
            self._global_plan,
            self._global_plan_world_coord,
        )
        self.initialized = True
        self.metric_info = {}

    def get_metric_info(self):
        """Per-frame ego pose / velocity snapshot. Inlined from leaderboard
        ``AutonomousAgent.get_metric_info``."""

        def v(vec, rot=False):
            return [vec.roll, vec.pitch, vec.yaw] if rot else [vec.x, vec.y, vec.z]

        hero = self.hero_actor
        return {
            "acceleration": v(hero.get_acceleration()),
            "angular_velocity": v(hero.get_angular_velocity()),
            "forward_vector": v(hero.get_transform().get_forward_vector()),
            "right_vector": v(hero.get_transform().get_right_vector()),
            "location": v(hero.get_transform().location),
            "rotation": v(hero.get_transform().rotation, rot=True),
        }

    # --- Per-tick inference ------------------------------------------------

    def _act(self) -> np.ndarray:
        """One inference tick + 2-D env-action conversion.

        Reads :meth:`CARLALeaderboardEnv._build_sensors_dict` (already
        in the leaderboard ``input_data`` shape ``{id: (frame, payload)}``)
        and remaps ``rgb`` → ``rgb_<N>`` per SimLingo's
        ``config.num_cameras`` so ``_tick`` sees the keys it expects.
        """
        sensors = self._env_unwrapped._build_sensors_dict()
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
            self._init()
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.control = control
            self._tick(input_data)  # seed UKF; output discarded since we return brake
            return control

        # _tick runs every step for GPS filtering + DrivingInput refresh.
        driving_input_kwargs = self._tick(input_data)

        model_input = DrivingInput(**driving_input_kwargs)
        pred_speed_wps, pred_route, _, driving_features = self.network(model_input)
        pred_speed_wps = pred_speed_wps.float() if pred_speed_wps is not None else None
        pred_route = pred_route.float() if pred_route is not None else None

        # ``pred_route`` / ``pred_speed_wps`` are exactly the waypoint heads
        # applied to ``driving_features`` — i.e. the deterministic μ(s).
        features = driving_features.squeeze(0).to(torch.float32)  # (30, hidden)
        action_mean = _waypoints_to_action_vec(pred_route, pred_speed_wps)
        noise = torch.randn_like(action_mean) * self.exploration_noise
        action_taken = action_mean + noise
        # Cache features + action for ``step`` to write into the buffer at
        # this timestep's index (no VLM re-forward needed at train time —
        # only the waypoint heads are re-applied to ``features``).
        self._current_state = features
        self._current_action_taken = action_taken

        # Feed the (noised) waypoints to the deterministic PID.
        pred_route, pred_speed_wps = _action_vec_to_waypoints(action_taken)

        # Cache the executed trajectory + its critic value Q(s, a) for the
        # bird's-eye visualization panel. ``s`` is the mean over the 30
        # waypoint queries, matching the critic's state convention in training.
        s_vec = features.mean(dim=0, keepdim=True)
        q = self._critic_value(self.critic(s_vec, action_taken.unsqueeze(0).unsqueeze(1))["output"])
        self._viz_q_value = float(q.item())
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

        metric_info = self.get_metric_info()
        self.metric_info[self._frame_step] = metric_info
        if self.save_path_metric is not None and self._frame_step % 1 == 0:
            # metric info
            outfile = open(f"{self.save_path_metric}/metric_info.json", "w")
            json.dump(self.metric_info, outfile, indent=4)
            outfile.close()

        return control

    # --- Off-policy training step -----------------------------------------

    def _policy_action(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the SimLingo waypoint heads to per-query features and
        return the flattened action μ(s).

        Args:
            features: ``(B, 30, hidden)`` per-query VLM features.

        Returns:
            ``(B, 60)`` action: route waypoints (20×2) then speed
            waypoints (10×2), matching ``_waypoints_to_action_vec``.
        """
        preds = self._driving_adaptor.get_predictions(features)
        b = features.shape[0]
        route = preds["route"].reshape(b, -1)  # (B, 40)
        speed = preds["speed_wps"].reshape(b, -1)  # (B, 20)
        return torch.cat([route, speed], dim=1)

    def _train(self, global_step: int, episode_done: bool) -> dict:
        if self.learning_mode == "streaming":
            return self._train_streaming(global_step, episode_done)
        if self.learning_mode == "awr":
            return self._train_awr(global_step)
        return self._maybe_train(global_step)

    def _critic_value(self, logits: torch.Tensor) -> torch.Tensor:
        """Map the critic's raw ``"output"`` to a scalar Q (B,).

        For the distributional (``num_bins > 1``) SimbaV2 critic the output is
        ``num_bins`` categorical logits; HL-Gauss returns their expected return.
        For a scalar critic it is just the value itself.
        """
        return self.critic.to_value(logits).view(-1)

    def _read_transition(self, seq: torch.Tensor) -> tuple:
        """Read one transition batch for index pairs ``seq`` (B, 2).

        ``seq[:, 0] = t``, ``seq[:, 1] = t+1``; the replay convention puts a_t
        at ``actions[t+1]``, r_t at ``rewards[t+1]``, done_t at ``dones[t+1]``.
        The per-query features were cached into the obs slot at write time; the
        critic state is their mean over the 30 waypoint queries. Returns
        ``(a, r, done, feat, feat_next, s, s_next)``.
        """
        a = self.rb.actions[seq[:, 1]].to(self.device)
        r = self.reward_processor.normalize(self.rb.rewards[seq[:, 1], 0].to(self.device))
        done = self.rb.dones[seq[:, 1], 0].to(self.device)
        feat = self.rb.observations[seq[:, 0]].to(self.device)  # (B, 30, hidden)
        feat_next = self.rb.observations[seq[:, 1]].to(self.device)
        s = feat.mean(dim=1)
        s_next = feat_next.mean(dim=1)
        return a, r, done, feat, feat_next, s, s_next

    def _critic_targets(
        self,
        a: torch.Tensor,
        r: torch.Tensor,
        done: torch.Tensor,
        s: torch.Tensor,
        s_next: torch.Tensor,
        feat_next: torch.Tensor,
    ) -> tuple:
        """TD target + current critic output for one transition batch.

        DDPG target: a' = μ(s'), y = r + γ(1-done) Q(s', a'). With the
        distributional critic the next-state value is the expectation of its
        categorical output (via HL-Gauss). Returns
        ``(current_logits, current_q, target_q)``.
        """
        with torch.no_grad():
            a_next = self._policy_action(feat_next)
            next_q = self._critic_value(self.critic(s_next, a_next.unsqueeze(1))["output"])
            target_q = r + self.gamma * (1.0 - done) * next_q
        current_logits = self.critic(s, a.unsqueeze(1))["output"]
        current_q = self._critic_value(current_logits)
        return current_logits, current_q, target_q

    def _ddpg_actor_loss(self, feat: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """L_actor = − Q(s, μ(s)) (deterministic policy gradient).

        Freeze the critic's params during this Q forward so the −Q gradient
        flows only into the waypoint heads, not the critic. ``s`` is a buffer
        read with no grad, so the heads are the sole path to the loss.
        """
        a_pred = self._policy_action(feat)
        for p in self.critic.parameters():
            p.requires_grad_(False)
        actor_q = self._critic_value(self.critic(s, a_pred.unsqueeze(1))["output"])
        for p in self.critic.parameters():
            p.requires_grad_(True)
        return -actor_q.mean()

    def _awr_actor_loss(self, feat: torch.Tensor, s: torch.Tensor) -> tuple:
        """Exp-weighted (advantage-weighted) supervised regression actor.

        Instead of ascending the critic's Q-gradient (``_ddpg_actor_loss``),
        sample ``awr_num_samples`` candidate actions around the current
        deterministic policy μ(s), score each with the critic Q(s, a_i),
        normalize the scores *per state* (zero-mean / unit-std advantages) and
        turn them into softmax weights ``w_i = softmax(adv_i / temperature)``,
        then regress μ(s) toward the candidates weighted by ``w_i``. This is
        reward/advantage-weighted regression (AWR): a supervised loss that pulls
        the policy toward the higher-Q samples *without* backpropagating through
        the critic — the Q forward (and hence the weights and the candidate
        actions) is fully detached, so the only gradient path is μ(s).

        Returns ``(actor_loss, weight_entropy)`` where ``weight_entropy`` is a
        telemetry scalar (mean per-state Shannon entropy of the softmax weights,
        nats) — low entropy means the weights collapsed onto a single candidate.
        """
        mu = self._policy_action(feat)  # (B, A), grad flows into the heads
        b, action_dim = mu.shape
        n = self.awr_num_samples

        # Candidates + their Q scores live entirely under no_grad: they are
        # regression *targets*, must not carry gradient into the heads/critic.
        with torch.no_grad():
            noise = torch.randn(b, n, action_dim, device=self.device) * self.awr_sample_noise
            # Keep μ(s) itself as the first candidate so the current policy is
            # always represented in the mix (zero noise on sample 0).
            noise[:, 0, :] = 0.0
            cand = mu.unsqueeze(1) + noise  # (B, N, A)

            # Q(s, a_i): broadcast the state over the N samples and score the
            # whole (B*N) batch in one critic forward.
            s_rep = s.unsqueeze(1).expand(b, n, s.shape[-1]).reshape(b * n, -1)
            a_rep = cand.reshape(b * n, action_dim).unsqueeze(1)  # (B*N, 1, A)
            q = self._critic_value(self.critic(s_rep, a_rep)["output"]).view(b, n)

            # Per-state exp weights via softmax (handles the exp and the
            # normalize-to-sum-1). softmax is shift-invariant, so centering the
            # Q's would be a no-op — we only rescale by the per-state std, which
            # keeps the temperature scale-invariant to the critic's magnitude.
            adv = q / (q.std(dim=1, keepdim=True) + 1e-8)
            weights = torch.softmax(adv / self.awr_temperature, dim=1)  # (B, N)
            weight_entropy = -(weights * (weights + 1e-8).log()).sum(dim=1).mean()

        # Weighted regression of μ(s) toward the candidates.
        sq = ((mu.unsqueeze(1) - cand) ** 2).sum(dim=-1)  # (B, N)
        actor_loss = (weights * sq).sum(dim=1).mean()
        return actor_loss, weight_entropy

    def _read_and_compute(self, seq: torch.Tensor) -> tuple:
        """Read one transition batch and build the DDPG losses' tensors.

        Returns ``(current_logits, current_q, target_q, actor_loss)`` with the
        deterministic-policy-gradient actor loss — used by ``off_policy`` and
        ``streaming``. ``current_logits`` is the raw critic output kept for the
        (distributional) critic loss; ``current_q`` / ``target_q`` are scalars.
        """
        a, r, done, feat, feat_next, s, s_next = self._read_transition(seq)
        current_logits, current_q, target_q = self._critic_targets(a, r, done, s, s_next, feat_next)
        actor_loss = self._ddpg_actor_loss(feat, s)
        return current_logits, current_q, target_q, actor_loss

    # --- Off-policy training step -----------------------------------------

    def _maybe_train(self, global_step: int) -> dict:
        # The buffer's ``sample`` requires ``curr_size > seq_len``; the
        # learning_starts warmup is set high enough that this is
        # implicitly satisfied.
        if global_step < self.learning_starts:
            return {}

        curr_size = self.rb.size if self.rb.full else self.rb.idx
        span = _SEQ_LEN + _HORIZON
        start = torch.randint(0, curr_size - span, (self.batch_size,))
        seq = start[:, None] + torch.arange(span)[None, :]
        current_logits, current_q, target_q, actor_loss = self._read_and_compute(seq)
        # Distributional critic: cross-entropy of the categorical output against
        # the (support-projected) scalar TD target via HL-Gauss; grow the support
        # first if the target exceeds the current range. Scalar critic: MSE.
        self.critic.update_value_range(target_q)
        critic_loss = self.critic.value_loss(current_logits, target_q)

        # One backward over (critic_loss + actor_loss) so no ``step`` lands
        # between the two graphs (which would invalidate them in-place). The
        # critic-freeze in ``_read_and_compute`` keeps the two losses on
        # disjoint param sets, so each optimizer only steps the params it owns.
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        (critic_loss + actor_loss).backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.actor_heads.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        self.actor_optimizer.step()

        info = {
            "losses/critic_loss": float(critic_loss.item()),
            "losses/actor_loss": float(actor_loss.item()),
            "losses/q_value": float(current_q.mean().item()),
            "losses/target_q": float(target_q.mean().item()),
        }
        if self.num_bins > 1:
            info["losses/value_range"] = float(self.critic.value_range)
        return info

    # --- Advantage-weighted regression (exp-weighted SL) training step -----

    def _train_awr(self, global_step: int) -> dict:
        """Off-policy critic (TD) + exp-weighted supervised-regression actor.

        Same random-replay critic update as ``_maybe_train`` (the critic still
        learns Q via the TD loss), but the waypoint heads are trained by
        :meth:`_awr_actor_loss` — sample multiple actions, score them with Q,
        normalize and exp-weight the scores, and regress the policy toward the
        higher-scoring samples (advantage-weighted regression) instead of the
        deterministic ``-Q`` policy gradient.
        """
        if global_step < self.learning_starts:
            return {}

        curr_size = self.rb.size if self.rb.full else self.rb.idx
        span = _SEQ_LEN + _HORIZON
        start = torch.randint(0, curr_size - span, (self.batch_size,))
        seq = start[:, None] + torch.arange(span)[None, :]

        a, r, done, feat, feat_next, s, s_next = self._read_transition(seq)
        current_logits, current_q, target_q = self._critic_targets(
            a, r, done, s, s_next, feat_next
        )
        self.critic.update_value_range(target_q)
        critic_loss = self.critic.value_loss(current_logits, target_q)
        actor_loss, weight_entropy = self._awr_actor_loss(feat, s)

        # One backward over (critic_loss + actor_loss): the AWR actor loss scores
        # candidates under ``no_grad``, so it touches the heads only and the
        # critic loss touches the critic only — each optimizer steps its own
        # params. (Same single-backward discipline as ``_maybe_train``.)
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        (critic_loss + actor_loss).backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.actor_heads.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        self.actor_optimizer.step()

        info = {
            "losses/critic_loss": float(critic_loss.item()),
            "losses/actor_loss": float(actor_loss.item()),
            "losses/q_value": float(current_q.mean().item()),
            "losses/target_q": float(target_q.mean().item()),
            "losses/awr_weight_entropy": float(weight_entropy.item()),
        }
        if self.num_bins > 1:
            info["losses/value_range"] = float(self.critic.value_range)
        return info

    # --- Streaming (online TD(λ)) training step ----------------------------

    def _train_streaming(self, global_step: int, episode_done: bool) -> dict:
        curr_size = self.rb.size if self.rb.full else self.rb.idx
        span = _SEQ_LEN + _HORIZON
        if curr_size < span:
            return {}

        # Latest contiguous transition: t = idx-2, t+1 = idx-1 (the two most
        # recent writes; modulo handles the ring wrap). Batch size is 1 —
        # online, no replay sampling.
        i_next = (self.rb.idx - 1) % self.rb.size
        i_curr = (self.rb.idx - 2) % self.rb.size
        seq = torch.tensor([[i_curr, i_next]], dtype=torch.long)
        # AdamET's eligibility trace is inherently scalar-TD, so streaming uses
        # the expected value (``current_q``) of the distributional critic rather
        # than the full categorical cross-entropy (which only off_policy uses).
        _current_logits, current_q, target_q, actor_loss = self._read_and_compute(seq)

        # TD error drives the eligibility-trace critic update; a detached
        # scalar (AdamET multiplies the per-parameter trace by it).
        delta = float((target_q - current_q).mean().item())

        # Backward BOTH losses before any optimizer step so the actor graph
        # still sees the pre-update critic weights (``AdamET.step`` mutates
        # critic params in place). The critic-freeze keeps actor grads off the
        # critic, so the eligibility trace gets only the value gradient.
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_heads.parameters(), self.max_grad_norm)

        self.critic_optimizer.zero_grad(set_to_none=True)
        (-current_q.mean()).backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step(delta=delta, reset=episode_done)

        info = {
            "losses/critic_loss": float(delta * delta),
            "losses/actor_loss": float(actor_loss.item()),
            "losses/q_value": float(current_q.mean().item()),
            "losses/target_q": float(target_q.mean().item()),
            "losses/delta": delta,
        }
        if self.num_bins > 1:
            info["losses/value_range"] = float(self.critic.value_range)
        return info
