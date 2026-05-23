# SPDX-License-Identifier: MIT
"""SimLingo zero-shot agent — drops into ``scripts/train.py`` like an RL agent.

Implements the same surface (``select_action`` / ``step`` /
``on_episode_end`` + ``network`` + ``reward_processor`` attributes) the
RL agents in this directory expose, so the trainer's loop runs unchanged.
The agent brings up the SimLingo VLM, then on each step feeds it the
per-frame sensor snapshot that :class:`CARLALeaderboardEnv` already
publishes in ``info["sensors"]`` (rgb / gps / imu / speed), and converts
the resulting ``carla.VehicleControl`` to the env's 2-D
``[steer, throttle_or_brake]`` action.

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
from leaderboard.utils.route_manipulation import downsample_route
from omegaconf import OmegaConf
from PIL import Image
from transformers import Qwen2Tokenizer

from vla_streaming_rl.reward_processor import RewardProcessor
from vla_streaming_rl.simlingo.simlingo_training.models.driving import DrivingModel
from vla_streaming_rl.simlingo.simlingo_training.models.encoder.internvl2_vendored.configuration_internvl_chat import (
    InternVLChatConfig,
)
from vla_streaming_rl.simlingo.simlingo_training.utils.custom_types import DrivingInput
from vla_streaming_rl.simlingo.simlingo_training.utils.internvl2_utils import (
    SIMLINGO_ADDITIONAL_SPECIAL_TOKENS,
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

# Configure pytorch for maximum performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


class SimLingoAgent:
    """SimLingo policy with the RL-agent surface ``scripts/train.py`` expects."""

    _HF_REPO_ID = "RenzKa/simlingo"
    _HF_CKPT_NAME = "pytorch_model.pt"

    def __init__(
        self,
        *,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        env: gym.Env,
        scratch_dir: Path,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self._env_unwrapped = env.unwrapped
        self._scratch_dir = Path(scratch_dir)
        self._scratch_dir.mkdir(parents=True, exist_ok=True)

        # Per-episode handover state; ``hero_actor`` and global plans are
        # filled in by ``_set_global_plan`` before the first ``run_step``.
        self.hero_actor = None
        self._global_plan = None
        self._global_plan_world_coord = None

        torch.cuda.empty_cache()
        self.config_path = str(self._resolve_checkpoint())
        print(f"Config path: {self.config_path}")
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

        # load config from .hydra folder
        config_load_path = Path(self.config_path).parent.parent.parent / ".hydra" / "config.yaml"
        with open(config_load_path, "r") as file:
            cfg = OmegaConf.load(file)
        self.cfg = cfg
        self.cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img

        # ``AutoProcessor`` + ``trust_remote_code=True`` resolves to
        # ``Qwen2Tokenizer`` for InternVL2-1B (the custom remote processor
        # class isn't actually used downstream). Name the concrete class
        # directly so we don't depend on HF-hosted remote code.
        processor = Qwen2Tokenizer.from_pretrained(cfg.model.vision_model.variant)
        if "tokenizer" in processor.__dict__:
            self.tokenizer = processor.tokenizer
        else:
            self.tokenizer = processor
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(SIMLINGO_ADDITIONAL_SPECIAL_TOKENS)}
        )
        self.tokenizer.padding_side = "left"

        # ``AutoConfig`` + ``trust_remote_code=True`` resolves to
        # ``InternVLChatConfig``. Use the vendored class directly so the
        # HF-hosted ``configuration_internvl_chat.py`` is no longer
        # downloaded / executed at runtime.
        tmp_config = InternVLChatConfig.from_pretrained(cfg.model.vision_model.variant)
        image_size = tmp_config.force_image_size or tmp_config.vision_config.image_size
        patch_size = tmp_config.vision_config.patch_size
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (tmp_config.downsample_ratio**2)
        )
        cache_dir = f"pretrained/{(cfg.model.vision_model.variant.split('/')[1])}"
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        # Construct ``DrivingModel`` directly instead of going through
        # ``hydra.utils.instantiate(cfg.model, ...)`` — the dispatch via
        # ``_target_`` was only useful for simlingo's experiment-yaml
        # model-class swap workflow, which we never exercise. Removing
        # it also frees us from having to rewrite stale absolute module
        # paths in the saved checkpoint config.
        model_kwargs = {k: v for k, v in cfg.model.items() if k != "_target_"}
        self.model = DrivingModel(
            cfg_data_module=cfg.data_module,
            processor=processor,
            cache_dir=cache_dir,
            **model_kwargs,
        ).to(self.device)
        torch.set_default_dtype(default_dtype)
        self.model.load_state_dict(torch.load(self.config_path))

        self.T = 1
        self.stuck_detector = 0
        self.force_move = 0

        self.ego_state_filter = EgoStateFilter(dt=1.0 / 20.0, state_log_maxlen=5)
        self.prompt_builder = PromptBuilder(
            config=self.config,
            tokenizer=self.tokenizer,
            num_image_token=self.num_image_token,
            device=self.device,
        )

        self.save_path_metric = str(self._scratch_dir) + "/metric"
        Path(self.save_path_metric).mkdir(parents=True, exist_ok=True)

        # Freeze the VLM — we never backprop. Empty trainable_state at
        # checkpoint time then writes a (harmless) empty dict.
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Duck-type the RL agent surface the trainer uses:
        # ``agent.network.parameters()`` for the parameter count print,
        # ``agent.network.named_parameters()`` for the checkpoint save.
        self.network = self.model
        self.reward_processor = RewardProcessor("none", 1.0)

        # ``CARLALeaderboardEnv`` re-spawns the ego on every reset; a
        # changed actor id is our "new episode" signal.
        self._attached_ego_id: int | None = None

    # --- RL-agent protocol surface used by scripts/train.py -----------------

    def select_action(self, global_step, obs, reward, terminated, truncated, task_prompt):
        self._maybe_handover_episode()
        return self._act(), self._dummy_agent_info()

    def step(self, global_step, obs, reward, terminated, truncated, task_prompt):
        self._maybe_handover_episode()
        return self._act(), self._dummy_agent_info()

    def _dummy_agent_info(self) -> dict:
        """Trainer reads ``agent_info["goal_image"]`` (and ``next_image``/
        ``next_reward`` via ``.get(...)``) for its rendering pipeline. The
        zero-shot policy doesn't predict a goal, so hand back zeroed
        arrays shaped like the obs the env returns."""
        h, w = self.observation_space.shape[1], self.observation_space.shape[2]
        zero_hwc = np.zeros((h, w, 3), dtype=np.float32)
        return {"goal_image": zero_hwc}

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        # Force re-init on the next select_action.
        self._attached_ego_id = None
        return {}

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

    @classmethod
    def _resolve_checkpoint(cls):
        """Pull the SimLingo checkpoint from HF and return its local path.

        ``snapshot_download`` populates the HF cache; we pick the single
        ``pytorch_model.pt`` inside (excluding the blob-store hardlinks,
        which point to the same file but live under a content-addressed
        path that breaks SimLingo's
        ``Path(...).parent.parent.parent / .hydra/config.yaml`` lookup).
        """
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(cls._HF_REPO_ID))
        candidates = [p for p in snapshot.rglob(cls._HF_CKPT_NAME) if "/blobs/" not in str(p)]
        if not candidates:
            raise RuntimeError(
                f"no {cls._HF_CKPT_NAME} in HF snapshot of {cls._HF_REPO_ID} at {snapshot}"
            )
        return candidates[0]

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

    @torch.inference_mode()  # Turns off gradient computation
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
        pred_speed_wps, pred_route, _ = self.model(model_input)
        pred_speed_wps = pred_speed_wps.float() if pred_speed_wps is not None else None
        pred_route = pred_route.float() if pred_route is not None else None

        gt_velocity = driving_input_kwargs["vehicle_speed"]

        steer, throttle, brake = self.trajectory_to_control(
            pred_route, gt_velocity, pred_speed_wps
        )

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
        if self._frame_step < self.config.inital_frames_delay:
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
