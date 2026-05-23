"""
partially taken from https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/team_code/sensor_agent.py
(MIT license)
"""

import json
import math
from collections import deque
from pathlib import Path

import carla
import cv2
import numpy as np
import torch
from leaderboard.autoagents import autonomous_agent
from omegaconf import OmegaConf
from PIL import Image
from scipy.optimize import fsolve
from transformers import Qwen2Tokenizer

from vla_streaming_rl.simlingo.simlingo_training.models.driving import DrivingModel
from vla_streaming_rl.simlingo.simlingo_training.models.encoder.internvl2_vendored import (
    conversation as conv_module,
)
from vla_streaming_rl.simlingo.simlingo_training.models.encoder.internvl2_vendored.configuration_internvl_chat import (
    InternVLChatConfig,
)
from vla_streaming_rl.simlingo.simlingo_training.utils.custom_types import (
    DrivingInput,
    LanguageLabel,
)
from vla_streaming_rl.simlingo.simlingo_training.utils.internvl2_utils import (
    SIMLINGO_ADDITIONAL_SPECIAL_TOKENS,
    build_transform,
    dynamic_preprocess,
)
from vla_streaming_rl.simlingo.team_code.config_simlingo import GlobalConfig
from vla_streaming_rl.simlingo.team_code.ego_state_filter import EgoStateFilter
from vla_streaming_rl.simlingo.team_code.route_planner import RoutePlanner
from vla_streaming_rl.simlingo.team_code.simlingo_utils import (
    command_to_one_hot,
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


# Leaderboard function that selects the class used as agent.
def get_entry_point():
    return "LingoAgent"


class LingoAgent(autonomous_agent.AutonomousAgent):
    """
    Main class that runs the agents with the run_step function
    """

    # ``AutonomousAgent.__init__`` originally needed ``carla_host`` /
    # ``carla_port`` to talk to a CARLA server directly, but in this
    # project the env owns the CARLA connection and the agent never
    # uses these. ``debug`` is likewise unused — the leaderboard's
    # debug-image saving path is disabled in this vendored copy.
    _HF_REPO_ID = "RenzKa/simlingo"
    _HF_CKPT_NAME = "pytorch_model.pt"

    def __init__(self, scratch_dir):
        # ``AutonomousAgent.__init__`` internally calls
        # ``self.get_hero()``; our override below short-circuits the
        # CarlaDataProvider lookup since the env hasn't reset yet at
        # construction time. Setup (checkpoint load, model build,
        # PID + UKF wiring) runs right after, so the agent is fully
        # ready by the time __init__ returns.
        super().__init__("", 2000, False)
        self.setup(str(self._resolve_checkpoint()), scratch_dir)

    def get_hero(self):
        # Defer the ego lookup — the env-driven handover in the wrapping
        # ``SimLingoAgent`` sets ``hero_actor`` explicitly once the env
        # has reset and spawned its ego.
        self.hero_actor = None

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

    def setup(self, checkpoint_path, scratch_dir):
        """Sets up the agent.

        Args:
            checkpoint_path: Path to the SimLingo VLM ``pytorch_model.pt``.
            scratch_dir: Output directory for the agent's per-step metric file.
        """

        torch.cuda.empty_cache()
        self.track = autonomous_agent.Track.SENSORS
        self.config_path = checkpoint_path
        print(f"Config path: {self.config_path}")
        self.step = -1
        self.initialized = False
        self.device = torch.device("cuda")
        self.DrivingInput = {}
        self.config = GlobalConfig()
        self.bias = {
            "speed_scale": 1.0,
            "speed_offset": 0.0,
            "gps_x": 0.0,
            "gps_y": 0.0,
            "compass_rad": 0.0,
            "cam_dx": 0.0,
            "cam_dy": 0.0,
            "cam_dz": 0.0,
            "cam_roll_deg": 0.0,
            "cam_pitch_deg": 0.0,
            "cam_yaw_deg": 0.0,
        }

        if self.config.eval_route_as == -1:
            self.config.eval_route_as = self.model.route_as

        self.last_command = -1
        self.last_command_tmp = -1

        self.trajectory_to_control = TrajectoryToControl(self.config)

        image_fps = 5
        image_history_length = 1

        self.image_buffer = deque(maxlen=image_fps * image_history_length)

        # config
        self.route_planner_max_distance = 50.0
        self.route_planner_min_distance = 7.5

        # load config from .hydra folder
        self.config_load_path = (
            Path(self.config_path).parent.parent.parent / ".hydra" / "config.yaml"
        )
        with open(self.config_load_path, "r") as file:
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
        # llm_tokenizer = AutoTokenizer.from_pretrained(cfg.model.language_model.variant)
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

        self.commands = deque(maxlen=2)
        self.commands.append(4)
        self.commands.append(4)
        self.target_point_prev = [1e5, 1e5, 1e5]

        self.ego_state_filter = EgoStateFilter(dt=1.0 / 20.0, state_log_maxlen=5)

        # Path to where visualizations and other debug output gets stored
        scratch_dir = str(scratch_dir)

        self.save_path_metric = scratch_dir + "/metric"
        Path(self.save_path_metric).mkdir(parents=True, exist_ok=True)

    def _init(self):
        # The CARLA leaderboard does not expose the lat lon reference value of the GPS which make it impossible to use the
        # GPS because the scale is not known. In the past this was not an issue since the reference was constant 0.0
        # But town 13 has a different value in CARLA 0.9.15. The following code, adapted from Bench2DriveZoo estimates the
        # lat, lon reference values by abusing the fact that the leaderboard exposes the route plan also in CARLA
        # coordinates. The GPS plan is compared to the CARLA coordinate plan to estimate the reference point / scale
        # of the GPS. It seems to work reasonably well, so we use this workaround for now.
        try:
            locx, locy = (
                self._global_plan_world_coord[0][0].location.x,
                self._global_plan_world_coord[0][0].location.y,
            )
            lon, lat = self._global_plan[0][0]["lon"], self._global_plan[0][0]["lat"]
            earth_radius_equa = 6378137.0  # Constant from CARLA leaderboard GPS simulation

            def equations(variables):
                x, y = variables
                eq1 = (
                    lon * math.cos(x * math.pi / 180.0)
                    - (locx * x * 180.0) / (math.pi * earth_radius_equa)
                    - math.cos(x * math.pi / 180.0) * y
                )
                eq2 = (
                    math.log(math.tan((lat + 90.0) * math.pi / 360.0))
                    * earth_radius_equa
                    * math.cos(x * math.pi / 180.0)
                    + locy
                    - math.cos(x * math.pi / 180.0)
                    * earth_radius_equa
                    * math.log(math.tan((90.0 + x) * math.pi / 360.0))
                )
                return [eq1, eq2]

            initial_guess = [0.0, 0.0]
            solution, _, ier, msg = fsolve(equations, initial_guess, full_output=True)
            if ier != 1:
                raise RuntimeError(msg)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            self.lat_ref, self.lon_ref = 0.0, 0.0
        self._route_planner = RoutePlanner(
            self.route_planner_min_distance,
            self.route_planner_max_distance,
            self.lat_ref,
            self.lon_ref,
        )
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}

    def sensors(self):
        sensors = []
        for num_cam in self.config.num_cameras:
            # get from config by name as string
            sensors += [
                {
                    "type": "sensor.camera.rgb",
                    "x": self.config.__dict__[f"camera_pos_{num_cam}"][0] + self.bias["cam_dx"],
                    "y": self.config.__dict__[f"camera_pos_{num_cam}"][1] + self.bias["cam_dy"],
                    "z": self.config.__dict__[f"camera_pos_{num_cam}"][2] + self.bias["cam_dz"],
                    "roll": self.config.__dict__[f"camera_rot_{num_cam}"][0]
                    + self.bias["cam_roll_deg"],
                    "pitch": self.config.__dict__[f"camera_rot_{num_cam}"][1]
                    + self.bias["cam_pitch_deg"],
                    "yaw": self.config.__dict__[f"camera_rot_{num_cam}"][2]
                    + self.bias["cam_yaw_deg"],
                    "width": self.config.__dict__[f"camera_width_{num_cam}"],
                    "height": self.config.__dict__[f"camera_height_{num_cam}"],
                    "fov": self.config.__dict__[f"camera_fov_{num_cam}"],
                    "id": f"rgb_{num_cam}",
                }
            ]

        sensors += [
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": self.config.carla_frame_rate,
                "id": "imu",
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "gps",
            },
            {
                "type": "sensor.speedometer",
                "reading_frequency": self.config.carla_fps,
                "id": "speed",
            },
        ]

        return sensors

    @torch.inference_mode()  # Turns off gradient computation
    def _tick(self, input_data):
        """Pre-processes sensor data and runs the Unscented Kalman Filter"""
        rgb = []

        for camera_pos in self.config.num_cameras:
            rgb_cam = "rgb_" + str(camera_pos)
            camera = input_data[rgb_cam][1][:, :, :3]

            # Also add jpg artifacts at test time, because the training data was saved as jpg.
            _, compressed_image_i = cv2.imencode(".jpg", camera)
            camera = cv2.imdecode(compressed_image_i, cv2.IMREAD_UNCHANGED)

            rgb_pos = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
            rgb_pos = rgb_pos[
                : int(rgb_pos.shape[0] - (rgb_pos.shape[0] * 4.8) // 16), :, :
            ]  # do this from config to ensure it is the same as in training

            # Switch to pytorch channel first order
            rgb_pos = np.transpose(rgb_pos, (2, 0, 1))
            rgb.append(rgb_pos)

        rgb = np.array(rgb)
        self.image_buffer.append(rgb)

        rgbs = rgb
        image_sizes = None

        T, C, H, W = rgbs.shape
        transform = build_transform(input_size=448)
        images_processed_tmp = []
        images_sizes_tmp = []

        image = Image.fromarray(rgbs.squeeze(0).transpose(1, 2, 0))
        images = dynamic_preprocess(
            image,
            image_size=448,
            use_thumbnail=self.cfg.model.vision_model.use_global_img,
            max_num=2,
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        images_processed_tmp.append(pixel_values)
        images_sizes_tmp.append([image.size[1], image.size[0]])

        images_processed = {
            "pixel_values": torch.stack(images_processed_tmp),
            "image_sizes": torch.tensor(images_sizes_tmp),
        }
        processed_image = images_processed["pixel_values"]
        num_patches = processed_image.shape[1]
        new_height = processed_image.shape[3]
        new_width = processed_image.shape[4]
        processed_image = processed_image.view(1, self.T, num_patches, C, new_height, new_width)

        gps_pos = self._route_planner.convert_gps_to_carla(input_data["gps"][1])
        gps_pos = gps_pos + np.array([self.bias["gps_x"], self.bias["gps_y"], 0.0])

        compass = preprocess_compass(input_data["imu"][1][-1]) + self.bias["compass_rad"]

        result = {
            "rgb": rgb,
            "compass": compass,
        }
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
        result["gps"] = filtered_state[0:2]

        speed = round(speed, 1)

        waypoint_route = self._route_planner.run_step(np.append(result["gps"], gps_pos[2]))

        if len(waypoint_route) > 2:
            target_point, far_command = waypoint_route[1]
            next_target_point, next_far_command = waypoint_route[2]
        elif len(waypoint_route) > 1:
            target_point, far_command = waypoint_route[1]
            next_target_point, next_far_command = waypoint_route[1]
        else:
            target_point, far_command = waypoint_route[0]
            next_target_point, next_far_command = waypoint_route[0]

        if self.last_command_tmp != far_command:
            self.last_command = self.last_command_tmp

        self.last_command_tmp = far_command
        if (target_point != self.target_point_prev).all():
            self.target_point_prev = target_point
            self.commands.append(far_command.value)

        one_hot_command = command_to_one_hot(self.commands[-2])
        result["command"] = torch.from_numpy(one_hot_command[np.newaxis]).to(
            self.device, dtype=torch.float32
        )

        ego_target_point = inverse_conversion_2d(target_point[:2], result["gps"], result["compass"])
        ego_target_point_torch = torch.from_numpy(ego_target_point[np.newaxis]).to(
            self.device, dtype=torch.float32
        )
        ego_next_target_point = inverse_conversion_2d(
            next_target_point[:2], result["gps"], result["compass"]
        )

        result["target_point"] = ego_target_point_torch

        placeholder_batch_list = []

        if (
            self.config.eval_route_as == "target_point"
            or self.config.eval_route_as == "target_point_command"
        ):
            target_points = [ego_target_point, ego_next_target_point]
            target_points_np = np.array(target_points)
            target_points = (
                torch.from_numpy(target_points_np).to(self.device, dtype=torch.float32).unsqueeze(0)
            )
            result["route"] = target_points

            placeholder_values = {"<TARGET_POINT>": target_points_np}
            tmp = {}
            for key, value in placeholder_values.items():
                token_nr_key = self.tokenizer.convert_tokens_to_ids(key)
                tmp[token_nr_key] = value
            placeholder_batch_list.append(tmp)

            prompt_tp = "Target waypoint: <TARGET_POINT><TARGET_POINT>."

        elif self.config.eval_route_as == "command":
            # get distance from target_point
            dist_to_command = np.linalg.norm(ego_target_point)
            dist_to_command = int(dist_to_command)
            map_command = {
                1: "go left at the next intersection",
                2: "go right at the next intersection",
                3: "go straight at the next intersection",
                4: "follow the road",
                5: "do a lane change to the left",
                6: "do a lane change to the right",
            }
            command = map_command[far_command]
            next_command = map_command[next_far_command]
            if self.last_command in [1, 2, 3] and far_command == 4:
                next_command = command
                command = map_command[self.last_command]

            if command != next_command:
                next_command = f" then {next_command}"
            else:
                next_command = ""

            if far_command == 4:
                prompt_tp = f"Command: {command}{next_command}."
            else:
                prompt_tp = f"Command: {command} in {dist_to_command} meter{next_command}."

        else:
            raise ValueError(f"Unsupported eval_route_as: {self.config.eval_route_as}")

        if self.config.use_cot:
            prompt = f"Current speed: {speed} m/s. {prompt_tp} What should the ego do next?"
        else:
            prompt = f"Current speed: {speed} m/s. {prompt_tp} Predict the waypoints."

        result["speed"] = (
            torch.FloatTensor([speed]).unsqueeze(0).to(self.device, dtype=torch.float32)
        )

        B, T, num_patches, C, H, W = processed_image.shape
        assert B == 1
        assert T == self.T
        assert C == 3

        speed = round(speed, 1)

        conversation_all = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Waypoints:"},
                ],
            },
        ]
        conv_batch_list = [conversation_all]
        questions = []
        for conv in conv_batch_list:
            for i in range(len(conv)):
                questions.append(conv[i]["content"][0]["text"])
                conv[i]["content"] = conv[i]["content"][0]["text"]

        prompt_batch_list = []
        for idx, conv in enumerate(conv_batch_list):
            question = questions[idx]
            if "<image>" not in question:
                question = "<image>\n" + question
            template = conv_module.get_conv_template("internlm2-chat")
            for conv_part_idx, conv_part in enumerate(conv):
                if conv_part["role"] == "assistant":
                    # template.append_message(template.roles[1], conv_part['content'])
                    template.append_message(template.roles[1], None)
                elif conv_part["role"] == "user":
                    if conv_part_idx == 0 and "<image>" not in conv_part["content"]:
                        # add image token
                        conv_part["content"] = "<image>\n" + conv_part["content"]
                    template.append_message(template.roles[0], conv_part["content"])
                else:
                    raise ValueError(f"Role {conv_part['role']} not supported")

            query = template.get_prompt()
            # remove system prompt
            system_prompt = (
                template.system_template.replace("{system_message}", template.system_message)
                + template.sep
            )
            query = query.replace(system_prompt, "")

            IMG_START_TOKEN = "<img>"
            IMG_END_TOKEN = "</img>"
            IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
            num_patches_all = 2  # sum(grid_nums)

            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches_all
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
            prompt_batch_list.append(query)

        prompt_tokenized = self.tokenizer(
            prompt_batch_list,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        prompt_tokenized_ids = prompt_tokenized["input_ids"]
        prompt_tokenized_valid = prompt_tokenized["input_ids"] != self.tokenizer.pad_token_id
        prompt_tokenized_mask = prompt_tokenized_valid

        ll = LanguageLabel(
            phrase_ids=prompt_tokenized_ids.to(self.device),
            phrase_valid=prompt_tokenized_valid.to(self.device),
            phrase_mask=prompt_tokenized_mask.to(self.device),
            placeholder_values=placeholder_batch_list,
            language_string=prompt_batch_list,
            loss_masking=None,
        )

        self.DrivingInput["camera_images"] = processed_image.to(self.device).bfloat16()
        self.DrivingInput["image_sizes"] = image_sizes
        self.DrivingInput["camera_intrinsics"] = (
            torch.repeat_interleave(get_camera_intrinsics(W, H, 110).unsqueeze(0), 1, dim=0)
            .view(1, 3, 3)
            .float()
            .to(self.device),
        )
        self.DrivingInput["camera_extrinsics"] = (
            torch.repeat_interleave(get_camera_extrinsics().unsqueeze(0), 1, dim=0)
            .view(1, 4, 4)
            .float()
            .to(self.device),
        )
        self.DrivingInput["vehicle_speed"] = result["speed"]
        self.DrivingInput["target_point"] = result["target_point"].to(self.device)
        self.DrivingInput["prompt"] = ll
        self.DrivingInput["prompt_inference"] = ll

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp, sensors=None):  # pylint: disable=locally-disabled, unused-argument
        self.step += 1

        if not self.initialized:
            self._init()
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.control = control
            tick_data = self._tick(input_data)
            return control

        # Need to run this every step for GPS filtering
        tick_data = self._tick(input_data)

        # initialize DrivingInput with dict self.DrivingInput
        model_input = DrivingInput(**self.DrivingInput)
        pred_speed_wps, pred_route, _ = self.model(model_input)
        pred_speed_wps = pred_speed_wps.float() if pred_speed_wps is not None else None
        pred_route = pred_route.float() if pred_route is not None else None

        # prepare velocity input
        gt_velocity = tick_data["speed"]

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
        if self.step < self.config.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)
        else:
            self.control = control

        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info
        if self.save_path_metric is not None and self.step % 1 == 0:
            # metric info
            outfile = open(f"{self.save_path_metric}/metric_info.json", "w")
            json.dump(self.metric_info, outfile, indent=4)
            outfile.close()

        return control

    def destroy(self, results=None):  # pylint: disable=locally-disabled, unused-argument
        """
        Gets called after a route finished.
        The leaderboard client doesn't properly clear up the agent after the route finishes so we need to do it here.
        Also writes logging files to disk.
        """

        del self.model
        del self.config
