# SPDX-License-Identifier: MIT
"""DACER2-based RL agent built on top of SimLingo.

The action trained against is SimLingo's full waypoint output:
``pred_route`` (20×2) concatenated with ``pred_speed_wps`` (10×2),
flattened to a 60-D continuous vector. The deterministic PID controller
that turns waypoints into ``carla.VehicleControl`` is treated as part
of the environment — the policy emits waypoints, PID emits 2-D control,
env sees ``[steer, throttle - brake]``.

  μ(s) = base_wps(VLM(s)) + δ_scale * Δ(features(s))
  Δ : DiffusionPolicy (DACER2; tanh-bounded flow-matching policy)
  Q(s, a) = ActionValueHead(features(s), a)

Critic update (off-policy, batch from replay):
  Δ' ~ π(·|s'), a' = base'(s') + δ_scale * Δ'
  y = r + γ (1 − done) Q_target(s', a')
  L_critic = MSE(Q(s, a), y)

Actor update (off-policy, batch from replay):
  Δ ~ π(·|s)
  L_actor = advantage-based loss + DACER2 flow-matching loss
           (via DiffusionPolicy.compute_actor_loss).
  A small wrapper presents Q(s, base + δ_scale * Δ) to that method, so
  the policy is trained on Δ while Q sees the full action.

VLM gradients only flow when ``use_lora`` is true: peft wraps the
SimLingo LLM after the checkpoint load, and both critic and actor losses
backprop into the LoRA parameters through the re-forwarded state at
each replay sample. With ``use_lora`` false, the VLM is fully frozen
and only Δ and Q learn.

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
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch import nn, optim
from transformers import Qwen2Tokenizer

from vla_streaming_rl.networks.policy_head import DiffusionPolicy
from vla_streaming_rl.networks.value_head import ActionValueHead
from vla_streaming_rl.replay_buffer import ReplayBuffer
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

# SimLingo's driving adaptor emits 20 route waypoints + 10 speed
# waypoints, each 2-D — see DrivingAdaptor in
# simlingo_training/models/adaptors.py.
_ROUTE_LEN = 20
_SPEED_WPS_LEN = 10
_WP_DIM = 2
_ACTION_DIM = (_ROUTE_LEN + _SPEED_WPS_LEN) * _WP_DIM

# Standard off-policy target needs two contiguous indices (s_t, s_{t+1}).
_SEQ_LEN = 2


def _apply_lora(module: nn.Module) -> nn.Module:
    """Wrap an nn.Module with peft LoRA adapters (``target_modules="all-linear"``)
    and return the wrapped model. Logs the trainable-parameter count.
    """
    wrapped = get_peft_model(
        module,
        LoraConfig(
            inference_mode=False,
            r=8,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules="all-linear",
        ),
    )
    wrapped.print_trainable_parameters()
    return wrapped


def _to_device(obj, device):
    """Recursively move every Tensor inside ``obj`` to ``device``.

    Handles nested tuples / lists / dicts and dataclass instances (so a
    ``LanguageLabel`` carrying tensor fields round-trips correctly when
    we shuffle a ``DrivingInput`` between CPU storage and GPU compute).
    Non-tensor leaves are returned as-is.
    """
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.detach().to(device)
    if isinstance(obj, tuple):
        return tuple.__new__(type(obj), (_to_device(x, device) for x in obj))
    if isinstance(obj, list):
        return [_to_device(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return type(obj)(
            **{f: _to_device(getattr(obj, f), device) for f in obj.__dataclass_fields__}
        )
    return obj


def _waypoints_to_action_vec(route: torch.Tensor, speed_wps: torch.Tensor) -> torch.Tensor:
    """Flatten one sample's (1, R, 2) + (1, S, 2) waypoints to a 1-D
    action vector of length (R+S)*2 in float32.
    """
    return torch.cat([route.reshape(-1), speed_wps.reshape(-1)], dim=0).to(torch.float32)


def _waypoints_batch_to_action_vec(route: torch.Tensor, speed_wps: torch.Tensor) -> torch.Tensor:
    """Batch version of the above: (B, R, 2) + (B, S, 2) → (B, (R+S)*2)."""
    bs = route.shape[0]
    return torch.cat([route.reshape(bs, -1), speed_wps.reshape(bs, -1)], dim=1).to(torch.float32)


def _action_vec_to_waypoints(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    route_flat = a[: _ROUTE_LEN * _WP_DIM]
    speed_flat = a[_ROUTE_LEN * _WP_DIM :]
    return (
        route_flat.view(1, _ROUTE_LEN, _WP_DIM),
        speed_flat.view(1, _SPEED_WPS_LEN, _WP_DIM),
    )


class _BasedValueHead:
    """Thin shim around ``ActionValueHead`` that adds a fixed base action
    before evaluating Q. Lets :meth:`DiffusionPolicy.compute_actor_loss`
    treat the policy's tanh-bounded Δ as the action (so the advantage /
    Q-grad machinery applies cleanly) while the real critic still sees
    the full ``base + δ_scale * Δ`` action.
    """

    def __init__(
        self,
        *,
        value_head: ActionValueHead,
        base_action: torch.Tensor,
        delta_scale: float,
    ) -> None:
        self.value_head = value_head
        self.base_action = base_action
        self.delta_scale = delta_scale

    def __call__(self, state: torch.Tensor, delta_chunk: torch.Tensor) -> dict:
        action = self.base_action + self.delta_scale * delta_chunk
        return self.value_head(state, action)

    def get_advantage(self, state: torch.Tensor, delta_chunk: torch.Tensor) -> dict:
        action = self.base_action + self.delta_scale * delta_chunk
        return self.value_head.get_advantage(state, action)

    def parameters(self):
        return self.value_head.parameters()


class SimLingoAgent:
    """DACER2 off-policy agent built on SimLingo's waypoint output.

    ``DrivingModel.forward`` returns ``(speed_wps, route, language,
    driving_features)``. The fourth element is pooled (mean over the 30
    waypoint-query positions) into the critic / actor's state vector.

    ``run_step`` is pure inference: VLM forward → base waypoints +
    sampled Δ from the DACER2 diffusion policy → PID. Training is
    off-policy: ``step`` writes (state, action, reward, done) into the
    replay buffer plus a CPU copy of the ``DrivingInput`` dict so
    ``_maybe_train`` can re-forward the VLM at each replay sample and
    keep critic / actor losses propagating fresh gradients into LoRA.
    """

    _HF_REPO_ID = "RenzKa/simlingo"
    _HF_CKPT_NAME = "pytorch_model.pt"

    def __init__(
        self,
        *,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        env: gym.Env,
        scratch_dir: Path,
        use_lora: bool,
        gamma: float,
        buffer_size: int,
        batch_size: int,
        learning_starts: int,
        critic_hidden_dim: int,
        critic_block_num: int,
        actor_hidden_dim: int,
        actor_block_num: int,
        denoising_time: float,
        denoising_steps: int,
        dacer_loss_weight: float,
        delta_scale: float,
        actor_lr: float,
        critic_lr: float,
        max_grad_norm: float,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self._env_unwrapped = env.unwrapped

        scratch_dir = Path(scratch_dir)
        scratch_dir.mkdir(parents=True, exist_ok=True)

        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.delta_scale = delta_scale
        self.max_grad_norm = max_grad_norm

        torch.cuda.empty_cache()
        config_path = str(self._resolve_checkpoint())
        print(f"Config path: {config_path}")
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
        config_load_path = Path(config_path).parent.parent.parent / ".hydra" / "config.yaml"
        with open(config_load_path, "r") as file:
            cfg = OmegaConf.load(file)
        self.cfg = cfg
        self.cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img

        # ``AutoProcessor`` + ``trust_remote_code=True`` resolves to
        # ``Qwen2Tokenizer`` for InternVL2-1B (the custom remote processor
        # class isn't actually used downstream). Name the concrete class
        # directly so we don't depend on HF-hosted remote code.
        tokenizer = Qwen2Tokenizer.from_pretrained(cfg.model.vision_model.variant)
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(SIMLINGO_ADDITIONAL_SPECIAL_TOKENS)}
        )
        tokenizer.padding_side = "left"

        # ``AutoConfig`` + ``trust_remote_code=True`` resolves to
        # ``InternVLChatConfig``. Use the vendored class directly so the
        # HF-hosted ``configuration_internvl_chat.py`` is no longer
        # downloaded / executed at runtime.
        tmp_config = InternVLChatConfig.from_pretrained(cfg.model.vision_model.variant)
        image_size = tmp_config.force_image_size or tmp_config.vision_config.image_size
        patch_size = tmp_config.vision_config.patch_size
        num_image_token = int((image_size // patch_size) ** 2 * (tmp_config.downsample_ratio**2))
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
            processor=tokenizer,
            cache_dir=cache_dir,
            **model_kwargs,
        ).to(self.device)
        torch.set_default_dtype(default_dtype)
        self.model.load_state_dict(torch.load(config_path))

        self.T = 1
        self.stuck_detector = 0
        self.force_move = 0

        self.ego_state_filter = EgoStateFilter(dt=1.0 / 20.0, state_log_maxlen=5)
        self.prompt_builder = PromptBuilder(
            config=self.config,
            tokenizer=tokenizer,
            num_image_token=num_image_token,
            device=self.device,
        )

        self.save_path_metric = str(scratch_dir) + "/metric"
        Path(self.save_path_metric).mkdir(parents=True, exist_ok=True)

        # --- RL setup ---------------------------------------------------

        # Freeze everything by default; LoRA (when enabled) re-introduces
        # trainable params on both the vision encoder and the LLM —
        # driving is visually dominated, so adapting the ViT alongside
        # the language model matters as much as (or more than) just the
        # LLM. The peft wrap happens AFTER the checkpoint load —
        # vanilla SimLingo cfg has ``lora=False`` at checkpoint time so
        # the loaded state_dict has no LoRA tensors.
        for p in self.model.parameters():
            p.requires_grad_(False)
        if use_lora:
            self.model.vision_model = _apply_lora(self.model.vision_model)
            self.model.language_model.model = _apply_lora(self.model.language_model.model)

        # SimLingo runs in bfloat16; the RL heads run in float32 to keep
        # value targets numerically stable.
        feature_dim = int(self.model.language_model.hidden_size)
        # DACER2 flow-matching policy that produces a tanh-bounded Δ.
        # The actual action sent to PID is ``base + delta_scale * Δ``.
        self.delta_head = DiffusionPolicy(
            state_dim=feature_dim,
            action_dim=_ACTION_DIM,
            hidden_dim=actor_hidden_dim,
            block_num=actor_block_num,
            denoising_time=denoising_time,
            sparsity=0.0,
            horizon=1,
            denoising_steps=denoising_steps,
            dacer_loss_weight=dacer_loss_weight,
        ).to(self.device)
        # Dueling Q(s, a) head from networks.value_head. ``num_bins=1``
        # collapses the distributional output to a scalar.
        self.critic = ActionValueHead(
            in_channels=feature_dim,
            action_dim=_ACTION_DIM,
            horizon=1,
            hidden_dim=critic_hidden_dim,
            block_num=critic_block_num,
            num_bins=1,
            sparsity=0.0,
        ).to(self.device)

        # Filter picks up LoRA params when use_lora=True, otherwise the
        # list reduces to the diffusion policy alone.
        actor_params = list(self.delta_head.parameters()) + [
            p for p in self.model.parameters() if p.requires_grad
        ]
        self.actor_optimizer = optim.AdamW(actor_params, lr=actor_lr, weight_decay=0.0)
        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(), lr=critic_lr, weight_decay=0.0
        )

        # Replay slots only carry (action, reward, done); state and base
        # action are recomputed at training time by re-forwarding the
        # VLM on the parallel ``_driving_input_buffer``. The unused
        # slots (obs, obs_z, rnn_state, log_prob, value, token ids) get
        # shape-(1,) / 0 / empty dummies so the buffer machinery still
        # type-checks.
        self.rb = ReplayBuffer(
            size=buffer_size,
            seq_len=_SEQ_LEN,
            obs_shape=(1,),
            obs_z_shape=(1,),
            rnn_state_shape=(1,),
            action_shape=(_ACTION_DIM,),
            output_device=self.device,
            storage_device=torch.device("cpu"),
            max_new_tokens=0,
            max_prompt_tokens=0,
            pad_token_id=0,
        )
        self._dummy_obs = torch.zeros(1, device=self.device)
        self._dummy_obs_z = torch.zeros(1, device=self.device)
        self._dummy_rnn_state = torch.zeros(1, device=self.device)

        # Parallel CPU storage of the ``DrivingInput`` kwargs at each
        # step, indexed identically to the main replay buffer. ``step``
        # appends the latest kwargs (captured by ``run_step`` into
        # ``self._last_driving_input_kwargs``); ``_maybe_train`` pulls
        # them out, moves them back to GPU, and re-runs the VLM so both
        # critic and actor losses flow fresh gradient through LoRA.
        self._driving_input_buffer: list = [None] * buffer_size
        self._last_driving_input_kwargs: dict | None = None

        # ``_current_action_taken`` carries the action just selected by
        # ``run_step`` across to the next ``step``, which writes it as
        # the buffer's ``actions[t]`` (= action that produced state t,
        # per the project's convention).
        self._current_action_taken: torch.Tensor = torch.zeros(_ACTION_DIM, device=self.device)

        self._attached_ego_id: int | None = None
        self._prev_action = torch.zeros(_ACTION_DIM, device=self.device)

        # Trainer-facing surface (parameter count print + checkpoint).
        self.network = self.model
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
    ) -> tuple[np.ndarray, dict]:
        self._maybe_handover_episode()
        env_action = self._act()
        return env_action, self._build_info(env_action)

    def step(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
    ) -> tuple[np.ndarray, dict]:
        self._maybe_handover_episode()
        episode_done = terminated or truncated
        env_action = self._act()
        info = self._build_info(env_action)

        # Add (action_{t-1}, reward, done). Mirrors
        # off_policy.OffPolicyAgent.select_action's add semantics so
        # later sampling and indexing match the project convention.
        self.rb.add(
            self._dummy_obs,
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
        # Mirror the buffer's write slot in the DrivingInput buffer so
        # ``_maybe_train`` can re-forward the VLM at the same indices.
        write_idx = (self.rb.idx - 1) % self.rb.size
        self._driving_input_buffer[write_idx] = _to_device(
            self._last_driving_input_kwargs, torch.device("cpu")
        )

        info.update(self._maybe_train(global_step))

        # Advance: the action just selected becomes the prev for the
        # next add. (off_policy carries it across episode boundaries
        # too — the buffer's done flag handles bootstrap correctness.)
        self._prev_action = self._current_action_taken

        return env_action, info

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        # Force re-init on the next select_action.
        self._attached_ego_id = None
        return {}

    def _build_info(self, env_action: np.ndarray) -> dict:
        """Trainer reads ``agent_info["goal_image"]`` for its rendering
        pipeline. The waypoint policy doesn't predict a goal, so hand
        back a zeroed frame shaped like the obs the env returns.
        ``action_norm`` is included as a scalar telemetry hook."""
        h, w = self.observation_space.shape[1], self.observation_space.shape[2]
        return {
            "goal_image": np.zeros((h, w, 3), dtype=np.float32),
            "action_norm": float(np.linalg.norm(env_action)),
        }

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
        """Pure inference: VLM forward → base waypoints → sample Δ from the
        DACER2 diffusion policy → ``base + δ_scale * Δ`` → PID. The
        DiffusionPolicy's tanh-bounded output ``[-1, 1]`` combined with
        ``delta_scale`` (e.g. 0.1) caps Δ at ``[-0.1, 0.1]`` per coord —
        exploration comes entirely from the diffusion sampler's noise.
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
        # Capture for the parallel ``_driving_input_buffer`` written by
        # ``step``; the training loop later re-forwards the VLM on a
        # CPU-stored copy of this dict to get fresh features + base.
        self._last_driving_input_kwargs = driving_input_kwargs

        model_input = DrivingInput(**driving_input_kwargs)
        pred_speed_wps, pred_route, _, driving_features = self.model(model_input)
        pred_speed_wps = pred_speed_wps.float() if pred_speed_wps is not None else None
        pred_route = pred_route.float() if pred_route is not None else None

        pooled = driving_features.mean(dim=1).squeeze(0).to(torch.float32)
        base_action = _waypoints_to_action_vec(pred_route, pred_speed_wps)
        delta_chunk, _ = self.delta_head.get_action(pooled.unsqueeze(0))
        delta = delta_chunk.squeeze(0).squeeze(0)
        action_taken = base_action + self.delta_scale * delta
        self._current_action_taken = action_taken

        # Feed the modified waypoints to the deterministic PID.
        pred_route, pred_speed_wps = _action_vec_to_waypoints(action_taken)

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

    def _maybe_train(self, global_step: int) -> dict:
        # The buffer's ``sample`` requires ``curr_size > seq_len``; the
        # learning_starts warmup is set high enough that this is
        # implicitly satisfied.
        if global_step < self.learning_starts:
            return {}

        # Sample transition indices directly so we can mirror them into
        # the parallel ``_driving_input_buffer``. seq[:, 0] = t,
        # seq[:, 1] = t+1; the replay convention puts a_t at
        # actions[t+1], r_t at rewards[t+1], done_t at dones[t+1].
        curr_size = self.rb.size if self.rb.full else self.rb.idx
        start = torch.randint(0, curr_size - _SEQ_LEN, (self.batch_size,))
        seq = start[:, None] + torch.arange(_SEQ_LEN)[None, :]
        a = self.rb.actions[seq[:, 1]].to(self.device)
        r = self.rb.rewards[seq[:, 1], 0].to(self.device)
        r = self.reward_processor.normalize(r)
        done = self.rb.dones[seq[:, 1], 0].to(self.device)

        # Re-forward the VLM at each replay index. The same forward
        # gives both the pooled state and the base waypoints, so
        # base_t / base_next come "for free" alongside features.
        states_t = []
        bases_t = []
        states_next = []
        bases_next = []
        for i_t, i_next in zip(seq[:, 0].tolist(), seq[:, 1].tolist()):
            di_t = _to_device(self._driving_input_buffer[i_t], self.device)
            sp_t, rt_t, _, df_t = self.model(DrivingInput(**di_t))
            states_t.append(df_t.mean(dim=1).squeeze(0).to(torch.float32))
            bases_t.append(_waypoints_to_action_vec(rt_t.float(), sp_t.float()))
            di_next = _to_device(self._driving_input_buffer[i_next], self.device)
            sp_n, rt_n, _, df_n = self.model(DrivingInput(**di_next))
            states_next.append(df_n.mean(dim=1).squeeze(0).to(torch.float32))
            bases_next.append(_waypoints_to_action_vec(rt_n.float(), sp_n.float()))
        s = torch.stack(states_t)
        base_t = torch.stack(bases_t)
        s_next = torch.stack(states_next)
        base_next = torch.stack(bases_next)

        # --- critic loss --------------------------------------------------
        with torch.no_grad():
            delta_next, _ = self.delta_head.get_action(s_next)
            a_next_target = base_next + self.delta_scale * delta_next.squeeze(1)
            next_q = self.critic(s_next, a_next_target.unsqueeze(1))["output"].view(-1)
            target_q = r + self.gamma * (1.0 - done) * next_q
        current_q = self.critic(s, a.unsqueeze(1))["output"].view(-1)
        critic_loss = nn.functional.mse_loss(current_q, target_q)

        # --- actor loss (delegated to DiffusionPolicy) --------------------
        # ``_BasedValueHead`` lets DiffusionPolicy.compute_actor_loss treat the
        # tanh-bounded Δ as the action, while the real ``self.critic``
        # still sees ``base + δ_scale * Δ``. ``base_t`` is detached so the
        # advantage gradient flows to Δ (and through state to LoRA), not
        # back through the base path.
        based_value_head = _BasedValueHead(
            value_head=self.critic,
            base_action=base_t.detach().unsqueeze(1),
            delta_scale=self.delta_scale,
        )
        actor_loss, _, actor_info = self.delta_head.compute_actor_loss(
            s,
            None,
            value_head=based_value_head,
            hl_gauss_loss=None,
            num_bins=1,
            detach_actor=False,
        )

        # --- backward + optimizer step ------------------------------------
        # Combined backward: critic_loss + actor_loss share the LoRA
        # graph via the re-forwarded state, so collapsing them into one
        # backward pass avoids needing retain_graph. Each optimizer only
        # owns its slice of parameters, so cross-bleed (e.g. actor's
        # q-grad path depositing on critic) only affects critic.grad and
        # is then stepped solely by critic_optimizer — same as a
        # critic_loss-only update at that magnitude.
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        (critic_loss + actor_loss).backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.actor_optimizer.param_groups[0]["params"], self.max_grad_norm)
        self.critic_optimizer.step()
        self.actor_optimizer.step()

        return {
            "losses/critic_loss": float(critic_loss.item()),
            "losses/actor_loss": float(actor_info["actor_loss"]),
            "losses/dacer_loss": float(actor_info["dacer_loss"]),
            "losses/advantage": float(actor_info["advantage"]),
            "losses/q_value": float(current_q.mean().item()),
            "losses/target_q": float(target_q.mean().item()),
        }
