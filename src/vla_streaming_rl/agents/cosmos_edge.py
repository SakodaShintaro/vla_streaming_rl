# SPDX-License-Identifier: MIT
"""Cosmos3-Edge av policy agent for CARLA/Bench2Drive.

Mirrors the off-policy structure of ``SimLingoAgent`` but drives the Cosmos3-Edge
av policy: the network turns the front camera + a fixed driving prompt into an
ego-pose action chunk (raw dim 9 = 3D translation + 6D rotation), which the agent
executes open-loop over ``chunk_size`` ticks (one row per tick, matching the
critic's per-step chunk reward/done window). The ego-frame translation is
transformed to a world target point and driven to via CARLA's
``VehiclePIDController`` (built against the live ego vehicle each episode).

NOTE (calibration): the exact av ego-pose convention (translation axis order /
units / frame) is not yet pinned against NVIDIA's av reference data, so the
ego->world target mapping and the SPEED_SCALE in ``_to_env_action`` are a
documented first guess and will need calibration on a live CARLA run.
"""

from typing import Any

import carla
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from agents.navigation.controller import VehiclePIDController
from torch import optim

from vla_streaming_rl.agents.base import Agent, StepResult
from vla_streaming_rl.networks.cosmos_edge_network import CosmosEdgeNetwork
from vla_streaming_rl.networks.interface import InferInput
from vla_streaming_rl.reward_processor import RewardProcessor
from vla_streaming_rl.utils import create_reward_image

FRAME_HW = 256
DT = 0.05  # CARLA control tick (20 FPS)

# Longitudinal calibration knobs mapping the av chunk's (unknown-unit) forward
# progress to a target speed in m/s. Provisional v1 values — calibrate against
# NVIDIA's av reference data / observed CARLA behaviour.
SPEED_SCALE = 20.0
MIN_SPEED = 2.0
MAX_SPEED = 6.0


class _TargetWaypoint:
    """Minimal duck-typed waypoint (VehiclePIDController's lateral controller only
    reads ``.transform.location``), so the av target point can be steered toward
    directly without snapping it to a lane."""

    def __init__(self, location: "carla.Location") -> None:
        self.transform = carla.Transform(location)


class CosmosEdgeAgent(Agent):
    def __init__(
        self,
        *,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Box,
        network: CosmosEdgeNetwork,
        env,
        learning_mode: str,
        horizon: int,
        buffer_size: int,
        batch_size: int,
        learning_starts: int,
        actor_lr: float,
        critic_lr: float,
        weight_decay: float,
        max_grad_norm: float,
        gamma: float,
    ) -> None:
        super().__init__(learning_mode=learning_mode, horizon=horizon)
        del observation_space, action_space
        assert learning_mode == "off_policy", "cosmos_edge supports off_policy only"

        self.device = torch.device("cuda")
        self.network = network
        self.env = env
        self.chunk_size = network.chunk_size
        self.action_dim = network.action_dim
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.max_grad_norm = max_grad_norm

        # CARLA VehiclePIDController, (re)built per episode against the live vehicle.
        self._controller: VehiclePIDController | None = None
        self._ctrl_vehicle = None

        self.actor_optimizer = optim.AdamW(
            network.actor_parameters, lr=actor_lr, weight_decay=weight_decay
        )
        self.critic_optimizer = optim.AdamW(
            network.critic.parameters(), lr=critic_lr, weight_decay=weight_decay
        )
        self.reward_processor = RewardProcessor("carla", 1.0)

        # Learn the flat-obs schema up front so the buffer can be sized.
        zero_frame = torch.zeros(3, FRAME_HW, FRAME_HW)
        network.pack_obs(zero_frame)
        from vla_streaming_rl.replay_buffer import ReplayBuffer

        # seq_len = chunk_size + 1: a sampled window reconstructs the executed
        # chunk (one av row per tick) plus the bootstrap next-observation.
        self.rb = ReplayBuffer(
            size=buffer_size,
            seq_len=self.chunk_size + 1,
            obs_shape=(network.obs_flat_dim,),
            obs_z_shape=(1,),
            rnn_state_shape=(1,),
            action_shape=(self.action_dim,),
            output_device=self.device,
            storage_device=torch.device("cpu"),
            max_prompt_tokens=0,
            pad_token_id=0,
        )
        self._dummy = torch.zeros(1, device=self.device)

        # Open-loop chunk execution state.
        self._chunk: torch.Tensor | None = None
        self._chunk_idx = 0
        self._prev_action = torch.zeros(self.action_dim, device=self.device)

    # --- required agent surface --------------------------------------------

    def _preprocess(self, obs: dict[str, Any], info: dict) -> torch.Tensor:
        del info
        image = torch.as_tensor(np.asarray(obs["image"]), dtype=torch.float32)  # (3, H, W) in [0,1]
        frame = F.interpolate(
            image[None], size=(FRAME_HW, FRAME_HW), mode="bilinear", align_corners=False
        )[0]
        return frame

    def _pid_controller(self) -> VehiclePIDController:
        """(Re)build the CARLA VehiclePIDController against the current ego vehicle."""
        vehicle = self.env.unwrapped.vehicle
        if self._controller is None or self._ctrl_vehicle is not vehicle:
            self._controller = VehiclePIDController(
                vehicle,
                args_lateral={"K_P": 1.5, "K_I": 0.05, "K_D": 0.1, "dt": DT},
                args_longitudinal={"K_P": 1.0, "K_I": 0.05, "K_D": 0.0, "dt": DT},
            )
            self._ctrl_vehicle = vehicle
        return self._controller

    def _to_env_action(self, chunk: torch.Tensor, idx: int, velocity: float):
        """av ego-pose chunk -> (steer, gas_or_brake) via CARLA's VehiclePIDController.

        The furthest remaining av translation point ``chunk[-1, :2]`` (ego frame:
        x forward, y left — av-convention-dependent, see module note) is transformed
        to world coordinates and passed as the target waypoint; the target speed is
        derived from the av chunk's forward progress."""
        path = chunk[idx:, :2].detach().float().cpu().numpy()
        target_local = path[-1]  # furthest lookahead point

        step = (
            float(np.linalg.norm(np.diff(path, axis=0), axis=1).mean()) if len(path) >= 2 else 0.0
        )
        desired_speed_kmh = float(np.clip(step * SPEED_SCALE, MIN_SPEED, MAX_SPEED)) * 3.6

        vehicle = self.env.unwrapped.vehicle
        # av ego (x fwd, y left) -> CARLA vehicle local (x fwd, y right): negate y.
        local = carla.Location(x=float(target_local[0]), y=-float(target_local[1]), z=0.0)
        world_loc = vehicle.get_transform().transform(local)
        control = self._pid_controller().run_step(desired_speed_kmh, _TargetWaypoint(world_loc))
        gas_or_brake = float(control.throttle) - float(control.brake)
        return np.array([float(control.steer), gas_or_brake], dtype=np.float32)

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
        frame = self._preprocess(obs, info)
        self.rb.add(
            self.network.pack_obs(frame).to(self.device),
            self._dummy,
            reward,
            terminated or truncated,
            self._dummy,
            self._prev_action,
            [],
        )

        # Re-plan a fresh av chunk at the start of each open-loop window.
        if self._chunk is None or self._chunk_idx == 0:
            live = self.network.infer(
                InferInput(
                    s_seq=frame.to(self.device),
                    obs_z_seq=self._dummy,
                    a_seq=self._dummy,
                    r_seq=self._dummy,
                    rnn_state=self._dummy,
                    task_prompts=[],
                )
            )
            self._chunk = live.action.squeeze(0)  # (chunk, 9)
            self._value_report = live.value_report

        row = self._chunk[self._chunk_idx]
        self._prev_action = row
        speed = float(obs["sensors"]["speed"][1]["speed"]) if "sensors" in obs else 0.0
        env_action = self._to_env_action(self._chunk, self._chunk_idx, speed)
        self._chunk_idx = (self._chunk_idx + 1) % self.chunk_size

        metrics = {
            "action_norm": float(np.linalg.norm(env_action)),
            "processed_reward": self.reward_processor.normalize(torch.tensor(reward)).item(),
            **self._value_report,
        }
        return StepResult(action=env_action, metrics=metrics, panels=self._panels(obs, reward))

    def _step_streaming(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult:
        raise NotImplementedError("cosmos_edge supports off_policy only")

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        del score, feedback_text
        self._chunk = None
        self._chunk_idx = 0
        self._controller = None  # rebuilt against the next episode's vehicle
        return {}

    def _panels(self, obs: dict[str, Any], reward: float) -> dict[str, np.ndarray]:
        del obs
        return {"reward": create_reward_image(None, reward)}
