# SPDX-License-Identifier: MIT
"""Cosmos3-Edge av policy agent for CARLA/Bench2Drive.

Mirrors the off-policy structure of ``SimLingoAgent`` but drives the Cosmos3-Edge
av policy: the network turns the front camera + a fixed driving prompt into an
ego-pose action chunk (raw dim 9 = 3D translation + 6D rotation), which the agent
executes open-loop over ``chunk_size`` ticks (one row per tick, matching the
critic's per-step chunk reward/done window). The ego-frame translation is
transformed to a world target point and driven to via CARLA's
``VehiclePIDController`` (built against the live ego vehicle each episode).

av action convention (Cosmos 3 paper arXiv:2606.02800 + NVIDIA action code,
confirmed against the HF ``edge_action_id_av_*`` examples):
  * 9D = [translation(3), rotation_6d(6)]; rotation_6d is the diffusion_policy /
    pytorch3d first-two-columns representation.
  * Each row is a *relative pose delta between consecutive frames* (not an
    absolute pose), expressed in the ego head-camera frame:
    x = right, y = down, z = forward (OpenCV camera convention).
  * Temporal spacing is 20 fps -> dt = 0.05 s, which equals the CARLA tick, so
    one av row is consumed per env step. Translation is physical (metres): the
    output is un-normalized (rotation_6d values are raw ~unit SO(3) columns).
The one residual assumption is the metre unit of translation; everything else is
sourced, not tuned.
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
from vla_streaming_rl.networks.cosmos_edge_network import MAX_PROMPT_TOKENS, CosmosEdgeNetwork
from vla_streaming_rl.networks.interface import InferInput
from vla_streaming_rl.reward_processor import RewardProcessor
from vla_streaming_rl.utils import create_reward_image

FRAME_HW = 256
DT = 0.05  # av frame == CARLA control tick (both 20 FPS)

# Target-speed floor (exploration: keep the car moving so RL gets a signal even
# when the untrained policy predicts ~0 forward motion) and cap (urban safety),
# in m/s. The speed itself is physical (mean forward delta / dt), not tuned.
MIN_SPEED = 2.0
MAX_SPEED = 8.0


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
        # chunk (one av row per tick) plus the bootstrap next-observation. The
        # navigation prompt (built by the env, in obs["language"]) is tokenized
        # into the buffer's prompt slot so the loss can rebuild the conditioning.
        self.rb = ReplayBuffer(
            size=buffer_size,
            seq_len=self.chunk_size + 1,
            obs_shape=(network.obs_flat_dim,),
            obs_z_shape=(1,),
            rnn_state_shape=(1,),
            action_shape=(self.action_dim,),
            output_device=self.device,
            storage_device=torch.device("cpu"),
            max_prompt_tokens=MAX_PROMPT_TOKENS,
            pad_token_id=network.pad_id,
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

        The remaining per-frame ego-camera translation deltas (x=right, y=down,
        z=forward; see module note) are summed into a lookahead target point and
        transformed to world coordinates; the target speed is the mean forward
        delta over dt. Small near-identity per-frame rotations are approximated by
        summing translations directly in the current ego frame."""
        deltas = chunk[idx:, :3].detach().float().cpu().numpy()  # (M, 3): right, down, forward
        forward_total = float(deltas[:, 2].sum())
        right_total = float(deltas[:, 0].sum())

        desired_speed = float(np.clip(deltas[:, 2].mean() / DT, MIN_SPEED, MAX_SPEED))  # m/s

        vehicle = self.env.unwrapped.vehicle
        # ego camera (x=right, z=forward) -> CARLA vehicle local (x=forward, y=right).
        local = carla.Location(x=forward_total, y=right_total, z=0.0)
        world_loc = vehicle.get_transform().transform(local)
        control = self._pid_controller().run_step(desired_speed * 3.6, _TargetWaypoint(world_loc))
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
        # The navigation prompt is built by the env and delivered in obs["language"];
        # store its token ids so the loss can rebuild the same conditioning.
        prompt = obs["language"]
        token_ids = self.network.tokenize_task_prompt(prompt)
        self.rb.add(
            self.network.pack_obs(frame).to(self.device),
            self._dummy,
            reward,
            terminated or truncated,
            self._dummy,
            self._prev_action,
            token_ids,
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
                    task_prompts=[prompt],
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
