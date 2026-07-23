# SPDX-License-Identifier: MIT
"""Cosmos3-Edge av policy agent for CARLA/Bench2Drive.

Mirrors the off-policy structure of ``SimLingoAgent`` but drives the Cosmos3-Edge
av policy with **history-conditioned joint generation**. Every tick the agent feeds
a short history clip of the last ``history_len`` front-camera frames + the
navigation prompt (obs["language"]) into Cosmos3-Edge: the first few latent frames
are anchored as clean history, the rest are generated jointly with the action, so
the model outputs the future ego-pose trajectory (raw dim 9 = 3D translation + 6D
rotation per row), grounded in the observed ego-motion / velocity and steered by
the navigation prompt. The ego-frame translation is transformed to a world target
point and driven to via CARLA's ``VehiclePIDController``.

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

import math
from collections import deque
from typing import Any

import carla
import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from agents.navigation.controller import VehiclePIDController
from torch import optim

from vla_streaming_rl.agents.base import Agent, StepResult
from vla_streaming_rl.networks.cosmos_edge_network import (
    MAX_PROMPT_TOKENS,
    CosmosEdgeNetwork,
    ego_history_av_deltas,
)
from vla_streaming_rl.networks.interface import InferInput
from vla_streaming_rl.replay_buffer import ReplayBuffer
from vla_streaming_rl.reward_processor import RewardProcessor
from vla_streaming_rl.utils import create_reward_image

FRAME_HW = 256
DT = 0.05  # av frame == CARLA control tick (both 20 FPS)

# Target-speed floor (exploration: keep the car moving so RL gets a signal even
# when the untrained policy predicts ~0 forward motion) and cap (urban safety),
# in m/s. The speed itself is physical (mean forward delta / dt), not tuned.
MIN_SPEED = 2.0
MAX_SPEED = 8.0

# The PID target is the net ego displacement over the *generated future* rows
# (``future_start`` onward, i.e. the farthest predicted future frame). Earlier rows
# connect the given history frames and are not steered along.


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
        # History-conditioned joint generation: the clip is ``history_len`` recent
        # frames; the action rows from ``future_start`` on are the generated future
        # (the forward plan we steer along).
        self.history_len = network.history_len
        self.future_start = network.future_start
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

        # seq_len = history_len + 1: the model conditions on a history clip of
        # ``history_len`` consecutive frames, so a sampled window holds that clip plus
        # the one-step-shifted bootstrap (next) clip. Per-step the buffer stores a
        # single frame; consecutive stored frames reconstruct the clip. The stored
        # action is the whole joint trajectory flattened (chunk_size * action_dim);
        # the critic scores it as one action with a single-step reward window. The
        # navigation prompt (in obs["language"]) is tokenized into the prompt slot so
        # the loss can rebuild the conditioning.
        self._clip_len = self.history_len
        self.rb = ReplayBuffer(
            size=buffer_size,
            seq_len=self.history_len + 1,
            obs_shape=(network.obs_flat_dim,),
            obs_z_shape=(3,),  # ego pose (x, y, yaw) -> real history action deltas at loss time
            rnn_state_shape=(1,),
            action_shape=(self.chunk_size * self.action_dim,),
            output_device=self.device,
            storage_device=torch.device("cpu"),
            max_prompt_tokens=MAX_PROMPT_TOKENS,
            pad_token_id=network.pad_id,
        )
        self._dummy = torch.zeros(1, device=self.device)

        # The previous tick's flattened trajectory, stored as the transition's action.
        self._prev_action = torch.zeros(self.chunk_size * self.action_dim, device=self.device)
        # Recent-frame + ego-pose history for the rollout clip (reset per episode). The
        # poses give the real history action rows that anchor the joint generation.
        self._frame_history: deque = deque(maxlen=self._clip_len)
        self._pose_history: deque = deque(maxlen=self._clip_len)

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

    def _to_env_action(self, trajectory: torch.Tensor):
        """av ego-pose trajectory -> (steer, gas_or_brake) via CARLA's VehiclePIDController.

        Only the *generated future* rows (``future_start`` onward) are used — the
        earlier rows connect the given history frames. Their per-frame ego-camera
        translation deltas (x=right, y=down, z=forward; see module note) are summed
        into the lookahead target point (the net ego displacement to the farthest
        predicted future frame) and transformed to world coordinates; the target
        speed is the mean forward delta over dt. Small near-identity per-frame
        rotations are approximated by summing translations in the current ego frame."""
        deltas = (
            trajectory[self.future_start :, :3].detach().float().cpu().numpy()
        )  # (future_len, 3): right, down, forward
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
        # Current ego pose (world x, y, yaw) -> stored so the loss can rebuild the real
        # history action rows; also drives the rollout history action below.
        tf = self.env.unwrapped.vehicle.get_transform()
        pose = torch.tensor(
            [tf.location.x, tf.location.y, math.radians(tf.rotation.yaw)],
            device=self.device,
            dtype=torch.float32,
        )
        self.rb.add(
            self.network.pack_obs(frame).to(self.device),
            pose,
            reward,
            terminated or truncated,
            self._dummy,
            self._prev_action,
            token_ids,
        )

        # Feed the recent-frame history clip + the real history action rows (from the
        # ego poses) to the joint generation, re-planned every tick. At episode start
        # the history is padded at the front with its oldest entry.
        self._frame_history.append(frame)
        self._pose_history.append(pose.cpu().numpy())
        clip = list(self._frame_history)
        clip = [clip[0]] * (self._clip_len - len(clip)) + clip
        poses = list(self._pose_history)
        poses = [poses[0]] * (self._clip_len - len(poses)) + poses
        history_action = torch.from_numpy(ego_history_av_deltas(np.stack(poses))).to(self.device)
        live = self.network.infer(
            InferInput(
                s_seq=torch.stack(clip).to(self.device),
                obs_z_seq=self._dummy,
                a_seq=history_action,
                r_seq=self._dummy,
                rnn_state=self._dummy,
                task_prompts=[prompt],
            )
        )
        trajectory = live.action.squeeze(0)  # (chunk, 9)
        self._prev_action = trajectory.reshape(-1)
        env_action = self._to_env_action(trajectory)

        metrics = {
            "action_norm": float(np.linalg.norm(env_action)),
            "processed_reward": self.reward_processor.normalize(torch.tensor(reward)).item(),
            **live.value_report,
        }
        panels = self._panels(reward, trajectory, live.value_report["value"], live.next_image)
        return StepResult(action=env_action, metrics=metrics, panels=panels)

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
        self._controller = None  # rebuilt against the next episode's vehicle
        self._frame_history.clear()  # history must not cross episode boundaries
        self._pose_history.clear()
        return {}

    def _panels(
        self, reward: float, trajectory: torch.Tensor, q: float, pred_image: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Reward panel, the Cosmos world model's predicted future frame, and a
        top-down (ego-frame) view of the predicted av trajectory annotated with the
        critic's value estimate Q(s, a).

        The per-row ego-camera deltas (x=right, z=forward; see module note) are
        accumulated into cumulative positions, re-centered on the current frame so the
        ego marker is the origin; forward -> up, right -> right. The history rows fall
        behind the ego and are dim; the generated future rows (``future_start`` onward)
        are cyan ahead of it, and the farthest future point actually fed to the PID is
        highlighted in orange. Ego is the green marker."""
        deltas = (
            trajectory[:, :3].detach().float().cpu().numpy()
        )  # (chunk, 3): right, down, forward
        positions = np.cumsum(deltas, axis=0)  # cumulative positions from the oldest frame
        # Re-center on the CURRENT frame (end of the history rows) so the ego marker
        # sits at the origin: history rows fall behind it, generated future ahead.
        positions = positions - positions[self.future_start - 1]

        size = 256
        scale = 6.0  # pixels per meter
        img = np.full((size, size, 3), 30, dtype=np.uint8)
        ego_px = (size // 2, int(size * 0.8))

        def to_px(forward: float, right: float) -> tuple[int, int]:
            return (int(ego_px[0] + right * scale), int(ego_px[1] - forward * scale))

        # range rings every 10 m for scale reference
        for r_m in (10, 20, 30):
            cv2.circle(img, ego_px, int(r_m * scale), (60, 60, 60), 1)

        hist_color = (90, 90, 90)
        traj_color = (0, 200, 255)
        look_color = (255, 165, 0)
        pts = [to_px(float(p[2]), float(p[0])) for p in positions]
        for i in range(len(pts) - 1):
            color = hist_color if i < self.future_start else traj_color
            cv2.line(img, pts[i], pts[i + 1], color, 2)
        for i, p in enumerate(pts):
            cv2.circle(img, p, 2, hist_color if i < self.future_start else traj_color, -1)
        cv2.circle(img, pts[-1], 5, look_color, -1)

        # ego marker pointing forward (up)
        cv2.drawMarker(img, ego_px, (0, 255, 0), cv2.MARKER_TRIANGLE_UP, markerSize=12, thickness=2)

        # value annotation (green if non-negative, red otherwise) + legend
        q_color = (0, 255, 0) if q >= 0.0 else (255, 80, 80)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"Q: {q:.3f}", (8, 22), font, 0.6, q_color, 2)
        cv2.putText(img, "trajectory", (8, size - 24), font, 0.45, traj_color, 1)
        cv2.putText(img, "lookahead", (8, size - 8), font, 0.45, look_color, 1)

        return {
            "reward": create_reward_image(None, reward),
            "prediction": pred_image,
            "bev_value": img,
        }
