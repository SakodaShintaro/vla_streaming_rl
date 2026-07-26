# SPDX-License-Identifier: MIT
"""Direct-control agent for the Cosmos3-Edge ``navigation`` action domain.

Every ``frame_stride`` env ticks (one model frame at ``action_fps`` Hz) the
agent stores one replay slot (frame, previously executed control row, the
reward accumulated over those ticks, done) and pops the next control row
``[steer, gas_or_brake]`` from its plan queue; the row is held for the whole
model frame. When the queue is empty it replans: the recent ``history_len``
frames + the really-executed history control rows + the navigation prompt
(obs["language"]) condition the joint generation, and the ``chunk_size``
generated future rows are queued for open-loop execution.

The action row IS the env action (no PID / trajectory conversion), clipped to
[-1, 1], so the same agent can drive other envs with the same 2D action space
by changing ``frame_stride`` to match the env's per-tick real-time duration
(CARLA ticks at 20 Hz -> frame_stride 2; Animal-AI's ``decisionPeriod=5``
already yields one env tick per 10 Hz decision -> frame_stride 1).
"""

from collections import deque
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import (
    _ACTION_RESOLUTION_BINS,
    VideoProcessor,
)
from torch import optim

from vla_streaming_rl.agents.base import Agent, StepResult
from vla_streaming_rl.networks.cosmos_edge_network import (
    MAX_PROMPT_TOKENS,
    NAVIGATION_ACTION_DIM,
    CosmosEdgeNetwork,
)
from vla_streaming_rl.networks.interface import InferInput
from vla_streaming_rl.replay_buffer import ReplayBuffer
from vla_streaming_rl.reward_processor import RewardProcessor
from vla_streaming_rl.utils import create_reward_image


class CosmosEdgeAgent(Agent):
    def __init__(
        self,
        *,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Box,
        network: CosmosEdgeNetwork,
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
        reward_processor_type: str,
    ) -> None:
        super().__init__(learning_mode=learning_mode, horizon=horizon)
        del action_space, gamma
        assert learning_mode == "off_policy", "cosmos_edge supports off_policy only"

        self.device = torch.device("cuda")
        self.network = network
        # Aspect-preserving conditioning size. The pipeline bins the *input's*
        # aspect ratio into a canvas of the resolution tier, downscales into it
        # (never upscales, ``scale = min(target/source, 1.0)``) and pads the rest
        # (``_prepare_action_video_conditioning``). Feeding frames at exactly the
        # content size for the obs aspect keeps the pipeline resize a no-op.
        _c, obs_h, obs_w = observation_space["image"].shape
        canvas_h, canvas_w = VideoProcessor.classify_height_width_bin(
            obs_h, obs_w, ratios=_ACTION_RESOLUTION_BINS[str(network.policy.resolution_tier)]
        )
        scale = min(canvas_w / obs_w, canvas_h / obs_h, 1.0)
        self.frame_h = max(1, int(scale * obs_h + 0.5))
        self.frame_w = max(1, int(scale * obs_w + 0.5))
        del observation_space

        self.chunk_size = network.chunk_size
        self.history_len = network.history_len
        self.future_start = network.future_start
        self.frame_stride = network.frame_stride
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.max_grad_norm = max_grad_norm

        # Only the navigation slot rows of the DomainAwareLinear banks receive
        # gradients, but the optimizer owns the whole embedding tensors — a
        # nonzero weight_decay would shrink every other domain's pretrained
        # rows, so the actor decay is pinned to 0 (see the network note).
        del weight_decay
        self.actor_optimizer = optim.AdamW(network.actor_parameters, lr=actor_lr, weight_decay=0.0)
        self.critic_optimizer = optim.AdamW(
            network.critic.parameters(), lr=critic_lr, weight_decay=0.0
        )
        self.reward_processor = RewardProcessor(reward_processor_type, 1.0)

        # Learn the flat-obs schema up front so the buffer can be sized.
        zero_frame = torch.zeros(3, self.frame_h, self.frame_w)
        network.pack_obs(zero_frame)

        # One replay slot per model frame (every ``frame_stride`` env ticks);
        # a sampled window of ``network.seq_len`` slots holds the conditioning
        # clip plus the executed chunk (see the network's ``_unpack_window``).
        self.rb = ReplayBuffer(
            size=buffer_size,
            seq_len=network.seq_len,
            obs_shape=(network.obs_flat_dim,),
            obs_z_shape=(1,),
            rnn_state_shape=(1,),
            # Per-slot action: the control row executed during the previous
            # model frame (the transition leading into this slot's frame).
            action_shape=(NAVIGATION_ACTION_DIM,),
            output_device=self.device,
            storage_device=torch.device("cpu"),
            max_prompt_tokens=MAX_PROMPT_TOKENS,
            pad_token_id=network.pad_id,
        )
        self._dummy = torch.zeros(1, device=self.device)

        # Rollout state (reset per episode).
        self._tick = 0
        self._reward_acc = 0.0
        self._row_queue: deque = deque()
        self._current_row = np.zeros(NAVIGATION_ACTION_DIM, dtype=np.float32)
        self._prev_row = torch.zeros(NAVIGATION_ACTION_DIM, device=self.device)
        self._frame_history: deque = deque(maxlen=self.history_len)
        self._row_history: deque = deque(maxlen=self.future_start)
        self._metrics: dict[str, float] = {}
        self._pred_image = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        self._plan_image = np.zeros((256, 256, 3), dtype=np.uint8)

    # --- required agent surface --------------------------------------------

    def _preprocess(self, obs: dict[str, Any], info: dict) -> torch.Tensor:
        del info
        image = torch.as_tensor(np.asarray(obs["image"]), dtype=torch.float32)  # (3, H, W) in [0,1]
        frame = F.interpolate(
            image[None], size=(self.frame_h, self.frame_w), mode="bilinear", align_corners=False
        )[0]
        return frame

    def _to_env_action(self, row: np.ndarray) -> np.ndarray:
        """A control row IS the env action [steer, gas_or_brake]; just clip."""
        return np.clip(row.astype(np.float32), -1.0, 1.0)

    def _replan(self, prompt: str) -> None:
        clip = list(self._frame_history)
        clip = [clip[0]] * (self.history_len - len(clip)) + clip
        rows = list(self._row_history)
        rows = [np.zeros(NAVIGATION_ACTION_DIM, dtype=np.float32)] * (
            self.future_start - len(rows)
        ) + rows
        live = self.network.infer(
            InferInput(
                s_seq=torch.stack(clip).to(self.device),
                obs_z_seq=self._dummy,
                a_seq=torch.from_numpy(np.stack(rows)).to(self.device),
                r_seq=self._dummy,
                rnn_state=self._dummy,
                task_prompts=[prompt],
            )
        )
        future = live.action.squeeze(0).float().cpu().numpy()  # (chunk_size, 2)
        self._row_queue.extend(self._to_env_action(row) for row in future)
        self._metrics = dict(live.value_report)
        self._pred_image = live.next_image
        self._plan_image = self._draw_plan(future, live.value_report["value"])

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
        self._reward_acc += reward
        prompt = obs["language"]
        boundary = self._tick % self.frame_stride == 0
        if boundary or terminated or truncated:
            frame = self._preprocess(obs, info)
            token_ids = self.network.tokenize_task_prompt(prompt)
            self.rb.add(
                self.network.pack_obs(frame).to(self.device),
                self._dummy,
                self._reward_acc,
                terminated or truncated,
                self._dummy,
                self._prev_row,
                token_ids,
            )
            self._reward_acc = 0.0
            self._frame_history.append(frame)
            self._row_history.append(self._prev_row.cpu().numpy())
            if len(self._row_queue) == 0:
                self._replan(prompt)
            self._current_row = self._row_queue.popleft()
            self._prev_row = torch.from_numpy(self._current_row).to(self.device)
        self._tick += 1

        metrics = {
            "action_norm": float(np.linalg.norm(self._current_row)),
            "processed_reward": self.reward_processor.normalize(torch.tensor(reward)).item(),
            **self._metrics,
        }
        panels = self._panels(reward)
        return StepResult(action=self._current_row.copy(), metrics=metrics, panels=panels)

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
        # Plans, histories and cadence must not cross episode boundaries.
        self._tick = 0
        self._reward_acc = 0.0
        self._row_queue.clear()
        self._current_row = np.zeros(NAVIGATION_ACTION_DIM, dtype=np.float32)
        self._prev_row = torch.zeros(NAVIGATION_ACTION_DIM, device=self.device)
        self._frame_history.clear()
        self._row_history.clear()
        return {}

    def _draw_plan(self, future: np.ndarray, q: float) -> np.ndarray:
        """Bar chart of the queued plan: steer (cyan, signed) and gas_or_brake
        (orange, signed) per future row, annotated with the critic value."""
        size = 256
        img = np.full((size, size, 3), 30, dtype=np.uint8)
        mid = size // 2
        cv2.line(img, (0, mid), (size, mid), (60, 60, 60), 1)
        num = future.shape[0]
        slot = size // max(num, 1)
        bar = max(2, slot // 3)
        for i, (steer, accel) in enumerate(future):
            x = i * slot + slot // 2
            steer_y = int(mid - np.clip(steer, -1, 1) * (size * 0.45))
            accel_y = int(mid - np.clip(accel, -1, 1) * (size * 0.45))
            cv2.line(img, (x - bar, mid), (x - bar, steer_y), (0, 200, 255), 2)
            cv2.line(img, (x + bar, mid), (x + bar, accel_y), (255, 165, 0), 2)
        q_color = (0, 255, 0) if q >= 0.0 else (255, 80, 80)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"Q: {q:.3f}", (8, 22), font, 0.6, q_color, 2)
        cv2.putText(img, "steer", (8, size - 24), font, 0.45, (0, 200, 255), 1)
        cv2.putText(img, "gas_or_brake", (8, size - 8), font, 0.45, (255, 165, 0), 1)
        return img

    def _panels(self, reward: float) -> dict[str, np.ndarray]:
        return {
            "reward": create_reward_image(None, reward),
            "prediction": self._pred_image,
            "plan": self._plan_image,
        }
