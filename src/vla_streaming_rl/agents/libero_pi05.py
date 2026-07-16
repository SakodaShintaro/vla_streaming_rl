# SPDX-License-Identifier: MIT
import collections
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from torch import nn, optim

from vla_streaming_rl.agents.base import Agent, StepResult
from vla_streaming_rl.networks.interface import InferInput
from vla_streaming_rl.networks.libero_pi05_network import (
    OBS_IMAGE_AGENTVIEW,
    OBS_IMAGE_WRIST,
    OBS_STATE,
    TASK_KEY,
    LiberoPi05Network,
)
from vla_streaming_rl.networks.vlac import VlacRewardRelabeler
from vla_streaming_rl.optimizers.adam_et import AdamET
from vla_streaming_rl.replay_buffer import ReplayBuffer
from vla_streaming_rl.reward_processor import RewardProcessor


def _sample_evenly(frames: list, n: int) -> list:
    """Sample ``n`` evenly-spaced frames (first + n-1 spaced), as VLAC's reference."""
    delta = (len(frames) - 1) / (n - 1)
    return [frames[0]] + [frames[int(i * delta)] for i in range(1, n)]


def _reward_image(
    total: float,
    sparse: float,
    reach_shaping: float,
    place_shaping: float,
) -> np.ndarray:
    """Fixed 200x200 panel breaking down the reward components."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"Total: {total:+.3f}", (8, 28), font, 0.5, (255, 255, 255), 2)
    cv2.putText(img, f"Sparse:{sparse:.3f}", (8, 56), font, 0.5, (255, 0, 0), 2)
    cv2.putText(img, f"Reach: {reach_shaping:+.4f}", (8, 84), font, 0.5, (0, 200, 255), 2)
    cv2.putText(img, f"Place: {place_shaping:+.4f}", (8, 112), font, 0.5, (0, 255, 0), 2)
    return img


class LiberoPi05Agent(Agent):
    def __init__(
        self,
        *,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Box,
        network: LiberoPi05Network,
        learning_mode: str,
        horizon: int,
        buffer_size: int,
        batch_size: int,
        learning_starts: int,
        actor_lr: float,
        critic_lr: float,
        weight_decay: float,
        max_grad_norm: float,
        et_lambda: float,
        gamma: float,
        relabeler: VlacRewardRelabeler | None,
        vlac_ref_num: int,
        reach_reward_scale: float,
    ) -> None:
        super().__init__(learning_mode=learning_mode, horizon=horizon)

        self.network = network
        self.preprocessor = network.preprocessor
        self.postprocessor = network.postprocessor
        self.device = torch.device(network.cfg.device)
        self.chunk_size = network.chunk_size
        self.action_dim = network.action_dim
        # A training transition spans the chunk plus the bootstrap state.
        self._seq_len = self.chunk_size + 1

        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.max_grad_norm = float(max_grad_norm)
        # Coefficient on the env's unscaled reach/place shaping signal (from
        # info); the scaled term is folded into the stored learning reward.
        self._reach_reward_scale = float(reach_reward_scale)
        self.reward_processor = RewardProcessor("none", 1.0)

        self._action_low = action_space.low
        self._action_high = action_space.high

        self.actor_optimizer = optim.AdamW(
            network.actor_parameters, lr=actor_lr, weight_decay=weight_decay
        )
        if learning_mode == "streaming":
            self.critic_optimizer = AdamET(
                network.critic.parameters(), lr=critic_lr, gamma=gamma, et_lambda=et_lambda
            )
        else:
            self.critic_optimizer = optim.AdamW(
                network.critic.parameters(), lr=critic_lr, weight_decay=weight_decay
            )

        # Dummy tensor for the InferInput / ReplayBuffer slots pi0.5 does not use
        # (recurrent state, encoded obs, log-probs, token ids …): pi0.5's
        # multimodal observation rides the packed obs slot instead.
        self._dummy = torch.zeros(1, device=self.device)

        # The replay buffer is created lazily on the first store, once the packed
        # observation width is known from a real preprocessed batch (the network
        # owns the pack/unpack schema, exposed as ``network.obs_flat_dim``).
        self.rb: ReplayBuffer | None = None

        # Open-loop chunk execution: the env actions to play and the matching
        # normalized actions to store, filled together when a chunk is planned.
        self._env_action_queue: collections.deque = collections.deque()
        self._raw_action_queue: collections.deque = collections.deque()
        # ``action_{t-1}`` (normalized) stored with state t; advanced to the action
        # executed this step after acting (project buffer convention).
        self._prev_action = torch.zeros(self.action_dim, device=self.device)
        self._episode_reset = False

        del observation_space

        # VLAC online dense reward: scored per env step against a key frame with a
        # one-shot in-context reference (the best episode so far, sampled to
        # ``vlac_ref_num`` frames), added to that step's stored reward. Disabled
        # when relabeler is None.
        self._relabeler = relabeler
        self._vlac_ref_num = vlac_ref_num
        self._cur_frames: list[Image.Image] = []
        self._cur_task = ""
        self._references: dict[str, list[Image.Image]] = {}
        self._best: dict[str, tuple[float, int]] = {}

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
        packed, batch, wrist = self._preprocess(obs, info)
        if self.rb is None:
            self.rb = ReplayBuffer(
                size=self.buffer_size,
                seq_len=self._seq_len,
                obs_shape=(self.network.obs_flat_dim,),
                obs_z_shape=(1,),
                rnn_state_shape=(1,),
                action_shape=(self.action_dim,),
                velocity_shape=(1,),
                output_device=self.device,
                storage_device=torch.device("cpu"),
                max_new_tokens=0,
                max_prompt_tokens=0,
                pad_token_id=0,
            )
        metrics = {}
        dense = 0.0
        sparse = float(reward)
        if self._relabeler is not None:
            frame = Image.fromarray(
                (np.transpose(obs["image"], (1, 2, 0)) * 255.0).astype(np.uint8)
            )
            if not self._cur_frames:
                self._cur_task = obs["language"]
                self._relabeler.set_reference(self._references.get(self._cur_task))
            self._cur_frames.append(frame)
            dense, metrics = self._relabeler.step(frame, obs["language"])
        reach_shaping = self._reach_reward_scale * info.get("reach_shaping", 0.0)
        place_shaping = self._reach_reward_scale * info.get("place_shaping", 0.0)
        reward = sparse + dense + reach_shaping + place_shaping
        self.rb.add(
            packed,
            self._dummy,
            reward,
            terminated or truncated,
            self._dummy,
            self._prev_action,
            [],
            [],
            self._dummy,
        )
        if terminated or truncated:
            self._episode_reset = True
        metrics.update(
            self._reward_metrics(reward, sparse, reach_shaping, place_shaping, dense, info)
        )
        panels = self._panels(reward, wrist, sparse, reach_shaping, place_shaping)

        if not self._env_action_queue:
            curr_size = self.rb.size if self.rb.full else self.rb.idx
            if curr_size >= self._seq_len:
                data = self.rb.get_latest(self._seq_len)
                data.rewards = self.reward_processor.normalize(data.rewards)
                result = self.network.infer_and_compute_loss(data)
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                result.et_info.actor_entropy_loss.backward(retain_graph=True)
                result.et_info.neg_value.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step(delta=result.et_info.delta, reset=self._episode_reset)
                self._episode_reset = False
                metrics.update(result.loss_result.info)
                infer_result = result.infer_result
            else:
                with torch.no_grad():
                    infer_result = self.network.infer(
                        InferInput(
                            s_seq=batch,
                            obs_z_seq=self._dummy,
                            a_seq=self._dummy,
                            r_seq=self._dummy,
                            rnn_state=self._dummy,
                            task_prompts=[],
                            velocity_seq=self._dummy,
                        )
                    )
            metrics.update(infer_result.value_report)
            raw_chunk = infer_result.action.squeeze(0)
            env_chunk = self.postprocessor(infer_result.action).squeeze(0).float().cpu().numpy()
            self._env_action_queue.extend(env_chunk)
            self._raw_action_queue.extend(raw_chunk)

        self._prev_action = self._raw_action_queue.popleft()
        env_action = self._to_env_action(self._env_action_queue.popleft())
        return StepResult(action=env_action, metrics=metrics, panels=panels)

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        del feedback_text
        # Drop any partially-executed chunk so the next episode plans fresh.
        self._env_action_queue.clear()
        self._raw_action_queue.clear()
        if self._relabeler is not None:
            length = len(self._cur_frames)
            best = self._best.get(self._cur_task)
            if length >= self._vlac_ref_num and (
                best is None or score > best[0] or (score == best[0] and length < best[1])
            ):
                self._best[self._cur_task] = (score, length)
                self._references[self._cur_task] = _sample_evenly(
                    self._cur_frames, self._vlac_ref_num
                )
            self._cur_frames = []
            self._relabeler.reset()
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
        packed, batch, wrist = self._preprocess(obs, info)
        if self.rb is None:
            self.rb = ReplayBuffer(
                size=self.buffer_size,
                seq_len=self._seq_len,
                obs_shape=(self.network.obs_flat_dim,),
                obs_z_shape=(1,),
                rnn_state_shape=(1,),
                action_shape=(self.action_dim,),
                velocity_shape=(1,),
                output_device=self.device,
                storage_device=torch.device("cpu"),
                max_new_tokens=0,
                max_prompt_tokens=0,
                pad_token_id=0,
            )
        metrics = {}
        dense = 0.0
        sparse = float(reward)
        if self._relabeler is not None:
            frame = Image.fromarray(
                (np.transpose(obs["image"], (1, 2, 0)) * 255.0).astype(np.uint8)
            )
            if not self._cur_frames:
                self._cur_task = obs["language"]
                self._relabeler.set_reference(self._references.get(self._cur_task))
            self._cur_frames.append(frame)
            dense, metrics = self._relabeler.step(frame, obs["language"])
        reach_shaping = self._reach_reward_scale * info.get("reach_shaping", 0.0)
        place_shaping = self._reach_reward_scale * info.get("place_shaping", 0.0)
        reward = sparse + dense + reach_shaping + place_shaping
        self.rb.add(
            packed,
            self._dummy,
            reward,
            terminated or truncated,
            self._dummy,
            self._prev_action,
            [],
            [],
            self._dummy,
        )

        if not self._env_action_queue:
            infer_result = self.network.infer(
                InferInput(
                    s_seq=batch,
                    obs_z_seq=self._dummy,
                    a_seq=self._dummy,
                    r_seq=self._dummy,
                    rnn_state=self._dummy,
                    task_prompts=[],
                    velocity_seq=self._dummy,
                )
            )
            metrics.update(infer_result.value_report)
            raw_chunk = infer_result.action.squeeze(0)
            env_chunk = self.postprocessor(infer_result.action).squeeze(0).float().cpu().numpy()
            self._env_action_queue.extend(env_chunk)
            self._raw_action_queue.extend(raw_chunk)

        self._prev_action = self._raw_action_queue.popleft()
        env_action = self._to_env_action(self._env_action_queue.popleft())
        metrics.update(
            self._reward_metrics(reward, sparse, reach_shaping, place_shaping, dense, info)
        )
        return StepResult(
            action=env_action,
            metrics=metrics,
            panels=self._panels(reward, wrist, sparse, reach_shaping, place_shaping),
        )

    def _preprocess(self, obs: dict[str, Any], info: dict) -> tuple[torch.Tensor, dict, np.ndarray]:
        """Build pi0.5's raw multimodal observation, run the (normalizing,
        tokenizing) preprocessor, and return the packed obs vector for the replay
        buffer together with the preprocessed batch (for inference) and the wrist
        frame (for the render panel)."""
        del info
        agentview = torch.from_numpy((obs["image"] * 255.0).astype(np.uint8)).float() / 255.0
        wrist_uint8 = obs["wrist_image"].copy()
        wrist = torch.from_numpy(wrist_uint8).permute(2, 0, 1).float() / 255.0
        raw_obs = {
            OBS_IMAGE_AGENTVIEW: agentview,
            OBS_IMAGE_WRIST: wrist,
            OBS_STATE: torch.from_numpy(obs["proprio"]).float(),
            TASK_KEY: obs["language"],
        }
        batch = self.preprocessor(raw_obs)
        return self.network.pack_obs(batch), batch, wrist_uint8

    def _to_env_action(self, net_action: np.ndarray) -> np.ndarray:
        """Clip an un-normalized pi0.5 action into the env's action space."""
        return np.clip(net_action, self._action_low, self._action_high).astype(np.float32)

    def _reward_metrics(
        self,
        total: float,
        sparse: float,
        reach_shaping: float,
        place_shaping: float,
        dense: float,
        info: dict,
    ) -> dict:
        return {
            "reward/total": float(total),
            "reward/sparse": float(sparse),
            "reward/reach_shaping": float(reach_shaping),
            "reward/place_shaping": float(place_shaping),
            "reward/vlac_dense": float(dense),
            "reach_dist": info.get("reach_dist", 0.0),
            "place_dist": info.get("place_dist", 0.0),
        }

    def _panels(
        self,
        total: float,
        wrist: np.ndarray,
        sparse: float,
        reach_shaping: float,
        place_shaping: float,
    ) -> dict:
        reward_img = _reward_image(total, sparse, reach_shaping, place_shaping)
        return {"wrist": wrist, "reward": reward_img}
