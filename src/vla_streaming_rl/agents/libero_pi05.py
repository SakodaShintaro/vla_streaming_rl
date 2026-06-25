# SPDX-License-Identifier: MIT
import collections

import cv2
import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from torch import nn, optim

from vla_streaming_rl.agents.step_result import StepResult
from vla_streaming_rl.networks.interface import InferInput
from vla_streaming_rl.networks.libero_pi05_network import (
    OBS_IMAGE_AGENTVIEW,
    OBS_IMAGE_WRIST,
    OBS_STATE,
    TASK_KEY,
    LiberoPi05Network,
)
from vla_streaming_rl.networks.vlac import VlacRewardRelabeler
from vla_streaming_rl.replay_buffer import ReplayBuffer, ReplayBufferData


def _build_obs_schema(batch: dict) -> tuple[list, int]:
    """Record the (key, per-sample shape, dtype) of every tensor in a preprocessed
    pi0.5 batch (B == 1), plus the total flat length. Used to pack the multimodal
    observation into the project's tensor ``ReplayBuffer`` and unpack it back."""
    schema = []
    for key in sorted(batch.keys()):
        value = batch[key]
        if isinstance(value, torch.Tensor):
            schema.append((key, tuple(value.shape[1:]), value.dtype))
    flat_dim = sum(int(np.prod(shape)) for _, shape, _ in schema)
    return schema, flat_dim


def _pack_obs(batch: dict, schema: list) -> torch.Tensor:
    """Flatten a single (B == 1) preprocessed pi0.5 batch into one float vector."""
    return torch.cat([batch[key].reshape(-1).to(torch.float32) for key, _, _ in schema])


def _unpack_obs(flat: torch.Tensor, schema: list) -> dict:
    """Inverse of :func:`_pack_obs` for a batch of packed rows ``(B, flat_dim)`` →
    a preprocessed pi0.5 batch of ``(B, *shape)`` tensors (dtypes restored)."""
    batch_size = flat.shape[0]
    batch = {}
    offset = 0
    for key, shape, dtype in schema:
        n = int(np.prod(shape))
        batch[key] = flat[:, offset : offset + n].reshape(batch_size, *shape).to(dtype)
        offset += n
    return batch


def _window_to_loss_input(data: ReplayBufferData, schema: list) -> ReplayBufferData:
    """Reshape a sampled window ``(B, chunk_size + 1, ...)`` into the inputs the
    pi0.5 network's loss reads: ``observations = [cur_batch, next_batch]`` (the
    window's first / last observation, unpacked), ``actions`` the executed
    normalized chunk ``a_t..a_{t+H-1}`` (rows 1..H), and ``rewards`` / ``dones``
    the per-step chunk vectors over the same rows."""
    cur_batch = _unpack_obs(data.observations[:, 0], schema)
    next_batch = _unpack_obs(data.observations[:, -1], schema)
    return ReplayBufferData(
        observations=[cur_batch, next_batch],
        actions=data.actions[:, 1:],
        rewards=data.rewards[:, 1:, 0],
        dones=data.dones[:, 1:, 0],
        obs_z=data.obs_z,
        rnn_state=data.rnn_state,
        log_probs=data.log_probs,
        values=data.values,
        action_token_ids=data.action_token_ids,
        task_prompt_token_ids=data.task_prompt_token_ids,
    )


def _sample_evenly(frames: list, n: int) -> list:
    """Sample ``n`` evenly-spaced frames (first + n-1 spaced), as VLAC's reference."""
    delta = (len(frames) - 1) / (n - 1)
    return [frames[0]] + [frames[int(i * delta)] for i in range(1, n)]


def _vlac_reward_image(sparse_reward: float, dense_reward: float, progress: float) -> np.ndarray:
    """Fixed 200x200 panel: env sparse reward, VLAC dense (pseudo) reward, progress."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"Env rew:  {sparse_reward:.3f}", (8, 40), font, 0.6, (255, 0, 0), 2)
    cv2.putText(img, f"VLAC rew: {dense_reward:+.4f}", (8, 80), font, 0.6, (0, 200, 255), 2)
    cv2.putText(img, f"Progress: {progress:.3f}", (8, 120), font, 0.6, (0, 255, 0), 2)
    return img


class LiberoPi05Agent:
    def __init__(
        self,
        *,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        network: LiberoPi05Network,
        learning_mode: str,
        buffer_size: int,
        batch_size: int,
        learning_starts: int,
        actor_lr: float,
        critic_lr: float,
        max_grad_norm: float,
        et_lambda: float,
        gamma: float,
        relabeler: VlacRewardRelabeler | None,
        vlac_ref_num: int,
    ) -> None:
        if learning_mode not in ("off_policy", "streaming"):
            raise ValueError(f"Unknown learning_mode: {learning_mode!r}")
        self.learning_mode = learning_mode

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

        self._action_low = action_space.low
        self._action_high = action_space.high

        # Separate optimizers for the policy μ (action expert) and the injected
        # critic. The critic uses AdamET (eligibility traces) in streaming mode,
        # AdamW for off-policy; the actor is always AdamW with a tiny LR so
        # fine-tuning the expert does not wash out the pretrained flow.
        from vla_streaming_rl.optimizers.adam_et import AdamET

        self.actor_optimizer = optim.AdamW(network.actor_parameters, lr=actor_lr, weight_decay=0.0)
        if learning_mode == "streaming":
            self.critic_optimizer = AdamET(
                network.critic.parameters(), lr=critic_lr, gamma=gamma, et_lambda=et_lambda
            )
        else:
            self.critic_optimizer = optim.AdamW(
                network.critic.parameters(), lr=critic_lr, weight_decay=0.0
            )

        # Dummy tensor for the InferInput / ReplayBuffer slots pi0.5 does not use
        # (recurrent state, encoded obs, log-probs, token ids …): pi0.5's
        # multimodal observation rides the packed obs slot instead.
        self._dummy = torch.zeros(1, device=self.device)

        # The replay buffer is created lazily on the first store, once the packed
        # observation width is known from a real preprocessed batch.
        self.rb: ReplayBuffer | None = None
        self._obs_schema: list | None = None

        # Open-loop chunk execution: the env actions to play and the matching
        # normalized actions to store, filled together when a chunk is planned.
        self._env_queue: collections.deque = collections.deque()
        self._norm_queue: collections.deque = collections.deque()
        # ``action_{t-1}`` (normalized) stored with state t; advanced to the action
        # executed this step after acting (project buffer convention).
        self._prev_action = torch.zeros(self.action_dim, device=self.device)
        self._current_action_taken = torch.zeros(self.action_dim, device=self.device)

        # Latest preprocessed batch (cached by ``_preprocess`` for ``_act``) and
        # the wrist frame backing the render panel.
        self._current_batch: dict | None = None
        self._last_wrist = np.zeros(observation_space.shape, dtype=np.uint8)
        # Latest critic read-out, for the telemetry path.
        self._value_report: dict[str, float] = {"value": 0.0}

        # VLAC online dense reward: scored per env step against a key frame with a
        # one-shot in-context reference (the best episode so far, sampled to
        # ``vlac_ref_num`` frames), added to that step's stored reward. Disabled
        # when relabeler is None.
        self._relabeler = relabeler
        self._vlac_ref_num = vlac_ref_num
        self._last_dense = 0.0
        self._last_progress = 0.0
        self._cur_frames: list[Image.Image] = []
        self._cur_task = ""
        self._references: dict[str, list[Image.Image]] = {}
        self._best: dict[str, tuple[float, int]] = {}

    # --- agent surface -----------------------------------------------------

    def select_action(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
        info: dict,
    ) -> StepResult:
        metrics = self._store_transition(obs, reward, terminated, truncated, task_prompt, info)
        action, act_metrics = self._act(global_step, obs, task_prompt, info)
        metrics.update(act_metrics)
        self._prev_action = self._current_action_taken
        return StepResult(action=action, metrics=metrics, panels=self._panels(obs, reward))

    def step(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
        info: dict,
    ) -> StepResult:
        if self.learning_mode == "off_policy":
            return self._step_offpolicy(
                global_step, obs, reward, terminated, truncated, task_prompt, info
            )
        return self._step_streaming(
            global_step, obs, reward, terminated, truncated, task_prompt, info
        )

    def _step_offpolicy(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
        info: dict,
    ) -> StepResult:
        train_metrics = self._train_offpolicy(global_step)
        result = self.select_action(
            global_step, obs, reward, terminated, truncated, task_prompt, info
        )
        result.metrics.update(train_metrics)
        return result

    def _step_streaming(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
        info: dict,
    ) -> StepResult:
        metrics = self._store_transition(obs, reward, terminated, truncated, task_prompt, info)
        action, act_metrics = self._act(global_step, obs, task_prompt, info)
        metrics.update(act_metrics)
        self._prev_action = self._current_action_taken

        # Online TD(λ): train on the latest chunk-window once one exists.
        curr_size = self.rb.size if self.rb.full else self.rb.idx
        if curr_size >= self._seq_len:
            self.network.policy.train()
            data = _window_to_loss_input(self.rb.get_latest(self._seq_len), self._obs_schema)
            result = self.network.infer_and_compute_loss(data)
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            result.et_info.actor_entropy_loss.backward(retain_graph=True)
            result.et_info.neg_value.backward()
            nn.utils.clip_grad_norm_(self.network.actor_parameters, self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.network.critic.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step(delta=result.et_info.delta, reset=terminated or truncated)
            metrics.update(result.loss_result.info)

        return StepResult(action=action, metrics=metrics, panels=self._panels(obs, reward))

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        del feedback_text
        # Drop any partially-executed chunk so the next episode plans fresh.
        self._env_queue.clear()
        self._norm_queue.clear()
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

    # --- training ----------------------------------------------------------

    def _train_offpolicy(self, global_step: int) -> dict:
        if self.rb is None or global_step < self.learning_starts:
            return {}
        curr_size = self.rb.size if self.rb.full else self.rb.idx
        if curr_size <= self._seq_len or curr_size < self.batch_size:
            return {}

        self.network.policy.train()
        data = _window_to_loss_input(self.rb.sample(self.batch_size), self._obs_schema)
        result = self.network.compute_loss(data)

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        actor_grad = nn.utils.clip_grad_norm_(self.network.actor_parameters, self.max_grad_norm)
        critic_grad = nn.utils.clip_grad_norm_(self.network.critic.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        return {
            **result.info,
            "losses/actor_grad_norm": float(actor_grad),
            "losses/critic_grad_norm": float(critic_grad),
        }

    # --- per-tick machinery ------------------------------------------------

    def _store_transition(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
        info: dict,
    ) -> dict:
        """Preprocess + pack the current observation and store one per-env-step
        transition ``(s_t, a_{t-1}, r_t, done_t)``. Creates the replay buffer on
        the first call, once the packed-observation width is known."""
        del task_prompt
        packed = self._preprocess(obs, info, task_prompt=info["task_prompt"])
        if self.rb is None:
            self._obs_schema, flat_dim = _build_obs_schema(self._current_batch)
            self.rb = ReplayBuffer(
                size=self.buffer_size,
                seq_len=self._seq_len,
                obs_shape=(flat_dim,),
                obs_z_shape=(1,),
                rnn_state_shape=(1,),
                action_shape=(self.action_dim,),
                output_device=self.device,
                storage_device=torch.device("cpu"),
                max_new_tokens=0,
                max_prompt_tokens=0,
                pad_token_id=0,
            )
        metrics = {}
        if self._relabeler is not None:
            frame = Image.fromarray((np.transpose(obs, (1, 2, 0)) * 255.0).astype(np.uint8))
            if not self._cur_frames:
                self._cur_task = info["task_prompt"]
                self._relabeler.set_reference(self._references.get(self._cur_task))
            self._cur_frames.append(frame)
            dense, metrics = self._relabeler.step(frame, info["task_prompt"])
            reward = reward + dense
            self._last_dense = dense
            self._last_progress = metrics["vlac/progress"]
        self.rb.add(
            packed,
            self._dummy,
            reward,
            terminated or truncated,
            self._dummy,
            self._prev_action,
            0.0,
            0.0,
            [],
            [],
        )
        return metrics

    @torch.no_grad()
    def _act(
        self, global_step: int, obs: np.ndarray, task_prompt: str, info: dict
    ) -> tuple[np.ndarray, dict]:
        """Open-loop chunk execution: plan a fresh chunk when the queue is empty
        (pi0.5 inference → normalized chunk + un-normalized env chunk), then pop
        the next action. Caches the executed *normalized* action for the buffer."""
        del global_step, obs, task_prompt, info
        if not self._env_queue:
            infer_result = self.network.infer(
                InferInput(
                    s_seq=self._current_batch,
                    obs_z_seq=self._dummy,
                    a_seq=self._dummy,
                    r_seq=self._dummy,
                    rnn_state=self._dummy,
                    task_prompts=[],
                )
            )
            self._value_report = infer_result.value_report
            norm_chunk = infer_result.action.squeeze(0)  # (chunk_size, action_dim), normalized
            env_chunk = self.postprocessor(infer_result.action).squeeze(0).float().cpu().numpy()
            self._env_queue.extend(env_chunk)
            self._norm_queue.extend(norm_chunk)

        self._current_action_taken = self._norm_queue.popleft()
        env_action = self._to_env_action(self._env_queue.popleft())
        return env_action, dict(self._value_report)

    def _preprocess(self, obs: np.ndarray, info: dict, task_prompt: str) -> torch.Tensor:
        """Build pi0.5's raw multimodal observation, run the (normalizing,
        tokenizing) preprocessor, cache the batch for ``_act`` and the wrist frame
        for the panel, and return the packed obs vector for the replay buffer."""
        agentview = torch.from_numpy((obs * 255.0).astype(np.uint8)).float() / 255.0
        wrist_uint8 = info["wrist_image"].copy()
        self._last_wrist = wrist_uint8
        wrist = torch.from_numpy(wrist_uint8).permute(2, 0, 1).float() / 255.0
        raw_obs = {
            OBS_IMAGE_AGENTVIEW: agentview,
            OBS_IMAGE_WRIST: wrist,
            OBS_STATE: torch.from_numpy(info["proprio"]).float(),
            TASK_KEY: task_prompt,
        }
        batch = self.preprocessor(raw_obs)
        self._current_batch = batch
        if self._obs_schema is None:
            schema, _ = _build_obs_schema(batch)
            return _pack_obs(batch, schema)
        return _pack_obs(batch, self._obs_schema)

    def _to_env_action(self, net_action: np.ndarray) -> np.ndarray:
        """Clip an un-normalized pi0.5 action into the env's action space."""
        return np.clip(net_action, self._action_low, self._action_high).astype(np.float32)

    def _panels(self, obs: np.ndarray, reward: float) -> dict:
        del obs
        panels = {"wrist": self._last_wrist}
        if self._relabeler is not None:
            panels["reward"] = _vlac_reward_image(reward, self._last_dense, self._last_progress)
        return panels
