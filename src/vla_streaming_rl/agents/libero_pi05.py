# SPDX-License-Identifier: MIT
"""Online return-weighted fine-tuning of pi0.5 on LIBERO.

This agent runs LeRobot's pi0.5 flow-matching VLA inside the repo's online RL
loop and fine-tunes it with **return-weighted regression (RWR)** — the
critic-free member of the advantage-weighted-regression family, a natural first
step before a Q-critic is added (mirroring the ``awr`` idea in
``SimLingoAgent``).

How it works each episode:
  - pi0.5 predicts an action *chunk* (``chunk_size`` steps); the agent executes
    it open-loop, one action per env step, then re-plans. Exploration noise is
    added to the executed actions.
  - Each completed chunk is stored with the observation that produced it, the
    executed actions, and the reward accumulated while it ran.
  - At episode end every stored chunk gets a discounted return-to-go; chunks are
    pushed to a replay buffer with a softmax weight over the per-batch
    advantages.

Training re-runs pi0.5's per-sample flow-matching loss on sampled chunks and
weights it by those advantages, so the (frozen-backbone) action expert is pulled
toward the actions of high-return chunks.

The wrist camera, proprioceptive state and language instruction that pi0.5 needs
but the single-RGB observation slot cannot carry are read from the env (see
``LiberoEnv``); the agentview image arrives through the normal ``obs`` argument.
"""

import collections

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

from vla_streaming_rl.agents.step_result import StepResult
from vla_streaming_rl.networks.libero_pi05_network import (
    ACTION_KEY,
    OBS_IMAGE_AGENTVIEW,
    OBS_IMAGE_WRIST,
    OBS_STATE,
    TASK_KEY,
    LiberoPi05Network,
)


class _ChunkRecord:
    """One executed action chunk and the observation that produced it."""

    def __init__(
        self,
        agentview_uint8: np.ndarray,
        wrist_uint8: np.ndarray,
        proprio: np.ndarray,
        instruction: str,
        actions: np.ndarray,
        reward: float,
    ) -> None:
        self.agentview_uint8 = agentview_uint8
        self.wrist_uint8 = wrist_uint8
        self.proprio = proprio
        self.instruction = instruction
        self.actions = actions
        self.reward = reward
        self.weight = 0.0


class LiberoPi05Agent:
    def __init__(
        self,
        *,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        env: gym.Env,
        network: LiberoPi05Network,
        gamma: float,
        buffer_size: int,
        batch_size: int,
        learning_starts: int,
        exploration_noise: float,
        learning_rate: float,
        max_grad_norm: float,
        awr_temperature: float,
    ) -> None:
        del observation_space
        self._env_unwrapped = env.unwrapped
        self.network = network
        self.policy = network.policy
        self.preprocessor = network.preprocessor
        self.device = torch.device(network.cfg.device)
        self.chunk_size = network.chunk_size
        self.action_dim = network.action_dim

        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.exploration_noise = float(exploration_noise)
        self.max_grad_norm = float(max_grad_norm)
        self.awr_temperature = float(awr_temperature)

        self._action_low = action_space.low
        self._action_high = action_space.high

        self.optimizer = optim.AdamW(
            network.trainable_parameters, lr=learning_rate, weight_decay=0.0
        )

        self._buffer: collections.deque[_ChunkRecord] = collections.deque(maxlen=int(buffer_size))

        # Current-chunk execution state.
        self._queue: collections.deque[np.ndarray] = collections.deque()
        self._cur_inputs: dict | None = None
        self._cur_actions: list[np.ndarray] = []
        self._cur_reward = 0.0
        # Chunks completed in the ongoing episode, pending return-to-go weighting.
        self._episode_chunks: list[_ChunkRecord] = []

        self._train_step = 0

    # --- input assembly ----------------------------------------------------

    def _capture_inputs(self, obs: np.ndarray) -> dict:
        """Snapshot the multi-modal observation pi0.5 consumes.

        ``obs`` is the agentview RGB as ``(3, H, W)`` float in [0, 1]; the wrist
        image and proprio come from the env (uint8 ``(H, W, 3)`` and the raw
        8-D state vector).
        """
        agentview_uint8 = (obs * 255.0).astype(np.uint8).transpose(1, 2, 0)
        return {
            "agentview_uint8": agentview_uint8,
            "wrist_uint8": self._env_unwrapped.last_wrist_image.copy(),
            "proprio": self._env_unwrapped.last_proprio.copy(),
            "instruction": self._env_unwrapped.last_instruction,
        }

    def _raw_obs_dict(self, inputs: dict) -> dict:
        """Build the raw observation dict the pi0.5 preprocessor expects."""
        agentview = torch.from_numpy(inputs["agentview_uint8"]).permute(2, 0, 1).float() / 255.0
        wrist = torch.from_numpy(inputs["wrist_uint8"]).permute(2, 0, 1).float() / 255.0
        return {
            OBS_IMAGE_AGENTVIEW: agentview,
            OBS_IMAGE_WRIST: wrist,
            OBS_STATE: torch.from_numpy(inputs["proprio"]).float(),
            TASK_KEY: inputs["instruction"],
        }

    @torch.no_grad()
    def _plan_chunk(self, obs: np.ndarray) -> None:
        """Run pi0.5 inference, capture inputs, and fill the action queue."""
        inputs = self._capture_inputs(obs)
        processed = self.preprocessor(self._raw_obs_dict(inputs))
        chunk = self.policy.predict_action_chunk(processed)  # (1, chunk_size, action_dim)
        chunk = chunk.squeeze(0).float().cpu().numpy()

        self._cur_inputs = inputs
        self._cur_actions = []
        self._cur_reward = 0.0
        self._queue = collections.deque(chunk)

    def _next_action(self) -> np.ndarray:
        """Pop the next planned action, apply exploration noise, and record it."""
        action = self._queue.popleft()
        noise = np.random.randn(self.action_dim).astype(np.float32) * self.exploration_noise
        action = np.clip(action + noise, self._action_low, self._action_high).astype(np.float32)
        self._cur_actions.append(action)
        return action

    def _finalize_chunk(self) -> None:
        """Store the just-executed chunk as a pending episode record.

        Only complete chunks (one executed action per predicted step) become
        training targets; a partial trailing chunk at episode end is dropped so
        the flow-matching target always spans the full prediction horizon.
        """
        if self._cur_inputs is None:
            return
        if len(self._cur_actions) == self.chunk_size:
            self._episode_chunks.append(
                _ChunkRecord(
                    agentview_uint8=self._cur_inputs["agentview_uint8"],
                    wrist_uint8=self._cur_inputs["wrist_uint8"],
                    proprio=self._cur_inputs["proprio"],
                    instruction=self._cur_inputs["instruction"],
                    actions=np.stack(self._cur_actions).astype(np.float32),
                    reward=self._cur_reward,
                )
            )
        self._cur_inputs = None

    # --- RL-agent protocol used by scripts/train.py ------------------------

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
        # ``info`` is the env's reset/step info dict; the wrist image and proprio
        # this agent needs are read directly off the env, so it is ignored here.
        del global_step, reward, terminated, truncated, task_prompt, info
        self._plan_chunk(obs)
        action = self._next_action()
        return StepResult(action=action, metrics={}, panels=self._panels())

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
        del task_prompt, info
        self._cur_reward += float(reward)

        if terminated or truncated:
            self._finalize_chunk()
            metrics = self._end_episode_and_train(global_step)
            return StepResult(
                action=np.zeros(self.action_dim, dtype=np.float32),
                metrics=metrics,
                panels=self._panels(),
            )

        if not self._queue:
            self._finalize_chunk()
            self._plan_chunk(obs)

        action = self._next_action()
        return StepResult(action=action, metrics={"chunk_reward": self._cur_reward}, panels=self._panels())

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        del score, feedback_text
        return {}

    # --- return-to-go weighting + training ---------------------------------

    def _end_episode_and_train(self, global_step: int) -> dict:
        """Assign discounted return-to-go weights to this episode's chunks,
        push them to the buffer, then run a training update.
        """
        running = 0.0
        for record in reversed(self._episode_chunks):
            running = record.reward + self.gamma * running
            record.weight = running
        self._buffer.extend(self._episode_chunks)
        self._episode_chunks = []

        return self._train(global_step)

    def _collate(self, records: list[_ChunkRecord]) -> dict:
        """Preprocess each record and stack the processed tensors into a batch."""
        processed_list = []
        for r in records:
            raw = self._raw_obs_dict(
                {
                    "agentview_uint8": r.agentview_uint8,
                    "wrist_uint8": r.wrist_uint8,
                    "proprio": r.proprio,
                    "instruction": r.instruction,
                }
            )
            raw[ACTION_KEY] = torch.from_numpy(r.actions).float()
            processed_list.append(self.preprocessor(raw))

        batch = {}
        for key, value in processed_list[0].items():
            if isinstance(value, torch.Tensor):
                batch[key] = torch.cat([p[key] for p in processed_list], dim=0)
            else:
                batch[key] = [item for p in processed_list for item in _as_list(p[key])]
        return batch

    def _train(self, global_step: int) -> dict:
        if global_step < self.learning_starts or len(self._buffer) < self.batch_size:
            return {}

        idx = np.random.randint(0, len(self._buffer), size=self.batch_size)
        records = [self._buffer[i] for i in idx]
        batch = self._collate(records)

        # Softmax advantage weights over the sampled batch's returns. softmax is
        # shift-invariant, so only the per-batch std rescaling sets the
        # temperature scale; weights sum to 1 across the batch.
        returns = torch.tensor([r.weight for r in records], device=self.device, dtype=torch.float32)
        adv = returns / (returns.std() + 1e-8)
        weights = torch.softmax(adv / self.awr_temperature, dim=0)

        per_sample_loss, _ = self.policy.forward(batch, reduction="none")
        loss = (weights * per_sample_loss).sum()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.network.trainable_parameters, self.max_grad_norm)
        self.optimizer.step()
        self._train_step += 1

        return {
            "losses/flow_matching_loss": float(per_sample_loss.mean().item()),
            "losses/weighted_loss": float(loss.item()),
            "losses/grad_norm": float(grad_norm),
            "losses/return_mean": float(returns.mean().item()),
            "losses/buffer_size": float(len(self._buffer)),
        }

    # --- render ------------------------------------------------------------

    def _panels(self) -> dict:
        """Wrist-camera panel (constant shape across the run)."""
        return {"wrist": self._env_unwrapped.last_wrist_image}


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    return [value]
