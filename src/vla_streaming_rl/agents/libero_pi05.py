# SPDX-License-Identifier: MIT
import collections

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

from vla_streaming_rl.agents.step_result import StepResult
from vla_streaming_rl.networks.interface import InferInput
from vla_streaming_rl.networks.libero_pi05_network import (
    ACTION_KEY,
    OBS_IMAGE_AGENTVIEW,
    OBS_IMAGE_WRIST,
    OBS_STATE,
    TASK_KEY,
    LiberoPi05Network,
)
from vla_streaming_rl.replay_buffer import ReplayBufferData


class _ChunkRecord:
    """One executed action chunk + the observation that produced it, as a fully
    formed RL transition: the current obs ``inputs``, the executed chunk, the
    per-env-step rewards/dones over the chunk, and the next chunk's obs
    (``next_inputs``) for the TD bootstrap. Built only once the episode ends, so
    every field — including ``next_inputs`` and the terminal ``dones`` — is known
    and passed explicitly (see ``_end_episode``)."""

    def __init__(
        self,
        inputs: dict,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_inputs: dict,
    ) -> None:
        self.inputs = inputs
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.next_inputs = next_inputs


class LiberoPi05Agent:
    def __init__(
        self,
        *,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        network: LiberoPi05Network,
        buffer_size: int,
        batch_size: int,
        learning_starts: int,
        actor_lr: float,
        critic_lr: float,
        max_grad_norm: float,
    ) -> None:
        # The wrist camera, proprio and instruction pi0.5 needs arrive through
        # the env's ``info`` dict (LiberoEnv publishes them), so the agent never
        # holds the env. ``_last_wrist`` backs the render panel; seed it with a
        # zero frame of the observation shape so the panel has a constant shape
        # before the first observation.
        self._last_wrist = np.zeros(observation_space.shape, dtype=np.uint8)
        self.network = network
        self.policy = network.policy
        self.preprocessor = network.preprocessor
        self.postprocessor = network.postprocessor
        self.device = torch.device(network.cfg.device)
        self.chunk_size = network.chunk_size
        self.action_dim = network.action_dim

        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.max_grad_norm = float(max_grad_norm)

        self._action_low = action_space.low
        self._action_high = action_space.high

        # Separate optimizers for the policy μ (action expert) and the injected
        # critic, mirroring SimLingoAgent — the actor LR is tiny so fine-tuning
        # the expert does not wash out the pretrained flow.
        self.actor_optimizer = optim.AdamW(network.actor_parameters, lr=actor_lr, weight_decay=0.0)
        self.critic_optimizer = optim.AdamW(
            network.critic.parameters(), lr=critic_lr, weight_decay=0.0
        )

        # Dummy tensor for the InferInput / ReplayBufferData slots pi0.5 does not
        # use (recurrent state, encoded obs, log-probs, token ids …): pi0.5's
        # multimodal observation rides ``s_seq`` / ``observations`` instead.
        self._dummy = torch.zeros(1, device=self.device)

        self._buffer: collections.deque[_ChunkRecord] = collections.deque(maxlen=int(buffer_size))

        # Current-chunk execution state.
        self._queue: collections.deque[np.ndarray] = collections.deque()
        self._cur_inputs: dict | None = None
        self._cur_actions: list[np.ndarray] = []
        self._cur_rewards: list[float] = []
        # Raw ``(inputs, actions, rewards)`` of chunks completed in the ongoing
        # episode; turned into linked ``_ChunkRecord`` transitions at episode end.
        self._episode_chunks: list[tuple[dict, np.ndarray, np.ndarray]] = []
        # Latest critic read-out, for the render/telemetry path.
        self._value_report: dict[str, float] = {"value": 0.0}

        self._train_step = 0

    # --- input assembly ----------------------------------------------------

    def _capture_inputs(self, obs: np.ndarray, info: dict) -> dict:
        """Snapshot the multi-modal observation pi0.5 consumes.

        ``obs`` is the agentview RGB as ``(3, H, W)`` float in [0, 1]; the wrist
        image (uint8 ``(H, W, 3)``), the raw 8-D proprio and the language
        instruction are read from the env ``info`` dict under the keys LiberoEnv
        publishes (``wrist_image`` / ``proprio`` / ``task_prompt``).
        """
        agentview_uint8 = (obs * 255.0).astype(np.uint8).transpose(1, 2, 0)
        wrist_uint8 = info["wrist_image"].copy()
        self._last_wrist = wrist_uint8
        return {
            "agentview_uint8": agentview_uint8,
            "wrist_uint8": wrist_uint8,
            "proprio": info["proprio"].copy(),
            "instruction": info["task_prompt"],
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

    def _plan_chunk(self, obs: np.ndarray, info: dict) -> None:
        """Run pi0.5 inference, capture inputs, and fill the action queue.

        ``network.infer`` returns the *normalized* action chunk plus the critic's
        value report from a single prefix forward; the postprocessor un-normalizes
        the chunk into the env's action space.
        """
        inputs = self._capture_inputs(obs, info)
        processed = self.preprocessor(self._raw_obs_dict(inputs))
        infer_result = self.network.infer(
            InferInput(
                s_seq=processed,
                obs_z_seq=self._dummy,
                a_seq=self._dummy,
                r_seq=self._dummy,
                rnn_state=self._dummy,
                task_prompts=[],
            )
        )
        chunk = self.postprocessor(infer_result.action)  # (1, chunk_size, action_dim)
        chunk = chunk.squeeze(0).float().cpu().numpy()
        self._value_report = infer_result.value_report

        self._cur_inputs = inputs
        self._cur_actions = []
        self._cur_rewards = []
        self._queue = collections.deque(chunk)

    def _next_action(self) -> np.ndarray:
        action = np.clip(self._queue.popleft(), self._action_low, self._action_high).astype(
            np.float32
        )
        self._cur_actions.append(action)
        return action

    def _finalize_chunk(self) -> None:
        """Store the just-executed chunk as a pending episode record.

        Only complete chunks (one executed action *and* one reward per predicted
        step) become training targets; a partial trailing chunk at episode end is
        dropped so both the flow-matching target and the per-step reward vector
        span the full prediction horizon.
        """
        if self._cur_inputs is None:
            return
        if len(self._cur_actions) == self.chunk_size and len(self._cur_rewards) == self.chunk_size:
            self._episode_chunks.append(
                (
                    self._cur_inputs,
                    np.stack(self._cur_actions).astype(np.float32),
                    np.asarray(self._cur_rewards, dtype=np.float32),
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
        del global_step, reward, terminated, truncated, task_prompt
        self._plan_chunk(obs, info)
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
        del task_prompt
        # This reward is the consequence of the action returned last call, so it
        # belongs to the current (not-yet-finalized) chunk.
        self._cur_rewards.append(float(reward))

        if terminated or truncated:
            self._finalize_chunk()
            metrics = self._end_episode(global_step, terminated)
            metrics.update(self._value_report)
            return StepResult(
                action=np.zeros(self.action_dim, dtype=np.float32),
                metrics=metrics,
                panels=self._panels(),
            )

        if not self._queue:
            self._finalize_chunk()
            self._plan_chunk(obs, info)

        action = self._next_action()
        return StepResult(
            action=action,
            metrics={"chunk_reward": float(np.sum(self._cur_rewards))},
            panels=self._panels(),
        )

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        del score, feedback_text
        return {}

    # --- transition linking + training -------------------------------------

    def _end_episode(self, global_step: int, terminated: bool) -> dict:
        """Link this episode's chunks into TD transitions, push to the buffer,
        then run a training update.

        Each chunk's ``next_inputs`` points at the next chunk's observation. The
        final chunk has no successor: on a true ``terminated`` it is marked
        terminal (last-step done = 1 → bootstrap masked); on truncation it
        bootstraps from its own state (``next_inputs = inputs``), the standard
        no-true-terminal approximation.
        """
        chunks = self._episode_chunks
        for i, (inputs, actions, rewards) in enumerate(chunks):
            dones = np.zeros(self.chunk_size, dtype=np.float32)
            if i + 1 < len(chunks):
                next_inputs = chunks[i + 1][0]
            else:
                next_inputs = inputs
                if terminated:
                    dones[-1] = 1.0
            self._buffer.append(
                _ChunkRecord(
                    inputs=inputs,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    next_inputs=next_inputs,
                )
            )
        self._episode_chunks = []
        return self._train(global_step)

    def _collate(self, records: list[_ChunkRecord]) -> ReplayBufferData:
        """Preprocess each record's current + next observation into the
        ``ReplayBufferData`` ``compute_loss`` consumes.

        pi0.5's multimodal observation cannot live in a tensor buffer, so
        ``observations`` is a length-2 ``[cur_batch, next_batch]`` list of
        preprocessed pi0.5 batches. ``cur`` carries the executed (real-space)
        action under ``ACTION_KEY`` so the preprocessor normalizes it into pi0.5's
        action space — that normalized chunk (read back as ``actions``) is what
        the critic and the DACER2 actor operate on. ``next`` is observation-only
        (the bootstrap re-samples π(s')). The remaining slots are unused dummies.
        """
        cur_list, next_list = [], []
        for r in records:
            cur = self._raw_obs_dict(r.inputs)
            cur[ACTION_KEY] = torch.from_numpy(r.actions).float().unsqueeze(0)
            cur_list.append(self.preprocessor(cur))
            next_list.append(self.preprocessor(self._raw_obs_dict(r.next_inputs)))

        cur_batch = _stack_processed(cur_list)
        next_batch = _stack_processed(next_list)
        rewards = torch.tensor(
            np.stack([r.rewards for r in records]), device=self.device, dtype=torch.float32
        )
        dones = torch.tensor(
            np.stack([r.dones for r in records]), device=self.device, dtype=torch.float32
        )
        return ReplayBufferData(
            observations=[cur_batch, next_batch],
            actions=cur_batch[ACTION_KEY],
            rewards=rewards,
            dones=dones,
            obs_z=self._dummy,
            rnn_state=self._dummy,
            log_probs=self._dummy,
            values=self._dummy,
            action_token_ids=self._dummy,
            task_prompt_token_ids=self._dummy,
        )

    def _train(self, global_step: int) -> dict:
        if global_step < self.learning_starts or len(self._buffer) < self.batch_size:
            return {}

        idx = np.random.randint(0, len(self._buffer), size=self.batch_size)
        records = [self._buffer[i] for i in idx]
        data = self._collate(records)

        self.policy.train()
        result = self.network.compute_loss(data)

        # Never let a non-finite loss reach an optimizer: a single NaN/Inf step
        # silently corrupts the expert/critic and every subsequent rollout.
        if not torch.isfinite(result.loss):
            return {"losses/skipped_nonfinite": 1.0}

        # One backward over the combined critic+actor loss, then step each
        # optimizer (the DACER2 actor loss froze the critic during its Q forwards,
        # so the gradients land on disjoint params) — same pattern as
        # SimLingoAgent._train_offpolicy.
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(self.network.critic.parameters(), self.max_grad_norm)
        actor_grad = nn.utils.clip_grad_norm_(self.network.actor_parameters, self.max_grad_norm)
        self.critic_optimizer.step()
        self.actor_optimizer.step()
        self._train_step += 1

        return {
            **result.info,
            "losses/critic_grad_norm": float(critic_grad),
            "losses/actor_grad_norm": float(actor_grad),
            "losses/buffer_size": float(len(self._buffer)),
        }

    # --- render ------------------------------------------------------------

    def _panels(self) -> dict:
        """Wrist-camera panel (constant shape across the run)."""
        return {"wrist": self._last_wrist}


def _stack_processed(processed_list: list[dict]) -> dict:
    """Concatenate a list of single-sample preprocessed batches along dim 0."""
    batch = {}
    for key, value in processed_list[0].items():
        if isinstance(value, torch.Tensor):
            batch[key] = torch.cat([p[key] for p in processed_list], dim=0)
        else:
            batch[key] = [item for p in processed_list for item in _as_list(p[key])]
    return batch


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    return [value]
