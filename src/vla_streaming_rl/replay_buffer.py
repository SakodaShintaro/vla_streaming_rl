# SPDX-License-Identifier: MIT
from dataclasses import dataclass

import torch

""" Note
Recording the convention for what data is stored at replay buffer index t,
since it can be confusing (the action selected after receiving timestep t
from the environment vs. the action selected before — the pairing is arbitrary).
In this code, index t stores information visible to the agent just before
selecting an action at timestep t. Specifically:
- obs, reward, done are obtained from the environment at timestep t as inputs
- rnn_state is the RNN hidden state at timestep t
- action is the action selected at timestep t-1
"""


@dataclass
class ReplayBufferData:
    observations: torch.Tensor  # (B, T, obs_shape)
    obs_z: torch.Tensor  # (B, T, obs_z_shape) - encoded observations
    rewards: torch.Tensor  # (B, T)
    dones: torch.Tensor  # (B, T)
    # rnn_state shape (SpatialTemporalEncoder): (B, T, space_len, state_size, n_layer)
    rnn_state: torch.Tensor
    actions: torch.Tensor  # (B, T, action_shape)
    action_token_ids: torch.Tensor  # (B, T, max_new_tokens)
    task_prompt_token_ids: torch.Tensor  # (B, T, max_prompt_tokens)


class ReplayBuffer:
    def __init__(
        self,
        size: int,
        seq_len: int,
        obs_shape: tuple[int, ...],
        obs_z_shape: tuple[int, ...],
        rnn_state_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        output_device: torch.device,
        storage_device: torch.device,
        max_new_tokens: int,
        max_prompt_tokens: int,
        pad_token_id: int,
    ) -> None:
        self.size = size
        self.seq_len = seq_len
        self.action_shape = action_shape
        self.output_device = output_device
        self.storage_device = storage_device
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.pad_token_id = pad_token_id

        assert self.seq_len <= self.size, "Replay buffer size must be >= sequence length."

        def init_tensor(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.zeros(
                shape,
                dtype=torch.float32,
                device=self.storage_device,
            )

        self.observations = init_tensor((size, *obs_shape))
        self.obs_z = init_tensor((size, *obs_z_shape))
        self.rewards = init_tensor((size, 1))
        self.dones = init_tensor((size, 1))
        self.rnn_states = init_tensor((size, *rnn_state_shape))
        self.actions = init_tensor((size, *action_shape))
        self.action_token_ids = torch.full(
            (size, max_new_tokens),
            pad_token_id,
            dtype=torch.long,
            device=self.storage_device,
        )
        self.task_prompt_token_ids = torch.full(
            (size, max_prompt_tokens),
            pad_token_id,
            dtype=torch.long,
            device=self.storage_device,
        )

        self.idx = 0
        self.full = False

    def is_full(self) -> bool:
        return self.full

    def num_stored(self) -> int:
        return self.size if self.full else self.idx

    def reset(self) -> None:
        self.idx = 0
        self.full = False

    def get_all_data(self) -> ReplayBufferData:
        """Get all data in the buffer (for on-policy training)"""
        curr_size = self.size if self.full else self.idx
        return ReplayBufferData(
            self.observations[:curr_size].to(self.output_device, non_blocking=True),
            self.obs_z[:curr_size].to(self.output_device, non_blocking=True),
            self.rewards[:curr_size].to(self.output_device, non_blocking=True),
            self.dones[:curr_size].to(self.output_device, non_blocking=True),
            self.rnn_states[:curr_size].to(self.output_device, non_blocking=True),
            self.actions[:curr_size].to(self.output_device, non_blocking=True),
            self.action_token_ids[:curr_size].to(self.output_device, non_blocking=True),
            self.task_prompt_token_ids[:curr_size].to(self.output_device, non_blocking=True),
        )

    def add(
        self,
        obs: torch.Tensor,
        obs_z: torch.Tensor,
        reward: float,
        done: bool,
        rnn_state: torch.Tensor,
        action: torch.Tensor,
        action_token_ids: list[int],
        task_prompt_token_ids: list[int],
    ) -> None:
        # Copy tensors to buffer storage
        self.observations[self.idx].copy_(obs.reshape(self.observations[self.idx].shape))
        self.obs_z[self.idx].copy_(obs_z.reshape(self.obs_z[self.idx].shape))
        self.rewards[self.idx].fill_(reward)
        self.dones[self.idx].fill_(done)
        self.rnn_states[self.idx].copy_(rnn_state.reshape(self.rnn_states[self.idx].shape))
        self.actions[self.idx].copy_(action.reshape(self.actions[self.idx].shape))

        self.action_token_ids[self.idx].fill_(self.pad_token_id)
        token_len = min(len(action_token_ids), self.max_new_tokens)
        self.action_token_ids[self.idx, :token_len] = torch.tensor(
            action_token_ids[:token_len], dtype=torch.long, device=self.storage_device
        )

        self.task_prompt_token_ids[self.idx].fill_(self.pad_token_id)
        if len(task_prompt_token_ids) > self.max_prompt_tokens:
            print(
                f"[WARNING] task_prompt_token_ids exceeds max_prompt_tokens: len={len(task_prompt_token_ids)}, max_prompt_tokens={self.max_prompt_tokens}"
            )
        prompt_len = min(len(task_prompt_token_ids), self.max_prompt_tokens)
        self.task_prompt_token_ids[self.idx, :prompt_len] = torch.tensor(
            task_prompt_token_ids[:prompt_len], dtype=torch.long, device=self.storage_device
        )

        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int) -> ReplayBufferData:
        curr_size = self.size if self.full else self.idx
        assert curr_size >= self.seq_len, "Not enough data to sample a sequence."

        # Generate base indices for each batch element
        indices = torch.randint(
            0, curr_size - self.seq_len, (batch_size,), device=self.storage_device
        )

        # Create vectorized sequence indices: (batch_size, seq_len)
        seq_indices = (
            indices[:, None] + torch.arange(self.seq_len, device=self.storage_device)[None, :]
        )

        # Vectorized slicing - much faster than loop + append
        return ReplayBufferData(
            self.observations[seq_indices].to(self.output_device, non_blocking=True),
            self.obs_z[seq_indices].to(self.output_device, non_blocking=True),
            self.rewards[seq_indices].to(self.output_device, non_blocking=True),
            self.dones[seq_indices].to(self.output_device, non_blocking=True),
            self.rnn_states[seq_indices].to(self.output_device, non_blocking=True),
            self.actions[seq_indices].to(self.output_device, non_blocking=True),
            self.action_token_ids[seq_indices].to(self.output_device, non_blocking=True),
            self.task_prompt_token_ids[seq_indices].to(self.output_device, non_blocking=True),
        )

    def get_latest(self, seq_len: int) -> ReplayBufferData:
        # Create vectorized indices for the latest sequence
        indices = (
            self.idx - seq_len + torch.arange(seq_len, device=self.storage_device)
        ) % self.size

        # Vectorized slicing
        return ReplayBufferData(
            self.observations[indices].unsqueeze(0).to(self.output_device, non_blocking=True),
            self.obs_z[indices].unsqueeze(0).to(self.output_device, non_blocking=True),
            self.rewards[indices].unsqueeze(0).to(self.output_device, non_blocking=True),
            self.dones[indices].unsqueeze(0).to(self.output_device, non_blocking=True),
            self.rnn_states[indices].unsqueeze(0).to(self.output_device, non_blocking=True),
            self.actions[indices].unsqueeze(0).to(self.output_device, non_blocking=True),
            self.action_token_ids[indices].unsqueeze(0).to(self.output_device, non_blocking=True),
            self.task_prompt_token_ids[indices]
            .unsqueeze(0)
            .to(self.output_device, non_blocking=True),
        )
