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
    observations: torch.Tensor  # (B, T, obs_shape) image only
    rewards: torch.Tensor  # (B, T)
    dones: torch.Tensor  # (B, T)
    rnn_state: torch.Tensor  # (B, T, space_len, state_size, n_layer)
    actions: torch.Tensor  # (B, T, action_shape)
    system_token_ids: torch.Tensor  # (B, T, max_prompt_tokens)
    turn_token_ids: torch.Tensor  # (B, T, max_prompt_tokens)
    reply_token_ids: torch.Tensor  # (B, T, max_prompt_tokens)
    velocity_x: torch.Tensor  # (B, T, 1)
    velocity_y: torch.Tensor  # (B, T, 1)
    velocity_z: torch.Tensor  # (B, T, 1)
    episode_return: torch.Tensor  # (B, T, 1)
    pass_mark: torch.Tensor  # (B, T, 1)
    remaining_return: torch.Tensor  # (B, T, 1)
    global_step: torch.Tensor  # (B, T, 1)
    episode_step: torch.Tensor  # (B, T, 1)
    health: torch.Tensor  # (B, T, 1)
    cot_activations: torch.Tensor  # (B, T, *cot_shape)
    cot_age: torch.Tensor  # (B, T, 1)


class ReplayBuffer:
    def __init__(
        self,
        size: int,
        seq_len: int,
        horizon: int,
        obs_shape: tuple[int, ...],
        rnn_state_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        cot_shape: tuple[int, ...],
        output_device: torch.device,
        storage_device: torch.device,
        max_prompt_tokens: int,
        pad_token_id: int,
    ) -> None:
        self.size = size
        self.seq_len = seq_len
        self.horizon = horizon
        self.action_shape = action_shape
        self.output_device = output_device
        self.storage_device = storage_device
        self.max_prompt_tokens = max_prompt_tokens
        self.pad_token_id = pad_token_id

        assert self.seq_len <= self.size, "Replay buffer size must be >= sequence length."
        assert self.horizon < self.seq_len, "The window needs a state slot before the chunk."

        def init_tensor(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.zeros(
                shape,
                dtype=torch.float32,
                device=self.storage_device,
            )

        self.observations = init_tensor((size, *obs_shape))
        self.rewards = init_tensor((size, 1))
        self.dones = init_tensor((size, 1))
        self.rnn_states = init_tensor((size, *rnn_state_shape))
        self.actions = init_tensor((size, *action_shape))
        self.velocity_x = init_tensor((size, 1))
        self.velocity_y = init_tensor((size, 1))
        self.velocity_z = init_tensor((size, 1))
        self.episode_return = init_tensor((size, 1))
        self.pass_mark = init_tensor((size, 1))
        self.remaining_return = init_tensor((size, 1))
        self.global_step = init_tensor((size, 1))
        self.episode_step = init_tensor((size, 1))
        self.health = init_tensor((size, 1))
        self.cot_activations = torch.zeros(
            (size, *cot_shape),
            dtype=torch.bfloat16,
            device=self.storage_device,
        )
        # How many steps ago each stored chain was generated. Kept per
        # transition because a sampled step is any step of an episode and the
        # age cannot be recovered from one: it is periodic in the writing
        # cadence, which no other stored field carries.
        self.cot_age = init_tensor((size, 1))
        # What the tick said, tokenized: the standing task, the text under the
        # frame, and the chain written on the tick. A conversation is rebuilt
        # from rows a fixed stride apart, each a frame under its text answered
        # by its reply, so the rows hold the parts and never the whole.
        self.system_token_ids = self._init_token_ids()
        self.turn_token_ids = self._init_token_ids()
        self.reply_token_ids = self._init_token_ids()

        self.pin_memory = self.storage_device.type == "cpu" and self.output_device.type == "cuda"
        self._staging = {}

        self.idx = 0
        self.full = False

    def _init_token_ids(self) -> torch.Tensor:
        return torch.full(
            (self.size, self.max_prompt_tokens),
            self.pad_token_id,
            dtype=torch.long,
            device=self.storage_device,
        )

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
            self.rewards[:curr_size].to(self.output_device, non_blocking=True),
            self.dones[:curr_size].to(self.output_device, non_blocking=True),
            self.rnn_states[:curr_size].to(self.output_device, non_blocking=True),
            self.actions[:curr_size].to(self.output_device, non_blocking=True),
            self.system_token_ids[:curr_size].to(self.output_device, non_blocking=True),
            self.turn_token_ids[:curr_size].to(self.output_device, non_blocking=True),
            self.reply_token_ids[:curr_size].to(self.output_device, non_blocking=True),
            self.velocity_x[:curr_size].to(self.output_device, non_blocking=True),
            self.velocity_y[:curr_size].to(self.output_device, non_blocking=True),
            self.velocity_z[:curr_size].to(self.output_device, non_blocking=True),
            self.episode_return[:curr_size].to(self.output_device, non_blocking=True),
            self.pass_mark[:curr_size].to(self.output_device, non_blocking=True),
            self.remaining_return[:curr_size].to(self.output_device, non_blocking=True),
            self.global_step[:curr_size].to(self.output_device, non_blocking=True),
            self.episode_step[:curr_size].to(self.output_device, non_blocking=True),
            self.health[:curr_size].to(self.output_device, non_blocking=True),
            self.cot_activations[:curr_size].to(self.output_device, non_blocking=True),
            self.cot_age[:curr_size].to(self.output_device, non_blocking=True),
        )

    def add(
        self,
        obs: torch.Tensor,
        reward: float,
        done: bool,
        rnn_state: torch.Tensor,
        action: torch.Tensor,
        system_token_ids: list[int],
        turn_token_ids: list[int],
        velocity_x: float,
        velocity_y: float,
        velocity_z: float,
        episode_return: float,
        pass_mark: float,
        remaining_return: float,
        global_step: float,
        episode_step: float,
        health: float,
    ) -> None:
        """Store the tick. What its chain then writes -- the activations, their
        age and the reply -- comes after, through :meth:`amend_latest`."""
        # Copy tensors to buffer storage
        self.observations[self.idx].copy_(obs.reshape(self.observations[self.idx].shape))
        self.rewards[self.idx].fill_(reward)
        self.dones[self.idx].fill_(done)
        self.rnn_states[self.idx].copy_(rnn_state.reshape(self.rnn_states[self.idx].shape))
        self.actions[self.idx].copy_(action.reshape(self.actions[self.idx].shape))
        self.velocity_x[self.idx].fill_(velocity_x)
        self.velocity_y[self.idx].fill_(velocity_y)
        self.velocity_z[self.idx].fill_(velocity_z)
        self.episode_return[self.idx].fill_(episode_return)
        self.pass_mark[self.idx].fill_(pass_mark)
        self.remaining_return[self.idx].fill_(remaining_return)
        self.global_step[self.idx].fill_(global_step)
        self.episode_step[self.idx].fill_(episode_step)
        self.health[self.idx].fill_(health)
        self._store_token_ids(self.system_token_ids, self.idx, system_token_ids)
        self._store_token_ids(self.turn_token_ids, self.idx, turn_token_ids)

        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0

    def amend_latest(
        self, cot_activation: torch.Tensor, cot_age: int, reply_token_ids: list[int]
    ) -> None:
        """Complete the newest row with what the chain wrote on it."""
        latest = (self.idx - 1) % self.size
        self.cot_activations[latest].copy_(cot_activation)
        self.cot_age[latest].fill_(cot_age)
        self._store_token_ids(self.reply_token_ids, latest, reply_token_ids)

    def _store_token_ids(self, storage: torch.Tensor, row: int, token_ids: list[int]) -> None:
        assert len(token_ids) <= self.max_prompt_tokens, (
            f"text of {len(token_ids)} tokens exceeds max_prompt_tokens="
            f"{self.max_prompt_tokens}; raise it"
        )
        storage[row].fill_(self.pad_token_id)
        storage[row, : len(token_ids)] = torch.tensor(
            token_ids, dtype=torch.long, device=self.storage_device
        )

    def _gather(self, name: str, storage: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """``storage[indices]`` on the output device, ``indices`` of any shape.

        Staged through a pinned buffer of the window's shape, reused across
        calls, so the copy to the device is asynchronous; the previous copy
        out of the same buffer is waited for before it is overwritten.
        """
        flat = indices.reshape(-1)
        key = (name, tuple(indices.shape))
        if key not in self._staging:
            self._staging[key] = torch.empty(
                (flat.numel(), *storage.shape[1:]),
                dtype=storage.dtype,
                device=self.storage_device,
                pin_memory=self.pin_memory,
            )
        staged = self._staging[key]
        if self.pin_memory:
            torch.cuda.current_stream(self.output_device).synchronize()
        torch.index_select(storage, 0, flat, out=staged)
        return staged.view(*indices.shape, *storage.shape[1:]).to(
            self.output_device, non_blocking=True
        )

    def _gather_all(self, indices: torch.Tensor) -> ReplayBufferData:
        return ReplayBufferData(
            self._gather("observations", self.observations, indices),
            self._gather("rewards", self.rewards, indices),
            self._gather("dones", self.dones, indices),
            self._gather("rnn_states", self.rnn_states, indices),
            self._gather("actions", self.actions, indices),
            self._gather("system_token_ids", self.system_token_ids, indices),
            self._gather("turn_token_ids", self.turn_token_ids, indices),
            self._gather("reply_token_ids", self.reply_token_ids, indices),
            self._gather("velocity_x", self.velocity_x, indices),
            self._gather("velocity_y", self.velocity_y, indices),
            self._gather("velocity_z", self.velocity_z, indices),
            self._gather("episode_return", self.episode_return, indices),
            self._gather("pass_mark", self.pass_mark, indices),
            self._gather("remaining_return", self.remaining_return, indices),
            self._gather("global_step", self.global_step, indices),
            self._gather("episode_step", self.episode_step, indices),
            self._gather("health", self.health, indices),
            self._gather("cot_activations", self.cot_activations, indices),
            self._gather("cot_age", self.cot_age, indices),
        )

    def valid_start_indices(self) -> torch.Tensor:
        """Window starts a learning batch may be drawn from.

        ``seq_len`` here is the whole window the learner reads: the state window
        plus the ``horizon`` action chunk that follows it. Two things make a
        start invalid.

        A ``done`` on the current state slot (the last of the state window) or
        inside the chunk but not on its last slot means the chunk carries
        actions the agent never took in the episode its state belongs to. A
        ``done`` earlier in the state window is allowed: the state encoder then
        sees frames of the previous episode, which is what it sees at every
        episode start anyway. A ``done`` on the last slot is the wanted case --
        the episode ends on the chunk's final action -- and the value head
        truncates the bootstrap there itself.

        A window containing the write head has the same problem in time: the
        slots on either side of it are the newest and the oldest transitions in
        the ring, with nothing connecting them.
        """
        curr_size = self.size if self.full else self.idx
        starts = torch.arange(
            0, curr_size - self.seq_len + 1, device=self.storage_device, dtype=torch.long
        )

        # Sum of dones over slots ``start + state_slot`` .. ``start + seq_len - 2``.
        state_slot = self.seq_len - self.horizon - 1
        cumulative_dones = torch.cat(
            [
                torch.zeros(1, device=self.storage_device),
                self.dones[:curr_size, 0].cumsum(dim=0),
            ]
        )
        crosses_episode = (
            cumulative_dones[starts + self.seq_len - 1] - cumulative_dones[starts + state_slot]
        ) > 0

        if self.full:
            # The head sits between slot ``idx - 1`` (newest) and ``idx`` (oldest).
            crosses_head = (starts <= self.idx - 1) & (starts + self.seq_len - 1 >= self.idx)
        else:
            crosses_head = torch.zeros_like(crosses_episode)

        return starts[~(crosses_episode | crosses_head)]

    def latest_window_is_clean(self, window_len: int) -> bool:
        """Whether the newest ``window_len`` slots pass the rule
        :meth:`valid_start_indices` draws under: from the current state slot on,
        a ``done`` only on the last slot, which is the episode ending on the
        window's own final action. The learner that reads the newest window
        instead of sampling asks this before it trains on it.
        """
        if not self.full and self.idx < window_len:
            return False
        state_slot = window_len - self.horizon - 1
        indices = (
            self.idx - window_len + torch.arange(window_len, device=self.storage_device)
        ) % self.size
        return bool(self.dones[indices[state_slot:-1], 0].sum() == 0)

    def sample(self, batch_size: int) -> ReplayBufferData:
        curr_size = self.size if self.full else self.idx
        assert curr_size >= self.seq_len, "Not enough data to sample a sequence."

        valid_starts = self.valid_start_indices()
        assert valid_starts.numel() > 0, (
            "No window of the buffer stays inside one episode: every start of "
            f"length {self.seq_len} crosses a done or the write head."
        )
        indices = valid_starts[
            torch.randint(0, valid_starts.numel(), (batch_size,), device=self.storage_device)
        ]

        # Create vectorized sequence indices: (batch_size, seq_len)
        seq_indices = (
            indices[:, None] + torch.arange(self.seq_len, device=self.storage_device)[None, :]
        )
        return self._gather_all(seq_indices)

    def get_latest(self, seq_len: int) -> ReplayBufferData:
        # Create vectorized indices for the latest sequence
        indices = (
            self.idx - seq_len + torch.arange(seq_len, device=self.storage_device)
        ) % self.size
        return self._gather_all(indices.unsqueeze(0))
