# SPDX-License-Identifier: MIT
from dataclasses import dataclass

import numpy as np
import torch

from vla_streaming_rl.scalar_obs import SCALAR_OBS_DIM

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
    rnn_state: torch.Tensor  # (B, min(T, horizon + 1), space_len, state_size, n_layer)
    actions: torch.Tensor  # (B, T, action_shape)
    states: torch.Tensor  # (B, T, *state_shape)
    reply_actions: torch.Tensor  # (B, T, action_shape)
    scalar_obs: torch.Tensor  # (B, T, SCALAR_OBS_DIM)、並びは SCALAR_OBS_NAMES


class ReplayBuffer:
    def __init__(
        self,
        size: int,
        seq_len: int,
        horizon: int,
        obs_shape: tuple[int, ...],
        rnn_state_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        state_shape: tuple[int, ...],
        output_device: torch.device,
        storage_device: torch.device,
    ) -> None:
        self.size = size
        self.seq_len = seq_len
        self.horizon = horizon
        self.action_shape = action_shape
        self.output_device = output_device
        self.storage_device = storage_device

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
        # ネットワークが行動を決めたステップで保存する状態の元。VLM の表現のように
        # 大きいので半精度で持つ
        self.states = torch.zeros(
            (size, *state_shape), dtype=torch.bfloat16, device=self.storage_device
        )
        self.scalar_obs = init_tensor((size, SCALAR_OBS_DIM))
        # ネットワークがそのステップに書いた返答が指す行動（正規化した行動空間）。返答を
        # 書かないネットワークでは 0 のまま
        self.reply_actions = init_tensor((size, *action_shape))

        self.pin_memory = self.storage_device.type == "cpu" and self.output_device.type == "cuda"
        self._staging = {}

        self.idx = 0
        self.full = False

    def num_stored(self) -> int:
        return self.size if self.full else self.idx

    def add(
        self,
        obs: torch.Tensor,
        reward: float,
        done: bool,
        rnn_state: torch.Tensor,
        action: torch.Tensor,
        scalar_obs: np.ndarray,
    ) -> None:
        """Store the tick. What the network then makes of it comes after,
        through :meth:`set_latest_state` and :meth:`set_latest_reply_action`."""
        # Copy tensors to buffer storage
        self.observations[self.idx].copy_(obs.reshape(self.observations[self.idx].shape))
        self.rewards[self.idx].fill_(reward)
        self.dones[self.idx].fill_(done)
        self.rnn_states[self.idx].copy_(rnn_state.reshape(self.rnn_states[self.idx].shape))
        self.actions[self.idx].copy_(action.reshape(self.actions[self.idx].shape))
        self.scalar_obs[self.idx].copy_(torch.from_numpy(scalar_obs))
        self.reply_actions[self.idx].zero_()
        self.states[self.idx].zero_()

        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0

    def set_latest_state(self, state: torch.Tensor) -> None:
        """最新の行に、そのステップでネットワークが作った状態の元を入れる。"""
        self.states[(self.idx - 1) % self.size].copy_(state)

    def set_latest_reply_action(self, reply_action: torch.Tensor) -> None:
        """最新の行に、そのステップでネットワークが書いた返答が指す行動を入れる。"""
        self.reply_actions[(self.idx - 1) % self.size].copy_(
            reply_action.reshape(self.reply_actions.shape[1:])
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
            # 状態は窓の先頭（現在状態の窓）と horizon 先（次状態の窓）でしか読まないので、
            # その範囲だけ集める。状態が大きい時間モデルでは、全ステップ分の転送が重い
            self._gather("rnn_states", self.rnn_states, indices[:, : self.horizon + 1]),
            self._gather("actions", self.actions, indices),
            self._gather("states", self.states, indices),
            self._gather("reply_actions", self.reply_actions, indices),
            self._gather("scalar_obs", self.scalar_obs, indices),
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
