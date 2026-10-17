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
    scalar_obs: torch.Tensor  # (B, T, SCALAR_OBS_DIM)、並びは SCALAR_OBS_NAMES
    # 窓の先頭の行。学習の後に、その窓の TD 誤差で優先度を更新するのに使う
    start_indices: torch.Tensor  # (B,)
    # 優先度で引いた偏りを打ち消す重要度の重み。最大が 1
    weights: torch.Tensor  # (B,)


class ReplayBuffer:
    """窓を、その先頭の行の優先度で引くリングバッファ（Prioritized Experience Replay,
    Schaul et al. 2016 の proportional 版）。

    優先度は窓の TD 誤差の絶対値に ``priority_epsilon`` を足したもので、窓を引く確率は
    その ``priority_alpha`` 乗に比例する。新しい行には、それまでの最大の優先度を入れて、
    一度は引かれるようにする。重要度の重みは ``(N * P)^(-priority_beta)`` を最大で割った
    もの。``priority_alpha`` が 0 なら一様に引き、重みはすべて 1 になる。
    """

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
        priority_alpha: float,
        priority_beta: float,
        priority_epsilon: float,
    ) -> None:
        self.size = size
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta
        self.priority_epsilon = priority_epsilon
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
        # 行 t を先頭とする窓の優先度（TD 誤差の絶対値 + priority_epsilon）
        self.priorities = init_tensor((size,))
        self.max_priority = 1.0

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
        through :meth:`set_latest_state`."""
        # Copy tensors to buffer storage
        self.observations[self.idx].copy_(obs.reshape(self.observations[self.idx].shape))
        self.rewards[self.idx].fill_(reward)
        self.dones[self.idx].fill_(done)
        self.rnn_states[self.idx].copy_(rnn_state.reshape(self.rnn_states[self.idx].shape))
        self.actions[self.idx].copy_(action.reshape(self.actions[self.idx].shape))
        self.scalar_obs[self.idx].copy_(torch.from_numpy(scalar_obs))
        self.states[self.idx].zero_()
        self.priorities[self.idx] = self.max_priority

        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0

    def set_latest_state(self, state: torch.Tensor) -> None:
        """最新の行に、そのステップでネットワークが作った状態の元を入れる。"""
        self.states[(self.idx - 1) % self.size].copy_(state)

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

    def _gather_all(self, indices: torch.Tensor, weights: torch.Tensor) -> ReplayBufferData:
        return ReplayBufferData(
            self._gather("observations", self.observations, indices),
            self._gather("rewards", self.rewards, indices),
            self._gather("dones", self.dones, indices),
            # 状態は窓の先頭（現在状態の窓）と horizon 先（次状態の窓）でしか読まないので、
            # その範囲だけ集める。状態が大きい時間モデルでは、全ステップ分の転送が重い
            self._gather("rnn_states", self.rnn_states, indices[:, : self.horizon + 1]),
            self._gather("actions", self.actions, indices),
            self._gather("states", self.states, indices),
            self._gather("scalar_obs", self.scalar_obs, indices),
            indices[:, 0].to(self.output_device),
            weights.to(self.output_device),
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
        scaled = self.priorities[valid_starts].pow(self.priority_alpha)
        probs = scaled / scaled.sum()
        picks = torch.multinomial(probs, batch_size, replacement=True)
        indices = valid_starts[picks]
        # 一番引かれにくい窓の重みが最大なので、それで割って最大を 1 にする
        weights = (valid_starts.numel() * probs[picks]).pow(-self.priority_beta)
        weights = weights / (valid_starts.numel() * probs.min()).pow(-self.priority_beta)

        # Create vectorized sequence indices: (batch_size, seq_len)
        seq_indices = (
            indices[:, None] + torch.arange(self.seq_len, device=self.storage_device)[None, :]
        )
        return self._gather_all(seq_indices, weights)

    def update_priorities(self, start_indices: torch.Tensor, td_errors: torch.Tensor) -> None:
        """引いた窓の優先度を、学習で出た TD 誤差の絶対値 (B,) で置き換える。同じ窓を
        何度か引いたときは、どれか1つの値が残る。"""
        priorities = td_errors.detach().abs().to(self.storage_device) + self.priority_epsilon
        self.priorities[start_indices.to(self.storage_device)] = priorities
        self.max_priority = max(self.max_priority, float(priorities.max()))

    def batch_stats(self, data: ReplayBufferData) -> dict[str, float]:
        """引いたバッチがバッファ全体と比べてどこに偏っているか。``data`` は :meth:`sample`
        が返したもので、報酬を正規化する前のもの。"""
        stored = self.num_stored()
        starts = data.start_indices.to(self.storage_device)
        # 窓の先頭が書き込まれてから何ステップ経ったか
        age = (self.idx - 1 - starts) % self.size
        # 窓の最後の行（遷移の結果）が終端か、大きな報酬（ゴールや赤い玉）を持つか
        terminal = data.dones[:, -1, 0].float()
        event = (data.rewards[:, -1, 0].abs() > 0.3).float()
        return {
            "replay/age_mean": age.float().mean().item(),
            "replay/terminal_frac": terminal.mean().item(),
            "replay/terminal_frac_buffer": self.dones[:stored, 0].mean().item(),
            "replay/event_frac": event.mean().item(),
            "replay/event_frac_buffer": (self.rewards[:stored, 0].abs() > 0.3)
            .float()
            .mean()
            .item(),
            "replay/priority_mean": self.priorities[starts].mean().item(),
            "replay/priority_mean_buffer": self.priorities[:stored].mean().item(),
            "replay/priority_max": self.max_priority,
        }

    def get_latest(self, seq_len: int) -> ReplayBufferData:
        # Create vectorized indices for the latest sequence
        indices = (
            self.idx - seq_len + torch.arange(seq_len, device=self.storage_device)
        ) % self.size
        return self._gather_all(indices.unsqueeze(0), torch.ones(1))
