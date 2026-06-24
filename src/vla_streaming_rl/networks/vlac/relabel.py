# SPDX-License-Identifier: MIT
"""Online milestone-based VLAC dense reward (arXiv:2512.14666 style).

A milestone (key) frame is held fixed for ``milestone_interval`` steps. Every
``check_interval`` steps the critic scores the current frame against the key,
``c in [-100, 100]`` = progress from the key to now. The absolute progress is
``v = v_banked + (100 - v_banked) * c / 100`` (diminishing returns), and the
potential is ``Phi = v / value_scale``. The per-step shaping reward is the PBRS
term ``alpha * (gamma * Phi_t - Phi_prev)``. When a milestone is reached the
progress is banked and the key advances to the current frame.
"""


class VlacRewardRelabeler:
    """Stateful per-episode online VLAC reward shaper."""

    def __init__(
        self,
        critic,
        milestone_interval: int,
        check_interval: int,
        alpha: float,
        gamma: float,
        value_scale: float,
    ) -> None:
        self._critic = critic
        self._milestone_interval = milestone_interval
        self._check_interval = check_interval
        self._alpha = alpha
        self._gamma = gamma
        self._value_scale = value_scale
        self.reset()

    def reset(self) -> None:
        """Clear per-episode state; the next ``step`` re-anchors the key frame."""
        self._key = None
        self._t = 0
        self._v_banked = 0.0
        self._phi_prev = 0.0
        self._last_c = 0.0

    def step(self, frame, task: str):
        """Advance one env step, returning ``(dense_reward, metrics)``.

        The key is set on the first call; the critic only runs on
        ``check_interval`` boundaries (dense reward is 0 in between)."""
        if self._key is None:
            self._key = frame
            self._t = 0
            self._phi_prev = 0.0
            return 0.0, {"vlac/dense_reward": 0.0, "vlac/progress": 0.0, "vlac/critic_c": 0.0}

        self._t += 1
        if self._t % self._check_interval != 0:
            return 0.0, {
                "vlac/dense_reward": 0.0,
                "vlac/progress": self._phi_prev,
                "vlac/critic_c": self._last_c,
            }

        c = self._critic.score_pair(self._key, frame, task)
        v = self._v_banked + (100.0 - self._v_banked) * c / 100.0
        phi = v / self._value_scale
        dense = self._alpha * (self._gamma * phi - self._phi_prev)
        self._phi_prev = phi
        self._last_c = c

        if self._t % self._milestone_interval == 0:
            self._v_banked = v
            self._key = frame

        return dense, {"vlac/dense_reward": dense, "vlac/progress": phi, "vlac/critic_c": c}
