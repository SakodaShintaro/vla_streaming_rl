# SPDX-License-Identifier: MIT


class VlacRewardRelabeler:
    """Stateful per-episode online VLAC reward shaper."""

    def __init__(
        self,
        critic,
        milestone_interval: int,
        check_interval: int,
        alpha: float,
        gamma: float,
    ) -> None:
        self._critic = critic
        self._milestone_interval = milestone_interval
        self._check_interval = check_interval
        self._alpha = alpha
        self._gamma = gamma
        self._value_scale = 100.0
        self._reference = None
        self.reset()

    def set_reference(self, reference) -> None:
        """Set the in-context reference trajectory (list of frames) or None."""
        self._reference = reference

    def reset(self) -> None:
        """Clear per-episode state; the next ``step`` re-anchors key/first."""
        self._key = None
        self._first = None
        self._t = 0
        self._v_banked = 0.0
        self._phi_prev = 0.0
        self._last_c = 0.0

    def step(self, frame, task: str):
        """Advance one env step, returning ``(dense_reward, metrics)``."""
        if self._key is None:
            self._key = frame
            self._first = frame
            self._t = 0
            self._phi_prev = 0.0
            return 0.0, {"vlac/dense_reward": 0.0, "vlac/progress": 0.0, "vlac/critic_c": 0.0}

        self._t += 1
        if self._t % self._check_interval != 0 or self._reference is None:
            return 0.0, {
                "vlac/dense_reward": 0.0,
                "vlac/progress": self._phi_prev,
                "vlac/critic_c": self._last_c,
            }

        c = self._critic.score_progress(self._reference, self._first, self._key, frame, task)
        v = self._v_banked + (100.0 - self._v_banked) * c / 100.0
        phi = v / self._value_scale
        dense = self._alpha * (self._gamma * phi - self._phi_prev)
        self._phi_prev = phi
        self._last_c = c

        if self._t % self._milestone_interval == 0:
            self._v_banked = v
            self._key = frame

        return dense, {"vlac/dense_reward": dense, "vlac/progress": phi, "vlac/critic_c": c}
