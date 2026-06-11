# SPDX-License-Identifier: MIT
from dataclasses import dataclass

import numpy as np


@dataclass
class StepResult:
    """Structured return value of an agent's ``select_action`` / ``step``.

    Replaces the old free-form ``info`` dict, which mixed three unrelated
    concerns and forced the trainer to know each agent's internal keys.
    The concerns are now split:

    - ``action``: the env action to execute this tick.
    - ``metrics``: flat scalar telemetry, logged verbatim to wandb.
    - ``panels``: named RGB image panels (HxWx3) for the render strip. The
      trainer concatenates whatever panels are present, in insertion order,
      so an agent can contribute extra panels (e.g. a bird's-eye trajectory
      view with the value estimate) without the trainer knowing about them.
    - ``next_image`` / ``next_reward``: the agent's one-step prediction, kept
      typed (and ``None`` when the agent makes no prediction this tick)
      because the trainer compares them against the next observation /
      reward to compute the prediction loss.
    """

    action: np.ndarray
    metrics: dict[str, float]
    panels: dict[str, np.ndarray]
    next_image: np.ndarray | None
    next_reward: float | None
