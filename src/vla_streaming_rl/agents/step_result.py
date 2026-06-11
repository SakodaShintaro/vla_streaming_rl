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
      so an agent can contribute extra panels (e.g. the next-frame prediction
      or a bird's-eye trajectory view) without the trainer knowing about them.
      Stable-panel contract: an agent must emit the SAME set of panel keys with
      the SAME shapes on every step of a run (use a blank placeholder when the
      content does not exist yet, e.g. before the first prediction). The render
      frames are encoded into a single video, which requires a constant size.
    """

    action: np.ndarray
    metrics: dict[str, float]
    panels: dict[str, np.ndarray]
