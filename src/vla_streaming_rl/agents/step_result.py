# SPDX-License-Identifier: MIT
from dataclasses import dataclass

import numpy as np


@dataclass
class StepResult:
    """Structured return value of the agent's ``select_action`` / ``step``.

    - ``action``: the env action to execute this tick.
    - ``metrics``: flat scalar telemetry, logged verbatim to wandb.
    - ``panels``: named RGB image panels (HxWx3) for the render strip. The
      trainer concatenates whatever panels are present, in insertion order,
      so the agent can contribute extra panels (e.g. the next-frame prediction
      or a bird's-eye trajectory view) without the trainer knowing about them.
      Stable-panel contract: the agent must emit the SAME set of panel keys with
      the SAME shapes on every step of a run (use a blank placeholder when the
      content does not exist yet, e.g. before the first prediction). The render
      frames are encoded into a single video, which requires a constant size.
    - ``texts``: named free-form text, written to the episode's ``texts.tsv``.
      What a panel shows as pixels this keeps as characters, so a chain of
      thought can be read back and searched after the run. The agent puts the
      instruction its policy read this tick under ``"prompt"``, which is also
      what the trainer captions the render with.
    """

    action: np.ndarray
    metrics: dict[str, float]
    panels: dict[str, np.ndarray]
    texts: dict[str, str]
