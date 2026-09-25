# SPDX-License-Identifier: MIT
"""The ritual every entry script performs before real work.

Imported and called at the very top of scripts/*.py, before the heavy imports,
so the warning filters are installed before the libraries that would trip them.
"""

import logging
import os
import warnings


def setup_runtime() -> None:
    """Warning filters and runtime knobs shared by every entry script. Called
    before the heavy imports, which is why torch is imported lazily here."""
    os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
    warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS.*")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    import torch

    torch.set_float32_matmul_precision("high")


def seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)


def resolve_seed(seed: int) -> int:
    """The seed itself, or a random one for the -1 sentinel."""
    if seed != -1:
        return seed

    import numpy as np

    return int(np.random.randint(0, 10000))


def disable_render_if_headless(render: bool) -> bool:
    has_display = "DISPLAY" in os.environ and os.environ["DISPLAY"] != ""
    if render and not has_display:
        print("Because a headless environment is detected, rendering is automatically disabled.")
        return False
    return render
