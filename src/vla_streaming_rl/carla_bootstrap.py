# SPDX-License-Identifier: MIT
"""Launch + wire up a CARLA server from pure Python.

``train_carla.sh`` does two shell-only things before ``train.py`` runs: it
exports ``PYTHONPATH`` (CARLA's ``PythonAPI/carla`` for the ``agents`` package,
plus Bench2Drive's ``leaderboard`` / ``scenario_runner``) and ``setsid``-launches
``CarlaUE4.sh``. This module reproduces both in-process so a sweep can be driven
by plain ``python scripts/train.py …`` with no bash wrapper and no pre-running
server — the pattern Bench2Drive's own
``leaderboard_evaluator._setup_simulation`` uses (subprocess.Popen + os.setsid +
atexit kill).

``carla`` itself is pip-installed in this repo's venv, so only the three source
directories above need to be added to ``sys.path``; do that via
:func:`setup_carla_paths` *before* importing anything under
``envs/carla_leaderboard_env`` or the SimLingo agent (both import
``leaderboard`` / ``srunner`` at module load). The env connects to
``localhost:2000`` (hard-coded in ``CARLALeaderboardEnv``), so that is the
launch port here. CARLA_ROOT is taken from ``$CARLA_ROOT`` (else ~/CARLA_0.9.16)
and the server runs on GPU 0 — neither is a sweep knob, so both are fixed below.
"""

import atexit
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# Aligned with carla.Client("localhost", 2000) hard-coded in CARLALeaderboardEnv.
_CARLA_HOST = "localhost"
_CARLA_PORT = 2000
_GPU_RANK = 0
_READY_TIMEOUT_S = 180.0
# Repo root is two levels up from src/vla_streaming_rl/carla_bootstrap.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_B2D_ROOT = _REPO_ROOT / "external" / "Bench2Drive"


def resolve_carla_root() -> Path:
    """Resolve the CARLA install dir from ``$CARLA_ROOT`` (else ~/CARLA_0.9.16)."""
    root = Path(os.environ.get("CARLA_ROOT", "~/CARLA_0.9.16")).expanduser()
    if not (root / "CarlaUE4.sh").exists():
        raise FileNotFoundError(
            f"CarlaUE4.sh not found under CARLA_ROOT={root}. "
            "Set $CARLA_ROOT to your CARLA 0.9.x install."
        )
    return root


def setup_carla_paths() -> None:
    """Put the CARLA ``PythonAPI/carla`` (for the ``agents`` package) and
    Bench2Drive's ``leaderboard`` / ``scenario_runner`` on ``sys.path``, and
    export ``CARLA_ROOT`` / ``SCENARIO_RUNNER_ROOT`` — the env that
    ``train_carla.sh`` sets up by hand. Idempotent.

    Must be called *before* importing ``carla_leaderboard_env`` or the SimLingo
    agent, which import ``leaderboard`` / ``srunner`` at module load.
    """
    root = resolve_carla_root()
    for p in (root / "PythonAPI" / "carla", _B2D_ROOT / "leaderboard", _B2D_ROOT / "scenario_runner"):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    os.environ["CARLA_ROOT"] = str(root)
    os.environ["SCENARIO_RUNNER_ROOT"] = str(_B2D_ROOT / "scenario_runner")


def carla_server_reachable() -> bool:
    """True if something is already accepting TCP on the CARLA rpc port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((_CARLA_HOST, _CARLA_PORT)) == 0


def ensure_carla_server() -> subprocess.Popen | None:
    """Make sure a CARLA server is up on the rpc port, launching one if needed.

    If the port is already serving (a server someone else started, or the
    train_carla.sh flow) this reuses it and returns ``None``. Otherwise it
    ``setsid``-launches ``CarlaUE4.sh -RenderOffScreen`` exactly like
    ``leaderboard_evaluator._setup_simulation``, registers an ``atexit`` hook to
    SIGKILL the whole process group on interpreter exit, and blocks until the
    rpc port is connectable. Returns the ``Popen`` for the launched server.
    """
    if carla_server_reachable():
        print(f"[carla_bootstrap] reusing CARLA already serving on port {_CARLA_PORT}", flush=True)
        return None

    root = resolve_carla_root()
    cmd = (
        f"{root / 'CarlaUE4.sh'} -RenderOffScreen -nosound "
        f"-carla-rpc-port={_CARLA_PORT} -graphicsadapter={_GPU_RANK}"
    )
    print(f"[carla_bootstrap] launching CARLA: {cmd}", flush=True)
    server = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)

    def _kill() -> None:
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    atexit.register(_kill)

    deadline = time.time() + _READY_TIMEOUT_S
    while time.time() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                f"CARLA server exited early (code {server.returncode}) — check the GPU "
                "is free and CarlaUE4.sh runs standalone."
            )
        if carla_server_reachable():
            # The rpc port is open; give the world a moment to finish booting
            # before the env's client.set_timeout / load_world.
            time.sleep(5.0)
            print(f"[carla_bootstrap] CARLA ready on port {_CARLA_PORT}", flush=True)
            return server
        time.sleep(2.0)

    _kill()
    raise TimeoutError(
        f"CARLA did not become reachable on port {_CARLA_PORT} within {_READY_TIMEOUT_S}s"
    )
