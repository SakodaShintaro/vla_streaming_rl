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

# The env's shared client (get_client) connects here; port 2000 is CARLA's rpc
# default that CARLALeaderboardEnv assumes.
_CARLA_HOST = "localhost"
_CARLA_PORT = 2000
_GPU_RANK = 0
_READY_TIMEOUT_S = 180.0
# Repo root is three levels up from src/vla_streaming_rl/envs/carla_bootstrap.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
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
    for p in (
        root / "PythonAPI" / "carla",
        _B2D_ROOT / "leaderboard",
        _B2D_ROOT / "scenario_runner",
    ):
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


# Process-wide CARLA server + client, created once and reused by every env
# instance (i.e. every trial). This is the crux of multi-trial support: CARLA's
# Python client keeps background streaming threads (rpc 2000 + streaming 2001)
# alive for the life of the interpreter, so constructing a *second* carla.Client
# in one process — what a fresh CARLALeaderboardEnv per trial would do — leaves
# the first session's threads interfering with the new one and the simulator
# becomes unreachable (300 s time-out / libcarla segfault on the 2nd trial).
# Reusing one server + one client for the whole process mirrors Bench2Drive's
# leaderboard_evaluator (one client across all routes) and makes ``trial_num>1``
# work in-process — no per-env special-casing in train.py.
_SERVER: subprocess.Popen | None = None
_CLIENT = None
_ATEXIT_REGISTERED = False


def kill_carla_server() -> None:
    """Reap every CARLA server so the next launch starts clean.

    CARLA segfaults are a normal, expected event (Bench2Drive's own
    leaderboard_evaluator detects ``crashed`` and ``kill -9``s the server, then
    relies on an outer harness to restart) — so cleanup must NOT assume a tidy
    exit. We SIGKILL the process group we launched *and* ``pkill`` any
    ``CarlaUE4`` left over from a crashed previous run / trial (whose handle we
    no longer hold), because reusing a worn or half-dead server is what
    corrupts the next ``carla.Client`` session. Then wait for the rpc port to
    free up. Safe to call when nothing is running.
    """
    global _SERVER
    if _SERVER is not None and _SERVER.poll() is None:
        try:
            os.killpg(os.getpgid(_SERVER.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    _SERVER = None
    # Backstop for orphans we have no handle to (e.g. a run that segfaulted, so
    # its atexit never ran). The env only ever talks to localhost:2000, so a
    # single CARLA is the only relevant one — matching CARLALeaderboardEnv.
    subprocess.run(["pkill", "-9", "-f", "CarlaUE4"], check=False)
    deadline = time.time() + 30.0
    while time.time() < deadline and carla_server_reachable():
        time.sleep(1.0)


def ensure_carla_server() -> subprocess.Popen:
    """Launch the process-wide CARLA server once and block until it is ready.

    Reuses the server this process already launched, so every trial shares ONE
    simulator (see the ``_SERVER`` / ``_CLIENT`` note above). On the first call
    it first reaps any stale CARLA left by a crashed earlier run
    (:func:`kill_carla_server` — ``atexit`` is skipped on SIGSEGV so orphans
    happen), then ``setsid``-launches a fresh server like
    ``leaderboard_evaluator._setup_simulation`` and registers an atexit
    group-kill.
    """
    global _SERVER, _ATEXIT_REGISTERED

    if _SERVER is not None and _SERVER.poll() is None:
        return _SERVER

    kill_carla_server()

    root = resolve_carla_root()
    cmd = (
        f"{root / 'CarlaUE4.sh'} -RenderOffScreen -nosound "
        f"-carla-rpc-port={_CARLA_PORT} -graphicsadapter={_GPU_RANK}"
    )
    print(f"[carla_bootstrap] launching CARLA: {cmd}", flush=True)
    _SERVER = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    # Register the group-kill once; it reads the module global so it always
    # targets the current server even after later relaunches.
    if not _ATEXIT_REGISTERED:
        atexit.register(kill_carla_server)
        _ATEXIT_REGISTERED = True

    deadline = time.time() + _READY_TIMEOUT_S
    while time.time() < deadline:
        if _SERVER.poll() is not None:
            raise RuntimeError(
                f"CARLA server exited early (code {_SERVER.returncode}) — check the GPU "
                "is free and CarlaUE4.sh runs standalone."
            )
        if carla_server_reachable():
            # The rpc port is open; give the world a moment to finish booting
            # before the env's client.set_timeout / load_world.
            time.sleep(5.0)
            print(f"[carla_bootstrap] CARLA ready on port {_CARLA_PORT}", flush=True)
            return _SERVER
        time.sleep(2.0)

    kill_carla_server()
    raise TimeoutError(
        f"CARLA did not become reachable on port {_CARLA_PORT} within {_READY_TIMEOUT_S}s"
    )


def get_client():
    """Return the process-wide ``carla.Client``, creating it once.

    ``CARLALeaderboardEnv`` must use this instead of constructing its own
    ``carla.Client``: a 2nd client in one process keeps the first's streaming
    threads alive and corrupts the session, so reusing a single client for the
    whole process is what lets ``trial_num>1`` run in-process (see the
    ``_SERVER`` / ``_CLIENT`` note above). Ensures the server is up first.
    """
    global _CLIENT
    ensure_carla_server()
    if _CLIENT is None:
        import carla

        _CLIENT = carla.Client(_CARLA_HOST, _CARLA_PORT)
        # Eval uses 300 s; large-map (Town12/13/15) actor spawns can blow past
        # 120 s while tiles stream in.
        _CLIENT.set_timeout(300.0)
    return _CLIENT
