# SPDX-License-Identifier: MIT
"""SimLingo zero-shot agent — drops into ``scripts/train.py`` like an RL agent.

Implements the same surface (``select_action`` / ``step`` /
``on_episode_end`` + ``network`` + ``reward_processor`` attributes) the
RL agents in this directory expose, so the trainer's loop runs unchanged.
The "agent" is a wrapper around the SimLingo
``team_code/agent_simlingo.py`` ``LingoAgent`` class — it brings up the
VLM, then on each step feeds the agent the per-frame sensor snapshot
that :class:`CARLALeaderboardEnv` already publishes in
``info["sensors"]`` (rgb / gps / imu / speed), and converts the
returned ``carla.VehicleControl`` to the env's 2-D
``[steer, throttle_or_brake]`` action.

Because the env owns the sensor lifecycle, SimLingoAgent does **not**
spawn its own multi-camera stack or wire a leaderboard
``SensorInterface``. New-episode handover (set_global_plan, hero_actor,
re-init) is detected automatically via ``env.unwrapped.vehicle.id``
changing between ticks.

Caveats:
- SimLingo's PID/speed controllers were tuned for 20 FPS leaderboard
  ticks; this env runs at 10 FPS. Behavior is slightly different than
  under ``leaderboard_evaluator.py`` — acceptable for relative
  comparison against the RL policy on the same env, not directly
  comparable to simlingo's published eval numbers.
- The env's front camera is at ``(x=+1.5, z=2.4)``, while simlingo was
  trained at ``(x=-1.5, z=2.0)`` (config_simlingo.camera_pos_0). The
  viewpoints differ; closed-loop performance reflects that mismatch.
"""
import os
from pathlib import Path

import carla
import gymnasium as gym
import numpy as np
import torch
from srunner.scenariomanager.timer import GameTime

from vla_streaming_rl.simlingo.team_code.agent_simlingo import LingoAgent


class _NoopRewardProcessor:
    """RL trainer calls ``reward_processor.update(score)`` after each
    episode when ``normalizing_by_return`` is on. SimLingo doesn't train,
    so this is a sink."""

    def update(self, score: float) -> None:  # noqa: D401
        return None


class SimLingoAgent:
    """SimLingo policy with the RL-agent surface ``scripts/train.py`` expects."""

    # huggingface_hub repo id of the published SimLingo VLM weights.
    # Used when ``checkpoint_path`` is not provided.
    _HF_REPO_ID = "RenzKa/simlingo"
    _HF_CKPT_NAME = "pytorch_model.pt"

    def __init__(
        self,
        *,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        env: gym.Env,
        checkpoint_path: Path | None,
        scratch_dir: Path,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

        self._env_unwrapped = env.unwrapped
        self._scratch_dir = Path(scratch_dir)
        self._scratch_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path = self._resolve_checkpoint(checkpoint_path)

        # ``LingoAgent.setup`` writes ``SAVE_PATH + save_path_root`` and
        # crashes if SAVE_PATH is unset. Redirect to a per-run scratch
        # dir; none of these files feed our downstream metrics
        # (CARLALeaderboardEnv's eval_writer owns those).
        os.environ["SAVE_PATH"] = str(self._scratch_dir) + "/"

        # ``AutonomousAgent.__init__`` calls ``self.get_hero()`` which scans
        # CarlaDataProvider for an actor with role_name='hero'. At
        # construction time the env hasn't reset yet so there is no ego;
        # defer the lookup to the first ``select_action`` (we set
        # hero_actor explicitly there).
        class _DeferredHero(LingoAgent):
            def get_hero(self):
                self.hero_actor = None

        self._lingo = _DeferredHero(carla_host="", carla_port=2000, debug=False)
        # Use simlingo's upstream "<ckpt>+<save_path_root>" convention
        # (eval_220routes.sh appends ``+$save_name`` via
        # ``--agent-config``). ``route_index=None`` keeps
        # ``self.save_path`` a plain str so the later
        # ``save_path + "/debug_viz"`` concat works (passing a non-None
        # value converts it to PosixPath, which doesn't support ``+``).
        agent_config = f"{self._checkpoint_path}+vla_streaming_rl"
        self._lingo.setup(agent_config, route_index=None)

        # Freeze the VLM — we never backprop. Empty trainable_state at
        # checkpoint time then writes a (harmless) empty dict.
        for p in self._lingo.model.parameters():
            p.requires_grad_(False)

        # Duck-type the RL agent surface the trainer uses:
        # ``agent.network.parameters()`` for the parameter count print,
        # ``agent.network.named_parameters()`` for the checkpoint save.
        self.network = self._lingo.model
        self.reward_processor = _NoopRewardProcessor()

        # ``CARLALeaderboardEnv`` re-spawns the ego on every reset; a
        # changed actor id is our "new episode" signal.
        self._attached_ego_id: int | None = None

    # --- RL-agent protocol surface used by scripts/train.py -----------------

    def select_action(self, global_step, obs, reward, terminated, truncated, task_prompt):
        self._maybe_handover_episode()
        return self._act(), self._dummy_agent_info()

    def step(self, global_step, obs, reward, terminated, truncated, task_prompt):
        self._maybe_handover_episode()
        return self._act(), self._dummy_agent_info()

    def _dummy_agent_info(self) -> dict:
        """Trainer reads ``agent_info["goal_image"]`` (and ``next_image``/
        ``next_reward`` via ``.get(...)``) for its rendering pipeline. The
        zero-shot policy doesn't predict a goal, so hand back zeroed
        arrays shaped like the obs the env returns."""
        h, w = self.observation_space.shape[1], self.observation_space.shape[2]
        zero_hwc = np.zeros((h, w, 3), dtype=np.float32)
        return {"goal_image": zero_hwc}

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        # Force re-init on the next select_action.
        self._attached_ego_id = None
        return {}

    # --- Internals ----------------------------------------------------------

    @classmethod
    def _resolve_checkpoint(cls, checkpoint_path: Path | None) -> Path:
        """Use ``checkpoint_path`` if given, else pull from HF.

        ``snapshot_download`` populates the HF cache; we pick the single
        ``pytorch_model.pt`` inside (excluding the blob-store hardlinks,
        which point to the same file but live under a content-addressed
        path that breaks SimLingo's
        ``Path(...).parent.parent.parent / .hydra/config.yaml`` lookup).
        """
        if checkpoint_path is not None and str(checkpoint_path):
            p = Path(checkpoint_path)
            if not p.is_file():
                raise FileNotFoundError(f"SimLingo checkpoint not found: {p}")
            return p

        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(cls._HF_REPO_ID))
        candidates = [
            p for p in snapshot.rglob(cls._HF_CKPT_NAME) if "/blobs/" not in str(p)
        ]
        if not candidates:
            raise RuntimeError(
                f"no {cls._HF_CKPT_NAME} in HF snapshot of {cls._HF_REPO_ID} at {snapshot}"
            )
        return candidates[0]

    def _maybe_handover_episode(self) -> None:
        """When the env reset to a new scenario, hand the agent the new
        ego + route plan and force ``LingoAgent._init`` to re-run.
        """
        ego = self._env_unwrapped.vehicle
        if ego is None:
            raise RuntimeError("SimLingoAgent: env has no live ego — was env.reset() called?")
        if ego.id == self._attached_ego_id:
            return

        runtime = self._env_unwrapped.runtime
        if runtime is None or runtime.route_scenario is None:
            raise RuntimeError(
                "SimLingoAgent requires Bench2DriveRuntime with an active scenario"
            )
        # ``set_global_plan`` is the standard leaderboard handover —
        # RouteScenario builds both gps_route and world-coord route.
        self._lingo.set_global_plan(
            runtime.route_scenario.gps_route, runtime.route_scenario.route
        )
        self._lingo.hero_actor = ego
        self._lingo.initialized = False
        self._attached_ego_id = ego.id

    def _act(self) -> np.ndarray:
        """One inference tick + 2-D env-action conversion.

        Reads :meth:`CARLALeaderboardEnv._build_sensors_dict` (already
        in the leaderboard ``input_data`` shape ``{id: (frame, payload)}``)
        and remaps ``rgb`` → ``rgb_<N>`` per SimLingo's
        ``config_simlingo.num_cameras`` so the agent's ``tick`` sees the
        keys it expects.
        """
        sensors = self._env_unwrapped._build_sensors_dict()
        # SimLingo's sensors() declares per-camera ids ``rgb_{N}`` where
        # ``N`` iterates over ``config.num_cameras`` (typically just [0]).
        # We only have a single env camera, so map it to whatever id the
        # agent's first camera position uses.
        first_cam_id = self._lingo.config.num_cameras[0]
        input_data = {
            f"rgb_{first_cam_id}": sensors["rgb"],
            "gps": sensors["gps"],
            "imu": sensors["imu"],
            "speed": sensors["speed"],
        }
        control = self._lingo.run_step(input_data, GameTime.get_time())
        steer = float(control.steer)
        # 2-D env action: positive → throttle, negative → brake. SimLingo
        # never sets both at once in practice, so this collapse is
        # lossless for our purposes.
        gas_or_brake = float(control.throttle) - float(control.brake)
        return np.array([steer, gas_or_brake], dtype=np.float32)
