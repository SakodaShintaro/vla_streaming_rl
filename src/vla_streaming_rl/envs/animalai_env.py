# SPDX-License-Identifier: MIT
"""Animal-AI Gymnasium environment.

Wraps Animal-AI v5 (which exposes a Unity ML-Agents BehaviorSpec interface)
as a single-agent gym.Env. The native AAI action is MultiDiscrete([3, 3]):
    dim 0 (forward/back): 0=noop, 1=forward, 2=back
    dim 1 (rotate):       0=noop, 1=right,   2=left
We expose this as Box(-1, 1, shape=(2,)) for parity with the project's other
environments (CARLA, GUI games), discretising with a +/-1/3 dead-zone.

Two curricula are provided, both built on the shared ``_AnimalAIBase``:
  - ``AnimalAIEnv``: success-driven progression/revisit curriculum that walks
    every one of the 900 Testbed arenas.
  - ``AnimalAIStagedEnv``: the cumulative 11-stage curriculum from the
    Animal-AI Environment paper (Testbed section), trained on a subset of
    each task's 3 variants and evaluated on all 900 by sweeping
    ``_all_arenas`` with a frozen policy (see scripts/test_trained_agent.py).
"""

import random
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import yaml
from animalai import AnimalAIEnvironment
from gymnasium import spaces
from mlagents_envs.base_env import ActionTuple


def _to_discrete(value: float) -> int:
    """Map a continuous control in [-1, 1] to AAI's {0=noop, 1, 2}."""
    if value >= 1.0 / 3.0:
        return 1
    elif value <= -1.0 / 3.0:
        return 2
    return 0


# Animal-AI yaml uses custom !ArenaConfig/!Item/!Vector3/!RGB tags. We register
# them as plain mappings so yaml.load can parse the full structure.
class _AAILoader(yaml.SafeLoader):
    pass


def _aai_tag_constructor(loader, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


for _tag in ("!ArenaConfig", "!Arena", "!Item", "!Vector3", "!RGB"):
    _AAILoader.add_constructor(_tag, _aai_tag_constructor)


def _parse_arena(yaml_path: str) -> tuple[float, list[dict]]:
    """Return (pass_mark, items) from an Animal-AI arena yaml.

    Each item dict carries one (position, size, rotation) triple in arena
    coordinates: {name, x, z, size_x, size_z, rotation}. yaml entries with
    multiple positions are expanded into multiple item dicts; if `sizes` is
    shorter than `positions` the last given size is reused (AAI's convention).
    """
    cfg = yaml.load(Path(yaml_path).read_text(), Loader=_AAILoader)
    arena = cfg["arenas"][0]
    pass_mark = float(arena.get("pass_mark", 0))
    items_out: list[dict] = []
    for item in arena.get("items", []) or []:
        name = item["name"]
        positions = item.get("positions") or []
        sizes = item.get("sizes") or []
        rotations = item.get("rotations") or []
        for i, pos in enumerate(positions):
            size = sizes[i] if i < len(sizes) else (sizes[-1] if sizes else None)
            rot = rotations[i] if i < len(rotations) else (rotations[-1] if rotations else 0)
            items_out.append(
                {
                    "name": name,
                    "x": float(pos["x"]),
                    "z": float(pos["z"]),
                    "size_x": float(size["x"]) if size else 1.0,
                    "size_z": float(size["z"]) if size else 1.0,
                    "rotation": float(rot),
                }
            )
    return pass_mark, items_out


def _discover_arena_sets(competition_dir: Path) -> np.ndarray:
    """Return a 3D ndarray of arena name stems, axes [level, task, variant].

    The Animal-AI Olympics directory contains files named XX-YY-ZZ.yaml where
    XX = level (01..10), YY = task (01..30), ZZ = variant (01..03). Filenames
    map directly: arena_sets[level_idx, task_idx, variant_idx] == "XX-YY-ZZ".
    Assumes the (level, task, variant) grid is dense.
    """
    by_key: dict[tuple[int, int, int], str] = {}
    for p in competition_dir.glob("*.yaml"):
        parts = p.stem.split("-")
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            by_key[(int(parts[0]), int(parts[1]), int(parts[2]))] = p.stem
    levels = sorted({k[0] for k in by_key})
    tasks = sorted({k[1] for k in by_key})
    variants = sorted({k[2] for k in by_key})
    arr = np.empty((len(levels), len(tasks), len(variants)), dtype=object)
    for i, lv in enumerate(levels):
        for j, tk in enumerate(tasks):
            for k, vr in enumerate(variants):
                arr[i, j, k] = by_key[(lv, tk, vr)]
    return arr


# RGB colors (matplotlib-style) for each AAI item type when drawn top-down.
_ITEM_COLORS: dict[str, tuple[int, int, int]] = {
    "GoodGoal": (40, 200, 40),
    "GoodGoalMulti": (40, 230, 80),
    "GoodGoalBounce": (40, 200, 120),
    "GoodGoalMultiBounce": (40, 230, 160),
    "BadGoal": (220, 40, 40),
    "BadGoalBounce": (220, 100, 40),
    "DeathZone": (140, 20, 20),
    "HotZone": (220, 120, 40),
    "Wall": (110, 110, 110),
    "WallTransparent": (200, 200, 220),
    "Ramp": (140, 100, 60),
    "Cardbox1": (210, 160, 80),
    "Cardbox2": (180, 130, 70),
    "LObject": (220, 200, 50),
    "LObject2": (200, 180, 50),
    "UObject": (220, 220, 50),
    "CylinderTunnel": (160, 190, 210),
    "CylinderTunnelTransparent": (200, 220, 230),
}
_DEFAULT_ITEM_COLOR: tuple[int, int, int] = (180, 180, 180)
_AGENT_COLOR: tuple[int, int, int] = (40, 80, 220)
_ARENA_SIZE_M = 40.0  # standard Animal-AI arena is 40 m square


class _AnimalAIBase(gym.Env):
    """Shared Unity lifecycle, observation decoding, and pass-mark reward
    logic for every Animal-AI curriculum variant.

    Subclasses implement:
      - `_select_arena_yaml()`: which arena to load on a (non-forced) reset.
      - `_on_forced_reset()`: hook run when `options["arena_stem"]` bypasses
        the curriculum (used by the eval sweep over all 900 arenas).
      - `_on_step()`: hook run on every `step()` call, before episode-end
        bookkeeping (e.g. to advance a global step counter).
      - `_record_episode_end(success)`: curriculum bookkeeping on episode end;
        returns extra `info` fields.
      - `_reset_info_fields()`: extra `info` fields to attach on reset.
      - `_render_header_text()`: one-line status string drawn on the top-down
        render.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, resolution: int, seed: int, base_port: int):
        super().__init__()
        self.competition_dir = Path("./external/animal-ai/configs/competition")
        self.arena_sets = _discover_arena_sets(self.competition_dir)
        if self.arena_sets.size == 0:
            raise ValueError(f"no XX-YY-ZZ.yaml files found under {self.competition_dir}")
        # Flat sorted list of every yaml stem (all 900 Testbed arenas).
        self._all_arenas: list[str] = sorted(self.arena_sets.flatten().tolist())
        self.prompt = "Find and reach the green goal sphere; avoid red zones and yellow goals."

        self.binary_path = str(Path.home() / "animalai_env" / "Linux" / "animalAI.x86_64")
        self.resolution = resolution
        self.seed_value = seed
        # Add jitter so parallel envs don't fight over the same socket.
        self.base_port = base_port + random.randint(0, 1000)
        self.render_mode = "rgb_array"

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(resolution, resolution, 3), dtype=np.uint8
        )

        self._aai: AnimalAIEnvironment | None = None
        self._behavior_name: str | None = None
        self._latest_image: np.ndarray | None = None
        self.episode_step = 0
        self._episode_return = 0.0
        self.arena_name: str = ""
        self.pass_mark: float = 0.0
        self._arena_items: list[dict] = []
        # (x, y, z) agent position in arena coords; populated each step from
        # the AAI vector observation. None before the first reset.
        self._agent_xyz: tuple[float, float, float] | None = None
        # (vx, vy, vz) agent velocity from the AAI vector observation.
        self._agent_velocity: np.ndarray = np.zeros(3, dtype=np.float32)

    def _select_arena_yaml(self) -> str:
        raise NotImplementedError

    def _on_forced_reset(self) -> None:
        raise NotImplementedError

    def _on_step(self) -> None:
        raise NotImplementedError

    def _record_episode_end(self, success: bool) -> dict:
        raise NotImplementedError

    def _reset_info_fields(self) -> dict:
        raise NotImplementedError

    def _render_header_text(self) -> str:
        raise NotImplementedError

    def _ensure_started(self):
        if self._aai is not None:
            return
        boot_yaml = str(self.competition_dir / f"{self._all_arenas[0]}.yaml")
        self._aai = AnimalAIEnvironment(
            file_name=self.binary_path,
            arenas_configurations=boot_yaml,
            seed=self.seed_value,
            play=False,
            useCamera=True,
            resolution=self.resolution,
            useRayCasts=False,
            decisionPeriod=5,
            # `--no-graphics-monitor` enables off-screen rendering on a host
            # without a window manager. `no_graphics=True` (the alternative
            # for headless) disables the renderer entirely and produces a
            # solid-colour image, which is unusable for vision policies.
            no_graphics=False,
            additional_args=["--no-graphics-monitor"],
            base_port=self.base_port,
            inference=False,
            use_YAML=True,
        )
        self._behavior_name = next(iter(self._aai.behavior_specs.keys()))

    def _decode_obs(self, obs_chw_float: np.ndarray) -> np.ndarray:
        # AAI emits float32 in [0, 1] with shape (3, H, W).
        return (obs_chw_float.transpose(1, 2, 0) * 255.0).astype(np.uint8)

    @staticmethod
    def _extract_agent_xyz(obs_list, idx: int) -> tuple[float, float, float]:
        # Vector obs layout (useCamera=True, useRayCasts=False): obs[1][idx] is
        # [health, vx, vy, vz, x, y, z].
        vec = obs_list[1][idx]
        return float(vec[4]), float(vec[5]), float(vec[6])

    @staticmethod
    def _extract_agent_velocity(obs_list, idx: int) -> np.ndarray:
        # Same vector obs layout: obs[1][idx][1:4] is (vx, vy, vz).
        vec = obs_list[1][idx]
        return np.array([float(vec[1]), float(vec[2]), float(vec[3])], dtype=np.float32)

    def _build_prompt(self) -> str:
        """Task text plus the current scalar observations, so a language-conditioned
        policy reads the same values the scalar observation vector carries."""
        vx, vy, vz = self._agent_velocity
        return (
            f"{self.prompt} "
            f"Velocity: ({vx:+.2f}, {vy:+.2f}, {vz:+.2f}). "
            f"Return so far: {self._episode_return:+.2f}. "
            f"Pass mark: {self.pass_mark:+.2f}."
        )

    def _render_topdown(self) -> np.ndarray:
        img_size = 256
        scale = img_size / _ARENA_SIZE_M
        canvas = np.full((img_size, img_size, 3), 240, dtype=np.uint8)

        # Flip z so the +z arena axis points up in the image (Unity-editor-like).
        def to_px(x: float, z: float) -> tuple[int, int]:
            return int(round(x * scale)), int(round((_ARENA_SIZE_M - z) * scale))

        cv2.rectangle(canvas, (0, 0), (img_size - 1, img_size - 1), (60, 60, 60), 2)

        for item in self._arena_items:
            if item["name"] == "Agent":
                continue
            # x or z = -1 in the yaml means Unity randomizes the position at
            # reset; we don't know the actual location, so skip drawing.
            if item["x"] < 0 or item["z"] < 0:
                continue
            cx, cy = to_px(item["x"], item["z"])
            color = _ITEM_COLORS.get(item["name"], _DEFAULT_ITEM_COLOR)
            # Goals are spherical in the Unity scene; draw as circles so they
            # are visually distinct from rectangular zones/walls/boxes.
            if "Goal" in item["name"]:
                radius = max(int(0.5 * item["size_x"] * scale), 3)
                cv2.circle(canvas, (cx, cy), radius, color, cv2.FILLED)
                cv2.circle(canvas, (cx, cy), radius, (20, 20, 20), 1)
                continue
            sx_px = max(item["size_x"] * scale, 3.0)
            sz_px = max(item["size_z"] * scale, 3.0)
            # Unity Y-axis rotation is CW from above (in left-handed world);
            # our z-flipped image preserves "north up" so we pass the raw angle
            # to cv2 (positive cv2 angle is CW in image after the z flip).
            rect = ((float(cx), float(cy)), (sx_px, sz_px), item["rotation"])
            box = np.intp(cv2.boxPoints(rect))
            transparent = "Transparent" in item["name"]
            thickness = 1 if transparent else cv2.FILLED
            cv2.drawContours(canvas, [box], 0, color, thickness)

        if self._agent_xyz is not None:
            ax, _ay, az = self._agent_xyz
            apx, apy = to_px(ax, az)
            radius = max(int(0.6 * scale), 4)
            cv2.circle(canvas, (apx, apy), radius, _AGENT_COLOR, cv2.FILLED)
            cv2.circle(canvas, (apx, apy), radius, (20, 20, 20), 1)

        header_height = 22
        header = np.full((header_height, img_size, 3), 215, dtype=np.uint8)
        if self.arena_name:
            cv2.putText(
                header,
                self._render_header_text(),
                (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        return np.vstack([header, canvas])

    def reset(
        self,
        seed: int | None,
        options: dict | None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._ensure_started()

        # `options={"arena_stem": ...}` bypasses the curriculum entirely
        # (used by scripts/test_trained_agent.py to sweep every arena exactly
        # once with a frozen policy).
        forced_stem = (
            options["arena_stem"] if options is not None and "arena_stem" in options else None
        )
        if forced_stem is not None:
            self._on_forced_reset()
            arena_yaml = str(self.competition_dir / f"{forced_stem}.yaml")
        else:
            arena_yaml = self._select_arena_yaml()

        self.arena_name = Path(arena_yaml).stem
        self.pass_mark, self._arena_items = _parse_arena(arena_yaml)
        self._aai.reset(arenas_configurations=arena_yaml)
        self.episode_step = 0
        self._episode_return = 0.0

        decision_steps, _ = self._aai.get_steps(self._behavior_name)
        self._latest_image = self._decode_obs(decision_steps.obs[0][0])
        self._agent_xyz = self._extract_agent_xyz(decision_steps.obs, 0)
        self._agent_velocity = self._extract_agent_velocity(decision_steps.obs, 0)
        info = {
            "task_prompt": self._build_prompt(),
            "arena_yaml": arena_yaml,
            "arena_name": self.arena_name,
            "pass_mark": self.pass_mark,
            "velocity": self._agent_velocity,
            "agent_xyz": self._agent_xyz,
        }
        info.update(self._reset_info_fields())
        return self._latest_image, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._on_step()
        a = np.asarray(action, dtype=np.float32)
        discrete = np.array(
            [[_to_discrete(float(a[0])), _to_discrete(float(a[1]))]], dtype=np.int32
        )
        action_tuple = ActionTuple(
            continuous=np.zeros((1, 0), dtype=np.float32),
            discrete=discrete,
        )
        self._aai.set_actions(self._behavior_name, action_tuple)
        self._aai.step()

        decision_steps, terminal_steps = self._aai.get_steps(self._behavior_name)
        episode_over = len(terminal_steps) > 0
        if episode_over:
            # `interrupted` is AAI's max-step (arena `t`) timeout, i.e. a time
            # limit rather than a real terminal (goal reached / death zone).
            interrupted = bool(terminal_steps.interrupted[0])
            reward = float(terminal_steps.reward[0])
            obs_source = terminal_steps
        else:
            interrupted = False
            reward = float(decision_steps.reward[0])
            obs_source = decision_steps
        self._latest_image = self._decode_obs(obs_source.obs[0][0])
        self._agent_xyz = self._extract_agent_xyz(obs_source.obs, 0)
        self._agent_velocity = self._extract_agent_velocity(obs_source.obs, 0)

        self.episode_step += 1
        self._episode_return += reward
        terminated = episode_over and not interrupted
        truncated = episode_over and interrupted

        info = {
            "task_prompt": self._build_prompt(),
            "episode_step": self.episode_step,
            "arena_name": self.arena_name,
            "pass_mark": self.pass_mark,
            "velocity": self._agent_velocity,
            "agent_xyz": self._agent_xyz,
        }
        if terminated or truncated:
            success = self._episode_return >= self.pass_mark
            # Pass-mark bonus: reward crossing the clear threshold, penalize missing it.
            # Computed from the pre-bonus return, then folded into this tick's reward.
            reward += 1.0 if success else -1.0
            info["success"] = bool(success)
            info.update(self._record_episode_end(success))
        return self._latest_image, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        return self._render_topdown()

    def close(self):
        if self._aai is not None:
            self._aai.close()
            self._aai = None


class AnimalAIEnv(_AnimalAIBase):
    """Success-driven progression/revisit curriculum over all 900 arenas.

    A pointer walks the flat, sorted list of every XX-YY-ZZ arena. On success
    the current yaml is added to `_cleared_arenas` and the pointer advances
    past any yamls already cleared. On failure the pointer stays. Each reset
    alternates between progression and revisit modes once any arena has been
    cleared. Revisits sample a cleared arena with probability
    softmax(failure_rate / revisit_temperature).
    """

    def __init__(
        self,
        resolution: int,
        seed: int,
        base_port: int,
        revisit_temperature: float,
    ):
        super().__init__(resolution=resolution, seed=seed, base_port=base_port)
        self.revisit_temperature = revisit_temperature
        self._next_yaml_idx = 0
        self._cleared_arenas: set[str] = set()
        self._is_revisit = False
        # Per-arena episode counts over every attempt (progression and revisit).
        self._arena_attempts: dict[str, int] = dict.fromkeys(self._all_arenas, 0)
        self._arena_successes: dict[str, int] = dict.fromkeys(self._all_arenas, 0)

    def _current_progression_stem(self) -> str:
        idx = min(self._next_yaml_idx, len(self._all_arenas) - 1)
        return self._all_arenas[idx]

    def _current_yaml_path(self) -> str:
        return str(self.competition_dir / f"{self._current_progression_stem()}.yaml")

    def _advance_progression(self) -> None:
        """Move the progression pointer past any yamls already in cleared."""
        n = len(self._all_arenas)
        while (
            self._next_yaml_idx < n
            and self._all_arenas[self._next_yaml_idx] in self._cleared_arenas
        ):
            self._next_yaml_idx += 1

    def _success_rate(self, arena_stem: str) -> float:
        """Success rate over all attempts. Cleared arenas always have >=1 attempt."""
        return self._arena_successes[arena_stem] / self._arena_attempts[arena_stem]

    def get_curriculum_state(self) -> dict:
        """Curriculum snapshot for resume: per-arena attempt/success counts
        (attempted arenas only) plus the progression/revisit alternation flag."""
        attempts = {k: v for k, v in self._arena_attempts.items() if v > 0}
        successes = {k: self._arena_successes[k] for k in attempts}
        return {
            "arena_attempts": attempts,
            "arena_successes": successes,
            "is_revisit": self._is_revisit,
        }

    def set_curriculum_state(
        self, arena_attempts: dict, arena_successes: dict, is_revisit: bool
    ) -> None:
        """Restore the curriculum from a previous run. Cleared arenas and the
        progression pointer are recomputed from the success counts (an arena is
        cleared iff it has ever succeeded). Arena names unknown to the current
        competition directory are ignored."""
        known = set(self._all_arenas)
        self._arena_attempts.update({k: int(v) for k, v in arena_attempts.items() if k in known})
        self._arena_successes.update({k: int(v) for k, v in arena_successes.items() if k in known})
        self._is_revisit = bool(is_revisit)
        self._cleared_arenas = {k for k, v in self._arena_successes.items() if v > 0}
        self._next_yaml_idx = 0
        self._advance_progression()

    def _select_arena_yaml(self) -> str:
        """Curriculum arena choice: alternate progression / revisit each reset."""
        all_done = self._next_yaml_idx >= len(self._all_arenas)
        if not self._cleared_arenas:
            self._is_revisit = False
        elif all_done:
            self._is_revisit = True
        else:
            self._is_revisit = not self._is_revisit
        if not self._is_revisit:
            return self._current_yaml_path()
        stems = sorted(self._cleared_arenas)
        failures = np.array([1.0 - self._success_rate(stem) for stem in stems], dtype=np.float64)
        logits = failures / self.revisit_temperature
        weights = np.exp(logits - logits.max())
        probs = weights / weights.sum()
        arena_stem = stems[int(self.np_random.choice(len(stems), p=probs))]
        return str(self.competition_dir / f"{arena_stem}.yaml")

    def _on_forced_reset(self) -> None:
        self._is_revisit = False

    def _on_step(self) -> None:
        pass

    def _record_episode_end(self, success: bool) -> dict:
        self._arena_attempts[self.arena_name] += 1
        self._arena_successes[self.arena_name] += int(success)
        if self._is_revisit:
            advanced = False
        else:
            before = self._next_yaml_idx
            if success:
                self._cleared_arenas.add(self.arena_name)
                self._advance_progression()
            advanced = self._next_yaml_idx != before
        return {"advanced": advanced, "cleared_count": len(self._cleared_arenas)}

    def _reset_info_fields(self) -> dict:
        return {"cleared_count": len(self._cleared_arenas), "is_revisit": self._is_revisit}

    def _render_header_text(self) -> str:
        tag = " R" if self._is_revisit else ""
        return (
            f"{self.arena_name}  cleared:{len(self._cleared_arenas)}/{len(self._all_arenas)}{tag}"
        )


class AnimalAIStagedEnv(_AnimalAIBase):
    """Cumulative 11-stage curriculum reproducing the Animal-AI Environment
    paper's Testbed protocol (Section 4.1.3).

    Stage i (0-indexed, 0..9) trains on `train_variant_count` of the 3
    variants of every task in levels 1..(i+1). Each of the first 9 stages
    runs for `steps_per_stage` steps; the final stage (index 9, all 10
    levels) then runs for the remainder of `step_limit` (the paper's "extra
    5M steps on the last stage" is simply the tail of this same phase, since
    the arena set does not change again). Training only ever samples the
    reserved `train_variant_count` variants per task -- the remaining
    variants (and, if `train_variant_count == 1`, the majority of the
    Testbed) are held out for the frozen 900-arena evaluation sweep in
    scripts/test_trained_agent.py.
    """

    def __init__(
        self,
        resolution: int,
        seed: int,
        base_port: int,
        train_variant_count: int,
        steps_per_stage: int,
    ):
        super().__init__(resolution=resolution, seed=seed, base_port=base_port)
        n_variants = self.arena_sets.shape[2]
        if not (1 <= train_variant_count < n_variants):
            raise ValueError(
                f"train_variant_count must be in [1, {n_variants - 1}], got {train_variant_count}"
            )
        self.train_variant_count = train_variant_count
        self.steps_per_stage = steps_per_stage
        self._stage_arenas: list[list[str]] = self._build_stage_arenas()
        self._global_step = 0

    def _build_stage_arenas(self) -> list[list[str]]:
        n_levels, n_tasks, _n_variants = self.arena_sets.shape
        stages = []
        for stage_i in range(n_levels):
            arenas = [
                self.arena_sets[lv, tk, vr]
                for lv in range(stage_i + 1)
                for tk in range(n_tasks)
                for vr in range(self.train_variant_count)
            ]
            stages.append(arenas)
        return stages

    def set_global_step(self, global_step: int) -> None:
        """Resume hook: called once after loading a checkpoint (see
        scripts/train.py's analogous `set_curriculum_state` for AnimalAIEnv)."""
        self._global_step = int(global_step)

    def _current_stage_index(self) -> int:
        n_stages = len(self._stage_arenas)
        return min(self._global_step // self.steps_per_stage, n_stages - 1)

    def _select_arena_yaml(self) -> str:
        stage = self._current_stage_index()
        arenas = self._stage_arenas[stage]
        arena_stem = arenas[int(self.np_random.integers(len(arenas)))]
        return str(self.competition_dir / f"{arena_stem}.yaml")

    def _on_forced_reset(self) -> None:
        pass

    def _on_step(self) -> None:
        self._global_step += 1

    def _record_episode_end(self, success: bool) -> dict:
        return {"stage": self._current_stage_index(), "global_step": self._global_step}

    def _reset_info_fields(self) -> dict:
        return {"stage": self._current_stage_index(), "global_step": self._global_step}

    def _render_header_text(self) -> str:
        return (
            f"{self.arena_name}  stage:{self._current_stage_index() + 1}/{len(self._stage_arenas)}"
            f"  step:{self._global_step}"
        )


if __name__ == "__main__":
    env = AnimalAIEnv(resolution=96, seed=0, base_port=5005, revisit_temperature=1.0)
    for ep in range(8):
        obs, info = env.reset(seed=ep, options=None)
        print(
            f"ep={ep} arena={info['arena_name']} "
            f"revisit={info['is_revisit']} cleared={info['cleared_count']}"
        )
        total_reward = 0.0
        for i in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        print(
            f"  return={total_reward:.4f} success={info.get('success')} "
            f"advanced={info.get('advanced')}"
        )
    env.close()
