# SPDX-License-Identifier: MIT
"""Animal-AI Gymnasium environment.

Wraps Animal-AI v5 (which exposes a Unity ML-Agents BehaviorSpec interface)
as a single-agent gym.Env. The native AAI action is MultiDiscrete([3, 3]):
    dim 0 (forward/back): 0=noop, 1=forward, 2=back
    dim 1 (rotate):       0=noop, 1=right,   2=left
We expose this as Box(-1, 1, shape=(2,)) for parity with the project's other
environments (CARLA, GUI games), discretising with a +/-1/3 dead-zone.

Everything about talking to Unity lives in `AnimalAIEnv`. The only thing that
differs between the three ways we run Animal-AI is *which arena each episode
loads*, so that is factored out into an `ArenaSelector`, picked by the
`mode` config field (see `build_selector`):

  - "staged"  training with the paper's cumulative 11-stage curriculum, one
              stage per subdirectory of the root (`StagedSelector`).
  - "success" training on the same arenas, ordered by which ones the agent
              has already passed (`SuccessDrivenSelector`).
  - "eval"    every configs/competition/ arena once, in order -- the
              900-arena Testbed sweep (`SequentialSelector`).

Which arenas a selector serves is just a root directory it reads recursively,
so an augmented training set is a matter of pointing at a different root. The
training root is configurable; "eval" is pinned to COMPETITION_DIR so the
benchmark cannot drift (see `build_selector`). Training and evaluation
therefore never share an arena.
"""

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import yaml
from animalai import AnimalAIEnvironment
from gymnasium import spaces
from mlagents_envs.base_env import ActionTuple

COMPETITION_DIR = Path("./external/animal-ai/configs/competition")


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


def _parse_arena(yaml_path: Path) -> tuple[float, list[dict]]:
    """Return (pass_mark, items) from an Animal-AI arena yaml.

    Each item dict carries one (position, size, rotation) triple in arena
    coordinates: {name, x, z, size_x, size_z, rotation}. yaml entries with
    multiple positions are expanded into multiple item dicts; if `sizes` is
    shorter than `positions` the last given size is reused (AAI's convention).
    """
    cfg = yaml.load(yaml_path.read_text(), Loader=_AAILoader)
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


@dataclass(frozen=True)
class Arena:
    """One arena yaml plus the label it is reported under (info, logs, render)."""

    path: Path
    name: str


def _arenas_in(directory: Path, label_root: Path) -> list[Arena]:
    """Arena yamls under `directory`, labelled by their path relative to
    `label_root` (minus the suffix): "01-01-01" for a flat set,
    "stage00/arena000" for a stage-split one, where the bare filename would
    repeat across stage dirs for unrelated tasks."""
    paths = sorted(directory.rglob("*.yaml"))
    assert paths, f"no arena yamls under {directory}"
    return [Arena(path, str(path.relative_to(label_root).with_suffix(""))) for path in paths]


def _arenas_under(root: Path) -> list[Arena]:
    """Every arena yaml under `root`, recursively, in path order."""
    return _arenas_in(root, root)


def _stages_under(root: Path) -> list[list[Arena]]:
    """Cumulative arena list per curriculum stage, one stage per subdirectory.

    scripts/split_paper_curriculum.py writes only each stage's *new* arenas
    into stageNN/, so stage i's arena set is stage00..stageNN(i) concatenated.
    The per-stage count therefore follows whatever is on disk -- 30 arenas per
    stage as split, 60 once mirrored copies are generated alongside them.
    """
    stage_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    assert stage_dirs, f"no stage subdirectories under {root}; expected one per curriculum stage"
    cumulative: list[Arena] = []
    stages: list[list[Arena]] = []
    for stage_dir in stage_dirs:
        cumulative = cumulative + _arenas_in(stage_dir, root)
        stages.append(cumulative)
    return stages


class ArenaSelector:
    """Picks the arena each episode runs, and owns whatever progress that
    choice depends on.

    `arenas` is every arena this selector can serve: `AnimalAIEnv` boots Unity
    on the first of them and resolves `options={"arena_stem": ...}` against
    their names.
    """

    arenas: list[Arena]

    @property
    def is_exhausted(self) -> bool:
        """True once there is nothing left to run (finite sweeps only)."""
        return False

    def next_arena(self, global_step: int) -> Arena:
        raise NotImplementedError

    def on_episode_end(self, arena: Arena, success: bool) -> None:
        """Progress bookkeeping. Selectors without progress do nothing."""

    def info(self, global_step: int) -> dict:
        """Selector-specific fields merged into each reset/step info dict."""
        return {}

    def status(self, global_step: int) -> str:
        """One-line progress summary drawn on the top-down render."""
        raise NotImplementedError

    def state(self) -> dict:
        """Resume snapshot; selectors without progress restore to a fresh run."""
        return {"arena_attempts": {}, "arena_successes": {}, "is_revisit": False}

    def load_state(self, arena_attempts: dict, arena_successes: dict, is_revisit: bool) -> None:
        """Restore a `state()` snapshot. Selectors without progress ignore it."""


class StagedSelector(ArenaSelector):
    """The paper's cumulative curriculum (Animal-AI paper, Section 4.1.3).

    One stage per subdirectory of `root`, plus a final stage repeating the
    last one, so the 10 split stages give 11. Stage i samples uniformly from
    every arena of stages 1..i, and runs for `steps_per_stage` steps; stage 11
    runs until train.py's step_limit ends the run -- the paper's "further five
    million steps on the last stage".
    """

    def __init__(self, root: Path, steps_per_stage: int, seed: int):
        stages = _stages_under(root)
        self._stages = stages + [stages[-1]]
        self.arenas = stages[-1]
        self.steps_per_stage = steps_per_stage
        self._rng = np.random.default_rng(seed)

    def _stage_index(self, global_step: int) -> int:
        return min(global_step // self.steps_per_stage, len(self._stages) - 1)

    def next_arena(self, global_step: int) -> Arena:
        stage_arenas = self._stages[self._stage_index(global_step)]
        return stage_arenas[int(self._rng.integers(len(stage_arenas)))]

    def info(self, global_step: int) -> dict:
        return {"stage": self._stage_index(global_step) + 1}

    def status(self, global_step: int) -> str:
        # Arena count included so a run's logs record which arena set was used
        # (e.g. 300 as split vs 600 with mirrored copies).
        return (
            f"stage:{self._stage_index(global_step) + 1}/{len(self._stages)}"
            f"  arenas:{len(self.arenas)}"
        )


class SuccessDrivenSelector(ArenaSelector):
    """Progression/revisit over every arena under `root`, in path order.

    A pointer walks the arena list; an arena becomes "cleared" the first time
    it is passed, and the pointer skips anything already cleared. Once
    anything is cleared, resets alternate progression and revisit; revisits
    sample a cleared arena with probability softmax(failure_rate /
    `revisit_temperature`), so the shakiest cleared arenas come back most
    often and a lower temperature concentrates harder on them.
    """

    def __init__(self, root: Path, revisit_temperature: float, seed: int):
        self.arenas = _arenas_under(root)
        self.revisit_temperature = revisit_temperature
        self._by_name = {arena.name: arena for arena in self.arenas}
        self._rng = np.random.default_rng(seed)
        self._next_index = 0
        self._cleared: set[str] = set()
        self._attempts = dict.fromkeys(self._by_name, 0)
        self._successes = dict.fromkeys(self._by_name, 0)
        self._is_revisit = False
        self._advanced = False

    def _skip_cleared(self) -> None:
        while (
            self._next_index < len(self.arenas)
            and self.arenas[self._next_index].name in self._cleared
        ):
            self._next_index += 1

    def _pick_revisit_mode(self) -> bool:
        if not self._cleared:
            return False
        if self._next_index >= len(self.arenas):
            return True
        return not self._is_revisit

    def _sample_cleared(self) -> Arena:
        names = sorted(self._cleared)
        failure_rate = np.array(
            [1.0 - self._successes[name] / self._attempts[name] for name in names]
        )
        logits = failure_rate / self.revisit_temperature
        weights = np.exp(logits - logits.max())
        index = self._rng.choice(len(names), p=weights / weights.sum())
        return self._by_name[names[int(index)]]

    def next_arena(self, global_step: int) -> Arena:
        self._is_revisit = self._pick_revisit_mode()
        if self._is_revisit:
            return self._sample_cleared()
        return self.arenas[min(self._next_index, len(self.arenas) - 1)]

    def on_episode_end(self, arena: Arena, success: bool) -> None:
        self._attempts[arena.name] += 1
        self._successes[arena.name] += int(success)
        before = self._next_index
        if success and not self._is_revisit:
            self._cleared.add(arena.name)
            self._skip_cleared()
        self._advanced = self._next_index != before

    def info(self, global_step: int) -> dict:
        return {
            "cleared_count": len(self._cleared),
            "is_revisit": self._is_revisit,
            "advanced": self._advanced,
        }

    def status(self, global_step: int) -> str:
        revisit_tag = " R" if self._is_revisit else ""
        return f"cleared:{len(self._cleared)}/{len(self.arenas)}{revisit_tag}"

    def state(self) -> dict:
        attempts = {name: n for name, n in self._attempts.items() if n > 0}
        return {
            "arena_attempts": attempts,
            "arena_successes": {name: self._successes[name] for name in attempts},
            "is_revisit": self._is_revisit,
        }

    def load_state(self, arena_attempts: dict, arena_successes: dict, is_revisit: bool) -> None:
        # An arena is cleared iff it has ever succeeded, so the pointer is
        # rebuilt from the counts rather than stored. Unknown names (a
        # different arena set) are dropped.
        known = set(self._by_name)
        self._attempts.update({k: int(v) for k, v in arena_attempts.items() if k in known})
        self._successes.update({k: int(v) for k, v in arena_successes.items() if k in known})
        self._is_revisit = bool(is_revisit)
        self._cleared = {name for name, n in self._successes.items() if n > 0}
        self._next_index = 0
        self._skip_cleared()


class SequentialSelector(ArenaSelector):
    """Every arena under `root` exactly once, in path order.

    Over COMPETITION_DIR this is the paper's Testbed evaluation protocol: one
    episode per arena, pass/fail by that arena's pass mark (see
    scripts/test_trained_agent.py, which loops until `is_exhausted`).
    """

    def __init__(self, root: Path):
        self.arenas = _arenas_under(root)
        self._next_index = 0

    @property
    def is_exhausted(self) -> bool:
        return self._next_index >= len(self.arenas)

    def next_arena(self, global_step: int) -> Arena:
        arena = self.arenas[self._next_index]
        self._next_index += 1
        return arena

    def info(self, global_step: int) -> dict:
        return {"arena_index": self._next_index - 1, "arena_total": len(self.arenas)}

    def status(self, global_step: int) -> str:
        return f"{self._next_index}/{len(self.arenas)}"


def build_selector(
    mode: str, train_arena_root: str, steps_per_stage: int, revisit_temperature: float, seed: int
) -> ArenaSelector:
    """Build the arena selector named by `mode` (see the module docstring).

    Only the training modes take their arenas from `train_arena_root` (so an
    augmented arena set is chosen in config, and recorded there). "eval" is
    pinned to COMPETITION_DIR: the Testbed is a fixed benchmark, and making it
    configurable would let a typo silently score a run on training arenas.

    Every mode's parameters are always supplied; a mode ignores the ones that
    do not apply to it.
    """
    train_root = Path(train_arena_root)
    builders = {
        "staged": lambda: StagedSelector(
            root=train_root, steps_per_stage=steps_per_stage, seed=seed
        ),
        "success": lambda: SuccessDrivenSelector(
            root=train_root, revisit_temperature=revisit_temperature, seed=seed
        ),
        "eval": lambda: SequentialSelector(root=COMPETITION_DIR),
    }
    assert mode in builders, f"unknown Animal-AI mode {mode!r}; expected one of {sorted(builders)}"
    return builders[mode]()


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
_RENDER_SIZE_PX = 256
_HEADER_HEIGHT_PX = 22


def _draw_header(text: str) -> np.ndarray:
    header = np.full((_HEADER_HEIGHT_PX, _RENDER_SIZE_PX, 3), 215, dtype=np.uint8)
    font_scale = 0.45
    max_width = _RENDER_SIZE_PX - 8
    (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    # Shrink rather than let a long status run off the right edge.
    font_scale *= min(1.0, max_width / width)
    cv2.putText(
        header, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), 1, cv2.LINE_AA
    )
    return header


def _render_topdown(
    items: list[dict], agent_xyz: tuple[float, float, float] | None, header_text: str
) -> np.ndarray:
    scale = _RENDER_SIZE_PX / _ARENA_SIZE_M
    canvas = np.full((_RENDER_SIZE_PX, _RENDER_SIZE_PX, 3), 240, dtype=np.uint8)

    # Flip z so the +z arena axis points up in the image (Unity-editor-like).
    def to_px(x: float, z: float) -> tuple[int, int]:
        return int(round(x * scale)), int(round((_ARENA_SIZE_M - z) * scale))

    cv2.rectangle(canvas, (0, 0), (_RENDER_SIZE_PX - 1, _RENDER_SIZE_PX - 1), (60, 60, 60), 2)

    for item in items:
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
        cv2.drawContours(canvas, [box], 0, color, 1 if transparent else cv2.FILLED)

    if agent_xyz is not None:
        ax, _ay, az = agent_xyz
        apx, apy = to_px(ax, az)
        radius = max(int(0.6 * scale), 4)
        cv2.circle(canvas, (apx, apy), radius, _AGENT_COLOR, cv2.FILLED)
        cv2.circle(canvas, (apx, apy), radius, (20, 20, 20), 1)

    return np.vstack([_draw_header(header_text), canvas])


class AnimalAIEnv(gym.Env):
    """Animal-AI environment with continuous Box action space.

    `selector` decides which arena each episode loads; see the module
    docstring for the three of them.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, resolution: int, seed: int, base_port: int, selector: ArenaSelector):
        super().__init__()
        self.selector = selector
        self._arena_by_name = {arena.name: arena for arena in selector.arenas}
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
        self.global_step = 0
        self.episode_step = 0
        self.arena_name: str = ""
        self.pass_mark: float = 0.0
        self._arena: Arena | None = None
        self._arena_items: list[dict] = []
        self._episode_return = 0.0
        # (x, y, z) agent position in arena coords; populated each step from
        # the AAI vector observation. None before the first reset.
        self._agent_xyz: tuple[float, float, float] | None = None
        # (vx, vy, vz) agent velocity from the AAI vector observation.
        self._agent_velocity: np.ndarray = np.zeros(3, dtype=np.float32)

    def set_global_step(self, global_step: int) -> None:
        """Resume hook: restore the step counter the curriculum schedules on."""
        self.global_step = int(global_step)

    def get_curriculum_state(self) -> dict:
        return self.selector.state()

    def set_curriculum_state(
        self, arena_attempts: dict, arena_successes: dict, is_revisit: bool
    ) -> None:
        self.selector.load_state(arena_attempts, arena_successes, is_revisit)

    def _ensure_started(self):
        if self._aai is not None:
            return
        self._aai = AnimalAIEnvironment(
            file_name=self.binary_path,
            arenas_configurations=str(self.selector.arenas[0].path),
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

    def _read_observation(self, steps) -> None:
        # Vector obs layout (useCamera=True, useRayCasts=False): steps.obs[1][0]
        # is [health, vx, vy, vz, x, y, z].
        vec = steps.obs[1][0]
        self._latest_image = self._decode_obs(steps.obs[0][0])
        self._agent_xyz = (float(vec[4]), float(vec[5]), float(vec[6]))
        self._agent_velocity = np.array(
            [float(vec[1]), float(vec[2]), float(vec[3])], dtype=np.float32
        )

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

    def _build_info(self) -> dict:
        info = {
            "task_prompt": self._build_prompt(),
            "arena_name": self.arena_name,
            "arena_yaml": str(self._arena.path),
            "pass_mark": self.pass_mark,
            "episode_step": self.episode_step,
            "velocity": self._agent_velocity,
            "agent_xyz": self._agent_xyz,
        }
        info.update(self.selector.info(self.global_step))
        return info

    def reset(self, seed: int | None, options: dict | None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._ensure_started()

        # `options={"arena_stem": name}` pins one arena instead of asking the
        # selector (scripts/collect_probe_data.py repeats a single arena).
        forced_name = options["arena_stem"] if options is not None else None
        self._arena = (
            self._arena_by_name[forced_name]
            if forced_name is not None
            else self.selector.next_arena(self.global_step)
        )

        self.arena_name = self._arena.name
        self.pass_mark, self._arena_items = _parse_arena(self._arena.path)
        self._aai.reset(arenas_configurations=str(self._arena.path))
        self.episode_step = 0
        self._episode_return = 0.0

        decision_steps, _ = self._aai.get_steps(self._behavior_name)
        self._read_observation(decision_steps)
        return self._latest_image, self._build_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.global_step += 1
        a = np.asarray(action, dtype=np.float32)
        action_tuple = ActionTuple(
            continuous=np.zeros((1, 0), dtype=np.float32),
            discrete=np.array(
                [[_to_discrete(float(a[0])), _to_discrete(float(a[1]))]], dtype=np.int32
            ),
        )
        self._aai.set_actions(self._behavior_name, action_tuple)
        self._aai.step()

        decision_steps, terminal_steps = self._aai.get_steps(self._behavior_name)
        episode_over = len(terminal_steps) > 0
        # `interrupted` is AAI's max-step (arena `t`) timeout, i.e. a time
        # limit rather than a real terminal (goal reached / death zone).
        interrupted = episode_over and bool(terminal_steps.interrupted[0])
        steps = terminal_steps if episode_over else decision_steps
        reward = float(steps.reward[0])
        self._read_observation(steps)

        self.episode_step += 1
        self._episode_return += reward
        terminated = episode_over and not interrupted
        truncated = interrupted

        if not episode_over:
            return self._latest_image, reward, terminated, truncated, self._build_info()

        success = self._episode_return >= self.pass_mark
        # Pass-mark bonus: reward crossing the clear threshold, penalize missing it.
        # Computed from the pre-bonus return, then folded into this tick's reward.
        reward += 1.0 if success else -1.0
        # Before _build_info, so the selector reports post-episode progress.
        self.selector.on_episode_end(self._arena, success)
        info = self._build_info()
        info["success"] = bool(success)
        return self._latest_image, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        header = f"{self.arena_name}  {self.selector.status(self.global_step)}"
        return _render_topdown(self._arena_items, self._agent_xyz, header)

    def close(self):
        if self._aai is not None:
            self._aai.close()
            self._aai = None


if __name__ == "__main__":
    env = AnimalAIEnv(
        resolution=96,
        seed=0,
        base_port=5005,
        selector=StagedSelector(steps_per_stage=2_000_000, seed=0),
    )
    for episode in range(8):
        obs, info = env.reset(seed=episode, options=None)
        print(f"ep={episode} arena={info['arena_name']} stage={info['stage']}")
        total_reward = 0.0
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total_reward += reward
            if terminated or truncated:
                break
        print(f"  return={total_reward:.4f} success={info.get('success')}")
    env.close()
