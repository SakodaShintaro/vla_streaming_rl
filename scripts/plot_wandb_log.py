"""Plot scalar metrics from a local wandb run file.

Reads the on-disk run-XXXX.wandb log and plots selected keys against
global_step (or _step). Useful when you want to inspect logs without
relying on the wandb web UI (offline runs, quick local diagnostics).

Examples:
    # List available keys in a run
    uv run python scripts/plot_wandb_log.py results/<run_dir> --list

    # Plot specific keys (auto-saves to <run_dir>/plot.png)
    uv run python scripts/plot_wandb_log.py results/<run_dir> \\
        -k agent_step_msec env_step_msec

    # Multiple keys + smoothing window + custom output
    uv run python scripts/plot_wandb_log.py results/<run_dir> \\
        -k reward losses/critic_loss --smooth 100 -o /tmp/out.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb.proto.wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore


def find_wandb_file(run_dir: Path) -> Path:
    candidates = list(run_dir.rglob("*.wandb"))
    if not candidates:
        raise FileNotFoundError(f"No .wandb file found under {run_dir}")
    if len(candidates) > 1:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_history(wandb_file: Path) -> dict[str, np.ndarray]:
    """Return {key: 1-D array} for every scalar logged via wandb.log."""
    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_file))

    columns: dict[str, list] = {}
    row_idx = 0

    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue

        for it in rec.history.item:
            key = "/".join(it.nested_key) if list(it.nested_key) else it.key
            try:
                value = json.loads(it.value_json)
            except (ValueError, json.JSONDecodeError):
                continue
            if not isinstance(value, (int, float)):
                continue
            col = columns.setdefault(key, [])
            col.extend([np.nan] * (row_idx - len(col)))
            col.append(float(value))

        row_idx += 1

    for key, col in columns.items():
        col.extend([np.nan] * (row_idx - len(col)))
        columns[key] = np.asarray(col, dtype=np.float64)

    return columns


def episode_boundaries(
    history: dict[str, np.ndarray], x_key: str, marker_key: str
) -> np.ndarray:
    """x positions where an episode ended.

    The trainer logs per-episode summaries (e.g. ``episodic_return``) in a
    separate ``wandb.log`` call at each episode boundary, so the rows where
    ``marker_key`` is non-NaN give the global_step of each episode switch.
    """
    if marker_key not in history or x_key not in history:
        return np.empty(0)
    marker = history[marker_key]
    x = history[x_key]
    rows = ~np.isnan(marker) & ~np.isnan(x)
    return x[rows]


def split_by_episode(
    gs: np.ndarray, y: np.ndarray, boundaries: np.ndarray
) -> list[np.ndarray]:
    """Split the per-step series ``y`` (logged at global steps ``gs``) into one
    array per episode, using the episode-end global steps ``boundaries``.

    Episode k holds the steps with ``boundary[k-1] < gs <= boundary[k]``; any
    steps past the last boundary form a trailing (in-progress) episode.
    """
    segments: list[np.ndarray] = []
    lo = -np.inf
    for b in boundaries:
        mask = (gs > lo) & (gs <= b)
        if mask.any():
            segments.append(y[mask])
        lo = b
    tail = gs > lo
    if tail.any():
        segments.append(y[tail])
    return segments


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    mask = ~np.isnan(y)
    out = np.full_like(y, np.nan)
    valid_y = y[mask]
    if len(valid_y) == 0:
        return out
    kernel = np.ones(window) / window
    smoothed = np.convolve(valid_y, kernel, mode="same")
    out[mask] = smoothed
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path, help="Run directory containing wandb/ or *.wandb")
    parser.add_argument(
        "-k", "--keys", nargs="+", help="Metric keys to plot (omit to list available keys)"
    )
    parser.add_argument("--list", action="store_true", help="List available keys and exit")
    parser.add_argument(
        "-x",
        "--x-key",
        default="global_step",
        help="Key used for the x-axis (fallback: _step)",
    )
    parser.add_argument(
        "--smooth", type=int, default=1, help="Rolling mean window (default: 1, no smoothing)"
    )
    parser.add_argument(
        "--episode-key",
        default="episodic_return",
        help="Per-episode marker key whose log rows mark episode boundaries "
        "(drawn as vertical dashed lines; default: episodic_return)",
    )
    parser.add_argument(
        "--no-episode-lines",
        action="store_true",
        help="Disable the episode-boundary vertical dashed lines",
    )
    parser.add_argument(
        "--per-episode",
        action="store_true",
        help="Overlay one line per episode, x = step within the episode, each "
        "line colored by episode id (gradient). Requires exactly one key.",
    )
    parser.add_argument(
        "--cumulative",
        action="store_true",
        help="In --per-episode mode, plot the within-episode cumulative sum of "
        "the key (e.g. -k reward --cumulative gives cumulative reward / return).",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Q-overestimation diagnostic: scatter Q (key, default q_value) vs "
        "the realized discounted return-to-go, colored by episode id, plus the "
        "per-episode mean gap (Q - return). Reveals critic divergence.",
    )
    parser.add_argument(
        "--reward-key",
        default="processed_reward",
        help="Per-step reward key used to build the return-to-go in "
        "--calibration (default: processed_reward; falls back to reward).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount for --calibration return-to-go (default: read from "
        "<run_dir>/.hydra/config.yaml, else 0.99).",
    )
    parser.add_argument(
        "--score-smooth",
        type=int,
        default=5,
        help="Smoothing window for the episodic score overlaid on the "
        "--calibration gap panel (default: 5).",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="Output image path (default: <run_dir>/plot.png)"
    )
    parser.add_argument("--show", action="store_true", help="Show the figure interactively")
    return parser.parse_args()


def _resolve_gamma(run_dir: Path, cli_gamma: float | None) -> float:
    if cli_gamma is not None:
        return cli_gamma
    cfg = run_dir / ".hydra" / "config.yaml"
    if cfg.exists():
        try:
            from omegaconf import OmegaConf

            data = OmegaConf.load(cfg)
            if data.get("gamma") is not None:
                return float(data["gamma"])
        except Exception:
            pass
    return 0.99


def _return_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Discounted sum of this-and-future rewards at each step (within episode)."""
    rtg = np.zeros_like(rewards, dtype=np.float64)
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = rewards[t] + gamma * acc
        rtg[t] = acc
    return rtg


def plot_q_calibration(
    args: argparse.Namespace, history: dict[str, np.ndarray], x: np.ndarray, x_key: str
) -> None:
    """Scatter Q vs realized discounted return-to-go (colored by episode id),
    plus the per-episode mean gap (Q - return). A growing positive gap for late
    episodes is the classic critic-overestimation collapse."""
    q_key = args.keys[0] if args.keys else "q_value"
    if q_key not in history:
        raise SystemExit(f"Q key '{q_key}' not found in run")
    r_key = args.reward_key
    if r_key not in history:
        if "reward" not in history:
            raise SystemExit(f"reward key '{r_key}' (and fallback 'reward') not found")
        print(f"'{r_key}' absent; falling back to raw 'reward' for return-to-go")
        r_key = "reward"
    gamma = _resolve_gamma(args.run_dir, args.gamma)

    valid = ~(np.isnan(x) | np.isnan(history[q_key]) | np.isnan(history[r_key]))
    gs = x[valid]
    q = history[q_key][valid]
    r = history[r_key][valid]
    boundaries = episode_boundaries(history, x_key, args.episode_key)
    idx_segments = split_by_episode(gs, np.arange(len(gs)), boundaries)
    if not idx_segments:
        raise SystemExit("No episodes to plot")

    n = len(idx_segments)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=max(1, n - 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    gaps = []
    lo = np.inf
    hi = -np.inf
    for i, idx in enumerate(idx_segments):
        qi = q[idx]
        rtg = _return_to_go(r[idx], gamma)
        ax1.scatter(rtg, qi, s=5, color=cmap(norm(i)), alpha=0.5, linewidths=0)
        gaps.append(float(np.mean(qi - rtg)))
        lo = min(lo, rtg.min(), qi.min())
        hi = max(hi, rtg.max(), qi.max())

    ax1.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, alpha=0.7, label="Q = return")
    ax1.set_xlabel(f"discounted return-to-go ({r_key}, gamma={gamma})")
    ax1.set_ylabel(f"Q ({q_key})")
    ax1.set_title("calibration (above diagonal = overestimation)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, label="episode id")

    (gap_line,) = ax2.plot(
        range(n), gaps, marker="o", markersize=3, linewidth=1.0, color="tab:blue",
        label="mean(Q - return)",
    )
    ax2.axhline(0.0, color="k", linewidth=0.8)
    ax2.set_xlabel("episode id")
    ax2.set_ylabel("mean(Q - return-to-go)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_title("overestimation gap vs actual score")
    ax2.grid(alpha=0.3)

    # actual score (episodic return) on a twin axis, smoothed
    handles = [gap_line]
    marker = history.get(args.episode_key)
    if marker is not None:
        scores = marker[~np.isnan(marker)]
        if len(scores):
            ax2b = ax2.twinx()
            ax2b.plot(
                range(len(scores)), scores, color="tab:red", alpha=0.25, linewidth=0.8
            )
            (score_line,) = ax2b.plot(
                range(len(scores)),
                smooth(scores, args.score_smooth),
                color="tab:red",
                linewidth=1.8,
                label=f"score (smooth={args.score_smooth})",
            )
            ax2b.set_ylabel(f"{args.episode_key}", color="tab:red")
            ax2b.tick_params(axis="y", labelcolor="tab:red")
            handles.append(score_line)
    ax2.legend(handles=handles, loc="upper left")

    fig.suptitle(f"{args.run_dir.name} — Q calibration ({n} episodes)")
    fig.tight_layout()
    output = args.output if args.output is not None else args.run_dir / "plot.png"
    fig.savefig(output, dpi=120)
    print(f"Saved {output}")
    if args.show:
        plt.show()


def plot_per_episode(
    args: argparse.Namespace, history: dict[str, np.ndarray], x: np.ndarray, x_key: str
) -> None:
    """One line per episode, x = step within the episode, colored by episode id."""
    if len(args.keys) != 1:
        raise SystemExit("--per-episode requires exactly one key")
    key = args.keys[0]

    valid = ~(np.isnan(x) | np.isnan(history[key]))
    gs = x[valid]
    y = history[key][valid]
    boundaries = episode_boundaries(history, x_key, args.episode_key)
    segments = split_by_episode(gs, y, boundaries)
    if not segments:
        raise SystemExit(f"No data to plot for key '{key}'")

    n = len(segments)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=max(1, n - 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, seg in enumerate(segments):
        if args.cumulative:
            seg = np.cumsum(seg)
        ys = smooth(seg, args.smooth)
        ax.plot(np.arange(len(ys)), ys, color=cmap(norm(i)), linewidth=1.0, alpha=0.8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="episode id")

    ax.set_xlabel("step within episode")
    ax.set_ylabel(f"cumulative {key}" if args.cumulative else key)
    ax.grid(alpha=0.3)
    label = f"cumulative {key}" if args.cumulative else key
    title = f"{args.run_dir.name} — {label} per episode ({n})"
    if args.smooth > 1:
        title += f" (smooth={args.smooth})"
    ax.set_title(title)
    fig.tight_layout()

    output = args.output if args.output is not None else args.run_dir / "plot.png"
    fig.savefig(output, dpi=120)
    print(f"Saved {output}")
    if args.show:
        plt.show()


def main() -> None:
    args = parse_args()

    wandb_file = find_wandb_file(args.run_dir)
    print(f"Reading {wandb_file}")
    history = load_history(wandb_file)

    # --calibration defaults its Q key to q_value so -k is optional.
    if args.calibration and not args.keys:
        args.keys = ["q_value"]

    if args.list or not args.keys:
        print(f"Available keys ({len(history)}):")
        for k in sorted(history):
            print(f"  {k}")
        if args.list:
            return
        if not args.keys:
            raise SystemExit("Specify -k <key> [<key> ...] to plot")

    missing = [k for k in args.keys if k not in history]
    if missing:
        raise SystemExit(f"Keys not found in run: {missing}")

    if args.x_key in history:
        x_key = args.x_key
    elif "_step" in history:
        x_key = "_step"
    else:
        raise SystemExit(f"x-axis key '{args.x_key}' not found and no '_step' fallback")
    x = history[x_key]

    if args.calibration:
        plot_q_calibration(args, history, x, x_key)
        return

    if args.per_episode:
        plot_per_episode(args, history, x, x_key)
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for key in args.keys:
        y = history[key]
        valid = ~(np.isnan(x) | np.isnan(y))
        xs = x[valid]
        ys = smooth(y[valid], args.smooth)
        ax.plot(xs, ys, label=key, linewidth=1.2)

    if not args.no_episode_lines:
        boundaries = episode_boundaries(history, x_key, args.episode_key)
        for i, xb in enumerate(boundaries):
            ax.axvline(
                xb,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
                label="episode boundary" if i == 0 else None,
            )
        if len(boundaries) == 0:
            print(f"No episode boundaries found (marker key '{args.episode_key}' absent)")

    ax.set_xlabel(x_key)
    ax.set_ylabel("value")
    ax.grid(alpha=0.3)
    ax.legend()
    title = args.run_dir.name
    if args.smooth > 1:
        title += f" (smooth={args.smooth})"
    ax.set_title(title)
    fig.tight_layout()

    output = args.output if args.output is not None else args.run_dir / "plot.png"
    fig.savefig(output, dpi=120)
    print(f"Saved {output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
