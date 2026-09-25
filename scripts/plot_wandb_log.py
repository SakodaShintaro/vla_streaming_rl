"""Plot scalar metrics from a local wandb run file.

Reads the on-disk run-XXXX.wandb log and plots selected keys against
global_step (or _step). Useful when you want to inspect logs without
relying on the wandb web UI (offline runs, quick local diagnostics).

Examples:
    # List available keys in a run
    uv run python scripts/plot_wandb_log.py results/<run_dir> --list

    # Plot specific keys (auto-named, saved into the run dir)
    uv run python scripts/plot_wandb_log.py results/<run_dir> \\
        -k agent_step_msec env_step_msec

    # Multiple keys + smoothing window + custom output directory
    uv run python scripts/plot_wandb_log.py results/<run_dir> \\
        -k reward losses/critic_loss --smooth 100 -o /tmp/plots

    # Q-overestimation diagnostic (saved as calibration_value.png)
    uv run python scripts/plot_wandb_log.py results/<run_dir> --calibration
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vla_streaming_rl.wandb_reader import find_wandb_file, load_history


def episode_boundaries(history: dict[str, np.ndarray], x_key: str, marker_key: str) -> np.ndarray:
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


def split_by_episode(gs: np.ndarray, y: np.ndarray, boundaries: np.ndarray) -> list[np.ndarray]:
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
    mask = ~np.isnan(y)
    valid_y = y[mask]
    # A series shorter than the window is returned as is: np.convolve with
    # mode="same" would return `window` values for it, which no longer line up
    # with the mask.
    if window <= 1 or len(valid_y) < window:
        return y
    out = np.full_like(y, np.nan)
    kernel = np.ones(window) / window
    out[mask] = np.convolve(valid_y, kernel, mode="same")
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
        "--per-gamma",
        metavar="BASE",
        default=None,
        help="Multi-gamma view: auto-discover all per-gamma series "
        "'<BASE>_g<gamma>' (BASE='value' for the per-step inference Q_g) and "
        "overlay one line per discount, colored by gamma. Pair with --smooth.",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Q-overestimation diagnostic: scatter the critic value (key, default value) vs "
        "the realized discounted return-to-go, colored by episode id, plus the "
        "per-episode mean gap (Q - return). Reveals critic divergence.",
    )
    parser.add_argument(
        "--calibration-per-gamma",
        action="store_true",
        help="One --calibration image per discount (gammas discovered from the "
        "logged per-gamma series): Q_g (value_g{g}, or gamma-mean 'value' if "
        "per-step per-gamma was not logged) vs the γ_g return-to-go.",
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
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to save the plot in (default: run_dir). The filename is "
        "auto-derived from the mode and keys.",
    )
    parser.add_argument("--show", action="store_true", help="Show the figure interactively")
    return parser.parse_args()


def resolve_output(args: argparse.Namespace) -> Path:
    """Output path: the chosen directory (default run_dir) plus an auto filename
    derived from the plot mode and keys (``/`` in keys becomes ``_``)."""
    out_dir = args.out_dir if args.out_dir is not None else args.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = "_".join(k.replace("/", "_") for k in (args.keys or ["value"]))
    if args.per_gamma:
        name = f"per_gamma_{args.per_gamma.replace('/', '_')}"
    elif args.calibration:
        name = f"calibration_{keys}"
    elif args.per_episode:
        name = f"{keys}_per_episode" + ("_cumulative" if args.cumulative else "")
    else:
        name = keys
    return out_dir / f"{name}.png"


def _resolve_gamma(run_dir: Path, cli_gamma: float | None) -> float:
    if cli_gamma is not None:
        return cli_gamma
    cfg = run_dir / ".hydra" / "config.yaml"
    if cfg.exists():
        from omegaconf import OmegaConf

        data = OmegaConf.load(cfg)
        if "gamma" in data and data["gamma"] is not None:
            return float(data["gamma"])
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
    q_key = args.keys[0]
    if q_key not in history:
        raise SystemExit(f"Q key '{q_key}' not found in run")
    gamma = _resolve_gamma(args.run_dir, args.gamma)
    _render_calibration(
        args, history, x, x_key, q_key=q_key, gamma=gamma, output=resolve_output(args)
    )


def plot_calibration_per_gamma(
    args: argparse.Namespace, history: dict[str, np.ndarray], x: np.ndarray, x_key: str
) -> None:
    """One Q-calibration image per discount: for each gamma g, scatter the
    per-gamma rollout value ``value_g{g}`` (Q of the executed action at each
    env-step, logged at inference) against the realized γ_g return-to-go.

    Gammas are discovered from the per-step ``value_g{g}`` keys. A run that did
    not log them (single-gamma, or pre-dating per-gamma inference logging) has
    nothing to plot here — falls back to the gamma-mean ``value`` for Q with a
    warning, in which case only the *discount* is per-gamma, not the Q itself.
    """
    gamma_keys = discover_gamma_keys(history, "value")
    if not gamma_keys:
        raise SystemExit(
            "No per-gamma rollout series ('value_g*') found in run — was it "
            "trained with per-gamma inference logging and multi_gammas set?"
        )
    out_dir = args.out_dir if args.out_dir is not None else args.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for gamma, _ in gamma_keys:
        per_step_key = f"value_g{gamma:.4f}"
        if per_step_key in history:
            q_key = per_step_key
        else:
            q_key = "value"
            print(
                f"γ={gamma:g}: per-step '{per_step_key}' absent; "
                f"falling back to gamma-mean 'value' (Q not per-gamma)"
            )
        _render_calibration(
            args,
            history,
            x,
            x_key,
            q_key=q_key,
            gamma=gamma,
            output=out_dir / f"calibration_g{gamma:.4f}.png",
        )


def _render_calibration(
    args: argparse.Namespace,
    history: dict[str, np.ndarray],
    x: np.ndarray,
    x_key: str,
    *,
    q_key: str,
    gamma: float,
    output: Path,
) -> None:
    """Render one calibration figure (Q vs γ return-to-go + per-episode gap)."""
    if q_key not in history:
        raise SystemExit(f"Q key '{q_key}' not found in run")
    r_key = args.reward_key
    if r_key not in history:
        if "reward" not in history:
            raise SystemExit(f"reward key '{r_key}' (and fallback 'reward') not found")
        print(f"'{r_key}' absent; falling back to raw 'reward' for return-to-go")
        r_key = "reward"

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
        range(n),
        gaps,
        marker="o",
        markersize=3,
        linewidth=1.0,
        color="tab:blue",
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
    if args.episode_key in history:
        marker = history[args.episode_key]
        scores = marker[~np.isnan(marker)]
        if len(scores):
            ax2b = ax2.twinx()
            ax2b.plot(range(len(scores)), scores, color="tab:red", alpha=0.25, linewidth=0.8)
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

    fig.suptitle(f"{args.run_dir.name} — Q calibration ({q_key}, γ={gamma:g}, {n} episodes)")
    fig.tight_layout()
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

    output = resolve_output(args)
    fig.savefig(output, dpi=120)
    print(f"Saved {output}")
    if args.show:
        plt.show()


def discover_gamma_keys(history: dict[str, np.ndarray], base: str) -> list[tuple[float, str]]:
    """Find logged keys of the form ``...<base>_g<gamma>`` and return
    ``[(gamma, key), ...]`` sorted by gamma (e.g. base='q_value' →
    'losses/q_value_g0.100'). The gamma is parsed from the ``_g<float>`` suffix."""
    pat = re.compile(rf"{re.escape(base)}_g(-?\d+(?:\.\d+)?)$")
    found = []
    for key in history:
        m = pat.search(key)
        if m:
            found.append((float(m.group(1)), key))
    found.sort(key=lambda gk: gk[0])
    return found


def plot_per_gamma(
    args: argparse.Namespace, history: dict[str, np.ndarray], x: np.ndarray, x_key: str
) -> None:
    """Overlay one curve per discount for a multi-gamma metric, colored by gamma."""
    gamma_keys = discover_gamma_keys(history, args.per_gamma)
    if not gamma_keys:
        avail = sorted(k for k in history if "_g" in k)
        raise SystemExit(
            f"No per-gamma series found for base '{args.per_gamma}'. Keys containing '_g': {avail}"
        )

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=max(1, len(gamma_keys) - 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (gamma, key) in enumerate(gamma_keys):
        y = history[key]
        valid = ~(np.isnan(x) | np.isnan(y))
        ys = smooth(y[valid], args.smooth)
        ax.plot(x[valid], ys, color=cmap(norm(i)), linewidth=1.3, label=f"γ={gamma:g}")

    if not args.no_episode_lines:
        for i, xb in enumerate(episode_boundaries(history, x_key, args.episode_key)):
            ax.axvline(
                xb,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                alpha=0.4,
                label="episode boundary" if i == 0 else None,
            )

    ax.set_xlabel(x_key)
    ax.set_ylabel(args.per_gamma)
    ax.grid(alpha=0.3)
    ax.legend(title="discount", ncol=2)
    title = f"{args.run_dir.name} — {args.per_gamma} per gamma ({len(gamma_keys)})"
    if args.smooth > 1:
        title += f" (smooth={args.smooth})"
    ax.set_title(title)
    fig.tight_layout()

    output = resolve_output(args)
    fig.savefig(output, dpi=120)
    print(f"Saved {output}")
    if args.show:
        plt.show()


def main() -> None:
    args = parse_args()

    wandb_file = find_wandb_file(args.run_dir)
    print(f"Reading {wandb_file}")
    history = load_history(wandb_file)

    # --calibration defaults its key to the critic value so -k is optional.
    if args.calibration and not args.keys:
        args.keys = ["value"]

    # --per-gamma / --calibration-per-gamma discover their own keys, so -k is
    # not required for them either.
    needs_keys = not (args.per_gamma or args.calibration_per_gamma or args.list)
    if args.list or (needs_keys and not args.keys):
        print(f"Available keys ({len(history)}):")
        for k in sorted(history):
            print(f"  {k}")
        if args.list:
            return
        if not args.keys:
            raise SystemExit("Specify -k <key> [<key> ...] to plot")

    if args.keys:
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

    # One plotting mode per flag; the first enabled one wins, the key-vs-step
    # plot is the fallback.
    modes = [
        (args.per_gamma, plot_per_gamma),
        (args.calibration_per_gamma, plot_calibration_per_gamma),
        (args.calibration, plot_q_calibration),
        (args.per_episode, plot_per_episode),
    ]
    for enabled, plot in modes:
        if enabled:
            plot(args, history, x, x_key)
            return
    plot_keys(args, history, x, x_key)


def plot_keys(
    args: argparse.Namespace, history: dict[str, np.ndarray], x: np.ndarray, x_key: str
) -> None:
    """The default mode: each requested key against the x-axis."""
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

    output = resolve_output(args)
    fig.savefig(output, dpi=120)
    print(f"Saved {output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
