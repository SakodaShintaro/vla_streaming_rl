"""Plot results (single-run layout) for one environment.

Given the parent directory produced by ``exp.sh`` (one subdirectory per
learning-mode variant, each holding ``log_episode.tsv`` and a local wandb
run), this renders three figures for the paper:

* ``<prefix>_score.pdf``      -- trailing-average score / success rate vs time.
* ``<prefix>_agent_step.pdf`` -- median agent step time (ms) per method.
* ``<prefix>_gpu_memory.pdf`` -- peak GPU memory allocated per method.

The score curve comes from ``log_episode.tsv``; the agent step time is the
median of the per-step ``agent_step_msec`` (wandb history, second half), which
isolates agent compute from the environment step; the GPU memory is read from
the wandb system metric ``gpu.0.memoryAllocatedBytes``. Per-environment labels
and output filenames are selected via the ``env`` argument; add a new entry to
``ENVS`` to support another task.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb.proto.wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

_TAB10 = plt.cm.tab10.colors

# (exp_name suffix, display label, color). Order = display order.
# Colors are kept consistent across the three figures. The suffix is matched
# against the end of each subdirectory name (e.g. ``..._pi05_streaming``).
METHODS = [
    ("streaming", "Streaming", _TAB10[3]),
    ("off_policy_bs1", "Off-policy bs1", _TAB10[1]),
    ("off_policy_bs16", "Off-policy bs16", _TAB10[2]),
]

# Per-environment presets: the score curve's y-axis label, the output filename
# for that curve, and an optional fixed y-limit (e.g. LIBERO success rate).
ENVS = {
    "car_racing": {
        "score_ylabel": "Moving Average Score",
        "score_fname": "car_racing_moving_average.pdf",
        "score_ymax": None,
        "prefix": "car_racing",
        "score_col": "recent_average_score",  # already a trailing average
        "smooth": None,
    },
    "libero": {
        "score_ylabel": "Success Rate",
        "score_fname": "libero_success_rate.pdf",
        "score_ymax": 1.0,
        "prefix": "libero",
        "score_col": "recent_average_score",
        "smooth": None,
    },
    "carla": {
        "score_ylabel": "Driving Score",
        "score_fname": "carla_score.pdf",
        "score_ymax": None,
        "prefix": "carla",
        # CARLA never fills recent_average_score, so smooth the raw Bench2Drive
        # driving score (0-100) over a small episode window here instead.
        "score_col": "eval/score_composed",
        "smooth": 10,
    },
}


def find_method_dir(data_dir: Path, suffix: str) -> Path | None:
    """Return the single subdirectory whose name ends with ``suffix``.

    ``off_policy_bs1`` must not accidentally match ``off_policy_bs16``, so the
    longest-suffix match wins when several candidates exist.
    """
    candidates = sorted(
        (d for d in data_dir.iterdir() if d.is_dir() and d.name.endswith(suffix)),
        key=lambda d: len(d.name),
    )
    return candidates[0] if candidates else None


def _scan_wandb(method_dir: Path):
    """Yield decoded records from the method's local wandb run file."""
    wandb_files = list(method_dir.rglob("*.wandb"))
    if not wandb_files:
        print(f"No .wandb file under {method_dir}")
        return
    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_files[0]))
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        yield rec


def peak_gpu_memory_gb(method_dir: Path) -> float | None:
    """Peak ``gpu.0.memoryAllocatedBytes`` (GB) from the local wandb run."""
    peak = 0.0
    for rec in _scan_wandb(method_dir):
        if rec.WhichOneof("record_type") != "stats":
            continue
        for it in rec.stats.item:
            if it.key == "gpu.0.memoryAllocatedBytes":
                try:
                    peak = max(peak, float(json.loads(it.value_json)))
                except (ValueError, json.JSONDecodeError):
                    pass
    return peak / 1e9 if peak > 0 else None


def agent_step_msec(method_dir: Path) -> float | None:
    """Mean per-env-step agent compute time (ms) over the run's second half.

    This isolates the agent's own inference + update cost from the environment
    step, so it stays meaningful even when the env dominates wall-clock (e.g.
    CARLA). Averaging over all steps gives the amortized agent cost per env step;
    for a chunked policy (pi0.5) that correctly credits streaming for reusing its
    acting forward instead of running a separate one for learning. Read from the
    per-step ``agent_step_msec`` in the wandb history.
    """
    vals = []
    for rec in _scan_wandb(method_dir):
        if rec.WhichOneof("record_type") != "history":
            continue
        for it in rec.history.item:
            key = "/".join(it.nested_key) if list(it.nested_key) else it.key
            if key == "agent_step_msec":
                try:
                    vals.append(float(json.loads(it.value_json)))
                except (ValueError, json.JSONDecodeError):
                    pass
    if not vals:
        return None
    return float(np.mean(np.asarray(vals)[len(vals) // 2 :]))


def find_method_dirs(data_dirs, suffix: str) -> list[Path]:
    """One matching subdir per parent run dir (trials), skipping misses."""
    dirs = []
    for dd in data_dirs:
        m = find_method_dir(dd, suffix)
        if m is not None:
            dirs.append(m)
    return dirs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env", choices=sorted(ENVS))
    parser.add_argument(
        "data_dir",
        type=Path,
        nargs="+",
        help="One or more parent run dirs; with several, results are averaged "
        "(mean +/- std band on the curve, mean on the bars) across them.",
    )
    return parser.parse_args()


def _apply_rcparams():
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
        }
    )


def _load_curve(method_dir, cfg):
    """(elapsed_time_hour, score) for one run, smoothed and NaN-filtered."""
    score_col, smooth = cfg["score_col"], cfg["smooth"]
    df = pd.read_csv(method_dir / "log_episode.tsv", sep="\t").sort_values("elapsed_time_hour")
    s = df[score_col]
    if smooth:
        s = s.rolling(smooth, min_periods=1).mean()
    valid = s.notna().to_numpy()
    return df["elapsed_time_hour"].to_numpy()[valid], s.to_numpy()[valid]


def plot_score_curve(out_dir, resolved, cfg):
    ylabel, fname, ymax = cfg["score_ylabel"], cfg["score_fname"], cfg["score_ymax"]
    fig, ax = plt.subplots(figsize=(12, 7.5))
    label_positions = []  # (x_end, y_end, label, color)
    x_max = 0

    for method_dirs, label, color in resolved:
        curves = [_load_curve(md, cfg) for md in method_dirs]
        if len(curves) == 1:
            times, scores = curves[0]
        else:
            # Average across trials on a common grid up to the shortest run,
            # with a +/- std band. Each trial ends at a different wall-clock time.
            hi = min(t[-1] for t, _ in curves)
            times = np.linspace(0, hi, 300)
            stack = np.stack([np.interp(times, t, s) for t, s in curves])
            scores = stack.mean(axis=0)
            ax.fill_between(
                times,
                scores - stack.std(axis=0),
                scores + stack.std(axis=0),
                color=color,
                alpha=0.2,
            )
        ax.plot(times, scores, color=color, linewidth=2)
        label_positions.append((times[-1], scores[-1], label, color))
        x_max = max(x_max, times[-1])

    ax.set_xlabel("Elapsed Time (hours)")
    ax.set_ylabel(ylabel)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(right=x_max)

    # Inline labels at the right end of each curve, nudged apart vertically.
    label_positions.sort(key=lambda p: p[1])
    y_lo, y_hi = ax.get_ylim()
    min_gap = (y_hi - y_lo) * 0.06
    adjusted, prev = [], -np.inf
    for _, y_end, _, _ in label_positions:
        y = max(y_end, prev + min_gap)
        adjusted.append(y)
        prev = y
    for (x_end, _, label, color), y in zip(label_positions, adjusted):
        ax.text(
            x_end,
            y,
            "  " + label,
            color=color,
            va="center",
            ha="left",
            fontsize=20,
            fontweight="bold",
            clip_on=False,
        )
    fig.subplots_adjust(right=0.75)

    output = out_dir / fname
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def _barh(data_dir, resolved, values, xlabel, fname, fmt):
    labels = [label for _, label, _ in resolved]
    colors = [color for _, _, color in resolved]
    fig, ax = plt.subplots(figsize=(10, 6.0))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=20)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    pad = max(values) * 0.01
    for yi, v in zip(y, values):
        ax.text(v + pad, yi, fmt.format(v), ha="left", va="center", fontsize=17)
    ax.set_xlim(right=max(values) * 1.15)
    output = data_dir / fname
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def _mean_over(dirs, fn):
    vals = [v for v in (fn(d) for d in dirs) if v is not None]
    return float(np.mean(vals)) if vals else None


def main():
    args = parse_args()
    cfg = ENVS[args.env]
    prefix = cfg["prefix"]
    _apply_rcparams()
    data_dirs = args.data_dir
    out_dir = data_dirs[0]

    resolved = []
    for suffix, label, color in METHODS:
        method_dirs = find_method_dirs(data_dirs, suffix)
        if not method_dirs:
            print(f"Skipping missing method: {suffix}")
            continue
        resolved.append((method_dirs, label, color))

    if not resolved:
        raise SystemExit(f"No method directories found under {data_dirs}")
    if len(data_dirs) > 1:
        print(f"Aggregating over {len(data_dirs)} runs.")

    plot_score_curve(out_dir, resolved, cfg)

    step_ms = [_mean_over(mds, agent_step_msec) for mds, _, _ in resolved]
    if all(v is not None for v in step_ms):
        _barh(
            out_dir,
            resolved,
            step_ms,
            "Agent Step Time (ms)",
            f"{prefix}_agent_step.pdf",
            " {:.0f}",
        )

    mem = [_mean_over(mds, peak_gpu_memory_gb) for mds, _, _ in resolved]
    if all(m is not None for m in mem):
        _barh(out_dir, resolved, mem, "Peak GPU Memory (GB)", f"{prefix}_gpu_memory.pdf", " {:.1f}")

    print("\nSummary")
    for (mds, label, _), s, m in zip(resolved, step_ms, mem):
        print(f"  {label:16s} runs={len(mds)} agent_step={s:6.1f} ms  GPU={m:.2f} GB")


if __name__ == "__main__":
    main()
