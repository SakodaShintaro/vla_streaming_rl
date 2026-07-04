"""Plot results (single-run layout) for one environment.

Given the parent directory produced by ``exp.sh`` (one subdirectory per
learning-mode variant, each holding ``log_episode.tsv`` and a local wandb
run), this renders three figures for the paper:

* ``<prefix>_score.pdf``      -- trailing-average score / success rate vs time.
* ``<prefix>_fps.pdf``        -- throughput (final SPS) per method.
* ``<prefix>_gpu_memory.pdf`` -- peak GPU memory allocated per method.

The score curve comes from ``log_episode.tsv`` (``recent_average_score``); the
throughput is the final cumulative ``SPS`` logged there; the GPU memory is read
from the wandb system metric ``gpu.0.memoryAllocatedBytes`` in the local
``*.wandb`` file. Per-environment labels and output filenames are selected via
the ``env`` argument; add a new entry to ``ENVS`` to support another task.
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
    },
    "libero": {
        "score_ylabel": "Success Rate",
        "score_fname": "libero_success_rate.pdf",
        "score_ymax": 1.0,
        "prefix": "libero",
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


def peak_gpu_memory_gb(method_dir: Path) -> float | None:
    """Peak ``gpu.0.memoryAllocatedBytes`` (GB) from the local wandb run."""
    wandb_files = list(method_dir.rglob("*.wandb"))
    if not wandb_files:
        print(f"No .wandb file under {method_dir}")
        return None
    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_files[0]))
    peak = 0.0
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "stats":
            continue
        for it in rec.stats.item:
            if it.key == "gpu.0.memoryAllocatedBytes":
                try:
                    peak = max(peak, float(json.loads(it.value_json)))
                except (ValueError, json.JSONDecodeError):
                    pass
    return peak / 1e9 if peak > 0 else None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env", choices=sorted(ENVS))
    parser.add_argument("data_dir", type=Path)
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


def plot_score_curve(data_dir, resolved, ylabel, fname, ymax):
    fig, ax = plt.subplots(figsize=(12, 7.5))
    label_positions = []  # (x_end, y_end, label, color)
    x_max = 0

    for method_dir, label, color in resolved:
        df = pd.read_csv(method_dir / "log_episode.tsv", sep="\t")
        df = df.dropna(subset=["recent_average_score"])
        times = df["elapsed_time_hour"].to_numpy()
        scores = df["recent_average_score"].to_numpy()
        order = np.argsort(times)
        times, scores = times[order], scores[order]
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

    output = data_dir / fname
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


def main():
    args = parse_args()
    cfg = ENVS[args.env]
    prefix = cfg["prefix"]
    _apply_rcparams()

    resolved = []
    for suffix, label, color in METHODS:
        method_dir = find_method_dir(args.data_dir, suffix)
        if method_dir is None:
            print(f"Skipping missing method: {suffix}")
            continue
        resolved.append((method_dir, label, color))

    if not resolved:
        raise SystemExit(f"No method directories found under {args.data_dir}")

    plot_score_curve(
        args.data_dir, resolved, cfg["score_ylabel"], cfg["score_fname"], cfg["score_ymax"]
    )

    fps = [
        pd.read_csv(d / "log_episode.tsv", sep="\t")["SPS"].to_numpy()[-1] for d, _, _ in resolved
    ]
    _barh(args.data_dir, resolved, fps, "Frames per Second (FPS)", f"{prefix}_fps.pdf", " {:.1f}")

    mem = [peak_gpu_memory_gb(d) for d, _, _ in resolved]
    if all(m is not None for m in mem):
        _barh(
            args.data_dir,
            resolved,
            mem,
            "Peak GPU Memory (GB)",
            f"{prefix}_gpu_memory.pdf",
            " {:.1f}",
        )

    print("\nSummary")
    for (d, label, _), f, m in zip(resolved, fps, mem):
        print(f"  {label:16s} FPS={f:5.1f}  GPU={m:.2f} GB")


if __name__ == "__main__":
    main()
