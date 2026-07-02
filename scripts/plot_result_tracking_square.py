"""Plot moving average score over Global Step from a W&B CSV export.

Reads a wandb CSV export where each agent has three columns:
  "agent: <name> - recent_average_score"
  "agent: <name> - recent_average_score__MIN"
  "agent: <name> - recent_average_score__MAX"

Plots the mean (center column) as a line and the [MIN, MAX] as a shaded band,
using the same method labels / colors as plot_result.py.
Legend is rendered as inline text at the right end of each curve.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    return parser.parse_args()


_TAB10 = plt.cm.tab10.colors

# (wandb exp_name key, display label, color).
# Runs now share agent=standard and are distinguished by exp_name (see train_all.sh).
# Same order and color mapping as plot_result.py's METHODS.
METHODS = [
    ("no_vlm_off_policy_bs16", "Off-policy bs16 (No VLM)", _TAB10[0]),
    ("vlm_off_policy_bs16", "Off-policy bs16 (VLM)", _TAB10[2]),
    ("vlm_streaming", "Streaming (VLM)", _TAB10[3]),
    ("vlm_off_policy_bs1", "Off-policy bs1 (VLM)", _TAB10[1]),
]


def main():
    args = parse_args()
    df = pd.read_csv(args.csv_path)

    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
        }
    )
    fig, ax = plt.subplots(figsize=(12, 7.5))

    steps = df["Step"].to_numpy()
    label_positions = []  # (x_end, y_end, label, color)

    for key, label, color in METHODS:
        mean_col = f"exp_name: {key} - recent_average_score"
        lo_col = f"{mean_col}__MIN"
        hi_col = f"{mean_col}__MAX"
        if mean_col not in df.columns:
            print(f"Skipping missing column: {mean_col}")
            continue
        mean = df[mean_col].to_numpy(dtype=float)
        lo = df[lo_col].to_numpy(dtype=float)
        hi = df[hi_col].to_numpy(dtype=float)
        valid = ~np.isnan(mean)
        x = steps[valid]
        y = mean[valid]
        ax.plot(x, y, color=color, linewidth=2)
        ax.fill_between(x, lo[valid], hi[valid], color=color, alpha=0.2)
        label_positions.append((x[-1], y[-1], label, color))

    ax.set_xlabel("Global Step")
    ax.set_ylabel("Moving Average Score")
    ax.set_xticks([0, 25_000, 50_000, 75_000, 100_000])
    ax.set_xticklabels(["0", "25k", "50k", "75k", "100k"])
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Inline text labels at the right end of each curve.
    x_max = max(max(p[0] for p in label_positions), 100_000)
    ax.set_xlim(right=x_max)
    label_positions_sorted = sorted(label_positions, key=lambda p: p[1])
    y_lo, y_hi = ax.get_ylim()
    min_gap = (y_hi - y_lo) * 0.05
    adjusted_ys = []
    prev_y = -np.inf
    for _, y_end, _, _ in label_positions_sorted:
        ny = max(y_end, prev_y + min_gap)
        adjusted_ys.append(ny)
        prev_y = ny
    for (x_end, _, label, color), ny in zip(label_positions_sorted, adjusted_ys):
        ax.text(
            x_end,
            ny,
            "  " + label,
            color=color,
            va="center",
            ha="left",
            fontsize=20,
            fontweight="bold",
            clip_on=False,
        )

    fig.subplots_adjust(right=0.72)

    output = args.csv_path.parent / "moving_average_tracking_square.pdf"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {output}")


if __name__ == "__main__":
    main()
