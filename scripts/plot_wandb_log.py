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
        "-o", "--output", type=Path, help="Output image path (default: <run_dir>/plot.png)"
    )
    parser.add_argument("--show", action="store_true", help="Show the figure interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    wandb_file = find_wandb_file(args.run_dir)
    print(f"Reading {wandb_file}")
    history = load_history(wandb_file)

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

    fig, ax = plt.subplots(figsize=(10, 6))
    for key in args.keys:
        y = history[key]
        valid = ~(np.isnan(x) | np.isnan(y))
        xs = x[valid]
        ys = smooth(y[valid], args.smooth)
        ax.plot(xs, ys, label=key, linewidth=1.2)

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
