"""Summarize last_round_success_rate (max / mean over closed rounds) of an Animal-AI run.

Usage:
    uv run python local/round_success_summary.py tsv <result_dir>
    uv run python local/round_success_summary.py wandb <run_id>

tsv reads <result_dir>/log_episode.tsv. Its columns are fixed by the first episode,
which has no last_round_success_rate, so the closed rounds are taken as the rows
whose round_index equals the number of arenas (round_success_rate there is the
closed round's rate, the same value wandb logs as last_round_success_rate).
"""

import argparse
import csv
import json
from pathlib import Path

WANDB_PROJECT = "shintaro-sakoda/vla_streaming_rl_AnimalAI-v0"


def rates_from_tsv(result_dir: Path) -> tuple[list[float], int, bool]:
    with open(result_dir / "log_episode.tsv") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(rows) > 0, f"no episode in {result_dir}"
    arena_num = max(int(row["round_index"]) for row in rows)
    rates = [
        float(row["round_success_rate"]) for row in rows if int(row["round_index"]) == arena_num
    ]
    episodes = len(rows)
    finished = (result_dir / "train_state.json").exists() and episodes >= 1200
    return rates, episodes, finished


def rates_from_wandb(run_id: str) -> tuple[list[float], int, bool]:
    import wandb

    run = wandb.Api().run(f"{WANDB_PROJECT}/{run_id}")
    history = run.history(keys=["last_round_success_rate"], pandas=False, samples=100000)
    rates = [float(row["last_round_success_rate"]) for row in history]
    episodes = int(run.summary["episode_id"]) + 1
    finished = run.state == "finished"
    return rates, episodes, finished


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="source", required=True)
    sub.add_parser("tsv").add_argument("result_dir", type=Path)
    sub.add_parser("wandb").add_argument("run_id")
    args = parser.parse_args()

    if args.source == "tsv":
        rates, episodes, finished = rates_from_tsv(args.result_dir)
    else:
        rates, episodes, finished = rates_from_wandb(args.run_id)
    assert len(rates) > 0, "no closed round yet"
    max_rate = max(rates)
    mean_rate = sum(rates) / len(rates)
    print(
        json.dumps(
            {
                "rounds": len(rates),
                "episodes": episodes,
                "finished": finished,
                "max": round(max_rate, 4),
                "mean": round(mean_rate, 4),
                "last5_mean": round(sum(rates[-5:]) / len(rates[-5:]), 4),
            }
        )
    )


if __name__ == "__main__":
    main()
