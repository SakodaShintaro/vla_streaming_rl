"""Summarize max_stage_reached of an Animal-AI run in life mode.

Usage:
    uv run python ralph/life_summary.py tsv <result_dir>
    uv run python ralph/life_summary.py wandb <run_id>

max_stage_reached is the furthest stage any game of the run has reached so far
(1-based; the number of stages + 1 once a game has cleared them all), so it never
decreases and its last value is the run's result. reached_at_episode is the first
episode whose row shows that value: of two runs with the same max_stage_reached,
the one that got there earlier is the better. tsv reads <result_dir>/log_episode.tsv.
"""

import argparse
import csv
import json
from pathlib import Path

WANDB_PROJECT = "shintaro-sakoda/vla_streaming_rl_AnimalAI-v0"
EPISODE_LIMIT = 2000


def summarize(rows: list[tuple[int, int, int]]) -> dict:
    """rows: (episode_id, max_stage_reached, games_ended) in episode order."""
    assert len(rows) > 0, "no episode yet"
    max_stage = rows[-1][1]
    reached_at = next(episode for episode, stage, _ in rows if stage == max_stage)
    return {
        "max_stage_reached": max_stage,
        "reached_at_episode": reached_at,
        "games_ended": rows[-1][2],
    }


def from_tsv(result_dir: Path) -> tuple[list[tuple[int, int, int]], int, bool]:
    with open(result_dir / "log_episode.tsv") as f:
        table = list(csv.DictReader(f, delimiter="\t"))
    rows = [
        (int(row["episode_id"]), int(row["max_stage_reached"]), int(row["games_ended"]))
        for row in table
    ]
    episodes = len(rows)
    finished = (result_dir / "train_state.json").exists() and episodes >= EPISODE_LIMIT
    return rows, episodes, finished


def from_wandb(run_id: str) -> tuple[list[tuple[int, int, int]], int, bool]:
    import wandb

    run = wandb.Api().run(f"{WANDB_PROJECT}/{run_id}")
    history = run.history(
        keys=["episode_id", "max_stage_reached", "games_ended"], pandas=False, samples=100000
    )
    rows = sorted(
        (int(row["episode_id"]), int(row["max_stage_reached"]), int(row["games_ended"]))
        for row in history
    )
    episodes = int(run.summary["episode_id"]) + 1
    finished = run.state == "finished"
    return rows, episodes, finished


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="source", required=True)
    sub.add_parser("tsv").add_argument("result_dir", type=Path)
    sub.add_parser("wandb").add_argument("run_id")
    args = parser.parse_args()

    if args.source == "tsv":
        rows, episodes, finished = from_tsv(args.result_dir)
    else:
        rows, episodes, finished = from_wandb(args.run_id)
    print(json.dumps({"episodes": episodes, "finished": finished, **summarize(rows)}))


if __name__ == "__main__":
    main()
