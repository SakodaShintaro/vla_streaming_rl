# SPDX-License-Identifier: MIT
"""Read scalar series out of a local run-XXXX.wandb file, without the wandb UI."""

import json
from pathlib import Path

import numpy as np
import wandb.proto.wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore


def find_wandb_file(run_dir: Path) -> Path:
    """The newest .wandb datastore file under ``run_dir``."""
    candidates = list(run_dir.rglob("*.wandb"))
    assert candidates, f"No .wandb file found under {run_dir}"
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def scan_records(wandb_file: Path):
    """Yield every decoded protobuf Record in the datastore, in order."""
    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_file))
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        yield rec


def load_history(wandb_file: Path) -> dict[str, np.ndarray]:
    """{key: 1-D array} for every scalar logged via wandb.log, NaN-padded so
    every column has one row per wandb.log call."""
    columns: dict[str, list] = {}
    row_idx = 0
    for rec in scan_records(wandb_file):
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
    return {
        key: np.asarray(col + [np.nan] * (row_idx - len(col)), dtype=np.float64)
        for key, col in columns.items()
    }
