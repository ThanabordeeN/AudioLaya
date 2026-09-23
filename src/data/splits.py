"""Stdlib-only call-level manifest splitting and JSONL I/O."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def assign_splits(records: list[dict], seed: int = 42) -> list[dict]:
    """Assign every original call to one split, stratified by class."""
    by_label: dict[int, set[str]] = defaultdict(set)
    for row in records:
        group = f"{row['source']}:{row['call_id']}"
        by_label[int(row["label"])].add(group)

    assignments: dict[str, str] = {}
    rng = random.Random(seed)
    for label in sorted(by_label):
        keys = sorted(by_label[label])
        rng.shuffle(keys)
        n = len(keys)
        if n < 3:
            raise ValueError("Need at least 3 original calls per class to make train/validation/test splits")
        n_train = max(1, math.floor(0.8 * n))
        n_validation = max(1, math.floor(0.1 * n))
        if n_train + n_validation >= n:
            n_train, n_validation = n - 2, 1
        for key in keys[:n_train]:
            assignments[key] = "train"
        for key in keys[n_train:n_train + n_validation]:
            assignments[key] = "validation"
        for key in keys[n_train + n_validation:]:
            assignments[key] = "test"

    result = []
    for row in records:
        item = dict(row)
        item["split"] = assignments[f"{row['source']}:{row['call_id']}"]
        result.append(item)

    seen: dict[str, str] = {}
    for row in result:
        group = f"{row['source']}:{row['call_id']}"
        previous = seen.setdefault(group, row["split"])
        if previous != row["split"]:
            raise AssertionError(f"Call {group} leaked across splits")
    return result


def write_manifest(records: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_manifests(paths: Iterable[str | Path]) -> list[dict]:
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    if not rows:
        raise ValueError("No records found in the supplied manifests")
    return rows
