"""Compare the summaries and the traces of two runs."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from model.core import V2_ROOT

TIMING = ("seconds", "seconds_per_round", "setup_seconds", "round_seconds_median", "round_seconds")


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(V2_ROOT, path)


def flatten(summary: dict, prefix: str = "") -> dict:
    flat: dict = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            flat.update(flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def load_summary(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "summary.json"), "r", encoding="utf-8") as handle:
        return flatten(json.load(handle))


def is_timing(key: str) -> bool:
    return key in TIMING or key.startswith("timing.")


def summary_line(key: str, left: object, right: object) -> str:
    head = f"{key:40s} {left!r:>26} {right!r:>26}"
    if is_timing(key) or left is None or right is None:
        return head + ("  (timing)" if is_timing(key) else "  (one side only)")
    if isinstance(left, bool) or isinstance(right, bool):
        return head + ("  same" if left == right else "  DIFFERS")
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return head + f"  diff={abs(float(left) - float(right)):.3e}"
    return head + ("  same" if left == right else "  DIFFERS")


def trace_diffs(base_dir: str, new_dir: str) -> tuple[int, dict] | None:
    paths = [os.path.join(run_dir, "trace.csv") for run_dir in (base_dir, new_dir)]
    if not all(os.path.exists(path) for path in paths):
        return None
    base, new = pd.read_csv(paths[0]), pd.read_csv(paths[1])
    rounds = min(len(base), len(new))
    shared = [column for column in base.columns if column in new.columns and column not in TIMING and column != "k"]
    diffs = {c: float(np.max(np.abs(base[c].to_numpy()[:rounds] - new[c].to_numpy()[:rounds]))) for c in shared}
    return rounds, diffs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("new")
    args = parser.parse_args()
    base_dir, new_dir = resolve(args.baseline), resolve(args.new)
    base, new = load_summary(base_dir), load_summary(new_dir)
    for key in sorted(set(base) | set(new)):
        print(summary_line(key, base.get(key), new.get(key)))
    traced = trace_diffs(base_dir, new_dir)
    if traced is None:
        print("trace: not compared, trace.csv missing in at least one run")
        return 0
    rounds, diffs = traced
    print(f"trace: {rounds} common rounds")
    for column, value in diffs.items():
        print(f"  {column:28s} max abs diff {value:.3e}")
    print("max trace diff over common rounds:", max(diffs.values()) if diffs else "no shared columns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
