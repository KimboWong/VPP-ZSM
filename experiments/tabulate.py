"""Collect the summaries of all runs of an experiment into one CSV file."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd

from model.core import V2_ROOT


def load_rows(experiment: str) -> pd.DataFrame:
    pattern = os.path.join(V2_ROOT, "results", experiment, "*", "summary.json")
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as handle:
            row = json.load(handle)
        row["run_dir"] = os.path.basename(os.path.dirname(path))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    table = load_rows(args.experiment)
    out = args.out or os.path.join(V2_ROOT, "results", args.experiment, "table.csv")
    table.to_csv(out, index=False, lineterminator="\n")
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(table.to_string(index=False))
    print("rows:", len(table), "->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
