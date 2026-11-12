"""Checks of the terminal price of every run: dual gap, price band and peak-charge component."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from algorithms.base import Workspace, aggregate_power_bound, build_workspace
from algorithms.dual_subgradient import dual_value
from model.core import V2_ROOT, load_instance


def run_dirs(experiment: str) -> list[str]:
    pattern = os.path.join(V2_ROOT, "results", experiment, "*", "hourly.csv")
    return sorted(os.path.dirname(path) for path in glob.glob(pattern))


def load_json(run_dir: str, name: str) -> dict:
    with open(os.path.join(run_dir, name), "r", encoding="utf-8") as handle:
        return json.load(handle)


def workspace_for(run_dir: str, summary: dict, cache: dict) -> Workspace:
    config = load_json(run_dir, "config.json")
    key = (config["prosumer_file"], summary["n_prosumers"], summary["horizon"])
    if key not in cache:
        inst = load_instance(config, n_prosumers=summary["n_prosumers"], horizon=summary["horizon"])
        cache[key] = build_workspace(inst, x_bound=aggregate_power_bound(inst))
    return cache[key]


def price_row(run_dir: str, solver: str | None, cache: dict) -> dict:
    summary = load_json(run_dir, "summary.json")
    ws = workspace_for(run_dir, summary, cache)
    inst = ws.inst
    mu = pd.read_csv(os.path.join(run_dir, "hourly.csv"))["mu_algorithm"].to_numpy()
    objective = float(summary["centralized_objective"])
    q_value = dual_value(ws, mu, solver or summary["solver"])
    band = np.maximum(np.clip(inst.sell - mu, 0.0, None), np.clip(mu - inst.buy - ws.pi_peak, 0.0, None))
    return {
        "run_dir": os.path.basename(run_dir),
        "algorithm": summary["algorithm"],
        "n_prosumers": inst.size,
        "converged": summary["converged"],
        "relative_gap": summary["relative_gap"],
        "dual_gap": (-objective - q_value) / abs(objective),
        "band_violation": float(band.max()),
        "capacity_component": float(np.clip(mu - inst.buy, 0.0, None).sum()),
        "pi_peak": ws.pi_peak,
        "mu_max_abs_diff_vs_centralized": summary["mu_max_abs_diff_vs_centralized"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--solver", default=None)
    args = parser.parse_args()
    cache: dict = {}
    rows = []
    for run_dir in run_dirs(args.experiment):
        rows.append(price_row(run_dir, args.solver, cache))
        last = rows[-1]
        print(
            f"{last['run_dir']}: dual_gap={last['dual_gap']:.3e} band={last['band_violation']:.2e} "
            f"capacity={last['capacity_component']:.4f}",
            flush=True,
        )
    out = os.path.join(V2_ROOT, "results", args.experiment, "price_checks.csv")
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    print("rows:", len(rows), "->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
