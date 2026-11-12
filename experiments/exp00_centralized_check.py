"""Centralized solution, the two benchmarks and the settlement checks."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from experiments.common import make_run_dir, write_json
from model.subproblems import ProsumerSubproblem
from model.core import (
    V2_ROOT,
    BenchmarkResult,
    CentralResult,
    Instance,
    load_config,
    load_instance,
    load_prices,
    solve_centralized,
    solve_uncoordinated,
)
from model.settlement import benchmark_payoffs, coordinated_payoffs, settlement_checks, solve_standalone


def price_consistency(config: dict) -> dict:
    buy_vpp, sell_vpp = load_prices(config)
    book = os.path.join(V2_ROOT, "data", config["prosumer_file"])
    buy_pro, sell_pro = load_prices(config, book)
    return {
        "max_abs_diff_buy": float(np.max(np.abs(buy_vpp - buy_pro))),
        "max_abs_diff_sell": float(np.max(np.abs(sell_vpp - sell_pro))),
    }


def time_subproblems(inst: Instance, solver: str, repeats: int = 5) -> dict:
    rng = np.random.default_rng(0)
    build_start = time.perf_counter()
    subs = [ProsumerSubproblem(par, inst.config) for par in inst.prosumers]
    build_time = time.perf_counter() - build_start
    zeros = np.zeros(inst.horizon)
    round_times = []
    for _ in range(repeats):
        mu_hat = 0.5 * (inst.buy + inst.sell) + rng.normal(0.0, 0.01, inst.horizon)
        start = time.perf_counter()
        for sub in subs:
            sub.solve(mu_hat, zeros, float(inst.config["prox_weight_prosumer"]), solver)
        round_times.append(time.perf_counter() - start)
    return {
        "build_seconds": build_time,
        "round_seconds_mean": float(np.mean(round_times)),
        "round_seconds_max": float(np.max(round_times)),
        "is_dpp": bool(subs[0].problem.is_dpp()),
    }


def write_tables(
    run_dir: str, inst: Instance, central: CentralResult, bench: BenchmarkResult, standalone: np.ndarray
) -> None:
    hourly = pd.DataFrame(
        {
            "buy": inst.buy,
            "sell": inst.sell,
            "mu": central.mu,
            "x_kw": central.x,
            "sum_p_kw": central.nets.sum(axis=0),
            "sum_p_uncoordinated_kw": bench.nets.sum(axis=0),
        }
    )
    hourly.index = np.arange(1, inst.horizon + 1)
    hourly.to_csv(os.path.join(run_dir, "hourly.csv"), index_label="hour", lineterminator="\n")
    per = pd.DataFrame(
        {
            "coordinated_payoff": coordinated_payoffs(central),
            "benchmark_payoff_pro_rata": benchmark_payoffs(inst, bench)[0],
            "standalone_payoff": standalone,
            "local_cost": central.local_costs,
        }
    )
    per.index = np.arange(1, inst.size + 1)
    per.to_csv(os.path.join(run_dir, "per_prosumer.csv"), index_label="prosumer", lineterminator="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default.json")
    parser.add_argument("--solver", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--pi-peak", type=float, default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.pi_peak is not None:
        config["pi_peak"] = args.pi_peak
    solver = args.solver or config["solver"]
    inst = load_instance(config, n_prosumers=args.n)
    central = solve_centralized(inst, solver)
    bench = solve_uncoordinated(inst, solver)
    standalone = solve_standalone(inst, solver)
    summary = settlement_checks(inst, central, bench, standalone)
    summary.update({"solver": solver, "n_prosumers": inst.size, "price_consistency": price_consistency(config)})
    summary["pi_peak"] = float(config["pi_peak"])
    summary["timing"] = time_subproblems(inst, solver)
    run_dir = make_run_dir("exp00", config, args.tag)
    write_json(run_dir, "summary.json", summary)
    write_tables(run_dir, inst, central, bench, standalone)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("run_dir:", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
