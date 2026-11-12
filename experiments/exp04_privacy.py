"""Run sharing ADMM with masked, noisy or encrypted aggregation."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict

import numpy as np

from algorithms.admm_sharing import AdmmSettings, run_admm_sharing
from algorithms.base import RunResult
from experiments.common import make_run_dir, write_json
from experiments.exp03_algorithms import write_tables
from experiments.summary import result_fields
from model.core import Instance, load_config, load_instance, solve_centralized
from privacy.aggregators import EncryptedSum, MaskedSum, NoisySum

PAILLIER_BITS = 2048
WINDOWS = (1, 10, 100)
Column = MaskedSum | NoisySum | EncryptedSum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["masked", "dp", "paillier"], required=True)
    parser.add_argument("--config", default="default.json")
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--kmax", type=int, default=None)
    parser.add_argument("--solver", default=None)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def admm_settings(config: dict, args: argparse.Namespace) -> AdmmSettings:
    return AdmmSettings(
        rho=float(config["admm_rho"]),
        k_max=args.kmax or int(config["k_max"]),
        eps_rel=float(config["eps_rel"]),
        solver=args.solver or config["solver"],
        init_price=config["init_price"],
    )


def make_column(inst: Instance, args: argparse.Namespace) -> Column:
    if args.mode == "masked":
        return MaskedSum(inst.size, inst.horizon, args.seed)
    if args.mode == "dp":
        if args.sigma is None:
            raise SystemExit("--mode dp needs --sigma (kW)")
        return NoisySum(inst, args.sigma, args.seed)
    return EncryptedSum(inst.size, inst.horizon, PAILLIER_BITS, args.seed)


def write_views(run_dir: str, result: RunResult, column: Column) -> None:
    views = {"nets_final": result.nets, "total_final": column.last_total, "mu_final": result.mu}
    if isinstance(column, NoisySum):
        views.update(column.window_means(WINDOWS))
    np.savez(os.path.join(run_dir, "views.npz"), **views)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    settings = admm_settings(config, args)
    inst = load_instance(config)
    print(f"mode={args.mode} N={inst.size} sigma={args.sigma} seed={args.seed} settings={asdict(settings)}", flush=True)
    central = solve_centralized(inst, settings.solver)
    column = make_column(inst, args)
    start = time.perf_counter()
    result = run_admm_sharing(inst, settings, aggregator=column)
    summary = {"algorithm": "admm", "privacy": args.mode, "n_prosumers": inst.size, "horizon": inst.horizon}
    summary.update(asdict(settings))
    summary.update(result_fields(inst, central, result, time.perf_counter() - start))
    summary.update(column.report())
    run_dir = make_run_dir("exp04", config, args.tag)
    write_json(run_dir, "summary.json", summary)
    write_tables(run_dir, inst, central, result)
    write_views(run_dir, result, column)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("run_dir:", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
