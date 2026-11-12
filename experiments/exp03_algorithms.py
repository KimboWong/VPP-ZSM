"""Run one coordination algorithm with plaintext aggregation."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from algorithms.admm_sharing import AdmmSettings, run_admm_sharing
from algorithms.base import RunResult
from algorithms.dual_subgradient import SubgradientSettings, run_dual_subgradient
from algorithms.pcpm import PcpmSettings, run_pcpm
from experiments.common import make_run_dir, write_json
from experiments.summary import result_fields
from model.core import CentralResult, Instance, load_config, load_instance, solve_centralized

Settings = PcpmSettings | AdmmSettings | SubgradientSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["pcpm", "admm", "subgradient"], required=True)
    parser.add_argument("--config", default="default.json")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--solver", default=None)
    parser.add_argument("--init", default=None)
    parser.add_argument("--kmax", type=int, default=None)
    parser.add_argument("--eps-rel", type=float, default=None)
    parser.add_argument("--wp", type=float, default=None)
    parser.add_argument("--wx", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--step0", type=float, default=None)
    parser.add_argument("--decay", default=None)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def pick(override: float | None, default: object) -> float:
    return float(override) if override is not None else float(str(default))


def common_settings(config: dict, args: argparse.Namespace) -> dict:
    return {
        "k_max": args.kmax or int(config["k_max"]),
        "eps_rel": pick(args.eps_rel, config["eps_rel"]),
        "solver": args.solver or config["solver"],
        "init_price": args.init or config["init_price"],
    }


def build_settings(config: dict, args: argparse.Namespace) -> Settings:
    common = common_settings(config, args)
    if args.algo == "pcpm":
        return PcpmSettings(
            prox_weight_prosumer=pick(args.wp, config["prox_weight_prosumer"]),
            prox_weight_vpp=pick(args.wx, config["prox_weight_vpp"]),
            dual_step=pick(args.alpha, config["dual_step"]),
            **common,
        )
    if args.algo == "admm":
        return AdmmSettings(rho=pick(args.rho, config["admm_rho"]), **common)
    decay = args.decay or config["subgradient_decay"]
    return SubgradientSettings(step0=pick(args.step0, config["subgradient_step0"]), decay=decay, **common)


def run(inst: Instance, settings: Settings) -> RunResult:
    if isinstance(settings, PcpmSettings):
        return run_pcpm(inst, settings)
    if isinstance(settings, AdmmSettings):
        return run_admm_sharing(inst, settings)
    return run_dual_subgradient(inst, settings)


def settings_fields(inst: Instance, settings: Settings) -> dict:
    fields = {"n_prosumers": inst.size, "horizon": inst.horizon, **asdict(settings)}
    if isinstance(settings, PcpmSettings):
        fields["theorem_bound"] = settings.step_bound(inst.size)
        fields["within_bound"] = settings.dual_step < settings.step_bound(inst.size)
    return fields


def write_tables(run_dir: str, inst: Instance, central: CentralResult, result: RunResult) -> None:
    trace = pd.DataFrame(result.trace.__dict__)
    trace.index = np.arange(1, len(trace) + 1)
    trace.to_csv(os.path.join(run_dir, "trace.csv"), index_label="k", lineterminator="\n")
    hourly = pd.DataFrame(
        {
            "buy": inst.buy,
            "sell": inst.sell,
            "mu_algorithm": result.mu,
            "mu_central": central.mu,
            "x_vpp_kw": result.x,
            "sum_p_kw": result.x_tilde,
            "x_central_kw": central.x,
        }
    )
    hourly.index = np.arange(1, inst.horizon + 1)
    hourly.to_csv(os.path.join(run_dir, "hourly.csv"), index_label="hour", lineterminator="\n")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    settings = build_settings(config, args)
    inst = load_instance(config, n_prosumers=args.n, horizon=args.horizon)
    print(f"algo={args.algo} N={inst.size} T={inst.horizon} settings={asdict(settings)}", flush=True)
    central = solve_centralized(inst, settings.solver)
    start = time.perf_counter()
    result = run(inst, settings)
    summary = {"algorithm": args.algo, **settings_fields(inst, settings)}
    summary.update(result_fields(inst, central, result, time.perf_counter() - start))
    run_dir = make_run_dir("exp03", config, args.tag)
    write_json(run_dir, "summary.json", summary)
    write_tables(run_dir, inst, central, result)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("run_dir:", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
