"""Device schedules and settlement of the masked protocol run, with both benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from algorithms.admm_sharing import AdmmSettings, run_admm_sharing
from algorithms.base import RunResult, Workspace, build_workspace
from experiments.common import make_run_dir, write_json
from model.core import (
    V2_ROOT,
    BenchmarkResult,
    CentralResult,
    Instance,
    device_arrays,
    load_config,
    load_instance,
    market_revenue_value,
    market_terms,
    solve_centralized,
    solve_uncoordinated,
    value_of,
)
from model.settlement import TOLERANCE, settled_payoffs, standalone_blocks
from privacy.aggregators import MaskedSum

MASK_SEED = 1
VIEW_KEYS = ("nets_final", "mu_final", "total_final")
REGIMES = ("coordinated", "uncoordinated", "standalone")
ACTIVE_KW = 1e-6
DEVICE_FORMAT = "%.10g"
Alone = tuple[np.ndarray, list]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default.json")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--solver", default=None)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def protocol_run(inst: Instance, config: dict, solver: str) -> tuple[RunResult, Workspace, MaskedSum]:
    settings = AdmmSettings(
        rho=float(config["admm_rho"]),
        k_max=int(config["k_max"]),
        eps_rel=float(config["eps_rel"]),
        solver=solver,
        init_price=config["init_price"],
    )
    column = MaskedSum(inst.size, inst.horizon, MASK_SEED)
    workspace = build_workspace(inst)
    result = run_admm_sharing(inst, settings, aggregator=column, workspace=workspace)
    return result, workspace, column


def regression(result: RunResult, column: MaskedSum, blocks: list, reference: str) -> dict:
    views = np.load(os.path.join(reference, "views.npz"))
    mine = {"nets_final": result.nets, "mu_final": result.mu, "total_final": column.last_total}
    flags = {f"{key}_identical": bool(np.array_equal(mine[key], views[key])) for key in VIEW_KEYS}
    with open(os.path.join(reference, "summary.json"), encoding="utf-8") as handle:
        flags["k_final_equal"] = result.k_final == json.load(handle)["k_final"]
    final_nets = np.array([device_arrays(block)["net"] for block in blocks])
    flags["device_nets_identical"] = bool(np.array_equal(final_nets, result.nets))
    return flags


def device_frame(regime: str, inst: Instance, blocks: list) -> pd.DataFrame:
    frames = []
    for index, (par, block) in enumerate(zip(inst.prosumers, blocks), start=1):
        frame = pd.DataFrame(device_arrays(block))
        frame.insert(0, "hour", np.arange(1, inst.horizon + 1))
        frame.insert(0, "prosumer", index)
        frame.insert(0, "regime", regime)
        frame["pv_max"], frame["load"], frame["t_out"] = par.pv_max, par.load, par.t_out
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def device_checks(devices: pd.DataFrame) -> dict:
    supply = devices["pv"] + devices["discharge"] - devices["charge"] - devices["hvac"] - devices["load"]
    both = (devices["charge"] > ACTIVE_KW) & (devices["discharge"] > ACTIVE_KW)
    return {
        "balance_residual_max_kw": float((supply - devices["net"]).abs().max()),
        "simultaneous_charge_discharge": {regime: int(both[devices["regime"] == regime].sum()) for regime in REGIMES},
        "pv_curtailed_kwh": {
            regime: float((devices["pv_max"] - devices["pv"])[devices["regime"] == regime].sum()) for regime in REGIMES
        },
    }


def prosumer_frame(inst: Instance, result: RunResult, bench: BenchmarkResult, alone: Alone) -> pd.DataFrame:
    payoffs, blocks = alone
    pi_peak = float(inst.config["pi_peak"])
    terms = [market_terms(device_arrays(block)["net"], inst.buy, inst.sell, pi_peak) for block in blocks]
    settled = settled_payoffs(result.nets, result.mu, result.local_costs)
    return pd.DataFrame(
        {
            "prosumer": np.arange(1, inst.size + 1),
            "mean_load_kw": [float(np.mean(par.load)) for par in inst.prosumers],
            "pv_peak_kw": [float(np.max(par.pv_max)) for par in inst.prosumers],
            "settlement_received": result.nets @ result.mu,
            "local_cost": result.local_costs,
            "payoff": settled,
            "standalone_payoff": payoffs,
            "margin": settled - payoffs,
            "standalone_purchase": [term["purchase"] for term in terms],
            "standalone_sales": [term["sales"] for term in terms],
            "standalone_peak_charge": [term["peak_charge"] for term in terms],
            "standalone_local_cost": [value_of(block.local_cost) for block in blocks],
            "uncoordinated_local_cost": bench.local_costs,
        }
    )


def coalition_costs(inst: Instance, result: RunResult, bench: BenchmarkResult, frame: pd.DataFrame) -> dict:
    pi_peak = float(inst.config["pi_peak"])
    coordinated = market_terms(result.x_tilde, inst.buy, inst.sell, pi_peak)
    coordinated["degradation"] = float(result.local_costs.sum())
    uncoordinated = market_terms(bench.nets.sum(axis=0), inst.buy, inst.sell, pi_peak)
    uncoordinated["degradation"] = float(bench.local_costs.sum())
    standalone = {
        key: float(frame[f"standalone_{key}"].sum()) for key in ("purchase", "sales", "peak_charge", "local_cost")
    }
    standalone["degradation"] = standalone.pop("local_cost")
    costs = {"coordinated": coordinated, "uncoordinated": uncoordinated, "standalone": standalone}
    for terms in costs.values():
        terms["payoff"] = terms["sales"] - terms["purchase"] - terms["peak_charge"] - terms["degradation"]
    return costs


def margin_fields(frame: pd.DataFrame) -> dict:
    worst = int(frame["margin"].idxmin())
    identity = frame["standalone_sales"] - frame["standalone_purchase"] - frame["standalone_peak_charge"]
    identity = identity - frame["standalone_local_cost"] - frame["standalone_payoff"]
    return {
        "margin_min": float(frame["margin"].min()),
        "margin_min_prosumer": int(frame["prosumer"][worst]),
        "worse_count": int((frame["margin"] < -TOLERANCE).sum()),
        "below_zero_count": int((frame["margin"] < 0).sum()),
        "standalone_identity_max": float(identity.abs().max()),
    }


def hourly_frame(
    inst: Instance, central: CentralResult, result: RunResult, bench: BenchmarkResult, alone: Alone
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "buy": inst.buy,
            "sell": inst.sell,
            "mu_final": result.mu,
            "mu_central": central.mu,
            "x_vpp_kw": result.x,
            "sum_p_kw": result.x_tilde,
            "x_central_kw": central.x,
            "sum_p_uncoordinated_kw": bench.nets.sum(axis=0),
            "sum_p_standalone_kw": np.sum([device_arrays(block)["net"] for block in alone[1]], axis=0),
        }
    )
    frame.index = np.arange(1, inst.horizon + 1)
    return frame


def reference_dir(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(V2_ROOT, path)


def collect(inst: Instance, config: dict, solver: str, reference: str) -> tuple[dict, dict]:
    central = solve_centralized(inst, solver)
    result, workspace, column = protocol_run(inst, config, solver)
    bench = solve_uncoordinated(inst, solver)
    alone = standalone_blocks(inst, solver)
    final_blocks = [sub.block for sub in workspace.subproblems]
    devices = pd.concat(
        [device_frame(name, inst, blocks) for name, blocks in zip(REGIMES, (final_blocks, bench.blocks, alone[1]))],
        ignore_index=True,
    )
    frame = prosumer_frame(inst, result, bench, alone)
    imbalance = float(result.mu @ result.x_tilde) - market_revenue_value(
        result.x_tilde, inst.buy, inst.sell, float(inst.config["pi_peak"])
    )
    summary = {"n_prosumers": inst.size, "solver": solver, "mask_seed": MASK_SEED, "reference": reference}
    summary.update({"k_final": result.k_final, "converged": result.converged, "settlement_imbalance": imbalance})
    summary["regression"] = regression(result, column, final_blocks, reference_dir(reference))
    summary["margins"] = margin_fields(frame)
    summary["coalition"] = coalition_costs(inst, result, bench, frame)
    summary["devices"] = device_checks(devices)
    summary.update(column.report())
    tables = {"hourly": hourly_frame(inst, central, result, bench, alone), "per_prosumer": frame, "devices": devices}
    return summary, tables


def write_outputs(run_dir: str, summary: dict, tables: dict) -> None:
    write_json(run_dir, "summary.json", summary)
    tables["hourly"].to_csv(os.path.join(run_dir, "hourly.csv"), index_label="hour", lineterminator="\n")
    tables["per_prosumer"].to_csv(os.path.join(run_dir, "per_prosumer.csv"), index=False, lineterminator="\n")
    tables["devices"].to_csv(
        os.path.join(run_dir, "devices.csv"), index=False, lineterminator="\n", float_format=DEVICE_FORMAT
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    solver = args.solver or config["solver"]
    inst = load_instance(config)
    print(f"exp06 N={inst.size} solver={solver} reference={args.reference}", flush=True)
    summary, tables = collect(inst, config, solver, args.reference)
    run_dir = make_run_dir("exp06", config, args.tag)
    write_outputs(run_dir, summary, tables)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("run_dir:", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
