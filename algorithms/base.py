"""Shared pieces of the coordination algorithms: cold start, trace, stopping test and aggregator interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from model.core import Instance, market_revenue_value, solve_uncoordinated
from model.subproblems import ProsumerSubproblem, VppSubproblem


@dataclass
class Trace:
    primal_residual: list = field(default_factory=list)
    change_residual: list = field(default_factory=list)
    threshold: list = field(default_factory=list)
    primal_residual_last: list = field(default_factory=list)
    objective_tilde: list = field(default_factory=list)
    settlement_imbalance: list = field(default_factory=list)
    peak_kw: list = field(default_factory=list)
    round_seconds: list = field(default_factory=list)


@dataclass
class RunResult:
    nets: np.ndarray
    x: np.ndarray
    mu: np.ndarray
    local_costs: np.ndarray
    k_final: int
    converged: bool
    trace: Trace

    @property
    def x_tilde(self) -> np.ndarray:
        return self.nets.sum(axis=0)


@dataclass(frozen=True)
class Workspace:
    inst: Instance
    subproblems: list
    vpp: VppSubproblem
    pi_peak: float


class Aggregator(Protocol):
    def total(self, nets: np.ndarray, round_index: int) -> np.ndarray: ...


class PlainSum:
    """Plaintext aggregation: the aggregator sees every message and adds them."""

    def total(self, nets: np.ndarray, round_index: int) -> np.ndarray:
        return nets.sum(axis=0)


@dataclass(frozen=True)
class RoundStats:
    primal: float
    change: float
    limit: float
    seconds: float


def build_workspace(inst: Instance, x_bound: float | None = None, solver_options: dict | None = None) -> Workspace:
    subproblems = [ProsumerSubproblem(par, inst.config, solver_options) for par in inst.prosumers]
    vpp = VppSubproblem(inst, x_bound, solver_options)
    return Workspace(inst, subproblems, vpp, float(inst.config["pi_peak"]))


def aggregate_power_bound(inst: Instance) -> float:
    """Largest |sum_i p_i,t| any feasible schedule can reach, from the device limits alone."""
    return float(sum(np.max(par.pv_max) + par.bess_pmax + par.hvac_max + np.max(par.load) for par in inst.prosumers))


def initial_price(inst: Instance, mode: str) -> np.ndarray:
    if mode == "zero":
        return np.zeros(inst.horizon)
    if mode == "midprice":
        return 0.5 * (inst.buy + inst.sell)
    raise ValueError(f"unknown init_price {mode}")


def cold_start(inst: Instance, solver: str) -> tuple[np.ndarray, np.ndarray]:
    nets = solve_uncoordinated(inst, solver).nets
    return nets, nets.sum(axis=0)


def residuals(x_new: np.ndarray, sum_new: np.ndarray, x_old: np.ndarray, sum_old: np.ndarray) -> tuple[float, float]:
    primal = float(np.linalg.norm(x_new - sum_new))
    change = float(np.linalg.norm(sum_new - sum_old) + np.linalg.norm(x_new - x_old))
    return primal, change


def tolerance(eps_rel: float, x: np.ndarray, sum_p: np.ndarray) -> float:
    return eps_rel * max(float(np.linalg.norm(x)), float(np.linalg.norm(sum_p)))


def local_costs_of(ws: Workspace) -> np.ndarray:
    return np.array([sub.local_cost for sub in ws.subproblems])


def record(
    trace: Trace, ws: Workspace, nets: np.ndarray, mu: np.ndarray, local_costs: np.ndarray, stats: RoundStats
) -> None:
    x_tilde = nets.sum(axis=0)
    g_value = market_revenue_value(x_tilde, ws.inst.buy, ws.inst.sell, ws.pi_peak)
    trace.primal_residual.append(stats.primal)
    trace.change_residual.append(stats.change)
    trace.threshold.append(stats.limit)
    trace.objective_tilde.append(g_value - float(local_costs.sum()))
    trace.settlement_imbalance.append(float(mu @ x_tilde) - g_value)
    trace.peak_kw.append(float(np.max(np.clip(-x_tilde, 0.0, None))))
    trace.round_seconds.append(stats.seconds)


def log_line(k: int, trace: Trace) -> None:
    print(
        f"  k={k:5d} r={trace.primal_residual[-1]:10.4f} s={trace.change_residual[-1]:10.4f} "
        f"thr={trace.threshold[-1]:8.4f} obj~={trace.objective_tilde[-1]:12.4f} "
        f"imb={trace.settlement_imbalance[-1]:10.4f} peak={trace.peak_kw[-1]:8.2f}",
        flush=True,
    )


def done(stats: RoundStats) -> bool:
    return stats.primal <= stats.limit and stats.change <= stats.limit
