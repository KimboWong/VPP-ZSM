"""Projected dual subgradient method."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from algorithms.base import (
    RoundStats,
    RunResult,
    Trace,
    Workspace,
    aggregate_power_bound,
    build_workspace,
    cold_start,
    done,
    initial_price,
    local_costs_of,
    log_line,
    record,
    residuals,
    tolerance,
)
from model.core import Instance, market_revenue_value


@dataclass(frozen=True)
class SubgradientSettings:
    step0: float
    decay: str
    k_max: int
    eps_rel: float
    solver: str
    init_price: str = "midprice"


@dataclass
class _Ergodic:
    nets: np.ndarray
    x: np.ndarray
    local_costs: np.ndarray

    def update(self, k: int, nets: np.ndarray, x: np.ndarray, local_costs: np.ndarray) -> None:
        weight = 1.0 / k
        self.nets = self.nets + weight * (nets - self.nets)
        self.x = self.x + weight * (x - self.x)
        self.local_costs = self.local_costs + weight * (local_costs - self.local_costs)


def step_size(settings: SubgradientSettings, k: int) -> float:
    if settings.decay == "constant":
        return settings.step0
    if settings.decay == "sqrt":
        return settings.step0 / math.sqrt(k)
    if settings.decay == "harmonic":
        return settings.step0 / k
    raise ValueError(f"unknown decay {settings.decay}")


def run_dual_subgradient(inst: Instance, settings: SubgradientSettings, log_every: int = 10) -> RunResult:
    ws = build_workspace(inst, x_bound=aggregate_power_bound(inst))
    nets, x = cold_start(inst, settings.solver)
    mu = initial_price(inst, settings.init_price)
    erg = _Ergodic(nets, x, np.zeros(inst.size))
    trace = Trace()
    converged, k = False, 0
    for k in range(1, settings.k_max + 1):
        mu, stats, last_residual = _iterate(ws, erg, mu, k, settings)
        record(trace, ws, erg.nets, mu, erg.local_costs, stats)
        trace.primal_residual_last.append(last_residual)
        if log_every and k % log_every == 0:
            log_line(k, trace)
        if k > 1 and done(stats):
            converged = True
            break
    return RunResult(erg.nets, erg.x, mu, erg.local_costs, k, converged, trace)


def _iterate(
    ws: Workspace, erg: _Ergodic, mu: np.ndarray, k: int, settings: SubgradientSettings
) -> tuple[np.ndarray, RoundStats, float]:
    tic = time.perf_counter()
    nets, x = _solve_all(ws, mu, settings.solver)
    lower, upper = ws.inst.sell, ws.inst.buy + ws.pi_peak
    mu_new = np.clip(mu + step_size(settings, k) * (x - nets.sum(axis=0)), lower, upper)
    erg_sum_old, erg_x_old = erg.nets.sum(axis=0), erg.x
    erg.update(k, nets, x, local_costs_of(ws))
    seconds = time.perf_counter() - tic
    erg_sum = erg.nets.sum(axis=0)
    primal, change = residuals(erg.x, erg_sum, erg_x_old, erg_sum_old)
    stats = RoundStats(primal, change, tolerance(settings.eps_rel, erg.x, erg_sum), seconds)
    return mu_new, stats, float(np.linalg.norm(x - nets.sum(axis=0)))


def _solve_all(ws: Workspace, mu: np.ndarray, solver: str) -> tuple[np.ndarray, np.ndarray]:
    zeros = np.zeros(ws.inst.horizon)
    nets = np.array([sub.solve(mu, zeros, 0.0, solver) for sub in ws.subproblems])
    x = ws.vpp.solve(mu, zeros, 0.0, solver)
    return nets, x


def dual_value(ws: Workspace, mu: np.ndarray, solver: str) -> float:
    """Dual function q(mu) = min over the decoupled subproblems; equals minus the centralized objective at mu*."""
    nets, x = _solve_all(ws, mu, solver)
    revenue = market_revenue_value(x, ws.inst.buy, ws.inst.sell, ws.pi_peak)
    return float(local_costs_of(ws).sum()) - float(mu @ nets.sum(axis=0)) - revenue + float(mu @ x)
