"""Predictor-corrector proximal multiplier method."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from algorithms.base import (
    RoundStats,
    RunResult,
    Trace,
    Workspace,
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
from model.core import Instance


@dataclass(frozen=True)
class PcpmSettings:
    prox_weight_prosumer: float
    prox_weight_vpp: float
    dual_step: float
    k_max: int
    eps_rel: float
    solver: str
    init_price: str = "midprice"

    def step_bound(self, size: int) -> float:
        return min(self.prox_weight_prosumer / (2.0 * size), self.prox_weight_vpp / 2.0)


def run_pcpm(inst: Instance, settings: PcpmSettings, log_every: int = 10) -> RunResult:
    ws = build_workspace(inst)
    nets, x = cold_start(inst, settings.solver)
    mu = initial_price(inst, settings.init_price)
    trace = Trace()
    converged, k = False, 0
    for k in range(1, settings.k_max + 1):
        tic = time.perf_counter()
        nets, x, mu, primal, change = _step(ws, nets, x, mu, settings)
        seconds = time.perf_counter() - tic
        stats = RoundStats(primal, change, tolerance(settings.eps_rel, x, nets.sum(axis=0)), seconds)
        record(trace, ws, nets, mu, local_costs_of(ws), stats)
        trace.primal_residual_last.append(primal)
        if log_every and k % log_every == 0:
            log_line(k, trace)
        if done(stats):
            converged = True
            break
    return RunResult(nets, x, mu, local_costs_of(ws), k, converged, trace)


def _step(
    ws: Workspace, nets: np.ndarray, x: np.ndarray, mu: np.ndarray, settings: PcpmSettings
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    sum_prev = nets.sum(axis=0)
    mu_hat = mu + settings.dual_step * (x - sum_prev)
    nets_new = np.array(
        [
            sub.solve(mu_hat, nets[i], settings.prox_weight_prosumer, settings.solver)
            for i, sub in enumerate(ws.subproblems)
        ]
    )
    x_new = ws.vpp.solve(mu_hat, x, settings.prox_weight_vpp, settings.solver)
    sum_new = nets_new.sum(axis=0)
    mu_new = mu + settings.dual_step * (x_new - sum_new)
    primal, change = residuals(x_new, sum_new, x, sum_prev)
    return nets_new, x_new, mu_new, primal, change
