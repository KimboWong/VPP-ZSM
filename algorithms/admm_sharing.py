"""Sharing ADMM for coordinating the households and the VPP."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from algorithms.base import (
    Aggregator,
    PlainSum,
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
class AdmmSettings:
    rho: float
    k_max: int
    eps_rel: float
    solver: str
    init_price: str = "midprice"


@dataclass
class _State:
    nets: np.ndarray
    x: np.ndarray
    u: np.ndarray
    total: np.ndarray


def run_admm_sharing(
    inst: Instance,
    settings: AdmmSettings,
    log_every: int = 10,
    aggregator: Aggregator | None = None,
    workspace: Workspace | None = None,
) -> RunResult:
    agg = aggregator or PlainSum()
    ws = workspace or build_workspace(inst)
    nets, _ = cold_start(inst, settings.solver)
    total = agg.total(nets, 0)
    state = _State(nets, total.copy(), -initial_price(inst, settings.init_price) / settings.rho, total)
    trace = Trace()
    converged, k = False, 0
    for k in range(1, settings.k_max + 1):
        tic = time.perf_counter()
        state, primal, change = _step(ws, state, settings, agg, k)
        seconds = time.perf_counter() - tic
        stats = RoundStats(primal, change, tolerance(settings.eps_rel, state.x, state.total), seconds)
        record(trace, ws, state.nets, -settings.rho * state.u, local_costs_of(ws), stats)
        trace.primal_residual_last.append(primal)
        if log_every and k % log_every == 0:
            log_line(k, trace)
        if done(stats):
            converged = True
            break
    return RunResult(state.nets, state.x, -settings.rho * state.u, local_costs_of(ws), k, converged, trace)


def _step(
    ws: Workspace, state: _State, settings: AdmmSettings, agg: Aggregator, round_index: int
) -> tuple[_State, float, float]:
    size = ws.inst.size
    zeros = np.zeros(ws.inst.horizon)
    # sum / N equals nets.mean(axis=0) bit for bit (numpy's mean is that reduction divided by the count)
    p_bar_old, x_old = state.total / size, state.x
    offset = p_bar_old - x_old / size + state.u
    weight = settings.rho / 2.0
    nets = np.array(
        [sub.solve(zeros, state.nets[i] - offset, weight, settings.solver) for i, sub in enumerate(ws.subproblems)]
    )
    total = agg.total(nets, round_index)
    p_bar = total / size
    x = ws.vpp.solve(zeros, size * (state.u + p_bar), settings.rho / (2.0 * size), settings.solver)
    u = state.u + p_bar - x / size
    primal, change = residuals(x, size * p_bar, x_old, size * p_bar_old)
    return _State(nets, x, u, total), primal, change
