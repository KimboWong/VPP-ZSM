"""Summary fields shared by the experiment scripts."""

from __future__ import annotations

import numpy as np

from algorithms.base import RunResult
from model.core import CentralResult, Instance, market_revenue_value


def result_fields(inst: Instance, central: CentralResult, result: RunResult, elapsed: float) -> dict:
    pi_peak = float(inst.config["pi_peak"])
    revenue = market_revenue_value(result.x_tilde, inst.buy, inst.sell, pi_peak)
    objective_tilde = revenue - float(result.local_costs.sum())
    trace = result.trace
    return {
        "k_final": result.k_final,
        "converged": result.converged,
        "seconds": elapsed,
        "seconds_per_round": elapsed / max(result.k_final, 1),
        "setup_seconds": elapsed - float(np.sum(trace.round_seconds)),
        "round_seconds_median": float(np.median(trace.round_seconds)),
        "primal_residual_final": trace.primal_residual[-1],
        "change_residual_final": trace.change_residual[-1],
        "threshold_final": trace.threshold[-1],
        "primal_residual_last_iterate": trace.primal_residual_last[-1],
        "centralized_objective": central.objective,
        "objective_tilde": objective_tilde,
        "relative_gap": abs(objective_tilde - central.objective) / abs(central.objective),
        "settlement_imbalance_final": trace.settlement_imbalance[-1],
        "peak_kw_final": trace.peak_kw[-1],
        "centralized_peak_kw": float(np.max(np.clip(-central.x, 0.0, None))),
        "mu_max_abs_diff_vs_centralized": float(np.max(np.abs(result.mu - central.mu))),
    }
