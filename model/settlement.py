"""Settlement checks and the benchmarks B1 (uncoordinated) and B2 (standalone)."""

from __future__ import annotations

import cvxpy as cvx
import numpy as np

from model.core import BenchmarkResult, CentralResult, Instance, market_revenue, market_revenue_value, prosumer_block

TOLERANCE = 1e-6


def benchmark_payoffs(inst: Instance, bench: BenchmarkResult) -> tuple[np.ndarray, float, float]:
    pi_peak = float(inst.config["pi_peak"])
    x_unc = bench.nets.sum(axis=0)
    peak_unc = float(np.max(np.clip(-x_unc, 0.0, None)))
    t_peak = int(np.argmax(np.clip(-x_unc, 0.0, None)))
    contribution = np.clip(-bench.nets[:, t_peak], 0.0, None)
    share = contribution / contribution.sum() if contribution.sum() > 0 else np.full(inst.size, 1.0 / inst.size)
    energy = np.minimum(inst.buy * bench.nets, inst.sell * bench.nets).sum(axis=1)
    payoff = energy - bench.local_costs - share * pi_peak * peak_unc
    objective = market_revenue_value(x_unc, inst.buy, inst.sell, pi_peak) - float(bench.local_costs.sum())
    return payoff, objective, peak_unc


def standalone_blocks(inst: Instance, solver: str) -> tuple[np.ndarray, list]:
    """Benchmark B2 with each prosumer's solved block, for the device schedules of standing alone."""
    pi_peak = float(inst.config["pi_peak"])
    payoffs, blocks = [], []
    for par in inst.prosumers:
        block = prosumer_block(par, inst.config)
        objective = market_revenue(block.net, inst.buy, inst.sell, pi_peak) - block.local_cost
        problem = cvx.Problem(cvx.Maximize(objective), block.constraints)
        problem.solve(solver=getattr(cvx, solver))
        if problem.status != cvx.OPTIMAL:
            raise RuntimeError(f"standalone subproblem status {problem.status}")
        payoffs.append(float(problem.value))
        blocks.append(block)
    return np.array(payoffs), blocks


def solve_standalone(inst: Instance, solver: str) -> np.ndarray:
    """Benchmark B2: each prosumer's best payoff trading alone at the wholesale prices with its own peak charge."""
    return standalone_blocks(inst, solver)[0]


def settled_payoffs(nets: np.ndarray, mu: np.ndarray, local_costs: np.ndarray) -> np.ndarray:
    """Each prosumer's payoff when settled at the price mu: mu'p_i minus its local cost, $."""
    return nets @ mu - local_costs


def coordinated_payoffs(central: CentralResult) -> np.ndarray:
    return settled_payoffs(central.nets, central.mu, central.local_costs)


def settlement_checks(inst: Instance, central: CentralResult, bench: BenchmarkResult, standalone: np.ndarray) -> dict:
    pi_peak = float(inst.config["pi_peak"])
    g_value = market_revenue_value(central.x, inst.buy, inst.sell, pi_peak)
    peak = float(np.max(np.clip(-central.x, 0.0, None)))
    _, unc_objective, peak_unc = benchmark_payoffs(inst, bench)
    in_band = bool(np.all(central.mu >= inst.sell - 1e-7) and np.all(central.mu <= inst.buy + pi_peak + 1e-7))
    checks = {
        "objective": central.objective,
        "market_term": g_value,
        "peak_kw": peak,
        "hours_at_peak": int(np.sum(np.abs(-central.x - peak) < 0.01)),
        "budget_gap": abs(float(central.mu @ central.x) - g_value),
        "mu_in_band": in_band,
        "capacity_component": float(np.clip(central.mu - inst.buy, 0.0, None).sum()),
        "simultaneous_charge_discharge": int(np.sum((central.charge > 1e-6) & (central.discharge > 1e-6))),
        "uncoordinated_objective": unc_objective,
        "uncoordinated_peak_kw": peak_unc,
        "coordination_gain": central.objective - unc_objective,
    }
    checks.update(benchmark_comparisons(inst, central, bench, standalone))
    return checks


def benchmark_comparisons(
    inst: Instance, central: CentralResult, bench: BenchmarkResult, standalone: np.ndarray
) -> dict:
    v_in = coordinated_payoffs(central)
    share_gap = v_in - benchmark_payoffs(inst, bench)[0]
    ir_gap = v_in - standalone
    return {
        "standalone_objective": float(standalone.sum()),
        "participation_value": central.objective - float(standalone.sum()),
        "ir_standalone_min": float(ir_gap.min()),
        "ir_standalone_worse_count": int(np.sum(ir_gap < -TOLERANCE)),
        "share_gap_min": float(share_gap.min()),
        "share_worse_count": int(np.sum(share_gap < -TOLERANCE)),
    }
