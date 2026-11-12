"""Each coordination algorithm against the centralized optimum on a small instance."""

from __future__ import annotations

import numpy as np
import pytest

from algorithms.admm_sharing import AdmmSettings, run_admm_sharing
from algorithms.base import RunResult, aggregate_power_bound, build_workspace
from algorithms.dual_subgradient import SubgradientSettings, dual_value, run_dual_subgradient
from algorithms.pcpm import PcpmSettings, run_pcpm
from model.core import CentralResult, Instance, load_config, load_instance, market_revenue_value, solve_centralized

SOLVER = "CLARABEL"
EPS_REL = 1e-4


@pytest.fixture(scope="module")
def tiny() -> Instance:
    return load_instance(load_config(), n_prosumers=3, horizon=4)


@pytest.fixture(scope="module")
def central(tiny: Instance) -> CentralResult:
    return solve_centralized(tiny, SOLVER)


def relative_gap(inst: Instance, result: RunResult, central_objective: float) -> float:
    pi_peak = float(inst.config["pi_peak"])
    objective = market_revenue_value(result.x_tilde, inst.buy, inst.sell, pi_peak) - float(result.local_costs.sum())
    return abs(objective - central_objective) / abs(central_objective)


def price_gap(result: RunResult, central: CentralResult) -> float:
    return float(np.max(np.abs(result.mu - central.mu)))


def test_admm_reaches_centralized_optimum(tiny: Instance, central: CentralResult) -> None:
    settings = AdmmSettings(rho=0.4, k_max=5000, eps_rel=EPS_REL, solver=SOLVER)
    result = run_admm_sharing(tiny, settings, log_every=0)
    assert result.converged
    assert relative_gap(tiny, result, central.objective) <= 2e-4
    assert price_gap(result, central) <= 1e-3


def test_pcpm_reaches_centralized_optimum(tiny: Instance, central: CentralResult) -> None:
    settings = PcpmSettings(
        prox_weight_prosumer=0.2, prox_weight_vpp=0.0025, dual_step=0.001, k_max=5000, eps_rel=EPS_REL, solver=SOLVER
    )
    result = run_pcpm(tiny, settings, log_every=0)
    assert result.converged
    assert relative_gap(tiny, result, central.objective) <= 2e-4
    assert price_gap(result, central) <= 1e-3


def test_strong_duality_through_decoupled_subproblems(tiny: Instance, central: CentralResult) -> None:
    ws = build_workspace(tiny, x_bound=aggregate_power_bound(tiny))
    assert abs(dual_value(ws, central.mu, SOLVER) + central.objective) <= 1e-6 * abs(central.objective)


def test_subgradient_ergodic_average_approaches_optimum(tiny: Instance, central: CentralResult) -> None:
    settings = SubgradientSettings(step0=0.03, decay="sqrt", k_max=500, eps_rel=EPS_REL, solver=SOLVER)
    result = run_dual_subgradient(tiny, settings, log_every=0)
    assert relative_gap(tiny, result, central.objective) <= 5e-3
    assert price_gap(result, central) <= 2e-2
