"""Model invariants on a small instance (3 households, 4 hours)."""

from __future__ import annotations

import numpy as np
import pytest

from model.core import (
    Instance,
    device_arrays,
    load_config,
    load_instance,
    market_revenue_value,
    market_terms,
    solve_centralized,
    solve_uncoordinated,
)
from model.settlement import settlement_checks, solve_standalone, standalone_blocks

SOLVER = "CLARABEL"


@pytest.fixture(scope="module")
def tiny() -> Instance:
    return load_instance(load_config(), n_prosumers=3, horizon=4)


@pytest.fixture(scope="module")
def checks(tiny: Instance) -> dict:
    central = solve_centralized(tiny, SOLVER)
    bench = solve_uncoordinated(tiny, SOLVER)
    return settlement_checks(tiny, central, bench, solve_standalone(tiny, SOLVER))


def test_budget_balance_identity(checks: dict) -> None:
    assert checks["budget_gap"] < 1e-6


def test_dual_price_in_band(checks: dict) -> None:
    assert checks["mu_in_band"]


def test_capacity_component_sums_to_peak_charge(tiny: Instance, checks: dict) -> None:
    assert abs(checks["capacity_component"] - float(tiny.config["pi_peak"])) < 1e-6


def test_no_simultaneous_charge_discharge(checks: dict) -> None:
    assert checks["simultaneous_charge_discharge"] == 0


def test_coordination_gain_nonnegative(checks: dict) -> None:
    assert checks["coordination_gain"] >= -1e-9


def test_participation_is_individually_rational(checks: dict) -> None:
    assert checks["ir_standalone_worse_count"] == 0
    assert checks["participation_value"] >= -1e-9


def test_prices_positive_and_spread(tiny: Instance) -> None:
    assert np.all(tiny.sell > 0) and np.all(tiny.buy > tiny.sell)


def test_devices_add_up_to_net(tiny: Instance) -> None:
    blocks = solve_uncoordinated(tiny, SOLVER).blocks + standalone_blocks(tiny, SOLVER)[1]
    for par, block in zip(list(tiny.prosumers) * 2, blocks):
        dev = device_arrays(block)
        supply = dev["pv"] + dev["discharge"] - dev["charge"] - dev["hvac"] - par.load
        assert np.max(np.abs(supply - dev["net"])) < 1e-6


def test_market_terms_add_up_to_revenue(tiny: Instance) -> None:
    pi_peak = float(tiny.config["pi_peak"])
    x = np.array([-3.0, 2.0, -1.0, 0.5])
    terms = market_terms(x, tiny.buy, tiny.sell, pi_peak)
    revenue = terms["sales"] - terms["purchase"] - terms["peak_charge"]
    assert abs(revenue - market_revenue_value(x, tiny.buy, tiny.sell, pi_peak)) < 1e-12
