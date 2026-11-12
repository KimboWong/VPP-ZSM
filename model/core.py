"""Household and VPP model, centralized solution and the uncoordinated benchmark.

Units: kW, kWh, $/kWh, one-hour slots. Sign: p > 0 is injection into the VPP, X > 0 is VPP export.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import cvxpy as cvx
import numpy as np
import pandas as pd

V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOUR_COLUMNS = [str(hour) for hour in range(1, 25)]


def load_config(name: str = "default.json") -> dict:
    with open(os.path.join(V2_ROOT, "config", name), "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if float(config["delta_t_hours"]) != 1.0:
        raise ValueError("the model equations assume one-hour slots")
    return config


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


@dataclass(frozen=True)
class Prosumer:
    pv_max: np.ndarray
    bess_pmax: float
    soc_max: float
    hvac_max: float
    t_min: float
    t_max: float
    alpha: float
    beta: float
    t_out: np.ndarray
    load: np.ndarray


@dataclass(frozen=True)
class Instance:
    prosumers: tuple
    buy: np.ndarray
    sell: np.ndarray
    config: dict

    @property
    def horizon(self) -> int:
        return len(self.buy)

    @property
    def size(self) -> int:
        return len(self.prosumers)


def _read_sheet(book: str, sheet: str) -> pd.DataFrame:
    return pd.read_excel(book, sheet_name=sheet, header=0, index_col=0)


def _hours(frame: pd.DataFrame) -> pd.DataFrame:
    # the workbooks label hour columns 1..24 as integers in some sheets and strings in others
    by_label = {str(column): column for column in frame.columns}
    missing = [hour for hour in HOUR_COLUMNS if hour not in by_label]
    if missing:
        raise KeyError(f"hour columns missing: {missing}")
    return frame.loc[:, [by_label[hour] for hour in HOUR_COLUMNS]]


def load_prices(config: dict, book: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    path = book or os.path.join(V2_ROOT, "data", config["price_file"])
    price = _hours(pd.read_excel(path, sheet_name=0 if book is None else "price", header=0, index_col=0))
    scale = float(config["price_scale"])
    buy = price.loc["Market_buy ($/Mwh)"].values.astype(float) * scale
    sell = price.loc["Market_sell ($/Mwh)"].values.astype(float) * scale
    return buy, sell


def load_instance(config: dict, n_prosumers: int | None = None, horizon: int | None = None) -> Instance:
    book = os.path.join(V2_ROOT, "data", config["prosumer_file"])
    pv = _hours(_read_sheet(book, "PV")).values.astype(float)
    bess = _read_sheet(book, "BESS")
    hvac = _read_sheet(book, "HVAC")
    t_out = _hours(_read_sheet(book, "T_out")).values.astype(float)
    load = _hours(_read_sheet(book, "Load")).values.astype(float)
    buy, sell = load_prices(config)
    count = n_prosumers or pv.shape[0]
    steps = horizon or int(config["horizon"])
    prosumers = tuple(
        Prosumer(
            pv[i, :steps],
            float(bess["max"].values[i]),
            float(bess["SOC_max"].values[i]),
            float(hvac["max"].values[i]),
            float(hvac["F_min"].values[i]),
            float(hvac["F_max"].values[i]),
            float(hvac["alpha"].values[i]),
            float(hvac["beta"].values[i]),
            t_out[i, :steps],
            load[i, :steps],
        )
        for i in range(count)
    )
    return Instance(prosumers, buy[:steps], sell[:steps], config)


@dataclass
class ProsumerBlock:
    net: cvx.Variable
    charge: cvx.Variable
    discharge: cvx.Variable
    constraints: list
    local_cost: cvx.Expression
    devices: _Devices


@dataclass(frozen=True)
class _Devices:
    pv: cvx.Variable
    charge: cvx.Variable
    discharge: cvx.Variable
    soc: cvx.Variable
    hvac: cvx.Variable
    indoor: cvx.Variable


def prosumer_block(par: Prosumer, config: dict) -> ProsumerBlock:
    steps = len(par.load)
    dev = _Devices(*(cvx.Variable(steps) for _ in range(6)))
    net = cvx.Variable(steps)
    cons = _bounds(par, dev)
    cons.append(net == dev.pv + dev.discharge - dev.charge - dev.hvac - par.load)
    cons += _initial_state(par, config, dev)
    cons += _dynamics(par, config, dev)
    local_cost = float(config["degradation_cost"]) * cvx.sum(dev.charge + dev.discharge)
    return ProsumerBlock(net, dev.charge, dev.discharge, cons, local_cost, dev)


def _bounds(par: Prosumer, dev: _Devices) -> list:
    return [
        dev.pv >= 0,
        dev.pv <= par.pv_max,
        dev.charge >= 0,
        dev.charge <= par.bess_pmax,
        dev.discharge >= 0,
        dev.discharge <= par.bess_pmax,
        dev.soc >= 0,
        dev.soc <= par.soc_max,
        dev.hvac >= 0,
        dev.hvac <= par.hvac_max,
        dev.indoor >= par.t_min,
        dev.indoor <= par.t_max,
    ]


def _initial_state(par: Prosumer, config: dict, dev: _Devices) -> list:
    soc_init = float(config["soc_init_fraction"]) * par.soc_max
    last = len(par.load) - 1
    return [
        dev.soc[0] == soc_init,
        dev.soc[last] == soc_init,
        dev.indoor[0] == float(config["indoor_init_c"]),
        dev.hvac[0] == 0,
        dev.charge[0] == 0,
        dev.discharge[0] == 0,
    ]


def _dynamics(par: Prosumer, config: dict, dev: _Devices) -> list:
    eta_c, eta_d = float(config["eta_charge"]), float(config["eta_discharge"])
    cons = []
    for k in range(1, len(par.load)):
        cons.append(dev.soc[k] == dev.soc[k - 1] + eta_c * dev.charge[k] - dev.discharge[k] / eta_d)
        drift = par.alpha * (par.t_out[k] - dev.indoor[k - 1])
        cons.append(dev.indoor[k] == dev.indoor[k - 1] + drift + par.beta * dev.hvac[k])
    return cons


def value_of(expr: cvx.Expression) -> float:
    if expr.value is None:
        raise RuntimeError("expression has no value; solve the problem first")
    return float(np.asarray(expr.value).ravel()[0]) if np.asarray(expr.value).size == 1 else float(expr.value)


def array_of(expr: cvx.Expression) -> np.ndarray:
    if expr.value is None:
        raise RuntimeError("expression has no value; solve the problem first")
    return np.asarray(expr.value, dtype=float).ravel()


def device_arrays(block: ProsumerBlock) -> dict[str, np.ndarray]:
    """Net power and device schedules of a solved block: kW, soc in kWh, indoor in degrees C."""
    dev = block.devices
    named = {"net": block.net, "pv": dev.pv, "charge": dev.charge, "discharge": dev.discharge}
    named.update({"soc": dev.soc, "hvac": dev.hvac, "indoor": dev.indoor})
    return {name: array_of(variable) for name, variable in named.items()}


def market_revenue(x: cvx.Expression, buy: np.ndarray, sell: np.ndarray, pi_peak: float) -> cvx.Expression:
    energy = cvx.sum(cvx.minimum(cvx.multiply(buy, x), cvx.multiply(sell, x)))
    return energy - pi_peak * cvx.max(cvx.pos(-x))


def market_revenue_value(x: np.ndarray, buy: np.ndarray, sell: np.ndarray, pi_peak: float) -> float:
    energy = float(np.minimum(buy * x, sell * x).sum())
    return energy - pi_peak * float(np.max(np.clip(-x, 0.0, None)))


def market_terms(x: np.ndarray, buy: np.ndarray, sell: np.ndarray, pi_peak: float) -> dict[str, float]:
    """Purchase cost, sales revenue and peak charge of a net position, $; revenue = sales - purchase - peak charge."""
    draw_kw = np.clip(-x, 0.0, None)
    return {
        "purchase": float((buy * draw_kw).sum()),
        "sales": float((sell * np.clip(x, 0.0, None)).sum()),
        "peak_charge": pi_peak * float(np.max(draw_kw)),
    }


@dataclass
class CentralResult:
    status: str
    objective: float
    x: np.ndarray
    mu: np.ndarray
    nets: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    local_costs: np.ndarray


def _dual_price(constraint: cvx.Constraint) -> np.ndarray:
    mu = np.asarray(constraint.dual_value, dtype=float).ravel()
    # cvxpy's dual sign for an equality depends on the problem sense; prices are positive here
    return -mu if mu.sum() < 0 else mu


def _solve(problem: cvx.Problem, solver: str) -> None:
    problem.solve(solver=getattr(cvx, solver))
    if problem.status != cvx.OPTIMAL:
        raise RuntimeError(f"solver status {problem.status}")


def solve_centralized(inst: Instance, solver: str) -> CentralResult:
    blocks = [prosumer_block(par, inst.config) for par in inst.prosumers]
    x = cvx.Variable(inst.horizon)
    coupling = x == sum(block.net for block in blocks)
    constraints = [c for block in blocks for c in block.constraints] + [coupling]
    revenue = market_revenue(x, inst.buy, inst.sell, float(inst.config["pi_peak"]))
    problem = cvx.Problem(cvx.Maximize(revenue - sum(block.local_cost for block in blocks)), constraints)
    _solve(problem, solver)
    return CentralResult(
        problem.status,
        float(problem.value),
        array_of(x),
        _dual_price(coupling),
        np.array([array_of(b.net) for b in blocks]),
        np.array([array_of(b.charge) for b in blocks]),
        np.array([array_of(b.discharge) for b in blocks]),
        np.array([value_of(b.local_cost) for b in blocks]),
    )


@dataclass
class BenchmarkResult:
    nets: np.ndarray
    local_costs: np.ndarray
    payoffs: np.ndarray
    blocks: list


def solve_uncoordinated(inst: Instance, solver: str) -> BenchmarkResult:
    nets, costs, payoffs, blocks = [], [], [], []
    for par in inst.prosumers:
        block = prosumer_block(par, inst.config)
        energy = cvx.sum(cvx.minimum(cvx.multiply(inst.buy, block.net), cvx.multiply(inst.sell, block.net)))
        problem = cvx.Problem(cvx.Maximize(energy - block.local_cost), block.constraints)
        _solve(problem, solver)
        nets.append(array_of(block.net))
        costs.append(value_of(block.local_cost))
        payoffs.append(float(problem.value))
        blocks.append(block)
    return BenchmarkResult(np.array(nets), np.array(costs), np.array(payoffs), blocks)
