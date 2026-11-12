"""Parametric subproblems of the households and of the VPP, re-solved with new parameters."""

from __future__ import annotations

import cvxpy as cvx
import numpy as np

from model.core import Instance, Prosumer, array_of, market_revenue, prosumer_block, value_of


def _solve_parametric(problem: cvx.Problem, solver: str, who: str, options: dict) -> None:
    problem.solve(solver=getattr(cvx, solver), warm_start=True, **options)
    if problem.status != cvx.OPTIMAL:
        raise RuntimeError(f"{who} subproblem status {problem.status}")


class ProsumerSubproblem:
    """min f_i(x) + lin^T p + w ||p||^2 on the prosumer's feasible set, with lin and w as parameters."""

    def __init__(self, par: Prosumer, config: dict, solver_options: dict | None = None) -> None:
        self.block = prosumer_block(par, config)
        self.lin = cvx.Parameter(len(par.load))
        self.weight = cvx.Parameter(nonneg=True)
        objective = self.block.local_cost + self.lin @ self.block.net + self.weight * cvx.sum_squares(self.block.net)
        self.problem = cvx.Problem(cvx.Minimize(objective), self.block.constraints)
        self.solver_options = dict(solver_options or {})

    def solve(self, mu_hat: np.ndarray, p_prev: np.ndarray, weight: float, solver: str) -> np.ndarray:
        self.lin.value = -(mu_hat + 2.0 * weight * p_prev)
        self.weight.value = np.asarray(weight, dtype=float)
        _solve_parametric(self.problem, solver, "prosumer", self.solver_options)
        return array_of(self.block.net)

    @property
    def local_cost(self) -> float:
        return value_of(self.block.local_cost)


class VppSubproblem:
    """min -g(X) + lin^T X + w ||X||^2, the proximal step for the VPP."""

    def __init__(self, inst: Instance, x_bound: float | None = None, solver_options: dict | None = None) -> None:
        self.x = cvx.Variable(inst.horizon)
        self.lin = cvx.Parameter(inst.horizon)
        self.weight = cvx.Parameter(nonneg=True)
        revenue = market_revenue(self.x, inst.buy, inst.sell, float(inst.config["pi_peak"]))
        objective = cvx.Minimize(-revenue + self.lin @ self.x + self.weight * cvx.sum_squares(self.x))
        constraints = [] if x_bound is None else [cvx.abs(self.x) <= x_bound]
        self.problem = cvx.Problem(objective, constraints)
        self.solver_options = dict(solver_options or {})

    def solve(self, mu_hat: np.ndarray, x_prev: np.ndarray, weight: float, solver: str) -> np.ndarray:
        self.lin.value = mu_hat - 2.0 * weight * x_prev
        self.weight.value = np.asarray(weight, dtype=float)
        _solve_parametric(self.problem, solver, "vpp", self.solver_options)
        return array_of(self.x)
