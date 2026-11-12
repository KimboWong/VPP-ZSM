"""Gaussian noise on the household messages and the privacy loss in zero-concentrated DP."""

from __future__ import annotations

import math

import numpy as np

from model.core import Instance

DELTA = 1e-5


def envelope(inst: Instance) -> tuple[np.ndarray, np.ndarray]:
    """Bounds on any prosumer's hourly net power pv + discharge - charge - hvac - load, from the device boxes."""
    high = np.max([par.pv_max + par.bess_pmax - par.load for par in inst.prosumers], axis=0)
    low = np.min([-par.bess_pmax - par.hvac_max - par.load for par in inst.prosumers], axis=0)
    return low, high


def sensitivities(inst: Instance) -> tuple[float, float]:
    """(day-level, hour-level) L2 sensitivity of one message, kW."""
    low, high = envelope(inst)
    return float(np.linalg.norm(high - low)), float(np.max(high - low))


def zcdp(sensitivity: float, sigma: float, releases: int) -> float:
    return releases * sensitivity**2 / (2.0 * sigma**2)


def epsilon(rho: float, delta: float = DELTA) -> float:
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
