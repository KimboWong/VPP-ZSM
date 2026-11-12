"""Fixed-point encoding of the schedules as signed 64-bit integers."""

from __future__ import annotations

import numpy as np

SCALE = 10**6
LIMIT_KW = 20.0


def encode(nets_kw: np.ndarray) -> np.ndarray:
    peak_kw = float(np.max(np.abs(nets_kw)))
    if peak_kw > LIMIT_KW:
        raise ValueError(f"net power {peak_kw:.3f} kW exceeds the encoding limit {LIMIT_KW} kW")
    return np.rint(nets_kw * SCALE).astype(np.int64)


def decode(total: np.ndarray) -> np.ndarray:
    """Signed kW from a modulo-2^64 total held as uint64."""
    return total.view(np.int64) / SCALE
