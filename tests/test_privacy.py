"""Invariants of the privacy layer on a small instance (3 households, 4 hours)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from algorithms.admm_sharing import AdmmSettings, run_admm_sharing
from model.core import Instance, load_config, load_instance, market_revenue_value, solve_centralized
from privacy import paillier
from privacy.aggregators import MaskedSum
from privacy.dp import zcdp, epsilon
from privacy.encoding import SCALE, decode, encode
from privacy.masking import PairwiseMasks

SOLVER = "CLARABEL"
TEST_KEY_BITS = 1024


@pytest.fixture(scope="module")
def tiny() -> Instance:
    return load_instance(load_config(), n_prosumers=3, horizon=4)


@pytest.fixture(scope="module")
def nets_kw() -> np.ndarray:
    return np.random.default_rng(7).uniform(-19.0, 19.0, size=(7, 24))


def test_encoding_round_trip_keeps_sign_and_resolution(nets_kw: np.ndarray) -> None:
    decoded = decode(encode(nets_kw).astype(np.uint64))
    assert np.max(np.abs(decoded - nets_kw)) <= 0.5 / SCALE
    with pytest.raises(ValueError):
        encode(np.array([[25.0]]))


def test_pairwise_masks_cancel_exactly(nets_kw: np.ndarray) -> None:
    encoded = encode(nets_kw)
    messages = PairwiseMasks(len(nets_kw), 24, master_seed=3).messages(encoded, round_index=5)
    assert np.array_equal(messages.sum(axis=0, dtype=np.uint64).view(np.int64), encoded.sum(axis=0))
    assert not np.any(messages == encoded.astype(np.uint64))


def test_masks_are_fresh_every_round() -> None:
    masks = PairwiseMasks(4, 24, master_seed=3)
    assert not np.any(masks.masks(1) == masks.masks(2))


def test_each_prosumer_computes_its_own_message(nets_kw: np.ndarray) -> None:
    encoded = encode(nets_kw)
    masks = PairwiseMasks(len(encoded), 24, master_seed=3)
    for round_index in (1, 2):
        rows = np.array([masks.message_of(i, encoded[i], round_index) for i in range(len(encoded))])
        assert np.array_equal(rows, masks.messages(encoded, round_index))


def test_packed_paillier_sum_is_exact(nets_kw: np.ndarray) -> None:
    key = paillier.keygen(TEST_KEY_BITS, seed=11)
    encoded = encode(nets_kw)
    assert paillier.fits(key, len(encoded), encoded.shape[1])
    rng = random.Random(12)
    ciphers = [paillier.encrypt(key, paillier.pack(row), rng) for row in encoded]
    total = paillier.unpack(paillier.decrypt(key, paillier.add(key, ciphers)), len(encoded), encoded.shape[1])
    assert np.array_equal(total, encoded.sum(axis=0))


def test_zcdp_accounting_composes_and_converts() -> None:
    assert zcdp(sensitivity=1.0, sigma=1.0, releases=1) == 0.5
    assert zcdp(sensitivity=2.0, sigma=1.0, releases=3) == 6.0
    assert epsilon(0.5, 1e-5) == pytest.approx(0.5 + 2.0 * math.sqrt(0.5 * math.log(1e5)))


def test_masked_admm_reaches_centralized_optimum(tiny: Instance) -> None:
    central = solve_centralized(tiny, SOLVER)
    column = MaskedSum(tiny.size, tiny.horizon, master_seed=1)
    settings = AdmmSettings(rho=0.4, k_max=5000, eps_rel=1e-4, solver=SOLVER)
    result = run_admm_sharing(tiny, settings, log_every=0, aggregator=column)
    pi_peak = float(tiny.config["pi_peak"])
    objective = market_revenue_value(result.x_tilde, tiny.buy, tiny.sell, pi_peak) - float(result.local_costs.sum())
    assert result.converged and column.mismatches == 0
    assert abs(objective - central.objective) <= 2e-4 * abs(central.objective)
    assert float(np.max(np.abs(result.mu - central.mu))) <= 1e-3
