"""The aggregator's sum of the household messages under masking, differential privacy and Paillier encryption."""

from __future__ import annotations

import random
import time

import numpy as np

from model.core import Instance
from privacy import paillier
from privacy.dp import DELTA, epsilon, sensitivities, zcdp
from privacy.encoding import SCALE, decode, encode
from privacy.masking import WORD_BYTES, PairwiseMasks


class MaskedSum:
    def __init__(self, size: int, horizon: int, master_seed: int) -> None:
        self.masks = PairwiseMasks(size, horizon, master_seed)
        self.master_seed, self.horizon = master_seed, horizon
        self.rounds, self.mismatches, self.max_encoding_error_kw = 0, 0, 0.0
        self.last_total = np.zeros(horizon)

    def total(self, nets: np.ndarray, round_index: int) -> np.ndarray:
        encoded = encode(nets)
        summed = self.masks.messages(encoded, round_index).sum(axis=0, dtype=np.uint64)
        self.rounds += 1
        self.mismatches += int(not np.array_equal(summed.view(np.int64), encoded.sum(axis=0)))
        self.last_total = decode(summed)
        error_kw = float(np.max(np.abs(self.last_total - nets.sum(axis=0))))
        self.max_encoding_error_kw = max(self.max_encoding_error_kw, error_kw)
        return self.last_total

    def report(self) -> dict:
        return {
            "master_seed": self.master_seed,
            "scale": SCALE,
            "rounds_checked": self.rounds,
            "mismatch_rounds": self.mismatches,
            "max_encoding_error_kw": self.max_encoding_error_kw,
            "bytes_per_message": WORD_BYTES * self.horizon,
        }


class NoisySum:
    def __init__(self, inst: Instance, sigma_kw: float, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.sigma_kw, self.seed, self.horizon = sigma_kw, seed, inst.horizon
        self.day_kw, self.hour_kw = sensitivities(inst)
        self.messages: list[np.ndarray] = []
        self.last_total = np.zeros(inst.horizon)

    def total(self, nets: np.ndarray, round_index: int) -> np.ndarray:
        noisy = nets + self.rng.normal(0.0, self.sigma_kw, size=nets.shape)
        self.messages.append(noisy)
        self.last_total = noisy.sum(axis=0)
        return self.last_total

    def window_means(self, windows: tuple[int, ...]) -> dict[str, np.ndarray]:
        """Each prosumer's mean noisy message over the last w releases, and over all of them."""
        stacked = np.array(self.messages)
        means = {f"window_{w}": stacked[-w:].mean(axis=0) for w in windows}
        means["window_all"] = stacked.mean(axis=0)
        return means

    def report(self) -> dict:
        releases = len(self.messages)
        rho_day = zcdp(self.day_kw, self.sigma_kw, releases)
        rho_hour = zcdp(self.hour_kw, self.sigma_kw, releases)
        return {
            "sigma_kw": self.sigma_kw,
            "noise_seed": self.seed,
            "releases": releases,
            "delta": DELTA,
            "sensitivity_day_kw": self.day_kw,
            "sensitivity_hour_kw": self.hour_kw,
            "zcdp_rho_day": rho_day,
            "epsilon_day": epsilon(rho_day),
            "zcdp_rho_hour": rho_hour,
            "epsilon_hour": epsilon(rho_hour),
            "bytes_per_message": WORD_BYTES * self.horizon,
        }


class EncryptedSum:
    def __init__(self, size: int, horizon: int, bits: int, seed: int) -> None:
        self.key = paillier.keygen(bits, seed)
        if not paillier.fits(self.key, size, horizon):
            raise ValueError(f"{size} prosumers x {horizon} hours do not fit the packing of a {bits}-bit key")
        self.rng = random.Random(seed + 1)
        self.bits, self.horizon = bits, horizon
        self.rounds, self.mismatches = 0, 0
        self.encrypt_seconds, self.aggregate_decrypt_seconds = 0.0, 0.0
        self.last_total = np.zeros(horizon)

    def total(self, nets: np.ndarray, round_index: int) -> np.ndarray:
        encoded = encode(nets)
        tic = time.perf_counter()
        ciphers = [paillier.encrypt(self.key, paillier.pack(row), self.rng) for row in encoded]
        self.encrypt_seconds += time.perf_counter() - tic
        tic = time.perf_counter()
        packed = paillier.decrypt(self.key, paillier.add(self.key, ciphers))
        summed = paillier.unpack(packed, len(encoded), self.horizon)
        self.aggregate_decrypt_seconds += time.perf_counter() - tic
        self.rounds += 1
        self.mismatches += int(not np.array_equal(summed, encoded.sum(axis=0)))
        self.last_total = summed / SCALE
        return self.last_total

    def report(self) -> dict:
        return {
            "key_bits": self.bits,
            "rounds_checked": self.rounds,
            "mismatch_rounds": self.mismatches,
            "bytes_per_message": (self.key.n_square.bit_length() + 7) // 8,
            "encrypt_seconds_total": self.encrypt_seconds,
            "aggregate_decrypt_seconds_total": self.aggregate_decrypt_seconds,
        }
