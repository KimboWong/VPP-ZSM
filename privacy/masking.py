"""Pairwise additive masking of the household messages."""

from __future__ import annotations

import hmac

import numpy as np

KEY_BYTES = 16
DIGEST_BYTES = 32
WORD_BYTES = 8


def pair_key(master_seed: int, first: int, second: int) -> bytes:
    seed = master_seed.to_bytes(16, "big")
    return hmac.digest(seed, f"pair:{first}:{second}".encode(), "sha256")[:KEY_BYTES]


class PairwiseMasks:
    def __init__(self, size: int, horizon: int, master_seed: int) -> None:
        self.first, self.second = np.triu_indices(size, k=1)
        self.horizon = horizon
        self.keys = [pair_key(master_seed, int(i), int(j)) for i, j in zip(self.first, self.second)]
        self.added = [np.flatnonzero(self.first == i) for i in range(size)]
        self.subtracted = [np.flatnonzero(self.second == i) for i in range(size)]

    def stream(self, pair: int, round_index: int) -> np.ndarray:
        """The hourly mask of one pair in one round, uint64."""
        blocks = -(-self.horizon * WORD_BYTES // DIGEST_BYTES)
        prefix = round_index.to_bytes(8, "big")
        key = self.keys[pair]
        raw = b"".join(hmac.digest(key, prefix + b.to_bytes(4, "big"), "sha256") for b in range(blocks))
        return np.frombuffer(raw[: self.horizon * WORD_BYTES], dtype="<u8")

    def masks(self, round_index: int) -> np.ndarray:
        """One row of hourly masks per pair, uint64."""
        out = np.empty((len(self.keys), self.horizon), dtype=np.uint64)
        for pair in range(len(self.keys)):
            out[pair] = self.stream(pair, round_index)
        return out

    def message_of(self, index: int, encoded_row: np.ndarray, round_index: int) -> np.ndarray:
        """One prosumer's masked message from its own N - 1 pair masks: the per-round work of one deployed prosumer."""
        out = encoded_row.astype(np.uint64)
        for pair in self.added[index]:
            out += self.stream(int(pair), round_index)
        for pair in self.subtracted[index]:
            out -= self.stream(int(pair), round_index)
        return out

    def messages(self, encoded: np.ndarray, round_index: int) -> np.ndarray:
        """The masked messages y_i, one row per prosumer, uint64."""
        masks = self.masks(round_index)
        out = encoded.astype(np.uint64)
        np.add.at(out, self.first, masks)
        np.subtract.at(out, self.second, masks)
        return out
