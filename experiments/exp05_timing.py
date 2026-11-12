"""Timing of one coordination round, by component."""

from __future__ import annotations

import argparse
import hmac
import os
import platform
import random
import sys
import time
from typing import Callable

import numpy as np
import pandas as pd

from algorithms.admm_sharing import AdmmSettings, run_admm_sharing
from algorithms.base import PlainSum, RunResult, Workspace, build_workspace
from experiments.common import make_run_dir, write_json
from experiments.summary import result_fields
from model.core import Instance, load_config, load_instance, solve_centralized, solve_uncoordinated
from model.subproblems import ProsumerSubproblem, VppSubproblem
from privacy import paillier
from privacy.encoding import decode, encode
from privacy.masking import WORD_BYTES, PairwiseMasks

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
SOLVER_OPTIONS = {"GUROBI": {"Threads": 1}}
PRIMITIVE_ROUNDS = 20
AGGREGATOR_REPEATS = 10
PAILLIER_BITS = 2048
KEY_SEED = 1
DH_MODULUS_BITS = 2048
DH_EXPONENT_BITS = 256
DH_SAMPLES = 50
DERIVE_SAMPLES = 1000
DP_SIGMA_KW = 1.0
FLOAT_BYTES = 8
PROBE_SAMPLES = 15
PROBE_SEED = 20261022


def timed(record: list[float], solve: Callable[..., np.ndarray], *args: object) -> np.ndarray:
    tic = time.perf_counter()
    out = solve(*args)
    record.append(time.perf_counter() - tic)
    return out


class TimedProsumer(ProsumerSubproblem):
    def __init__(self, inner: ProsumerSubproblem, record: list[float]) -> None:
        self.inner, self.record = inner, record

    def solve(self, mu_hat: np.ndarray, p_prev: np.ndarray, weight: float, solver: str) -> np.ndarray:
        return timed(self.record, self.inner.solve, mu_hat, p_prev, weight, solver)

    @property
    def local_cost(self) -> float:
        return self.inner.local_cost


class TimedVpp(VppSubproblem):
    def __init__(self, inner: VppSubproblem, record: list[float]) -> None:
        self.inner, self.record = inner, record

    def solve(self, mu_hat: np.ndarray, x_prev: np.ndarray, weight: float, solver: str) -> np.ndarray:
        return timed(self.record, self.inner.solve, mu_hat, x_prev, weight, solver)


class TimedSum(PlainSum):
    def __init__(self, record: list[float]) -> None:
        self.record = record

    def total(self, nets: np.ndarray, round_index: int) -> np.ndarray:
        return timed(self.record, super().total, nets, round_index)


def load_probe() -> float:
    """Median time of one fixed 2048-bit modular exponentiation: the same work in every run, so a slow probe marks a
    loaded machine."""
    rng = random.Random(PROBE_SEED)
    modulus = rng.getrandbits(DH_MODULUS_BITS) | (1 << (DH_MODULUS_BITS - 1)) | 1
    base = rng.getrandbits(DH_MODULUS_BITS - 1)
    exponent = rng.getrandbits(DH_EXPONENT_BITS) | (1 << (DH_EXPONENT_BITS - 1))
    times = []
    for _ in range(PROBE_SAMPLES):
        tic = time.perf_counter()
        pow(base, exponent, modulus)
        times.append(time.perf_counter() - tic)
    return float(np.median(times))


def thread_settings() -> dict[str, str]:
    values = {name: os.environ.get(name, "") for name in THREAD_VARIABLES}
    if any(value != "1" for value in values.values()):
        raise SystemExit(f"set {', '.join(THREAD_VARIABLES)} to 1 before timing; now {values}")
    return values


def timed_workspace(inst: Instance, options: dict, records: dict[str, list[float]]) -> Workspace:
    base = build_workspace(inst, solver_options=options)
    prosumers = [TimedProsumer(sub, records["prosumer"]) for sub in base.subproblems]
    return Workspace(inst, prosumers, TimedVpp(base.vpp, records["vpp"]), base.pi_peak)


def round_table(records: dict[str, list[float]], result: RunResult, size: int) -> pd.DataFrame:
    solves = np.array(records["prosumer"]).reshape(result.k_final, size)
    return pd.DataFrame(
        {
            "k": np.arange(1, result.k_final + 1),
            "prosumer_max": solves.max(axis=1),
            "prosumer_p90": np.quantile(solves, 0.9, axis=1),
            "prosumer_median": np.median(solves, axis=1),
            "prosumer_sum": solves.sum(axis=1),
            "aggregate": records["aggregate"][1:],
            "vpp": records["vpp"],
            "round": result.trace.round_seconds,
        }
    )


def admm_part(inst: Instance, config: dict, summary: dict) -> pd.DataFrame:
    settings = AdmmSettings(
        rho=float(config["admm_rho"]),
        k_max=int(config["k_max"]),
        eps_rel=float(config["eps_rel"]),
        solver=summary["solver"],
        init_price=config["init_price"],
    )
    records: dict[str, list[float]] = {"prosumer": [], "vpp": [], "aggregate": []}
    central = solve_centralized(inst, settings.solver)
    workspace = timed_workspace(inst, summary["solver_options"], records)
    tic = time.perf_counter()
    result = run_admm_sharing(inst, settings, 0, TimedSum(records["aggregate"]), workspace)
    summary.update(result_fields(inst, central, result, time.perf_counter() - tic))
    summary["k_max"] = settings.k_max
    rounds = round_table(records, result, inst.size)
    summary["round_medians"] = {column: float(rounds[column].median()) for column in rounds.columns if column != "k"}
    return rounds


def sample(mode: str, party: str, index: int, seconds: float) -> dict:
    return {"mode": mode, "party": party, "index": index, "seconds": seconds}


def plaintext_samples(nets: np.ndarray) -> list[dict]:
    rows = []
    for k in range(1, AGGREGATOR_REPEATS + 1):
        tic = time.perf_counter()
        nets.sum(axis=0)
        rows.append(sample("plaintext", "aggregator", k, time.perf_counter() - tic))
    return rows


def masking_samples(nets: np.ndarray) -> list[dict]:
    masks = PairwiseMasks(len(nets), nets.shape[1], master_seed=1)
    encoded = encode(nets)
    rows = []
    for k in range(1, PRIMITIVE_ROUNDS + 1):
        for i in range(len(nets)):
            tic = time.perf_counter()
            masks.message_of(i, encode(nets[i]), k)
            rows.append(sample("masked", "prosumer", k, time.perf_counter() - tic))
        messages = masks.messages(encoded, k)
        tic = time.perf_counter()
        decode(messages.sum(axis=0, dtype=np.uint64))
        rows.append(sample("masked", "aggregator", k, time.perf_counter() - tic))
    return rows


def dp_samples(nets: np.ndarray, rng: np.random.Generator) -> list[dict]:
    rows = []
    for k in range(1, PRIMITIVE_ROUNDS + 1):
        noisy = np.empty_like(nets)
        for i in range(len(nets)):
            tic = time.perf_counter()
            noisy[i] = nets[i] + rng.normal(0.0, DP_SIGMA_KW, size=nets.shape[1])
            rows.append(sample("dp", "prosumer", k, time.perf_counter() - tic))
        tic = time.perf_counter()
        noisy.sum(axis=0)
        rows.append(sample("dp", "aggregator", k, time.perf_counter() - tic))
    return rows


def paillier_samples(nets: np.ndarray, summary: dict) -> list[dict]:
    tic = time.perf_counter()
    key = paillier.keygen(PAILLIER_BITS, KEY_SEED)
    rows = [sample("paillier", "keygen", 1, time.perf_counter() - tic)]
    encoded, rng, ciphers = encode(nets), random.Random(KEY_SEED + 1), []
    for i, row in enumerate(encoded):
        tic = time.perf_counter()
        ciphers.append(paillier.encrypt(key, paillier.pack(row), rng))
        rows.append(sample("paillier", "prosumer", i + 1, time.perf_counter() - tic))
    for k in range(1, AGGREGATOR_REPEATS + 1):
        tic = time.perf_counter()
        total = paillier.unpack(paillier.decrypt(key, paillier.add(key, ciphers)), len(encoded), encoded.shape[1])
        rows.append(sample("paillier", "aggregator", k, time.perf_counter() - tic))
    summary["paillier_exact"] = bool(np.array_equal(total, encoded.sum(axis=0)))
    summary["bytes_per_message"]["paillier"] = (key.n_square.bit_length() + 7) // 8
    return rows


def key_agreement_samples(summary: dict) -> list[dict]:
    rng = random.Random(KEY_SEED)
    modulus = rng.getrandbits(DH_MODULUS_BITS) | (1 << (DH_MODULUS_BITS - 1)) | 1
    rows = []
    for s in range(1, DH_SAMPLES + 1):
        base = rng.getrandbits(DH_MODULUS_BITS - 1)
        exponent = rng.getrandbits(DH_EXPONENT_BITS) | (1 << (DH_EXPONENT_BITS - 1))
        tic = time.perf_counter()
        pow(base, exponent, modulus)
        rows.append(sample("masked", "modexp", s, time.perf_counter() - tic))
    shared = rng.getrandbits(DH_MODULUS_BITS).to_bytes(DH_MODULUS_BITS // 8, "big")
    for s in range(1, DERIVE_SAMPLES + 1):
        tic = time.perf_counter()
        hmac.digest(shared, b"pair key", "sha256")
        rows.append(sample("masked", "derive", s, time.perf_counter() - tic))
    summary["key_agreement_public_bytes"] = (modulus.bit_length() + 7) // 8
    return rows


def primitives_part(inst: Instance, summary: dict) -> pd.DataFrame:
    nets = solve_uncoordinated(inst, summary["solver"]).nets
    summary["bytes_per_message"] = {"plaintext": FLOAT_BYTES * inst.horizon, "dp": FLOAT_BYTES * inst.horizon}
    summary["bytes_per_message"]["masked"] = WORD_BYTES * inst.horizon
    summary["broadcast_bytes"] = FLOAT_BYTES * inst.horizon
    rows = plaintext_samples(nets) + masking_samples(nets) + dp_samples(nets, np.random.default_rng(summary["repeat"]))
    rows += paillier_samples(nets, summary) + key_agreement_samples(summary)
    samples = pd.DataFrame(rows)
    grouped = samples.groupby(["mode", "party"])["seconds"]
    summary["medians"] = {f"{mode}/{party}": float(value) for (mode, party), value in grouped.median().items()}
    summary["sample_counts"] = {f"{mode}/{party}": int(count) for (mode, party), count in grouped.size().items()}
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["admm", "primitives"], required=True)
    parser.add_argument("--config", default="default.json")
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--solver", default=None)
    return parser.parse_args()


def run_summary(args: argparse.Namespace, inst: Instance, solver: str) -> dict:
    return {
        "part": args.part,
        "n_prosumers": inst.size,
        "repeat": args.repeat,
        "solver": solver,
        "solver_options": SOLVER_OPTIONS.get(solver, {}),
        "threads": thread_settings(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    inst = load_instance(config)
    summary = run_summary(args, inst, args.solver or config["solver"])
    summary["probe_start"] = load_probe()
    if args.part == "admm":
        table, name = admm_part(inst, config, summary), "rounds.csv"
    else:
        table, name = primitives_part(inst, summary), "samples.csv"
    summary["probe_end"] = load_probe()
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_dir = make_run_dir("exp05", config, f"{args.part}_n{inst.size}_r{args.repeat}")
    table.to_csv(os.path.join(run_dir, name), index=False, lineterminator="\n")
    write_json(run_dir, "summary.json", summary)
    print(f"{args.part} N={inst.size} repeat {args.repeat} -> {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
