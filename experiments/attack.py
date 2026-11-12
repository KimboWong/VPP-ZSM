"""Reconstruction attack on the aggregator's view and its hit rate."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from model.core import V2_ROOT

DRAWS = 1000
HIT_TOLERANCE = 0.1
ATTACK_SEED = 20261021
DP_WINDOWS = ("window_1", "window_10", "window_100", "window_all")
PRIMARY_MASK_SEED = 1


def score(estimate: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    scale = float(np.max(np.abs(truth)))
    return float(np.mean(np.abs(estimate - truth) <= HIT_TOLERANCE * scale)), float(np.mean((estimate - truth) ** 2))


def summarize(column: str, setting: str, n_prosumers: int, scores: list[tuple[float, float]]) -> dict:
    hits, errors = np.array([s[0] for s in scores]), np.array([s[1] for s in scores])
    return {
        "column": column,
        "setting": setting,
        "n_prosumers": n_prosumers,
        "draws": len(scores),
        "hit_rate_mean": float(hits.mean()),
        "hit_rate_std": float(hits.std()),
        "mse_mean": float(errors.mean()),
        "mse_std": float(errors.std()),
    }


def masked_estimate(views: dict, target: int, colluders: np.ndarray, with_bill: bool) -> np.ndarray:
    nets = views["nets_final"]
    estimate = (views["total_final"] - nets[colluders].sum(axis=0)) / (len(nets) - len(colluders))
    if with_bill:
        mu = views["mu_final"]
        estimate = estimate + (float(mu @ nets[target]) - float(mu @ estimate)) * mu / float(mu @ mu)
    return estimate


def masked_rows(views: dict, rng: np.random.Generator) -> list[dict]:
    size = len(views["nets_final"])
    rows = []
    for colluding in (0, size // 2, size - 2):
        for with_bill in (False, True):
            scores = []
            for _ in range(DRAWS):
                target = int(rng.integers(size))
                colluders = rng.choice(np.delete(np.arange(size), target), size=colluding, replace=False)
                scores.append(score(masked_estimate(views, target, colluders, with_bill), views["nets_final"][target]))
            setting = f"colluders={colluding}" + (" +bill" if with_bill else "")
            rows.append(summarize("masked", setting, size, scores))
    return rows


def plaintext_row(views: dict, rng: np.random.Generator) -> dict:
    nets = views["nets_final"]
    targets = rng.integers(len(nets), size=DRAWS)
    return summarize("plaintext", "messages in the clear", len(nets), [score(nets[t], nets[t]) for t in targets])


def dp_groups(runs: list[tuple[dict, dict]]) -> list[tuple[tuple[int, int, float], list[tuple[dict, dict]]]]:
    groups: dict[tuple[int, int, float], list[tuple[dict, dict]]] = {}
    for summary, views in runs:
        groups.setdefault((summary["n_prosumers"], summary["k_max"], summary["sigma_kw"]), []).append((summary, views))
    return sorted(groups.items())


def dp_rows(runs: list[tuple[dict, dict]]) -> list[dict]:
    rows = []
    for (size, k_max, sigma), group in dp_groups(runs):
        for window in DP_WINDOWS:
            scores = [score(v[window][i], v["nets_final"][i]) for _, v in group for i in range(size)]
            setting = f"sigma={sigma} {window} k_max={k_max} seeds={len(group)}"
            rows.append(summarize("dp", setting, size, scores))
    return rows


def shrink_weight(views: dict, sigma: float, releases: int) -> float:
    nets = views["nets_final"]
    noise_var = sigma**2 / releases
    spread_var = float(np.mean(np.var(views["window_all"] - nets.sum(axis=0) / len(nets), axis=0)))
    signal_var = max(spread_var - noise_var, 0.0)
    return signal_var / (signal_var + noise_var)


def bill_matched(estimate: np.ndarray, mu: np.ndarray, bill: float) -> np.ndarray:
    return estimate + (bill - float(mu @ estimate)) * mu / float(mu @ mu)


def settlement_estimates(views: dict, target: int, weight: float) -> tuple[np.ndarray, np.ndarray]:
    nets, mu = views["nets_final"], views["mu_final"]
    prior = nets.sum(axis=0) / len(nets)
    bill = float(mu @ nets[target])
    shrunk = prior + weight * (views["window_all"][target] - prior)
    return bill_matched(prior, mu, bill), bill_matched(shrunk, mu, bill)


def dp_settlement_rows(runs: list[tuple[dict, dict]]) -> list[dict]:
    rows = []
    for (size, k_max, sigma), group in dp_groups(runs):
        prior_scores, shrunk_scores = [], []
        for summary, views in group:
            weight = shrink_weight(views, sigma, summary["releases"])
            for i in range(size):
                prior, shrunk = settlement_estimates(views, i, weight)
                prior_scores.append(score(prior, views["nets_final"][i]))
                shrunk_scores.append(score(shrunk, views["nets_final"][i]))
        tail = f"k_max={k_max} seeds={len(group)}"
        rows.append(summarize("dp", f"sigma={sigma} aggregate+bill {tail}", size, prior_scores))
        rows.append(summarize("dp", f"sigma={sigma} window_all+aggregate+bill {tail}", size, shrunk_scores))
    return rows


def load_runs(experiment: str) -> list[tuple[dict, dict]]:
    runs = []
    for path in sorted(glob.glob(os.path.join(V2_ROOT, "results", experiment, "*", "views.npz"))):
        with open(os.path.join(os.path.dirname(path), "summary.json"), "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        with np.load(path) as data:
            runs.append((summary, {key: data[key] for key in data.files}))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    args = parser.parse_args()
    runs = load_runs(args.experiment)
    rows = []
    for summary, views in runs:
        if summary["privacy"] == "masked" and summary["master_seed"] == PRIMARY_MASK_SEED:
            # one generator per instance, so the draws do not depend on which other runs are in the directory
            rng = np.random.default_rng([ATTACK_SEED, summary["n_prosumers"]])
            rows.append(plaintext_row(views, rng))
            rows.extend(masked_rows(views, rng))
    dp_runs = [run for run in runs if run[0]["privacy"] == "dp"]
    rows.extend(dp_rows(dp_runs))
    rows.extend(dp_settlement_rows(dp_runs))
    out = os.path.join(V2_ROOT, "results", args.experiment, "attack.csv")
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    for row in rows:
        print(
            f"{row['column']:9s} N={row['n_prosumers']:3d} {row['setting']:38s} hit {row['hit_rate_mean']:.3f} "
            f"+- {row['hit_rate_std']:.3f}  mse {row['mse_mean']:.4f}"
        )
    print("rows:", len(rows), "->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
