"""Timing table: per-round cost of each part and the parallel estimate of a round."""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from model.core import V2_ROOT

REPEAT_TOLERANCE = 0.10
MODES = ("plaintext", "masked", "dp", "paillier")
PROBE = "probe"
CHECKED = {
    ("admm", "round"),
    ("admm", "prosumer_median"),
    ("primitives", "masked/prosumer"),
    ("primitives", "paillier/prosumer"),
    ("primitives", "paillier/aggregator"),
    ("primitives", "masked/modexp"),
}
OUT_DIR = os.path.join(V2_ROOT, "results", "exp05")


def load_runs() -> list[dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "*", "summary.json"))):
        with open(path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary["run_dir"] = os.path.basename(os.path.dirname(path))
        runs.append(summary)
    return runs


def run_rows(run: dict) -> list[dict]:
    medians = dict(run["round_medians"] if run["part"] == "admm" else run["medians"])
    medians[PROBE] = max(run["probe_start"], run["probe_end"])
    keys = {"part": run["part"], "n_prosumers": run["n_prosumers"], "repeat": run["repeat"], "run_dir": run["run_dir"]}
    return [{**keys, "quantity": quantity, "value": value} for quantity, value in medians.items()]


def repeat_table(runs: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame([row for run in runs for row in run_rows(run)])
    probe = (frame["quantity"] == PROBE).to_numpy()
    cell_fastest = frame.groupby(["part", "n_prosumers", "quantity"])["value"].transform("min")
    frame["fastest"] = np.where(probe, frame.loc[probe, "value"].min(), cell_fastest)
    frame["ratio"] = frame["value"] / frame["fastest"]
    in_checked = np.array([(part, quantity) in CHECKED for part, quantity in zip(frame["part"], frame["quantity"])])
    frame["checked"] = probe | in_checked
    frame["slow"] = frame["checked"] & (frame["ratio"] > 1.0 + REPEAT_TOLERANCE)
    frame["run_flagged"] = frame.groupby("run_dir")["slow"].transform("any")
    return frame


def chosen_runs(repeats: pd.DataFrame) -> tuple[set[str], set[tuple[str, int]]]:
    """The unflagged runs; a part and instance without one keeps its least slow run so that the table is complete."""
    checked = repeats[repeats["checked"]]
    per_run = checked.groupby(["part", "n_prosumers", "run_dir"]).agg(
        flagged=("run_flagged", "first"), worst=("ratio", "max")
    )
    chosen: set[str] = set()
    fallback: set[tuple[str, int]] = set()
    for (part, size), group in per_run.reset_index().groupby(["part", "n_prosumers"]):
        clean = group[~group["flagged"]]
        if len(clean):
            chosen.update(clean["run_dir"])
        else:
            chosen.add(str(group.loc[group["worst"].idxmin(), "run_dir"]))
            fallback.add((str(part), int(size)))
    return chosen, fallback


def median_of(valid: pd.DataFrame, part: str, size: int, quantity: str) -> float:
    picked = valid[(valid.part == part) & (valid.n_prosumers == size) & (valid.quantity == quantity)]["value"]
    return float(picked.median())


def run_fields(runs: list[dict], size: int, chosen: set[str], fallback: set[tuple[str, int]]) -> dict:
    admm = [r for r in runs if r["part"] == "admm" and r["n_prosumers"] == size and r["run_dir"] in chosen]
    prims = [r for r in runs if r["part"] == "primitives" and r["n_prosumers"] == size and r["run_dir"] in chosen]
    return {
        "rounds": int(np.median([r["k_final"] for r in admm])),
        "k_max": int(admm[0]["k_max"]),
        "bytes": prims[0]["bytes_per_message"],
        "broadcast": prims[0]["broadcast_bytes"],
        "clean": ("admm", size) not in fallback and ("primitives", size) not in fallback,
    }


def setup_seconds(valid: pd.DataFrame, size: int, mode: str) -> float:
    """One prosumer's key agreement: its own public key plus one shared secret per peer, and a key per peer."""
    if mode != "masked":
        return 0.0
    modexp = median_of(valid, "primitives", size, "masked/modexp")
    derive = median_of(valid, "primitives", size, "masked/derive")
    return size * modexp + (size - 1) * derive


def mode_row(valid: pd.DataFrame, size: int, mode: str, fields: dict) -> dict:
    solve_max = median_of(valid, "admm", size, "prosumer_max")
    privacy = 0.0 if mode == "plaintext" else median_of(valid, "primitives", size, f"{mode}/prosumer")
    aggregator = median_of(valid, "primitives", size, f"{mode}/aggregator")
    vpp = median_of(valid, "admm", size, "vpp")
    rounds = fields["k_max"] if mode == "dp" else fields["rounds"]
    round_parallel = solve_max + privacy + aggregator + vpp
    return {
        "n_prosumers": size,
        "mode": mode,
        "prosumer_solve_max_s": solve_max,
        "prosumer_solve_median_s": median_of(valid, "admm", size, "prosumer_median"),
        "prosumer_privacy_s": privacy,
        "aggregator_s": aggregator,
        "vpp_s": vpp,
        "round_parallel_s": round_parallel,
        "rounds": rounds,
        "run_parallel_s": round_parallel * rounds,
        "uplink_bytes_per_prosumer": fields["bytes"][mode],
        "broadcast_bytes": fields["broadcast"],
        "setup_per_prosumer_s": setup_seconds(valid, size, mode),
        "key_holder_setup_s": median_of(valid, "primitives", size, "paillier/keygen") if mode == "paillier" else 0.0,
        "simulated_round_plaintext_s": median_of(valid, "admm", size, "round"),
        "clean": fields["clean"],
    }


def main() -> int:
    runs = load_runs()
    repeats = repeat_table(runs)
    chosen, fallback = chosen_runs(repeats)
    valid = repeats[repeats["run_dir"].isin(chosen)]
    rows: list[dict] = []
    for size in sorted({r["n_prosumers"] for r in runs}):
        fields = run_fields(runs, size, chosen, fallback)
        rows.extend(mode_row(valid, size, mode, fields) for mode in MODES)
    table = pd.DataFrame(rows)
    repeats.to_csv(os.path.join(OUT_DIR, "timing_repeats.csv"), index=False, lineterminator="\n")
    table.to_csv(os.path.join(OUT_DIR, "timing_table.csv"), index=False, lineterminator="\n")
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(table.to_string(index=False))
    flagged = sorted(set(repeats.loc[repeats["run_flagged"], "run_dir"]))
    print("runs:", len(runs), "flagged:", flagged or "none", "kept from flagged runs:", sorted(fallback) or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
