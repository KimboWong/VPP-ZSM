"""Run-directory bookkeeping shared by the experiment scripts."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time

import cvxpy
import numpy

from model.core import V2_ROOT, config_hash


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=V2_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def make_run_dir(experiment: str, config: dict, tag: str = "") -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{config_hash(config)}" + (f"_{tag}" if tag else "")
    run_dir = os.path.join(V2_ROOT, "results", experiment, name)
    os.makedirs(run_dir, exist_ok=False)
    write_json(run_dir, "config.json", config)
    meta = {
        "git_commit": git_commit(),
        "config_hash": config_hash(config),
        "python": platform.python_version(),
        "cvxpy": cvxpy.__version__,
        "numpy": numpy.__version__,
        "machine": platform.node(),
    }
    write_json(run_dir, "meta.json", meta)
    return run_dir


def write_json(run_dir: str, name: str, payload: dict) -> None:
    with open(os.path.join(run_dir, name), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
