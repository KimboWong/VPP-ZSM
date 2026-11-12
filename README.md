# Privacy-preserving coordination and settlement of residential virtual power plants via secure aggregation

Code for the paper of the same title. A virtual power plant (VPP) coordinates the batteries and air conditioners
of N households with sharing ADMM; each household masks its schedule with pairwise keys, so the VPP recovers
only the sum of the schedules.

## Requirements

Python 3.12 or later (the paper used 3.13.5).

    pip install -r requirements.txt

The experiments use Gurobi through cvxpy and need a Gurobi license (free for academic use).
The tests use Clarabel and need no license. `python check.py` runs the code checks and the 20 tests.

## Experiments

Run from the repository root. Each run writes a folder under `results/`. For the instance with 100 households:

    python -m experiments.exp00_centralized_check --config n100.json
    python -m experiments.exp03_algorithms --algo admm --config n100.json
    python -m experiments.exp04_privacy --mode masked --config n100.json
    python -m experiments.exp04_privacy --mode dp --sigma 10 --seed 1 --config n100.json
    python -m experiments.exp04_privacy --mode paillier --kmax 3
    python -m experiments.attack exp04
    python -m experiments.exp06_energy --config n100.json --reference results/exp04/<masked run folder>
    python -m experiments.exp05_timing --part admm --config n100.json --repeat 1
    python -m experiments.timing_table

Use `n200.json` or `n300.json` for the other instances, or no `--config` for 50 households.
The settings of the paper's runs are given in the paper. For timing, set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`
and `OPENBLAS_NUM_THREADS` to 1.

In this code all parties run in one process: the pairwise keys are derived from a recorded seed so that runs
can be repeated, and the time per round is a parallel estimate built from measured parts.

## Data

The household and price data come from https://github.com/hkuedl/Distributed-differentially-private-virtual-power-plant
. Their license is in
`data/LICENSE_Huang2024`.

## License

The code is released under the MIT License (`LICENSE`).
