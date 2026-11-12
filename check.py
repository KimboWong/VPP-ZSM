"""Quality gate: formatting, lint, types, tests. Run from the repository root: python check.py"""

from __future__ import annotations

import subprocess
import sys

PACKAGES = ["model", "algorithms", "privacy", "experiments", "tests"]
GATES = [
    ["black", "--check", "--line-length", "120", *PACKAGES, "check.py"],
    ["flake8", "--max-line-length", "120", *PACKAGES, "check.py"],
    ["mypy", "--ignore-missing-imports", "--explicit-package-bases", *PACKAGES],
    ["pytest", "-q", "tests"],
]


def main() -> int:
    failed = []
    for gate in GATES:
        print("==", " ".join(gate), flush=True)
        completed = subprocess.run([sys.executable, "-m", *gate])
        if completed.returncode != 0:
            failed.append(gate[0])
    print("FAILED:", failed if failed else "none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
