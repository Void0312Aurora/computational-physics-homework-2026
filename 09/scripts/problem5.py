from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from hw09_analysis import SEED, problem5


def main() -> None:
    result_dir = Path(__file__).resolve().parents[1] / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    log = io.StringIO()
    result = problem5(np.random.default_rng(SEED), result_dir, log)
    print(log.getvalue(), end="")
    print(f"Problem 5 keys: {', '.join(sorted(result))}")


if __name__ == "__main__":
    main()
