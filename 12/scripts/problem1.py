from __future__ import annotations

import numpy as np

from hw12.config import PROBLEM_SEEDS
from hw12.io_utils import ensure_dirs
from hw12.problem1 import problem1


def main() -> None:
    ensure_dirs()
    log: list[str] = []
    result = problem1(np.random.default_rng(PROBLEM_SEEDS["problem1"]), log)
    print("\n".join(log))
    print(f"Problem 1 keys: {', '.join(sorted(result))}")


if __name__ == "__main__":
    main()
