from __future__ import annotations

import numpy as np

from hw11_monte_carlo import RESULT, SEED, problem1


def main() -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    result = problem1(np.random.default_rng(SEED))
    print(f"Problem 1 keys: {', '.join(sorted(result))}")


if __name__ == "__main__":
    main()
