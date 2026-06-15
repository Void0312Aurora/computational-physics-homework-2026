from __future__ import annotations

import numpy as np

from hw11_monte_carlo import RESULT, SEED, problem5


def main() -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    result = problem5(np.random.default_rng(SEED))
    print(f"Problem 5 keys: {', '.join(sorted(result))}")


if __name__ == "__main__":
    main()
