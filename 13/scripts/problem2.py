from __future__ import annotations

import numpy as np

from hw13_ising import SEED, ensure_result_dir, run_problem2


def main() -> None:
    ensure_result_dir()
    result = run_problem2(np.random.default_rng(SEED))
    print(f"Problem 2 keys: {', '.join(sorted(result))}")


if __name__ == "__main__":
    main()
