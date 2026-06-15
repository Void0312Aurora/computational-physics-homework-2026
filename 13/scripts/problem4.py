from __future__ import annotations

import numpy as np

from hw13_ising import SEED, ensure_result_dir, run_problem3_and_4


def main() -> None:
    ensure_result_dir()
    result = run_problem3_and_4(np.random.default_rng(SEED))
    print("Problem 4 entry uses the shared 2D Ising workflow that also produces Problem 3 artifacts.")
    print(f"Generated keys: {', '.join(sorted(result))}")


if __name__ == "__main__":
    main()
