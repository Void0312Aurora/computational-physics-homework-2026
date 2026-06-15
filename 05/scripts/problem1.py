from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solution import ensure_result_dir, solve_problem1


def main() -> None:
    ensure_result_dir()
    result = solve_problem1()
    print(f"Problem 1 curve rows: {len(result['curve_summary'])}")


if __name__ == "__main__":
    main()
