from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.solution import ensure_result_dir, problem1


def main() -> None:
    if "--import-smoke" in sys.argv:
        print("Problem 1 import smoke: ok")
        return
    ensure_result_dir()
    rows, resources = problem1()
    print(f"Problem 1 rows: {len(rows)}")
    print(f"elapsed_seconds: {resources['elapsed_seconds']:.2f}")


if __name__ == "__main__":
    main()
