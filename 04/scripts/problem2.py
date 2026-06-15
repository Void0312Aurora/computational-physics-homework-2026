from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solution import ensure_result_dir, problem2


def main() -> None:
    ensure_result_dir()
    rows = problem2()
    print(f"Problem 2 rows: {len(rows)}")


if __name__ == "__main__":
    main()
