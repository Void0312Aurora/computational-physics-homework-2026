from __future__ import annotations

import io
from pathlib import Path

from hw08_analysis import problem1


def main() -> None:
    result_dir = Path(__file__).resolve().parents[1] / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    log = io.StringIO()
    rows = problem1(result_dir, log)
    print(log.getvalue(), end="")
    print(f"Problem 1 fit rows: {len(rows)}")


if __name__ == "__main__":
    main()
