from __future__ import annotations

import io
from pathlib import Path

from hw09_analysis import problem3


def main() -> None:
    result_dir = Path(__file__).resolve().parents[1] / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    log = io.StringIO()
    result = problem3(result_dir, log)
    print(log.getvalue(), end="")
    print(f"Problem 3 keys: {', '.join(sorted(result))}")


if __name__ == "__main__":
    main()
