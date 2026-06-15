from __future__ import annotations

import io
from pathlib import Path

from hw08_analysis import problem2


def main() -> None:
    result_dir = Path(__file__).resolve().parents[1] / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    log = io.StringIO()
    result = problem2(result_dir, log)
    print(log.getvalue(), end="")
    print(f"Problem 2 tau: {result['tau']:.10f}")


if __name__ == "__main__":
    main()
