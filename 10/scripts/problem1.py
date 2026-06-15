from __future__ import annotations

import io

from hw10_random_tests import RESULT_DIR, problem1


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log = io.StringIO()
    result = problem1(log)
    print(log.getvalue(), end="")
    print(f"Problem 1 keys: {', '.join(sorted(result))}")


if __name__ == "__main__":
    main()
