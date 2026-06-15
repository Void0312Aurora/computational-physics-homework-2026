from __future__ import annotations

import io

from hw10_random_tests import RESULT_DIR, problem3


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log = io.StringIO()
    result = problem3(log)
    print(log.getvalue(), end="")
    print(f"Problem 3 keys: {', '.join(sorted(result))}")


if __name__ == "__main__":
    main()
