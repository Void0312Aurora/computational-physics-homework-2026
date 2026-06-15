from __future__ import annotations

from hw12.io_utils import ensure_dirs
from hw12.problem3 import problem3


def main() -> None:
    ensure_dirs()
    log: list[str] = []
    result = problem3(log)
    print("\n".join(log))
    print(f"Problem 3 keys: {', '.join(sorted(result))}")


if __name__ == "__main__":
    main()
