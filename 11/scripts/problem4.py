from __future__ import annotations

from hw11_monte_carlo import RESULT, problem4


def main() -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    result = problem4()
    print(f"Problem 4 keys: {', '.join(sorted(result))}")


if __name__ == "__main__":
    main()
