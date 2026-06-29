from __future__ import annotations

from project2_pi.homework_bridge import solve_project2


def solve_problem4_project2() -> dict[str, object]:
    return solve_project2()


def main() -> None:
    result = solve_problem4_project2()
    print(f"Project 2 keys: {', '.join(sorted(result))}")


if __name__ == "__main__":
    main()
