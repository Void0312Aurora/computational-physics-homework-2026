from __future__ import annotations

from project2_pi.homework_bridge import solve_project2
from result_paths import ensure_result_dir


def main() -> None:
    ensure_result_dir()
    result = solve_project2()
    print(f"Project 2 keys: {', '.join(sorted(result))}")


if __name__ == "__main__":
    main()
