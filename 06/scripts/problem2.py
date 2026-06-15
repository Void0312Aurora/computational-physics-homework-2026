from __future__ import annotations

from hw06_analysis import ensure_result_dir, run_problem2


def main() -> None:
    ensure_result_dir()
    result = run_problem2()
    print(f"Problem 2 best table formula: {result['table_best_formula']}")


if __name__ == "__main__":
    main()
