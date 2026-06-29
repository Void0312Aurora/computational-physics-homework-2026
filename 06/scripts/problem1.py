from __future__ import annotations

from hw06_analysis import ensure_result_dir, build_problem1_formula_report


def main() -> None:
    ensure_result_dir()
    result = build_problem1_formula_report()
    print(f"Problem 1 formulas written to {result['report']}")


if __name__ == "__main__":
    main()
