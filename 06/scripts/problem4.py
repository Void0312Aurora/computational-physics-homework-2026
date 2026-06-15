from __future__ import annotations

from hw06_analysis import ensure_result_dir, run_problem4


def main() -> None:
    ensure_result_dir()
    result = run_problem4()
    print(f"Problem 4 empirical best h: {result['part_c_empirical_best_h']}")


if __name__ == "__main__":
    main()
