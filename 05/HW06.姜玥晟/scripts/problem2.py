from __future__ import annotations

import numpy as np

from interpolation_common import (
    InterpolationSummary,
    interpolation_case,
    plot_error_vs_node_count,
    plot_interpolation_panels,
    write_interpolation_summary,
    write_log_error_model,
)
from result_paths import PROBLEM2_RESULT_DIR, ensure_result_dir


def solve_problem2() -> list[InterpolationSummary]:
    ensure_result_dir(PROBLEM2_RESULT_DIR)

    def runge_function(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + 16.0 * x**2)

    x_dense = np.linspace(-1.0, 1.0, 2001)
    node_counts = [5, 7, 9, 17, 19, 21]
    summaries = [interpolation_case(runge_function, node_count, x_dense)[4] for node_count in node_counts]

    write_interpolation_summary(PROBLEM2_RESULT_DIR / "problem2_runge_error_summary.csv", summaries)
    write_log_error_model(PROBLEM2_RESULT_DIR / "problem2_runge_error_model.csv", summaries, min_node_count=7)
    plot_interpolation_panels(
        runge_function,
        [5],
        PROBLEM2_RESULT_DIR / "problem2_runge_5_nodes.png",
        "Problem 2(1): Runge function",
        y_limits=(-0.45, 1.1),
    )
    plot_interpolation_panels(
        runge_function,
        [7, 9, 17, 19, 21],
        PROBLEM2_RESULT_DIR / "problem2_runge_more_nodes.png",
        "Problem 2(2): Runge function",
        y_limits=(-0.45, 1.1),
        ncols=2,
    )
    plot_error_vs_node_count(
        summaries,
        PROBLEM2_RESULT_DIR / "problem2_runge_error_vs_b.png",
        "Problem 2: error growth as b increases",
    )
    return summaries


def main() -> None:
    rows = solve_problem2()
    print(f"Problem 2 rows: {len(rows)}")


if __name__ == "__main__":
    main()
