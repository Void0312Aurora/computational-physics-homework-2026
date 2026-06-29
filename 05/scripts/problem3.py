from __future__ import annotations

import math

import numpy as np

from interpolation_common import (
    InterpolationSummary,
    interpolation_case,
    plot_error_vs_node_count,
    plot_interpolation_panels,
    write_interpolation_summary,
    write_log_error_model,
)
from result_paths import PROBLEM3_RESULT_DIR, ensure_result_dir


def solve_problem3() -> list[InterpolationSummary]:
    ensure_result_dir(PROBLEM3_RESULT_DIR)

    def sine_function(x: np.ndarray) -> np.ndarray:
        return x * np.sin(2.0 * math.pi * x + 1.0)

    x_dense = np.linspace(-1.0, 1.0, 2001)
    node_counts = [7, 9, 17, 19, 21]
    summaries = [interpolation_case(sine_function, node_count, x_dense)[4] for node_count in node_counts]

    write_interpolation_summary(PROBLEM3_RESULT_DIR / "problem3_sine_error_summary.csv", summaries)
    write_log_error_model(PROBLEM3_RESULT_DIR / "problem3_sine_error_model.csv", summaries, min_node_count=7)
    plot_interpolation_panels(
        sine_function,
        [7],
        PROBLEM3_RESULT_DIR / "problem3_sine_7_nodes.png",
        "Problem 3(1): x sin(2 pi x + 1)",
        y_limits=(-1.35, 0.65),
    )
    plot_interpolation_panels(
        sine_function,
        [9, 17, 19, 21],
        PROBLEM3_RESULT_DIR / "problem3_sine_more_nodes.png",
        "Problem 3(2): x sin(2 pi x + 1)",
        y_limits=(-1.35, 0.65),
        ncols=2,
    )
    plot_error_vs_node_count(
        summaries,
        PROBLEM3_RESULT_DIR / "problem3_sine_error_vs_b.png",
        "Problem 3: error decay as b increases",
    )
    return summaries


def main() -> None:
    rows = solve_problem3()
    print(f"Problem 3 rows: {len(rows)}")


if __name__ == "__main__":
    main()
