from __future__ import annotations

import argparse
import csv
from pathlib import Path

from problem5_core import write_csv


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results"

FIELDS = [
    "dimension",
    "selection_rule",
    "q_pattern",
    "q2_axis_count",
    "q3_axis_count",
    "q4_axis_count",
    "q5_axis_count",
    "q6_axis_count",
    "total_points",
    "coefficient_inside_points",
    "coefficient_estimate",
    "reference_volume",
    "predicted_relative_error",
    "estimated_strict_count_runtime_h",
    "estimated_strict_count_runtime_min",
    "points_ratio_vs_q345_main",
    "points_ratio_vs_q3456_fast",
    "predictor_kind",
]


def load_rows() -> list[dict[str, str]]:
    path = RESULT_DIR / "q23456_coefficient_search_predictions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_rows(
    rows: list[dict[str, str]],
    dimensions: list[int],
    max_predicted_error: float,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for dimension in dimensions:
        candidates = [
            row
            for row in rows
            if int(row["dimension"]) == dimension
            and int(row["q2_axis_count"]) > 0
            and row["predicted_within_10_percent"] == "true"
            and float(row["predicted_relative_error"]) <= max_predicted_error
        ]
        if not candidates:
            raise RuntimeError(f"no q23456 candidate found for n={dimension}")
        best = min(candidates, key=lambda row: (int(row["total_points"]), float(row["predicted_relative_error"])))
        out = {field: best.get(field, "") for field in FIELDS}
        out["selection_rule"] = f"minimum_points_with_predicted_relative_error_le_{max_predicted_error:g}"
        selected.append(out)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", default="27,28,29,30")
    parser.add_argument("--max-predicted-error", type=float, default=0.02)
    args = parser.parse_args()

    dimensions = [int(item) for item in args.dimensions.split(",") if item.strip()]
    rows = select_rows(load_rows(), dimensions, args.max_predicted_error)
    output_csv = RESULT_DIR / "q23456_selected_under30_results.csv"
    write_csv(output_csv, rows, FIELDS)

    lines = [
        "q23456 selected results up to n=30",
        "===================================",
        f"dimensions={','.join(str(item) for item in dimensions)}",
        f"selection=max_predicted_error<={args.max_predicted_error:g}, minimum total_points",
        "",
    ]
    for row in rows:
        lines.append(
            f"n={row['dimension']}: pattern={row['q_pattern']}, "
            f"estimate={row['coefficient_estimate']}, "
            f"relerr={row['predicted_relative_error']}, "
            f"time_h={row['estimated_strict_count_runtime_h']}"
        )
    lines.append("")
    lines.append(f"Selected CSV written to {output_csv}")
    (RESULT_DIR / "q23456_selected_under30_results_run.log").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
