from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import time

from problem5_core import (
    Q_VALUES,
    coefficient_estimate,
    coefficient_inside_points,
    point_count,
    q345_main_counts,
    q_pattern,
    reference_volume,
    roughness,
    write_csv,
)


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results"
DEFAULT_STRICT_POINTS_PER_S = 2.935576436e12

FIELDS = [
    "candidate_label",
    "candidate_group",
    "dimension",
    "q_pattern",
    "q2_axis_count",
    "q3_axis_count",
    "q4_axis_count",
    "q5_axis_count",
    "q6_axis_count",
    "l1_distance_from_q345_main",
    "total_points",
    "points_ratio_vs_q345_main",
    "points_ratio_vs_q3456_fast",
    "roughness_ratio_vs_q345_main",
    "min_radius_squared",
    "coefficient_inside_points",
    "coefficient_estimate",
    "reference_volume",
    "predicted_signed_relative_error",
    "predicted_relative_error",
    "predicted_within_10_percent",
    "estimated_strict_count_runtime_s",
    "estimated_strict_count_runtime_min",
    "estimated_strict_count_runtime_h",
    "throughput_basis_points_per_s",
    "predictor_kind",
]

SUMMARY_FIELDS = [
    "dimension",
    "candidate_count",
    "q2_candidate_count",
    "q2_passing_count",
    "q2_passing_lower_than_q345_count",
    "q2_passing_lower_than_q3456_fast_count",
    "smallest_passing_q2_label",
    "smallest_passing_q2_pattern",
    "smallest_passing_q2_relative_error",
    "smallest_passing_q2_points_ratio_vs_q345_main",
    "smallest_passing_q2_points_ratio_vs_q3456_fast",
    "smallest_passing_q2_runtime_h",
    "best_error_q2_label",
    "best_error_q2_pattern",
    "best_error_q2_relative_error",
    "best_error_q2_points_ratio_vs_q345_main",
    "best_error_q2_points_ratio_vs_q3456_fast",
    "q345_main_pattern",
    "q345_main_points",
    "q345_main_estimated_runtime_h",
    "q3456_fast_pattern",
    "q3456_fast_points",
    "q3456_fast_estimated_runtime_h",
]


def all_compositions(dimension: int):
    for q2 in range(dimension + 1):
        for q3 in range(dimension - q2 + 1):
            for q4 in range(dimension - q2 - q3 + 1):
                for q5 in range(dimension - q2 - q3 - q4 + 1):
                    q6 = dimension - q2 - q3 - q4 - q5
                    yield (q2, q3, q4, q5, q6)


def q3456_fast_counts(dimension: int) -> tuple[int, int, int, int, int]:
    if dimension < 14:
        return q345_main_counts(dimension)
    return (0, 9, dimension - 14, 5, 0)


def min_radius_squared(counts: tuple[int, int, int, int, int]) -> float:
    return sum(count / (4.0 * q * q) for q, count in zip(Q_VALUES, counts))


def pattern_slug(counts: tuple[int, int, int, int, int]) -> str:
    return "_".join(f"{q}^{count}" for q, count in zip(Q_VALUES, counts))


def read_strict_points_per_s() -> tuple[float, str]:
    summary = RESULT_DIR / "q3456_strict_lowrisk_optimization_summary.csv"
    if summary.exists():
        with summary.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows and rows[0].get("best_count_points_per_s"):
            return float(rows[0]["best_count_points_per_s"]), "q3456_strict_lowrisk_best"

    local_scan = RESULT_DIR / "q23456_local_scan.csv"
    if local_scan.exists():
        rates: list[float] = []
        with local_scan.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                value = row.get("count_points_per_s", "")
                if value:
                    rates.append(float(value))
        if rates:
            return min(rates), "q23456_local_scan_min_rate"

    return DEFAULT_STRICT_POINTS_PER_S, "default_q23456_reference"


def add_candidate(
    candidates: dict[tuple[int, int, int, int, int], set[str]],
    counts: tuple[int, int, int, int, int],
    group: str,
) -> None:
    if min(counts) < 0:
        return
    if sum(counts) == 0:
        return
    candidates.setdefault(counts, set()).add(group)


def q26_formula_counts(family: str, dimension: int, k: int) -> tuple[int, int, int, int, int] | None:
    if family == "pair_q2q5":
        q4 = dimension - 2 * k
        return None if q4 < 0 else (k, 0, q4, k, 0)
    if family == "pair_q2q6":
        q4 = dimension - 2 * k
        return None if q4 < 0 else (k, 0, q4, 0, k)
    if family == "triple_q2_2q5":
        q4 = dimension - 3 * k
        return None if q4 < 0 else (k, 0, q4, 2 * k, 0)
    raise ValueError(f"unknown q26 family: {family}")


def formula_ks(dimension: int) -> list[tuple[str, int]]:
    return [
        ("round_n_over_4", round(dimension / 4.0)),
        ("round_n_over_5", round(dimension / 5.0)),
        ("floor_n_over_5", math.floor(dimension / 5.0)),
        ("round_n_over_6", round(dimension / 6.0)),
    ]


def build_candidates(
    dimensions: list[int],
    *,
    max_q2: int,
    max_q6: int,
    max_l1: int,
    min_ratio_vs_q345: float,
    max_ratio_vs_q345: float,
    max_roughness_ratio: float,
    max_min_radius_squared: float,
) -> list[tuple[str, str, tuple[int, int, int, int, int]]]:
    candidates: dict[tuple[int, int, int, int, int], set[str]] = {}

    for dimension in dimensions:
        base = q345_main_counts(dimension)
        base_points = point_count(base)
        base_roughness = roughness(base)

        for counts in all_compositions(dimension):
            if counts[0] <= 0 or counts[0] > max_q2 or counts[4] > max_q6:
                continue
            l1_distance = sum(abs(counts[index] - base[index]) for index in range(5))
            if l1_distance > max_l1:
                continue
            ratio = point_count(counts) / base_points
            if not (min_ratio_vs_q345 <= ratio <= max_ratio_vs_q345):
                continue
            roughness_ratio = roughness(counts) / base_roughness
            if roughness_ratio > max_roughness_ratio:
                continue
            if min_radius_squared(counts) > max_min_radius_squared:
                continue
            add_candidate(candidates, counts, "local_q2_filtered")

        if dimension >= 13:
            add_candidate(candidates, (1, 7, dimension - 13, 5, 0), "observed_q2_q3_7_q5_5_extension")
            add_candidate(candidates, (1, 6, dimension - 13, 6, 0), "observed_q2_q3_6_q5_6_extension")

        for family in ["pair_q2q5", "pair_q2q6", "triple_q2_2q5"]:
            for k_rule, k in formula_ks(dimension):
                counts = q26_formula_counts(family, dimension, k)
                if counts is not None:
                    add_candidate(candidates, counts, f"q26_{family}_{k_rule}_extension")

    rows: list[tuple[str, str, tuple[int, int, int, int, int]]] = []
    for counts, groups in candidates.items():
        group = "+".join(sorted(groups))
        label = f"n{sum(counts)}_{group}_{pattern_slug(counts)}"
        rows.append((label, group, counts))
    return sorted(rows, key=lambda item: (sum(item[2]), point_count(item[2]), item[2], item[1]))


def prediction_row(
    label: str,
    group: str,
    counts: tuple[int, int, int, int, int],
    throughput: float,
) -> dict[str, object]:
    dimension = sum(counts)
    base_q345 = q345_main_counts(dimension)
    base_q3456 = q3456_fast_counts(dimension)
    start = time.perf_counter()
    inside_points = coefficient_inside_points(counts)
    coefficient_runtime = time.perf_counter() - start
    estimate = coefficient_estimate(counts, inside_points)
    reference = reference_volume(dimension)
    signed_error = (estimate - reference) / reference
    relative_error = abs(signed_error)
    total_points = point_count(counts)
    runtime_s = total_points / throughput
    q2, q3, q4, q5, q6 = counts
    return {
        "candidate_label": label,
        "candidate_group": group,
        "dimension": dimension,
        "q_pattern": q_pattern(counts),
        "q2_axis_count": q2,
        "q3_axis_count": q3,
        "q4_axis_count": q4,
        "q5_axis_count": q5,
        "q6_axis_count": q6,
        "l1_distance_from_q345_main": sum(abs(counts[index] - base_q345[index]) for index in range(5)),
        "total_points": total_points,
        "points_ratio_vs_q345_main": f"{total_points / point_count(base_q345):.9e}",
        "points_ratio_vs_q3456_fast": f"{total_points / point_count(base_q3456):.9e}",
        "roughness_ratio_vs_q345_main": f"{roughness(counts) / roughness(base_q345):.9e}",
        "min_radius_squared": f"{min_radius_squared(counts):.9e}",
        "coefficient_inside_points": inside_points,
        "coefficient_estimate": f"{estimate:.18e}",
        "reference_volume": f"{reference:.16e}",
        "predicted_signed_relative_error": f"{signed_error:.6e}",
        "predicted_relative_error": f"{relative_error:.6e}",
        "predicted_within_10_percent": str(relative_error <= 0.10).lower(),
        "estimated_strict_count_runtime_s": f"{runtime_s:.6f}",
        "estimated_strict_count_runtime_min": f"{runtime_s / 60.0:.6f}",
        "estimated_strict_count_runtime_h": f"{runtime_s / 3600.0:.6f}",
        "throughput_basis_points_per_s": f"{throughput:.9e}",
        "predictor_kind": f"coefficient_oracle_for_parameter_selection_runtime={coefficient_runtime:.6f}s",
    }


def baseline_row(counts: tuple[int, int, int, int, int], throughput: float) -> dict[str, object]:
    points = point_count(counts)
    runtime_s = points / throughput
    return {
        "pattern": q_pattern(counts),
        "points": points,
        "runtime_h": runtime_s / 3600.0,
    }


def summarize(rows: list[dict[str, object]], dimensions: list[int], throughput: float) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for dimension in dimensions:
        dim_rows = [row for row in rows if int(row["dimension"]) == dimension]
        q2_rows = [row for row in dim_rows if int(row["q2_axis_count"]) > 0]
        passing = [row for row in q2_rows if row["predicted_within_10_percent"] == "true"]
        lower_than_q345 = [row for row in passing if float(row["points_ratio_vs_q345_main"]) < 1.0]
        lower_than_q3456 = [row for row in passing if float(row["points_ratio_vs_q3456_fast"]) < 1.0]
        smallest = min(passing, key=lambda row: int(row["total_points"])) if passing else None
        best_error = min(q2_rows, key=lambda row: float(row["predicted_relative_error"])) if q2_rows else None
        q345 = baseline_row(q345_main_counts(dimension), throughput)
        q3456 = baseline_row(q3456_fast_counts(dimension), throughput)
        summary_rows.append(
            {
                "dimension": dimension,
                "candidate_count": len(dim_rows),
                "q2_candidate_count": len(q2_rows),
                "q2_passing_count": len(passing),
                "q2_passing_lower_than_q345_count": len(lower_than_q345),
                "q2_passing_lower_than_q3456_fast_count": len(lower_than_q3456),
                "smallest_passing_q2_label": "" if smallest is None else smallest["candidate_label"],
                "smallest_passing_q2_pattern": "" if smallest is None else smallest["q_pattern"],
                "smallest_passing_q2_relative_error": "" if smallest is None else smallest["predicted_relative_error"],
                "smallest_passing_q2_points_ratio_vs_q345_main": "" if smallest is None else smallest["points_ratio_vs_q345_main"],
                "smallest_passing_q2_points_ratio_vs_q3456_fast": "" if smallest is None else smallest["points_ratio_vs_q3456_fast"],
                "smallest_passing_q2_runtime_h": "" if smallest is None else smallest["estimated_strict_count_runtime_h"],
                "best_error_q2_label": "" if best_error is None else best_error["candidate_label"],
                "best_error_q2_pattern": "" if best_error is None else best_error["q_pattern"],
                "best_error_q2_relative_error": "" if best_error is None else best_error["predicted_relative_error"],
                "best_error_q2_points_ratio_vs_q345_main": "" if best_error is None else best_error["points_ratio_vs_q345_main"],
                "best_error_q2_points_ratio_vs_q3456_fast": "" if best_error is None else best_error["points_ratio_vs_q3456_fast"],
                "q345_main_pattern": q345["pattern"],
                "q345_main_points": q345["points"],
                "q345_main_estimated_runtime_h": f"{q345['runtime_h']:.6f}",
                "q3456_fast_pattern": q3456["pattern"],
                "q3456_fast_points": q3456["points"],
                "q3456_fast_estimated_runtime_h": f"{q3456['runtime_h']:.6f}",
            }
        )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", default="27,28")
    parser.add_argument("--max-q2", type=int, default=6)
    parser.add_argument("--max-q6", type=int, default=6)
    parser.add_argument("--max-l1", type=int, default=24)
    parser.add_argument("--min-ratio-vs-q345", type=float, default=0.03)
    parser.add_argument("--max-ratio-vs-q345", type=float, default=1.0)
    parser.add_argument("--max-roughness-ratio", type=float, default=1.35)
    parser.add_argument("--max-min-radius-squared", type=float, default=1.0)
    args = parser.parse_args()

    dimensions = [int(item) for item in args.dimensions.split(",") if item.strip()]
    throughput, throughput_source = read_strict_points_per_s()
    candidates = build_candidates(
        dimensions,
        max_q2=args.max_q2,
        max_q6=args.max_q6,
        max_l1=args.max_l1,
        min_ratio_vs_q345=args.min_ratio_vs_q345,
        max_ratio_vs_q345=args.max_ratio_vs_q345,
        max_roughness_ratio=args.max_roughness_ratio,
        max_min_radius_squared=args.max_min_radius_squared,
    )

    rows = [prediction_row(label, group, counts, throughput) for label, group, counts in candidates]
    rows.sort(
        key=lambda row: (
            int(row["dimension"]),
            row["predicted_within_10_percent"] != "true",
            int(row["total_points"]),
            float(row["predicted_relative_error"]),
            row["q_pattern"],
        )
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_csv = RESULT_DIR / "q23456_coefficient_search_predictions.csv"
    write_csv(prediction_csv, rows, FIELDS)

    summary_rows = summarize(rows, dimensions, throughput)
    summary_csv = RESULT_DIR / "q23456_coefficient_search_summary.csv"
    write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)

    lines = [
        "q=2/3/4/5/6 coefficient search",
        "================================",
        f"dimensions={','.join(str(item) for item in dimensions)}",
        f"candidate_rows={len(rows)}",
        f"throughput_basis={throughput_source}:{throughput:.9e}",
        "",
    ]
    for summary in summary_rows:
        lines.append(
            f"n={summary['dimension']}: q2_passing={summary['q2_passing_count']}, "
            f"q2_passing_lower_than_q345={summary['q2_passing_lower_than_q345_count']}, "
            f"q2_passing_lower_than_q3456_fast={summary['q2_passing_lower_than_q3456_fast_count']}"
        )
        if summary["smallest_passing_q2_pattern"]:
            lines.append(
                f"  smallest_q2_pass={summary['smallest_passing_q2_pattern']}, "
                f"relerr={summary['smallest_passing_q2_relative_error']}, "
                f"ratio_vs_q345={summary['smallest_passing_q2_points_ratio_vs_q345_main']}, "
                f"ratio_vs_q3456_fast={summary['smallest_passing_q2_points_ratio_vs_q3456_fast']}, "
                f"runtime_h={summary['smallest_passing_q2_runtime_h']}"
            )
        if summary["best_error_q2_pattern"]:
            lines.append(
                f"  best_error_q2={summary['best_error_q2_pattern']}, "
                f"relerr={summary['best_error_q2_relative_error']}, "
                f"ratio_vs_q345={summary['best_error_q2_points_ratio_vs_q345_main']}, "
                f"ratio_vs_q3456_fast={summary['best_error_q2_points_ratio_vs_q3456_fast']}"
            )
    lines.extend(
        [
            "",
            f"Prediction CSV written to {prediction_csv}",
            f"Summary CSV written to {summary_csv}",
        ]
    )
    (RESULT_DIR / "q23456_coefficient_search_run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
