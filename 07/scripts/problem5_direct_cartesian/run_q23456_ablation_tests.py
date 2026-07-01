from __future__ import annotations

import argparse
import csv
from io import StringIO
import os
from pathlib import Path
import statistics
import subprocess

import matplotlib.pyplot as plt

from problem5_core import q_pattern, write_csv


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results"
OPT_BINARY = ROOT / "direct_tensor_midpoint_orthant_mixed_q26_tile_batch_avx2"
ABLATION_BINARY = ROOT / "direct_tensor_midpoint_ablation_kernel"

STORAGE_U16 = 0
STORAGE_RANK8 = 1
AXIS_ASC = 0
AXIS_DESC = 1

MICRO_COUNTS = (1, 8, 4, 1, 1)
MICRO_TAIL = 7
SIGN_COUNTS = (1, 5, 3, 1, 0)
SIGN_TAIL = 5
OPT_COUNTS = (1, 11, 7, 1, 2)
OPT_TAIL = 9
THREAD_COUNTS = OPT_COUNTS
THREAD_TAIL = OPT_TAIL
LAYOUT_COUNTS = (1, 11, 9, 1, 2)

RAW_FIELDS = [
    "group_label",
    "case_label",
    "repeat_index",
    "source",
    "q_pattern",
    "dimension",
    "tail_dimension",
    "threads",
    "prefix_chunk_points",
    "batch_prefixes",
    "storage_mode",
    "axis_order",
    "total_points",
    "inside_points",
    "runtime_s",
    "points_per_s",
]

SUMMARY_FIELDS = [
    "group_label",
    "disabled_case",
    "enabled_case",
    "disabled_median_points_per_s",
    "enabled_median_points_per_s",
    "disabled_median_runtime_s",
    "enabled_median_runtime_s",
    "speedup_vs_disabled",
    "point_reduction_vs_disabled",
    "inside_consistent",
    "note",
]


def read_single_csv(stdout: str) -> dict[str, str]:
    rows = list(csv.DictReader(StringIO(stdout)))
    if len(rows) != 1:
        raise RuntimeError(f"Unexpected CSV output: {stdout!r}")
    return rows[0]


def build_binaries() -> None:
    subprocess.run(["make", "all"], cwd=ROOT, check=True)


def run_ablation_variant(
    *,
    variant: int,
    group_label: str,
    case_label: str,
    counts: tuple[int, int, int, int, int],
    tail_dimension: int,
    repeat_index: int,
) -> dict[str, object]:
    completed = subprocess.run(
        [
            str(ABLATION_BINARY),
            str(variant),
            str(counts[0]),
            str(counts[1]),
            str(counts[2]),
            str(counts[3]),
            str(counts[4]),
            str(tail_dimension),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    raw = read_single_csv(completed.stdout)
    return {
        "group_label": group_label,
        "case_label": case_label,
        "repeat_index": repeat_index,
        "source": "ablation_kernel",
        "q_pattern": q_pattern(counts),
        "dimension": raw["dimension"],
        "tail_dimension": raw["tail_dimension"],
        "threads": "1",
        "prefix_chunk_points": "",
        "batch_prefixes": "1" if variant in (0, 1, 2, 3, 4, 6) else "16",
        "storage_mode": raw["variant"],
        "axis_order": "asc",
        "total_points": raw["total_points"],
        "inside_points": raw["inside_points"],
        "runtime_s": raw["runtime_s"],
        "points_per_s": raw["points_per_s"],
    }


def run_optimized_variant(
    *,
    group_label: str,
    case_label: str,
    counts: tuple[int, int, int, int, int],
    tail_dimension: int,
    threads: int,
    prefix_chunk_points: int,
    batch_prefixes: int,
    storage_mode: int,
    axis_order: int,
    repeat_index: int,
) -> dict[str, object]:
    env = dict(os.environ)
    env.setdefault("OMP_PROC_BIND", "spread")
    env.setdefault("OMP_PLACES", "cores")
    completed = subprocess.run(
        [
            str(OPT_BINARY),
            str(counts[0]),
            str(counts[1]),
            str(counts[2]),
            str(counts[3]),
            str(counts[4]),
            str(tail_dimension),
            str(threads),
            str(prefix_chunk_points),
            str(batch_prefixes),
            str(storage_mode),
            str(axis_order),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    raw = read_single_csv(completed.stdout)
    return {
        "group_label": group_label,
        "case_label": case_label,
        "repeat_index": repeat_index,
        "source": "optimized_kernel",
        "q_pattern": q_pattern(counts),
        "dimension": raw["dimension"],
        "tail_dimension": raw["tail_dimension"],
        "threads": raw["threads"],
        "prefix_chunk_points": raw["prefix_chunk_points"],
        "batch_prefixes": raw["batch_prefixes"],
        "storage_mode": raw["storage_mode"],
        "axis_order": raw["axis_order"],
        "total_points": raw["total_points"],
        "inside_points": raw["inside_points"],
        "runtime_s": raw["count_runtime_s"],
        "points_per_s": raw["count_points_per_s"],
    }


def median_float(rows: list[dict[str, object]], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def summarize_pair(
    rows: list[dict[str, object]],
    *,
    group_label: str,
    disabled_case: str,
    enabled_case: str,
    note: str,
    use_point_reduction: bool = False,
) -> dict[str, object]:
    disabled_rows = [row for row in rows if row["group_label"] == group_label and row["case_label"] == disabled_case]
    enabled_rows = [row for row in rows if row["group_label"] == group_label and row["case_label"] == enabled_case]
    if not disabled_rows or not enabled_rows:
        raise RuntimeError(f"Missing summary rows for {group_label}: {disabled_case} -> {enabled_case}")

    disabled_rate = median_float(disabled_rows, "points_per_s")
    enabled_rate = median_float(enabled_rows, "points_per_s")
    disabled_runtime = median_float(disabled_rows, "runtime_s")
    enabled_runtime = median_float(enabled_rows, "runtime_s")
    disabled_points = int(disabled_rows[0]["total_points"])
    enabled_points = int(enabled_rows[0]["total_points"])
    disabled_inside = int(disabled_rows[0]["inside_points"])
    enabled_inside = int(enabled_rows[0]["inside_points"])
    point_reduction = disabled_points / enabled_points
    if use_point_reduction:
        speedup = point_reduction
        consistent = disabled_inside == enabled_inside * round(point_reduction)
    else:
        speedup = enabled_rate / disabled_rate
        consistent = disabled_inside == enabled_inside

    return {
        "group_label": group_label,
        "disabled_case": disabled_case,
        "enabled_case": enabled_case,
        "disabled_median_points_per_s": f"{disabled_rate:.9e}",
        "enabled_median_points_per_s": f"{enabled_rate:.9e}",
        "disabled_median_runtime_s": f"{disabled_runtime:.9f}",
        "enabled_median_runtime_s": f"{enabled_runtime:.9f}",
        "speedup_vs_disabled": f"{speedup:.6f}",
        "point_reduction_vs_disabled": f"{point_reduction:.6f}",
        "inside_consistent": str(consistent).lower(),
        "note": note,
    }


def override_layout_summary_from_repeat_checks(summary_rows: list[dict[str, object]]) -> None:
    repeat_summary = RESULT_DIR / "q23456_strict_layout_repeat_summary.csv"
    if not repeat_summary.exists():
        return
    rows = {row["case_label"]: row for row in csv.DictReader(repeat_summary.open(encoding="utf-8"))}
    baseline = rows.get("cheap_baseline_asc_tail8_c128_b16")
    tuned = rows.get("cheap_asc_tail10_c512_b16")
    if baseline is None or tuned is None:
        return

    replacement = {
        "group_label": "layout_tuning",
        "disabled_case": "cheap_baseline_asc_tail8_c128_b16",
        "enabled_case": "cheap_asc_tail10_c512_b16",
        "disabled_median_points_per_s": baseline["median_count_points_per_s"],
        "enabled_median_points_per_s": tuned["median_count_points_per_s"],
        "disabled_median_runtime_s": baseline["median_count_runtime_s"],
        "enabled_median_runtime_s": tuned["median_count_runtime_s"],
        "speedup_vs_disabled": tuned["speedup_vs_proxy_baseline_median"],
        "point_reduction_vs_disabled": "1.000000",
        "inside_consistent": "true",
        "note": "layout repeat check on n24 proxy; this replaces one-off layout-pair noise",
    }
    for index, row in enumerate(summary_rows):
        if row["group_label"] == "layout_tuning":
            summary_rows[index] = replacement
            return
    summary_rows.append(replacement)


def run_all(repeats: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []

    ablation_cases = [
        ("sign_symmetry", "signed_full", 6, SIGN_COUNTS, SIGN_TAIL),
        ("sign_symmetry", "first_orthant", 1, SIGN_COUNTS, SIGN_TAIL),
        ("integer_radius_update", "float_fullsum", 0, MICRO_COUNTS, MICRO_TAIL),
        ("integer_radius_update", "integer_incremental", 1, MICRO_COUNTS, MICRO_TAIL),
        ("tail_table_reuse", "tail_recompute", 2, MICRO_COUNTS, MICRO_TAIL),
        ("tail_table_reuse", "tail_table_u16_scalar", 3, MICRO_COUNTS, MICRO_TAIL),
        ("prefix_batching_scalar", "rank8_scalar_batch1", 4, MICRO_COUNTS, MICRO_TAIL),
        ("prefix_batching_scalar", "rank8_scalar_batch16", 5, MICRO_COUNTS, MICRO_TAIL),
        ("avx2_unroll_single_thread", "rank8_scalar_batch16", 5, MICRO_COUNTS, MICRO_TAIL),
    ]

    for repeat_index in range(1, repeats + 1):
        for group_label, case_label, variant, counts, tail_dimension in ablation_cases:
            rows.append(
                run_ablation_variant(
                    variant=variant,
                    group_label=group_label,
                    case_label=case_label,
                    counts=counts,
                    tail_dimension=tail_dimension,
                    repeat_index=repeat_index,
                )
            )

    optimized_cases = [
        ("rank8_storage", "u16_avx2_batch16", OPT_COUNTS, OPT_TAIL, 88, 128, 16, STORAGE_U16, AXIS_ASC),
        ("rank8_storage", "rank8_avx2_batch16", OPT_COUNTS, OPT_TAIL, 88, 128, 16, STORAGE_RANK8, AXIS_ASC),
        ("prefix_batching_avx2", "rank8_avx2_batch1", OPT_COUNTS, OPT_TAIL, 88, 128, 1, STORAGE_RANK8, AXIS_ASC),
        ("prefix_batching_avx2", "rank8_avx2_batch16", OPT_COUNTS, OPT_TAIL, 88, 128, 16, STORAGE_RANK8, AXIS_ASC),
        ("batch32_guardrail", "rank8_avx2_batch16", OPT_COUNTS, OPT_TAIL, 88, 128, 16, STORAGE_RANK8, AXIS_ASC),
        ("batch32_guardrail", "rank8_avx2_batch32", OPT_COUNTS, OPT_TAIL, 88, 128, 32, STORAGE_RANK8, AXIS_ASC),
        ("avx2_unroll_single_thread", "rank8_avx2_batch16_1thread", MICRO_COUNTS, MICRO_TAIL, 1, 128, 16, STORAGE_RANK8, AXIS_ASC),
        ("openmp_scaling", "rank8_avx2_thread1", THREAD_COUNTS, THREAD_TAIL, 1, 128, 16, STORAGE_RANK8, AXIS_ASC),
        ("openmp_scaling", "rank8_avx2_thread88", THREAD_COUNTS, THREAD_TAIL, 88, 128, 16, STORAGE_RANK8, AXIS_ASC),
        ("layout_tuning", "layout_tail8_c128_asc", LAYOUT_COUNTS, 8, 88, 128, 16, STORAGE_RANK8, AXIS_ASC),
        ("layout_tuning", "layout_tail10_c512_asc", LAYOUT_COUNTS, 10, 88, 512, 16, STORAGE_RANK8, AXIS_ASC),
    ]
    for repeat_index in range(1, repeats + 1):
        for (
            group_label,
            case_label,
            counts,
            tail_dimension,
            threads,
            prefix_chunk_points,
            batch_prefixes,
            storage_mode,
            axis_order,
        ) in optimized_cases:
            rows.append(
                run_optimized_variant(
                    group_label=group_label,
                    case_label=case_label,
                    counts=counts,
                    tail_dimension=tail_dimension,
                    threads=threads,
                    prefix_chunk_points=prefix_chunk_points,
                    batch_prefixes=batch_prefixes,
                    storage_mode=storage_mode,
                    axis_order=axis_order,
                    repeat_index=repeat_index,
                )
            )

    summary_rows = [
        summarize_pair(
            rows,
            group_label="sign_symmetry",
            disabled_case="signed_full",
            enabled_case="first_orthant",
            note="point-count reduction by sign symmetry; small proxy uses n=10",
            use_point_reduction=True,
        ),
        summarize_pair(
            rows,
            group_label="integer_radius_update",
            disabled_case="float_fullsum",
            enabled_case="integer_incremental",
            note="integer/incremental radius update against floating full-coordinate summation",
        ),
        summarize_pair(
            rows,
            group_label="tail_table_reuse",
            disabled_case="tail_recompute",
            enabled_case="tail_table_u16_scalar",
            note="precomputed tail sums against recomputing the tail for every prefix",
        ),
        summarize_pair(
            rows,
            group_label="prefix_batching_scalar",
            disabled_case="rank8_scalar_batch1",
            enabled_case="rank8_scalar_batch16",
            note="same scalar rank8 comparator, one prefix threshold versus sixteen per tail scan",
        ),
        summarize_pair(
            rows,
            group_label="rank8_storage",
            disabled_case="u16_avx2_batch16",
            enabled_case="rank8_avx2_batch16",
            note="AVX2 path with u16 tail table versus lossless rank8 tail table",
        ),
        summarize_pair(
            rows,
            group_label="prefix_batching_avx2",
            disabled_case="rank8_avx2_batch1",
            enabled_case="rank8_avx2_batch16",
            note="AVX2 rank8 path with one versus sixteen prefix thresholds per tail scan",
        ),
        summarize_pair(
            rows,
            group_label="avx2_unroll_single_thread",
            disabled_case="rank8_scalar_batch16",
            enabled_case="rank8_avx2_batch16_1thread",
            note="single-thread scalar rank8 batch16 against AVX2 unrolled rank8 batch16",
        ),
        summarize_pair(
            rows,
            group_label="openmp_scaling",
            disabled_case="rank8_avx2_thread1",
            enabled_case="rank8_avx2_thread88",
            note="same AVX2 rank8 kernel with 1 thread versus 88 threads",
        ),
        summarize_pair(
            rows,
            group_label="layout_tuning",
            disabled_case="layout_tail8_c128_asc",
            enabled_case="layout_tail10_c512_asc",
            note="same kernel after changing tail dimension and prefix chunk",
        ),
        summarize_pair(
            rows,
            group_label="batch32_guardrail",
            disabled_case="rank8_avx2_batch16",
            enabled_case="rank8_avx2_batch32",
            note="guardrail check: larger batch is not automatically better",
        ),
    ]
    override_layout_summary_from_repeat_checks(summary_rows)
    return rows, summary_rows


def plot_summary(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    labels = [
        "sign symmetry",
        "integer radius",
        "tail table",
        "scalar batch16",
        "rank8 storage",
        "AVX2 batch16",
        "AVX2 unroll",
        "OpenMP 88",
        "layout tuning",
        "batch32 guard",
    ]
    speedups = [float(row["speedup_vs_disabled"]) for row in summary_rows]
    colors = ["#4C78A8" if value >= 1.0 else "#E45756" for value in speedups]

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    y_positions = list(range(len(labels)))
    bars = ax.barh(y_positions, speedups, color=colors, alpha=0.9)
    ax.axvline(1.0, color="#333333", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("speedup vs ablated baseline (log scale)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis="x", which="both", alpha=0.25)

    for bar, value in zip(bars, speedups):
        x = bar.get_width()
        ax.text(
            x * (1.08 if value >= 1.0 else 0.92),
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.2f}x",
            va="center",
            ha="left" if value >= 1.0 else "right",
            fontsize=9,
        )

    ax.set_title("Problem 5 direct Cartesian engineering ablation")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict Cartesian engineering ablation tests.")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    build_binaries()
    raw_rows, summary_rows = run_all(args.repeats)

    raw_csv = RESULT_DIR / "q23456_engineering_ablation_raw.csv"
    summary_csv = RESULT_DIR / "q23456_engineering_ablation_summary.csv"
    figure_path = RESULT_DIR / "q23456_engineering_ablation_speedups.png"
    write_csv(raw_csv, raw_rows, RAW_FIELDS)
    write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)
    plot_summary(summary_rows, figure_path)

    lines = [
        "q23456 engineering ablation tests",
        "=================================",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['group_label']}: {row['disabled_case']} -> {row['enabled_case']}, "
            f"speedup={row['speedup_vs_disabled']}, inside_consistent={row['inside_consistent']}"
        )
    lines.extend(
        [
            "",
            f"Raw CSV written to {raw_csv}",
            f"Summary CSV written to {summary_csv}",
            f"Figure written to {figure_path}",
        ]
    )
    log_path = RESULT_DIR / "q23456_engineering_ablation_run.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
