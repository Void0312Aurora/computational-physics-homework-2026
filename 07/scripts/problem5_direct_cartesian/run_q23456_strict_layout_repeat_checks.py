from __future__ import annotations

import csv
from io import StringIO
import os
from pathlib import Path
import statistics
import subprocess

from problem5_core import build_binary, write_csv


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results"
BINARY = ROOT / "direct_tensor_midpoint_orthant_mixed_q26_tile_batch_avx2"

THREADS = 88
STORAGE_RANK8 = 1
AXIS_ASC = 0
AXIS_DESC = 1
REPEATS = 3

PROXY_CHEAP_31 = (1, 11, 9, 1, 2)
PROXY_2PCT_31 = (1, 12, 8, 1, 2)

CASES = [
    ("cheap_baseline_asc_tail8_c128_b16", "n24_proxy_31_cheapest", PROXY_CHEAP_31, 8, 128, 16, AXIS_ASC),
    ("cheap_asc_tail10_c512_b16", "n24_proxy_31_cheapest", PROXY_CHEAP_31, 10, 512, 16, AXIS_ASC),
    ("cheap_desc_tail12_c128_b16", "n24_proxy_31_cheapest", PROXY_CHEAP_31, 12, 128, 16, AXIS_DESC),
    ("cheap_batch32_asc_tail10_c512", "n24_proxy_31_cheapest", PROXY_CHEAP_31, 10, 512, 32, AXIS_ASC),
    ("below2_baseline_asc_tail9_c128_b16", "n24_proxy_31_below_2pct", PROXY_2PCT_31, 9, 128, 16, AXIS_ASC),
    ("below2_asc_tail10_c128_b16", "n24_proxy_31_below_2pct", PROXY_2PCT_31, 10, 128, 16, AXIS_ASC),
    ("below2_asc_tail10_c512_b16", "n24_proxy_31_below_2pct", PROXY_2PCT_31, 10, 512, 16, AXIS_ASC),
    ("below2_batch32_asc_tail10_c512", "n24_proxy_31_below_2pct", PROXY_2PCT_31, 10, 512, 32, AXIS_ASC),
]

BENCHMARK_FIELDS = [
    "case_label",
    "proxy_label",
    "repeat_index",
    "tail_dimension",
    "axis_order",
    "prefix_chunk_points",
    "batch_prefixes",
    "total_points",
    "inside_points",
    "count_runtime_s",
    "count_points_per_s",
]

SUMMARY_FIELDS = [
    "case_label",
    "proxy_label",
    "median_count_points_per_s",
    "min_count_points_per_s",
    "max_count_points_per_s",
    "median_count_runtime_s",
    "speedup_vs_proxy_baseline_median",
]


def read_single_csv(stdout: str) -> dict[str, str]:
    rows = list(csv.DictReader(StringIO(stdout)))
    if len(rows) != 1:
        raise RuntimeError(f"Unexpected CSV output: {stdout!r}")
    return rows[0]


def run_case(
    counts: tuple[int, int, int, int, int],
    *,
    tail_dimension: int,
    chunk: int,
    batch_prefixes: int,
    axis_order: int,
) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("OMP_PROC_BIND", "spread")
    env.setdefault("OMP_PLACES", "cores")
    completed = subprocess.run(
        [
            str(BINARY),
            str(counts[0]),
            str(counts[1]),
            str(counts[2]),
            str(counts[3]),
            str(counts[4]),
            str(tail_dimension),
            str(THREADS),
            str(chunk),
            str(batch_prefixes),
            str(STORAGE_RANK8),
            str(axis_order),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    return read_single_csv(completed.stdout)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    build_binary()

    rows: list[dict[str, object]] = []
    for case_label, proxy_label, counts, tail_dimension, chunk, batch_prefixes, axis_order in CASES:
        for repeat_index in range(1, REPEATS + 1):
            raw = run_case(
                counts,
                tail_dimension=tail_dimension,
                chunk=chunk,
                batch_prefixes=batch_prefixes,
                axis_order=axis_order,
            )
            rows.append(
                {
                    "case_label": case_label,
                    "proxy_label": proxy_label,
                    "repeat_index": repeat_index,
                    "tail_dimension": raw["tail_dimension"],
                    "axis_order": raw.get("axis_order", "asc"),
                    "prefix_chunk_points": raw["prefix_chunk_points"],
                    "batch_prefixes": raw["batch_prefixes"],
                    "total_points": raw["total_points"],
                    "inside_points": raw["inside_points"],
                    "count_runtime_s": raw["count_runtime_s"],
                    "count_points_per_s": raw["count_points_per_s"],
                }
            )

    benchmark_csv = RESULT_DIR / "q23456_strict_layout_repeat_benchmark.csv"
    write_csv(benchmark_csv, rows, BENCHMARK_FIELDS)

    medians: dict[tuple[str, str], float] = {}
    for proxy_label in sorted({str(row["proxy_label"]) for row in rows}):
        proxy_cases = sorted({str(row["case_label"]) for row in rows if row["proxy_label"] == proxy_label})
        baseline = proxy_cases[0]
        if proxy_label == "n24_proxy_31_cheapest":
            baseline = "cheap_baseline_asc_tail8_c128_b16"
        elif proxy_label == "n24_proxy_31_below_2pct":
            baseline = "below2_baseline_asc_tail9_c128_b16"
        baseline_rates = [
            float(row["count_points_per_s"])
            for row in rows
            if row["proxy_label"] == proxy_label and row["case_label"] == baseline
        ]
        medians[(proxy_label, baseline)] = statistics.median(baseline_rates)

    summary_rows: list[dict[str, object]] = []
    for case_label, proxy_label, *_rest in CASES:
        case_rows = [row for row in rows if row["case_label"] == case_label]
        rates = [float(row["count_points_per_s"]) for row in case_rows]
        runtimes = [float(row["count_runtime_s"]) for row in case_rows]
        baseline_key = (
            proxy_label,
            "cheap_baseline_asc_tail8_c128_b16"
            if proxy_label == "n24_proxy_31_cheapest"
            else "below2_baseline_asc_tail9_c128_b16",
        )
        median_rate = statistics.median(rates)
        summary_rows.append(
            {
                "case_label": case_label,
                "proxy_label": proxy_label,
                "median_count_points_per_s": f"{median_rate:.9e}",
                "min_count_points_per_s": f"{min(rates):.9e}",
                "max_count_points_per_s": f"{max(rates):.9e}",
                "median_count_runtime_s": f"{statistics.median(runtimes):.9f}",
                "speedup_vs_proxy_baseline_median": f"{median_rate / medians[baseline_key]:.6f}",
            }
        )

    summary_csv = RESULT_DIR / "q23456_strict_layout_repeat_summary.csv"
    write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)

    lines = [
        "q23456 strict layout repeat checks",
        "==================================",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['case_label']}: median_rate={row['median_count_points_per_s']}, "
            f"median_speedup={row['speedup_vs_proxy_baseline_median']}"
        )
    lines.extend(
        [
            "",
            f"Benchmark CSV written to {benchmark_csv}",
            f"Summary CSV written to {summary_csv}",
        ]
    )
    (RESULT_DIR / "q23456_strict_layout_repeat_run.log").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
