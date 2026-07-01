from __future__ import annotations

import csv
from io import StringIO
import os
from pathlib import Path
import subprocess

from problem5_core import build_binary, point_count, write_csv


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results"
BINARY = ROOT / "direct_tensor_midpoint_orthant_mixed_q26_tile_batch_avx2"

THREADS = 88
STORAGE_RANK8 = 1
AXIS_ASC = 0
AXIS_DESC = 1

# n=24 proxies shaped like the n=31 q23456 candidates, but cheap enough for
# layout benchmarks.
PROXY_CHEAP_31 = (1, 11, 9, 1, 2)
PROXY_2PCT_31 = (1, 12, 8, 1, 2)
N31_CHEAP = (1, 15, 12, 1, 2)
N31_2PCT = (1, 16, 10, 1, 3)

BROAD_CONFIGS = [
    ("asc_tail8_c128_b16", 8, 128, 16, AXIS_ASC),
    ("asc_tail9_c128_b16", 9, 128, 16, AXIS_ASC),
    ("asc_tail10_c128_b16", 10, 128, 16, AXIS_ASC),
    ("asc_tail11_c128_b16", 11, 128, 16, AXIS_ASC),
    ("asc_tail12_c128_b16", 12, 128, 16, AXIS_ASC),
    ("desc_tail8_c128_b16", 8, 128, 16, AXIS_DESC),
    ("desc_tail10_c128_b16", 10, 128, 16, AXIS_DESC),
    ("desc_tail12_c128_b16", 12, 128, 16, AXIS_DESC),
    ("asc_tail10_c512_b16", 10, 512, 16, AXIS_ASC),
    ("asc_tail10_c2048_b16", 10, 2048, 16, AXIS_ASC),
    ("asc_tail10_c512_b32", 10, 512, 32, AXIS_ASC),
    ("asc_tail11_c512_b16", 11, 512, 16, AXIS_ASC),
    ("asc_tail11_c512_b32", 11, 512, 32, AXIS_ASC),
    ("desc_tail10_c512_b16", 10, 512, 16, AXIS_DESC),
    ("desc_tail10_c512_b32", 10, 512, 32, AXIS_DESC),
]

CONFIRM_CONFIGS = [
    ("confirm_asc_tail9_c128_b16", 9, 128, 16, AXIS_ASC),
    ("confirm_asc_tail10_c128_b16", 10, 128, 16, AXIS_ASC),
    ("confirm_asc_tail10_c512_b16", 10, 512, 16, AXIS_ASC),
    ("confirm_asc_tail10_c2048_b16", 10, 2048, 16, AXIS_ASC),
    ("confirm_asc_tail11_c128_b16", 11, 128, 16, AXIS_ASC),
    ("confirm_asc_tail11_c512_b16", 11, 512, 16, AXIS_ASC),
    ("confirm_asc_tail10_c512_b32", 10, 512, 32, AXIS_ASC),
    ("confirm_asc_tail11_c512_b32", 11, 512, 32, AXIS_ASC),
    ("confirm_desc_tail10_c512_b16", 10, 512, 16, AXIS_DESC),
    ("confirm_desc_tail10_c512_b32", 10, 512, 32, AXIS_DESC),
]

BENCHMARK_FIELDS = [
    "case_label",
    "proxy_label",
    "dimension",
    "q_pattern",
    "tail_dimension",
    "axis_order",
    "prefix_chunk_points",
    "batch_prefixes",
    "storage_mode",
    "total_points",
    "inside_points",
    "estimate",
    "count_runtime_s",
    "count_points_per_s",
    "speedup_vs_proxy_baseline",
    "tail_points",
    "prefix_points",
    "tail_bytes",
    "unique_tail_values",
]

SUMMARY_FIELDS = [
    "proxy_label",
    "baseline_case_label",
    "baseline_count_points_per_s",
    "best_case_label",
    "best_count_points_per_s",
    "best_count_runtime_s",
    "best_speedup_vs_proxy_baseline",
]

PROJECTION_FIELDS = [
    "target_label",
    "q_pattern",
    "total_points",
    "rate_source",
    "rate_points_per_s",
    "estimated_runtime_s",
    "estimated_runtime_h",
    "speedup_vs_n30_measured_rate",
]


def q_pattern(counts: tuple[int, int, int, int, int]) -> str:
    return " ".join(f"{q}^{count}" for q, count in zip((2, 3, 4, 5, 6), counts))


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


def n30_measured_rate() -> float:
    path = RESULT_DIR / "q23456_selected_under30_direct_results.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["dimension"]) == 30:
                return float(row["direct_count_points_per_s"])
    raise RuntimeError("missing n=30 direct rate")


def append_case(
    rows: list[dict[str, object]],
    *,
    case_label: str,
    proxy_label: str,
    counts: tuple[int, int, int, int, int],
    baseline_rate: float,
    raw: dict[str, str],
) -> None:
    rate = float(raw["count_points_per_s"])
    rows.append(
        {
            "case_label": case_label,
            "proxy_label": proxy_label,
            "dimension": raw["dimension"],
            "q_pattern": q_pattern(counts),
            "tail_dimension": raw["tail_dimension"],
            "axis_order": raw.get("axis_order", "asc"),
            "prefix_chunk_points": raw["prefix_chunk_points"],
            "batch_prefixes": raw["batch_prefixes"],
            "storage_mode": raw["storage_mode"],
            "total_points": raw["total_points"],
            "inside_points": raw["inside_points"],
            "estimate": raw["estimate"],
            "count_runtime_s": raw["count_runtime_s"],
            "count_points_per_s": raw["count_points_per_s"],
            "speedup_vs_proxy_baseline": f"{rate / baseline_rate:.6f}",
            "tail_points": raw["tail_points"],
            "prefix_points": raw["prefix_points"],
            "tail_bytes": raw["tail_bytes"],
            "unique_tail_values": raw["unique_tail_values"],
        }
    )


def run_proxy(
    rows: list[dict[str, object]],
    *,
    proxy_label: str,
    counts: tuple[int, int, int, int, int],
    configs: list[tuple[str, int, int, int, int]],
) -> dict[str, object]:
    baseline_raw: dict[str, str] | None = None
    proxy_start = len(rows)
    for i, (case_label, tail_dimension, chunk, batch_prefixes, axis_order) in enumerate(configs):
        raw = run_case(
            counts,
            tail_dimension=tail_dimension,
            chunk=chunk,
            batch_prefixes=batch_prefixes,
            axis_order=axis_order,
        )
        if i == 0:
            baseline_raw = raw
        assert baseline_raw is not None
        append_case(
            rows,
            case_label=case_label,
            proxy_label=proxy_label,
            counts=counts,
            baseline_rate=float(baseline_raw["count_points_per_s"]),
            raw=raw,
        )
    proxy_rows = rows[proxy_start:]
    best = max(proxy_rows, key=lambda row: float(row["count_points_per_s"]))
    return {
        "proxy_label": proxy_label,
        "baseline_case_label": proxy_rows[0]["case_label"],
        "baseline_count_points_per_s": proxy_rows[0]["count_points_per_s"],
        "best_case_label": best["case_label"],
        "best_count_points_per_s": best["count_points_per_s"],
        "best_count_runtime_s": best["count_runtime_s"],
        "best_speedup_vs_proxy_baseline": best["speedup_vs_proxy_baseline"],
    }


def build_projection_rows(cheap_proxy_rate: float, below_2pct_proxy_rate: float) -> list[dict[str, object]]:
    n30_rate = n30_measured_rate()
    rows = []
    for label, counts, rate in [
        ("n31_cheapest_predicted_pass", N31_CHEAP, cheap_proxy_rate),
        ("n31_predicted_below_2pct", N31_2PCT, below_2pct_proxy_rate),
    ]:
        points = point_count(counts)
        runtime_s = points / rate
        rows.append(
            {
                "target_label": label,
                "q_pattern": q_pattern(counts),
                "total_points": points,
                "rate_source": "matching_proxy_best_layout_rate",
                "rate_points_per_s": f"{rate:.9e}",
                "estimated_runtime_s": f"{runtime_s:.6f}",
                "estimated_runtime_h": f"{runtime_s / 3600.0:.6f}",
                "speedup_vs_n30_measured_rate": f"{rate / n30_rate:.6f}",
            }
        )
    return rows


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    build_binary()

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    summaries.append(
        run_proxy(
            rows,
            proxy_label="n24_proxy_31_cheapest",
            counts=PROXY_CHEAP_31,
            configs=BROAD_CONFIGS,
        )
    )
    summaries.append(
        run_proxy(
            rows,
            proxy_label="n24_proxy_31_below_2pct",
            counts=PROXY_2PCT_31,
            configs=CONFIRM_CONFIGS,
        )
    )

    benchmark_csv = RESULT_DIR / "q23456_strict_layout_optimization_benchmark.csv"
    write_csv(benchmark_csv, rows, BENCHMARK_FIELDS)

    summary_csv = RESULT_DIR / "q23456_strict_layout_optimization_summary.csv"
    write_csv(summary_csv, summaries, SUMMARY_FIELDS)

    rate_by_proxy = {
        str(summary["proxy_label"]): float(summary["best_count_points_per_s"])
        for summary in summaries
    }
    projection_rows = build_projection_rows(
        rate_by_proxy["n24_proxy_31_cheapest"],
        rate_by_proxy["n24_proxy_31_below_2pct"],
    )
    projection_csv = RESULT_DIR / "q23456_strict_layout_optimization_projection.csv"
    write_csv(projection_csv, projection_rows, PROJECTION_FIELDS)

    lines = [
        "q23456 strict layout optimization",
        "=================================",
    ]
    for summary in summaries:
        lines.append(
            f"{summary['proxy_label']}: best={summary['best_case_label']}, "
            f"rate={summary['best_count_points_per_s']}, "
            f"speedup={summary['best_speedup_vs_proxy_baseline']}"
        )
    lines.append("")
    for row in projection_rows:
        lines.append(
            f"{row['target_label']}: pattern={row['q_pattern']}, "
            f"estimated_runtime_h={row['estimated_runtime_h']}, "
            f"rate_source={row['rate_source']}"
        )
    lines.extend(
        [
            "",
            f"Benchmark CSV written to {benchmark_csv}",
            f"Summary CSV written to {summary_csv}",
            f"Projection CSV written to {projection_csv}",
        ]
    )
    (RESULT_DIR / "q23456_strict_layout_optimization_run.log").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
