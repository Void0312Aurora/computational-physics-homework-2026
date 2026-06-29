from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
RESULT_DIR = ROOT / "result" / "analysis"
BINARY = ROOT / "problem1_cpu_ultra"

RAW_FIELDNAMES = [
    "run_prefix",
    "sample_role",
    "run_index",
    "compute_grid",
    "render_grid",
    "factor",
    "tile_rows",
    "threads",
    "max_iter",
    "tol",
    "warmup_runs",
    "timed_repeats",
    "timing_scope",
    "elapsed_seconds",
    "throughput_gpoints_per_s",
    "peak_rss_gib",
    "convergence_fraction",
    "mean_iterations",
    "root0_fraction",
    "root1_fraction",
    "root2_fraction",
    "source_command",
    "source_csv",
]

SUMMARY_FIELDNAMES = [
    "run_prefix",
    "compute_grid",
    "render_grid",
    "factor",
    "tile_rows",
    "threads",
    "max_iter",
    "tol",
    "warmup_runs",
    "timed_repeats",
    "sample_count",
    "timing_scope",
    "elapsed_seconds",
    "elapsed_seconds_mean",
    "elapsed_seconds_std",
    "elapsed_seconds_median",
    "elapsed_seconds_iqr",
    "elapsed_seconds_best",
    "throughput_gpoints_per_s",
    "throughput_gpoints_per_s_mean",
    "throughput_gpoints_per_s_std",
    "throughput_gpoints_per_s_median",
    "throughput_gpoints_per_s_iqr",
    "throughput_gpoints_per_s_best",
    "peak_rss_gib",
    "convergence_fraction",
    "mean_iterations",
    "root0_fraction",
    "root1_fraction",
    "root2_fraction",
    "source_command",
    "source_csv",
]


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def load_single_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one result row in {csv_path}")
    return rows[0]


def numeric_stats(values: list[float], best_mode: str) -> dict[str, float]:
    array = np.array(values, dtype=np.float64)
    best = float(np.min(array) if best_mode == "min" else np.max(array))
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "best": best,
    }


def run_prefix_for(
    base_prefix: str, compute_grid: int, run_index: int, repeats: int, sample_role: str
) -> str:
    prefix = f"{base_prefix}_{compute_grid}"
    if sample_role == "warmup":
        return f"{prefix}_warmup{run_index:02d}"
    if repeats > 1:
        return f"{prefix}_r{run_index:02d}"
    return prefix


def run_cpu_ultra(
    compute_grid: int,
    run_prefix: str,
    args: argparse.Namespace,
) -> tuple[dict[str, str], list[str], Path]:
    cmd = [
        str(BINARY),
        "--compute-grid",
        str(compute_grid),
        "--render-grid",
        str(args.render_grid),
        "--tile-rows",
        str(args.tile_rows),
        "--threads",
        str(args.threads),
        "--max-iter",
        str(args.max_iter),
        "--tol",
        str(args.tol),
        "--output-prefix",
        run_prefix,
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    source_csv = RESULT_DIR / f"{run_prefix}.csv"
    row = load_single_row(source_csv)
    row["run_prefix"] = run_prefix
    row["source_command"] = shlex.join(["./problem1_cpu_ultra", *cmd[1:]])
    row["source_csv"] = source_csv.relative_to(ROOT).as_posix()
    return row, cmd, source_csv


def timed_summary(
    base_run_prefix: str,
    rows: list[dict[str, Any]],
    warmup_runs: int,
    timed_repeats: int,
    timing_scope: str,
) -> dict[str, Any]:
    first = rows[0]
    summary: dict[str, Any] = {
        "run_prefix": base_run_prefix,
        "compute_grid": first["compute_grid"],
        "render_grid": first["render_grid"],
        "factor": first["factor"],
        "tile_rows": first["tile_rows"],
        "threads": first["threads"],
        "max_iter": first["max_iter"],
        "tol": first["tol"],
        "warmup_runs": warmup_runs,
        "timed_repeats": timed_repeats,
        "sample_count": len(rows),
        "timing_scope": timing_scope,
    }

    elapsed_stats = numeric_stats(
        [float(row["elapsed_seconds"]) for row in rows], best_mode="min"
    )
    throughput_stats = numeric_stats(
        [float(row["throughput_gpoints_per_s"]) for row in rows],
        best_mode="max",
    )
    summary["elapsed_seconds"] = elapsed_stats["mean"]
    summary["throughput_gpoints_per_s"] = throughput_stats["mean"]
    for suffix, value in elapsed_stats.items():
        summary[f"elapsed_seconds_{suffix}"] = value
    for suffix, value in throughput_stats.items():
        summary[f"throughput_gpoints_per_s_{suffix}"] = value

    for key in [
        "peak_rss_gib",
        "convergence_fraction",
        "mean_iterations",
        "root0_fraction",
        "root1_fraction",
        "root2_fraction",
    ]:
        summary[key] = float(np.mean([float(row[key]) for row in rows]))
    summary["source_command"] = " | ".join(str(row["source_command"]) for row in rows)
    summary["source_csv"] = " | ".join(str(row["source_csv"]) for row in rows)
    return summary


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a benchmark sweep for problem1_cpu_ultra."
    )
    parser.add_argument("--compute-grids", default="20000,40000,80000")
    parser.add_argument("--render-grid", type=int, default=5000)
    parser.add_argument("--tile-rows", type=int, default=64)
    parser.add_argument("--threads", type=int, default=88)
    parser.add_argument("--max-iter", type=int, default=55)
    parser.add_argument("--tol", type=float, default=5.0e-7)
    parser.add_argument("--output-prefix", default="problem1_cpu_ultra_bench")
    parser.add_argument(
        "--repeats", type=int, default=1, help="Timed repeats per compute-grid case."
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Untimed warm-up runs per compute-grid case.",
    )
    parser.add_argument(
        "--timing-scope",
        default="process_full_run",
        help="Free-form label describing what elapsed_seconds includes.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for compute_grid in parse_int_list(args.compute_grids):
        for run_index in range(1, args.warmup_runs + 1):
            run_prefix = run_prefix_for(
                args.output_prefix, compute_grid, run_index, args.repeats, "warmup"
            )
            row, _, _ = run_cpu_ultra(compute_grid, run_prefix, args)
            row["sample_role"] = "warmup"
            row["run_index"] = run_index
            row["warmup_runs"] = args.warmup_runs
            row["timed_repeats"] = args.repeats
            row["timing_scope"] = args.timing_scope
            raw_rows.append(row)

        timed_rows: list[dict[str, Any]] = []
        for run_index in range(1, args.repeats + 1):
            run_prefix = run_prefix_for(
                args.output_prefix, compute_grid, run_index, args.repeats, "timed"
            )
            row, _, _ = run_cpu_ultra(compute_grid, run_prefix, args)
            row["sample_role"] = "timed"
            row["run_index"] = run_index
            row["warmup_runs"] = args.warmup_runs
            row["timed_repeats"] = args.repeats
            row["timing_scope"] = args.timing_scope
            raw_rows.append(row)
            timed_rows.append(row)

        base_run_prefix = f"{args.output_prefix}_{compute_grid}"
        summary_rows.append(
            timed_summary(
                base_run_prefix,
                timed_rows,
                warmup_runs=args.warmup_runs,
                timed_repeats=args.repeats,
                timing_scope=args.timing_scope,
            )
        )

    samples_path = RESULT_DIR / f"{args.output_prefix}_samples.csv"
    summary_path = RESULT_DIR / f"{args.output_prefix}_summary.csv"
    write_csv(raw_rows, samples_path, RAW_FIELDNAMES)
    write_csv(summary_rows, summary_path, SUMMARY_FIELDNAMES)


if __name__ == "__main__":
    main()
