from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solution import (
    FRACTAL_NEWTON_MAX_ITER,
    FRACTAL_TOL,
    RESULT_DIR,
    ResourceMonitor,
    compute_parallel_fractal,
)

ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
ANALYSIS_DIR = RESULT_DIR / "analysis"

RAW_FIELDNAMES = [
    "profile_mode",
    "run_index",
    "sample_role",
    "grid_size",
    "render_grid",
    "workers",
    "warmup_runs",
    "timed_repeats",
    "timing_scope",
    "elapsed_seconds",
    "avg_cpu_percent",
    "peak_cpu_percent",
    "peak_rss_gib",
    "convergence_fraction",
    "mean_iterations",
    "root0_fraction",
    "root1_fraction",
    "root2_fraction",
    "throughput_mpoints_per_s",
]

AGGREGATE_FIELDNAMES = [
    "profile_mode",
    "grid_size",
    "render_grid",
    "workers",
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
    "throughput_mpoints_per_s",
    "throughput_mpoints_per_s_mean",
    "throughput_mpoints_per_s_std",
    "throughput_mpoints_per_s_median",
    "throughput_mpoints_per_s_iqr",
    "throughput_mpoints_per_s_best",
    "avg_cpu_percent",
    "peak_cpu_percent",
    "peak_rss_gib",
    "convergence_fraction",
    "mean_iterations",
    "root0_fraction",
    "root1_fraction",
    "root2_fraction",
]


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def deduplicate_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    unique: list[int] = []
    for value in values:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def profile_case(grid_size: int, render_grid: int, workers: int) -> dict[str, float]:
    monitor = ResourceMonitor()
    timer_start = time.perf_counter()
    monitor.start()
    summary = compute_parallel_fractal(
        "newton",
        grid_size=grid_size,
        render_grid=render_grid,
        tol=FRACTAL_TOL,
        max_iter=FRACTAL_NEWTON_MAX_ITER,
        workers=workers,
    )
    resources = monitor.stop()
    elapsed_seconds = time.perf_counter() - timer_start
    total_points = float(grid_size * grid_size)
    return {
        "grid_size": float(grid_size),
        "render_grid": float(render_grid),
        "workers": float(workers),
        "elapsed_seconds": elapsed_seconds,
        "avg_cpu_percent": float(resources["avg_cpu_percent"]),
        "peak_cpu_percent": float(resources["peak_cpu_percent"]),
        "peak_rss_gib": float(resources["peak_rss_gib"]),
        "convergence_fraction": float(summary["convergence_fraction"]),
        "mean_iterations": float(summary["mean_iterations"]),
        "root0_fraction": float(summary["root0_fraction"]),
        "root1_fraction": float(summary["root1_fraction"]),
        "root2_fraction": float(summary["root2_fraction"]),
        "throughput_mpoints_per_s": total_points / elapsed_seconds / 1.0e6,
    }


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


def aggregate_case(
    profile_mode: str,
    rows: list[dict[str, Any]],
    warmup_runs: int,
    timed_repeats: int,
    timing_scope: str,
) -> dict[str, Any]:
    first = rows[0]
    aggregate: dict[str, Any] = {
        "profile_mode": profile_mode,
        "grid_size": first["grid_size"],
        "render_grid": first["render_grid"],
        "workers": first["workers"],
        "warmup_runs": warmup_runs,
        "timed_repeats": timed_repeats,
        "sample_count": len(rows),
        "timing_scope": timing_scope,
    }
    elapsed_stats = numeric_stats(
        [float(row["elapsed_seconds"]) for row in rows], best_mode="min"
    )
    throughput_stats = numeric_stats(
        [float(row["throughput_mpoints_per_s"]) for row in rows],
        best_mode="max",
    )
    aggregate["elapsed_seconds"] = elapsed_stats["mean"]
    aggregate["throughput_mpoints_per_s"] = throughput_stats["mean"]
    for suffix, value in elapsed_stats.items():
        aggregate[f"elapsed_seconds_{suffix}"] = value
    for suffix, value in throughput_stats.items():
        aggregate[f"throughput_mpoints_per_s_{suffix}"] = value

    for key in [
        "avg_cpu_percent",
        "peak_cpu_percent",
        "peak_rss_gib",
        "convergence_fraction",
        "mean_iterations",
        "root0_fraction",
        "root1_fraction",
        "root2_fraction",
    ]:
        aggregate[key] = float(np.mean([float(row[key]) for row in rows]))
    return aggregate


def run_profile_repeats(
    profile_mode: str,
    grid_size: int,
    render_grid: int,
    workers: int,
    warmup_runs: int,
    timed_repeats: int,
    timing_scope: str,
) -> list[dict[str, Any]]:
    for _ in range(warmup_runs):
        profile_case(grid_size=grid_size, render_grid=render_grid, workers=workers)

    rows: list[dict[str, Any]] = []
    for run_index in range(1, timed_repeats + 1):
        row: dict[str, Any] = profile_case(
            grid_size=grid_size,
            render_grid=render_grid,
            workers=workers,
        )
        row["profile_mode"] = profile_mode
        row["run_index"] = run_index
        row["sample_role"] = "timed"
        row["warmup_runs"] = warmup_runs
        row["timed_repeats"] = timed_repeats
        row["timing_scope"] = timing_scope
        rows.append(row)
    return rows


def plot_profile(
    worker_rows: list[dict[str, Any]],
    grid_rows: list[dict[str, Any]],
    output_png: Path,
) -> None:
    worker_counts = np.array([row["workers"] for row in worker_rows], dtype=np.float64)
    worker_elapsed = np.array(
        [row["elapsed_seconds"] for row in worker_rows], dtype=np.float64
    )
    worker_elapsed_err = np.array(
        [row.get("elapsed_seconds_std", 0.0) for row in worker_rows], dtype=np.float64
    )
    worker_speedup = worker_elapsed[0] / worker_elapsed
    ideal_speedup = worker_counts / worker_counts[0]

    grid_sizes = np.array([row["grid_size"] for row in grid_rows], dtype=np.float64)
    grid_elapsed = np.array(
        [row["elapsed_seconds"] for row in grid_rows], dtype=np.float64
    )
    grid_elapsed_err = np.array(
        [row.get("elapsed_seconds_std", 0.0) for row in grid_rows], dtype=np.float64
    )
    grid_memory = np.array([row["peak_rss_gib"] for row in grid_rows], dtype=np.float64)
    grid_throughput = np.array(
        [row["throughput_mpoints_per_s"] for row in grid_rows], dtype=np.float64
    )
    grid_throughput_err = np.array(
        [row.get("throughput_mpoints_per_s_std", 0.0) for row in grid_rows],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)

    axes[0, 0].errorbar(
        worker_counts,
        worker_elapsed,
        yerr=worker_elapsed_err,
        marker="o",
        linewidth=2.0,
        capsize=4,
        color="#1f77b4",
    )
    axes[0, 0].set_title("CPU worker sweep")
    axes[0, 0].set_xlabel("workers")
    axes[0, 0].set_ylabel("elapsed time (s)")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(
        worker_counts,
        worker_speedup,
        marker="o",
        linewidth=2.0,
        color="#d62728",
        label="measured",
    )
    axes[0, 1].plot(
        worker_counts,
        ideal_speedup,
        linestyle="--",
        linewidth=1.5,
        color="#7f7f7f",
        label="ideal",
    )
    axes[0, 1].set_title("CPU parallel speedup")
    axes[0, 1].set_xlabel("workers")
    axes[0, 1].set_ylabel("speedup vs 1 worker")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(frameon=False)

    axes[1, 0].errorbar(
        grid_sizes,
        grid_elapsed,
        yerr=grid_elapsed_err,
        marker="o",
        linewidth=2.0,
        capsize=4,
        color="#2ca02c",
    )
    axes[1, 0].set_title("CPU grid sweep")
    axes[1, 0].set_xlabel("compute grid size")
    axes[1, 0].set_ylabel("elapsed time (s)")
    axes[1, 0].grid(alpha=0.25)

    ax_mem = axes[1, 1]
    ax_mem.plot(
        grid_sizes,
        grid_memory,
        marker="o",
        linewidth=2.0,
        color="#9467bd",
        label="peak RSS",
    )
    ax_mem.set_title("CPU memory / throughput")
    ax_mem.set_xlabel("compute grid size")
    ax_mem.set_ylabel("peak RSS (GiB)", color="#9467bd")
    ax_mem.tick_params(axis="y", labelcolor="#9467bd")
    ax_mem.grid(alpha=0.25)
    ax_tp = ax_mem.twinx()
    ax_tp.errorbar(
        grid_sizes,
        grid_throughput,
        yerr=grid_throughput_err,
        marker="s",
        linewidth=2.0,
        capsize=4,
        color="#ff7f0e",
        label="throughput",
    )
    ax_tp.set_ylabel("throughput (Mpoints/s)", color="#ff7f0e")
    ax_tp.tick_params(axis="y", labelcolor="#ff7f0e")

    fig.suptitle("Problem 1 CPU supplementary profile for Newton fractal", fontsize=15)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def write_csv(
    rows: list[dict[str, Any]], output_csv: Path, fieldnames: list[str]
) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supplementary CPU profile for HW/04 Problem 1."
    )
    parser.add_argument("--worker-grid", type=int, default=4096)
    parser.add_argument("--worker-render-grid", type=int, default=1024)
    parser.add_argument("--worker-counts", default="1,8,16,32,64")
    parser.add_argument("--grid-sizes", default="1024,2048,4096,8192")
    parser.add_argument(
        "--grid-workers", type=int, default=min(64, os.cpu_count() or 8)
    )
    parser.add_argument("--output-prefix", default="problem1_cpu_profile")
    parser.add_argument(
        "--repeats", type=int, default=1, help="Timed repeats per profile case."
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Untimed warm-up runs before each profile case.",
    )
    parser.add_argument(
        "--timing-scope",
        default="full_compute_plus_resource_monitor",
        help="Free-form label describing what elapsed_seconds includes.",
    )
    parser.add_argument(
        "--include-cpu-count",
        dest="include_cpu_count",
        action="store_true",
        default=True,
        help="Append os.cpu_count() to the worker sweep.",
    )
    parser.add_argument(
        "--no-include-cpu-count",
        dest="include_cpu_count",
        action="store_false",
        help="Use only the worker counts explicitly supplied by --worker-counts.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")

    worker_counts = parse_int_list(args.worker_counts)
    if args.include_cpu_count:
        cpu_count = os.cpu_count() or max(worker_counts)
        worker_counts.append(cpu_count)
    worker_counts = deduplicate_preserve_order(worker_counts)
    grid_sizes = parse_int_list(args.grid_sizes)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    worker_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []

    for workers in worker_counts:
        samples = run_profile_repeats(
            profile_mode="worker_sweep",
            grid_size=args.worker_grid,
            render_grid=args.worker_render_grid,
            workers=workers,
            warmup_runs=args.warmup_runs,
            timed_repeats=args.repeats,
            timing_scope=args.timing_scope,
        )
        raw_rows.extend(samples)
        worker_rows.append(
            aggregate_case(
                "worker_sweep",
                samples,
                args.warmup_runs,
                args.repeats,
                args.timing_scope,
            )
        )

    for grid_size in grid_sizes:
        samples = run_profile_repeats(
            profile_mode="grid_sweep",
            grid_size=grid_size,
            render_grid=grid_size // 4,
            workers=args.grid_workers,
            warmup_runs=args.warmup_runs,
            timed_repeats=args.repeats,
            timing_scope=args.timing_scope,
        )
        raw_rows.extend(samples)
        grid_rows.append(
            aggregate_case(
                "grid_sweep", samples, args.warmup_runs, args.repeats, args.timing_scope
            )
        )

    aggregate_rows = worker_rows + grid_rows
    raw_csv = ANALYSIS_DIR / f"{args.output_prefix}_raw.csv"
    output_csv = ANALYSIS_DIR / f"{args.output_prefix}.csv"
    output_png = ANALYSIS_DIR / f"{args.output_prefix}.png"
    write_csv(raw_rows, raw_csv, RAW_FIELDNAMES)
    write_csv(aggregate_rows, output_csv, AGGREGATE_FIELDNAMES)
    plot_profile(worker_rows, grid_rows, output_png)


if __name__ == "__main__":
    main()
