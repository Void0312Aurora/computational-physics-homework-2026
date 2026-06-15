from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

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


def plot_profile(
    worker_rows: list[dict[str, float]],
    grid_rows: list[dict[str, float]],
    output_png: Path,
) -> None:
    worker_counts = np.array([row["workers"] for row in worker_rows], dtype=np.float64)
    worker_elapsed = np.array([row["elapsed_seconds"] for row in worker_rows], dtype=np.float64)
    worker_speedup = worker_elapsed[0] / worker_elapsed
    ideal_speedup = worker_counts / worker_counts[0]

    grid_sizes = np.array([row["grid_size"] for row in grid_rows], dtype=np.float64)
    grid_elapsed = np.array([row["elapsed_seconds"] for row in grid_rows], dtype=np.float64)
    grid_memory = np.array([row["peak_rss_gib"] for row in grid_rows], dtype=np.float64)
    grid_throughput = np.array([row["throughput_mpoints_per_s"] for row in grid_rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)

    axes[0, 0].plot(worker_counts, worker_elapsed, marker="o", linewidth=2.0, color="#1f77b4")
    axes[0, 0].set_title("CPU worker sweep")
    axes[0, 0].set_xlabel("workers")
    axes[0, 0].set_ylabel("elapsed time (s)")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(worker_counts, worker_speedup, marker="o", linewidth=2.0, color="#d62728", label="measured")
    axes[0, 1].plot(worker_counts, ideal_speedup, linestyle="--", linewidth=1.5, color="#7f7f7f", label="ideal")
    axes[0, 1].set_title("CPU parallel speedup")
    axes[0, 1].set_xlabel("workers")
    axes[0, 1].set_ylabel("speedup vs 1 worker")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(grid_sizes, grid_elapsed, marker="o", linewidth=2.0, color="#2ca02c")
    axes[1, 0].set_title("CPU grid sweep")
    axes[1, 0].set_xlabel("compute grid size")
    axes[1, 0].set_ylabel("elapsed time (s)")
    axes[1, 0].grid(alpha=0.25)

    ax_mem = axes[1, 1]
    ax_mem.plot(grid_sizes, grid_memory, marker="o", linewidth=2.0, color="#9467bd", label="peak RSS")
    ax_mem.set_title("CPU memory / throughput")
    ax_mem.set_xlabel("compute grid size")
    ax_mem.set_ylabel("peak RSS (GiB)", color="#9467bd")
    ax_mem.tick_params(axis="y", labelcolor="#9467bd")
    ax_mem.grid(alpha=0.25)
    ax_tp = ax_mem.twinx()
    ax_tp.plot(grid_sizes, grid_throughput, marker="s", linewidth=2.0, color="#ff7f0e", label="throughput")
    ax_tp.set_ylabel("throughput (Mpoints/s)", color="#ff7f0e")
    ax_tp.tick_params(axis="y", labelcolor="#ff7f0e")

    fig.suptitle("Problem 1 CPU supplementary profile for Newton fractal", fontsize=15)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def write_csv(rows: list[dict[str, float]], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "profile_mode",
                "grid_size",
                "render_grid",
                "workers",
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Supplementary CPU profile for HW/04 Problem 1.")
    parser.add_argument("--worker-grid", type=int, default=4096)
    parser.add_argument("--worker-render-grid", type=int, default=1024)
    parser.add_argument("--worker-counts", default="1,8,16,32,64")
    parser.add_argument("--grid-sizes", default="1024,2048,4096,8192")
    parser.add_argument("--grid-workers", type=int, default=min(64, os.cpu_count() or 8))
    parser.add_argument("--output-prefix", default="problem1_cpu_profile")
    args = parser.parse_args()

    worker_counts = parse_int_list(args.worker_counts)
    cpu_count = os.cpu_count() or max(worker_counts)
    worker_counts.append(cpu_count)
    worker_counts = deduplicate_preserve_order(worker_counts)
    grid_sizes = parse_int_list(args.grid_sizes)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    worker_rows: list[dict[str, float]] = []
    grid_rows: list[dict[str, float]] = []

    for workers in worker_counts:
        row = profile_case(
            grid_size=args.worker_grid,
            render_grid=args.worker_render_grid,
            workers=workers,
        )
        row["profile_mode"] = "worker_sweep"  # type: ignore[assignment]
        worker_rows.append(row)

    for grid_size in grid_sizes:
        row = profile_case(
            grid_size=grid_size,
            render_grid=grid_size // 4,
            workers=args.grid_workers,
        )
        row["profile_mode"] = "grid_sweep"  # type: ignore[assignment]
        grid_rows.append(row)

    rows = worker_rows + grid_rows
    output_csv = ANALYSIS_DIR / f"{args.output_prefix}.csv"
    output_png = ANALYSIS_DIR / f"{args.output_prefix}.png"
    write_csv(rows, output_csv)
    plot_profile(worker_rows, grid_rows, output_png)


if __name__ == "__main__":
    main()
