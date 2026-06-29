from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
RESULT_DIR = ROOT / "result"
ANALYSIS_DIR = RESULT_DIR / "analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Problem 1 scaling and precision-frontier curves.")
    parser.add_argument(
        "--extreme-summary",
        type=Path,
        default=None,
        help="Path to an extreme_stats_summary.csv file. Defaults to the latest matching analysis output.",
    )
    parser.add_argument(
        "--sparse-validation",
        type=Path,
        default=None,
        help="Path to a sparse_reference_validation.csv file. Defaults to the latest tracked run.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="problem1_frontier_curves",
        help="Prefix for the analysis output directory.",
    )
    return parser.parse_args()


def latest_matching(pattern: str) -> Path:
    matches = sorted(ANALYSIS_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern!r} under {ANALYSIS_DIR}.")
    return max(matches, key=lambda path: path.stat().st_mtime)


def make_run_dir(base_dir: Path, output_prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = base_dir / f"{output_prefix}__{stamp}"
    attempt = 1
    while candidate.exists():
        attempt += 1
        candidate = base_dir / f"{output_prefix}__{stamp}_{attempt:02d}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_scaling_rows(path: Path) -> list[dict[str, float | int | str]]:
    rows = read_csv_rows(path)
    converted: list[dict[str, float | int | str]] = []
    for row in rows:
        converted.append(
            {
                "prefix": row["prefix"],
                "compute_grid": int(row["compute_grid"]),
                "compute_tile": int(row["compute_tile"]),
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "peak_gpu_mem_gib": float(row["peak_gpu_mem_gib"]),
                "convergence_fraction": float(row["convergence_fraction"]),
                "mean_iterations": float(row["mean_iterations"]),
                "grid_to_float32_ulp_ratio": float(row["grid_to_float32_ulp_ratio"]),
                "effective_points_per_second": float(row["effective_points_per_second"]),
            }
        )
    converted.sort(key=lambda row: int(row["compute_grid"]))
    return converted


def load_precision_rows(path: Path) -> list[dict[str, float | int | str]]:
    rows = read_csv_rows(path)
    converted: list[dict[str, float | int | str]] = []
    for row in rows:
        converted.append(
            {
                "global_grid": int(row["global_grid"]),
                "patch_grid": int(row["patch_grid"]),
                "backend": row["backend"],
                "dx_to_float32_ulp_ratio": float(row["dx_to_float32_ulp_ratio"]),
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "root_diff_vs_fp64": int(row["root_diff_vs_fp64"]),
                "iter_diff_vs_fp64": int(row["iter_diff_vs_fp64"]),
                "root_diff_ratio_vs_fp64": float(row["root_diff_ratio_vs_fp64"]),
                "iter_diff_ratio_vs_fp64": float(row["iter_diff_ratio_vs_fp64"]),
                "unique_x_ratio": float(row["unique_x_ratio"]),
                "unique_y_ratio": float(row["unique_y_ratio"]),
            }
        )
    converted.sort(key=lambda row: float(row["dx_to_float32_ulp_ratio"]))
    return converted


def summarize_scaling_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, object]]:
    base_grid = int(rows[0]["compute_grid"])
    base_time = float(rows[0]["elapsed_seconds"])
    summary: list[dict[str, object]] = []
    for row in rows:
        grid = int(row["compute_grid"])
        elapsed = float(row["elapsed_seconds"])
        quadratic_reference = base_time * (grid / base_grid) ** 2
        summary.append(
            {
                "compute_grid": grid,
                "compute_tile": int(row["compute_tile"]),
                "elapsed_seconds": elapsed,
                "quadratic_reference_seconds": quadratic_reference,
                "relative_to_quadratic": elapsed / quadratic_reference,
                "peak_gpu_mem_gib": float(row["peak_gpu_mem_gib"]),
                "throughput_gpoints_per_second": float(row["effective_points_per_second"]) / 1.0e9,
                "grid_to_float32_ulp_ratio": float(row["grid_to_float32_ulp_ratio"]),
                "convergence_fraction": float(row["convergence_fraction"]),
                "mean_iterations": float(row["mean_iterations"]),
            }
        )
    return summary


def summarize_precision_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for row in rows:
        summary.append(
            {
                "global_grid": int(row["global_grid"]),
                "backend": str(row["backend"]),
                "dx_to_float32_ulp_ratio": float(row["dx_to_float32_ulp_ratio"]),
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "root_diff_vs_fp64": int(row["root_diff_vs_fp64"]),
                "root_diff_percent_vs_fp64": 100.0 * float(row["root_diff_ratio_vs_fp64"]),
                "iter_diff_percent_vs_fp64": 100.0 * float(row["iter_diff_ratio_vs_fp64"]),
                "unique_x_percent": 100.0 * float(row["unique_x_ratio"]),
                "unique_y_percent": 100.0 * float(row["unique_y_ratio"]),
            }
        )
    return summary


def style_palette() -> dict[str, str]:
    return {
        "fp32_triton": "#c4473a",
        "doublefloat_triton_base": "#2b6cb0",
        "doublefloat_triton_t8": "#2f855a",
    }


def backend_label(backend: str) -> str:
    labels = {
        "fp32_triton": "fp32 Triton",
        "doublefloat_triton_base": "double-float Triton",
        "doublefloat_triton_t8": "double-float + t8 replay",
    }
    return labels.get(backend, backend)


def short_grid_label(grid: float) -> str:
    if grid >= 1_000_000:
        return f"{int(grid / 1_000_000)}m"
    return f"{int(grid / 1000)}k"


def plot_scaling(rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    grids = np.array([int(row["compute_grid"]) for row in rows], dtype=np.float64)
    elapsed = np.array([float(row["elapsed_seconds"]) for row in rows], dtype=np.float64)
    throughput = np.array([float(row["effective_points_per_second"]) / 1.0e9 for row in rows], dtype=np.float64)
    peak_mem = np.array([float(row["peak_gpu_mem_gib"]) for row in rows], dtype=np.float64)
    ulp_ratio = np.array([float(row["grid_to_float32_ulp_ratio"]) for row in rows], dtype=np.float64)

    quadratic = elapsed[0] * (grids / grids[0]) ** 2

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10), constrained_layout=True)
    axes = axes.ravel()

    axes[0].loglog(grids, elapsed, marker="o", linewidth=2.2, color="#144b7f", label="Measured")
    axes[0].loglog(grids, quadratic, linestyle="--", linewidth=1.8, color="#d17b0f", label="N^2 reference")
    axes[0].set_title("Scaling: elapsed time vs compute grid")
    axes[0].set_xlabel("Compute grid")
    axes[0].set_ylabel("Elapsed time (s)")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    axes[1].semilogx(grids, throughput, marker="o", linewidth=2.2, color="#2f855a")
    axes[1].set_title("Throughput stability")
    axes[1].set_xlabel("Compute grid")
    axes[1].set_ylabel("Effective throughput (Gpoints/s)")
    axes[1].grid(True, which="both", alpha=0.25)

    axes[2].semilogx(grids, peak_mem, marker="o", linewidth=2.2, color="#6b46c1")
    axes[2].set_title("Peak GPU memory")
    axes[2].set_xlabel("Compute grid")
    axes[2].set_ylabel("Peak GPU memory (GiB)")
    axes[2].grid(True, which="both", alpha=0.25)

    axes[3].semilogx(grids, ulp_ratio, marker="o", linewidth=2.2, color="#c05621")
    axes[3].axhline(1.0, linestyle="--", linewidth=1.2, color="#444444", label="dx = ULP32")
    axes[3].set_title("Distance from the fp32 coordinate limit")
    axes[3].set_xlabel("Compute grid")
    axes[3].set_ylabel("dx / ULP32")
    axes[3].grid(True, which="both", alpha=0.25)
    axes[3].legend()

    for ax in axes:
        for grid, value in zip(grids, ax.lines[0].get_ydata(), strict=True):
            ax.annotate(short_grid_label(grid), (grid, value), textcoords="offset points", xytext=(5, 5), fontsize=8)

    fig.suptitle("Problem 1 Triton Newton scaling frontier", fontsize=15)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_precision(rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    palette = style_palette()
    backends = ["fp32_triton", "doublefloat_triton_base", "doublefloat_triton_t8"]
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 10.5), sharex=True, constrained_layout=True)

    for backend in backends:
        backend_rows = [row for row in rows if row["backend"] == backend]
        x = np.array([float(row["dx_to_float32_ulp_ratio"]) for row in backend_rows], dtype=np.float64)
        root_diff_pct = np.array([100.0 * float(row["root_diff_ratio_vs_fp64"]) for row in backend_rows], dtype=np.float64)
        iter_diff_pct = np.array([100.0 * float(row["iter_diff_ratio_vs_fp64"]) for row in backend_rows], dtype=np.float64)
        unique_y_pct = np.array([100.0 * float(row["unique_y_ratio"]) for row in backend_rows], dtype=np.float64)
        color = palette[backend]
        label = backend_label(backend)

        axes[0].plot(x, root_diff_pct, marker="o", linewidth=2.2, color=color, label=label)
        axes[1].plot(x, iter_diff_pct, marker="o", linewidth=2.2, color=color, label=label)
        axes[2].plot(x, unique_y_pct, marker="o", linewidth=2.2, color=color, label=label)

    titles = [
        "Root mismatch frontier",
        "Iteration mismatch frontier",
        "Coordinate uniqueness frontier",
    ]
    ylabels = [
        "Pixels differing from fp64 (%)",
        "Iteration-map differences (%)",
        "Unique y coordinates preserved (%)",
    ]
    for ax, title, ylabel in zip(axes, titles, ylabels, strict=True):
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.axvline(1.0, linestyle="--", linewidth=1.2, color="#444444")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)

    axes[-1].set_xlabel("dx / ULP32")
    axes[0].legend(loc="upper left")
    fig.suptitle("Problem 1 precision frontier near the fp32 coordinate limit", fontsize=15)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_summary_log(
    path: Path,
    extreme_path: Path,
    sparse_path: Path,
    scaling_rows: list[dict[str, object]],
    precision_rows: list[dict[str, object]],
) -> None:
    max_scaling = scaling_rows[-1]
    fp32_rows = [row for row in precision_rows if row["backend"] == "fp32_triton"]
    t8_rows = [row for row in precision_rows if row["backend"] == "doublefloat_triton_t8"]
    hardest_fp32 = min(fp32_rows, key=lambda row: float(row["dx_to_float32_ulp_ratio"]))
    hardest_t8 = min(t8_rows, key=lambda row: float(row["dx_to_float32_ulp_ratio"]))

    with path.open("w", encoding="utf-8") as fh:
        fh.write("Problem 1 frontier analysis\n")
        fh.write(f"extreme_summary_source={extreme_path}\n")
        fh.write(f"sparse_validation_source={sparse_path}\n")
        fh.write("\n")
        fh.write(
            "largest_stats_only_run="
            f"{int(max_scaling['compute_grid'])} "
            f"elapsed={float(max_scaling['elapsed_seconds']):.3f}s "
            f"throughput={float(max_scaling['throughput_gpoints_per_second']):.3f} Gpoints/s "
            f"dx_to_ulp32={float(max_scaling['grid_to_float32_ulp_ratio']):.3f}\n"
        )
        fh.write(
            "hardest_fp32_patch="
            f"global_grid={int(hardest_fp32['global_grid'])} "
            f"dx_to_ulp32={float(hardest_fp32['dx_to_float32_ulp_ratio']):.3f} "
            f"root_diff={int(hardest_fp32['root_diff_vs_fp64'])} "
            f"unique_y={float(hardest_fp32['unique_y_percent']):.3f}%\n"
        )
        fh.write(
            "hardest_t8_patch="
            f"global_grid={int(hardest_t8['global_grid'])} "
            f"dx_to_ulp32={float(hardest_t8['dx_to_float32_ulp_ratio']):.3f} "
            f"root_diff={int(hardest_t8['root_diff_vs_fp64'])} "
            f"unique_y={float(hardest_t8['unique_y_percent']):.3f}%\n"
        )


def main() -> None:
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    extreme_path = args.extreme_summary or latest_matching("problem1_gpu_triton_extreme_stats*/extreme_stats_summary.csv")
    sparse_path = args.sparse_validation or latest_matching("sparse_reference_validation_tracked*/sparse_reference_validation.csv")

    scaling_rows = load_scaling_rows(extreme_path)
    precision_rows = load_precision_rows(sparse_path)
    scaling_summary = summarize_scaling_rows(scaling_rows)
    precision_summary = summarize_precision_rows(precision_rows)

    run_dir = make_run_dir(ANALYSIS_DIR, args.output_prefix)
    scaling_plot_path = run_dir / "problem1_scaling_frontier.png"
    precision_plot_path = run_dir / "problem1_precision_frontier.png"
    scaling_csv_path = run_dir / "problem1_scaling_frontier.csv"
    precision_csv_path = run_dir / "problem1_precision_frontier.csv"
    log_path = run_dir / "problem1_frontier_curves.log"

    plot_scaling(scaling_rows, scaling_plot_path)
    plot_precision(precision_rows, precision_plot_path)
    write_csv(scaling_csv_path, list(scaling_summary[0].keys()), scaling_summary)
    write_csv(precision_csv_path, list(precision_summary[0].keys()), precision_summary)
    write_summary_log(log_path, extreme_path, sparse_path, scaling_summary, precision_summary)

    print(f"analysis_dir={run_dir}")
    print(f"scaling_plot={scaling_plot_path.name}")
    print(f"precision_plot={precision_plot_path.name}")
    print(f"scaling_csv={scaling_csv_path.name}")
    print(f"precision_csv={precision_csv_path.name}")
    print(f"log={log_path.name}")


if __name__ == "__main__":
    main()
