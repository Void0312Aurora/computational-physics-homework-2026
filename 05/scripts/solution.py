from __future__ import annotations

import csv
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = PROJECT_ROOT
if PROJECT_ROOT.name == "scripts":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
RESULT_DIR = PROJECT_ROOT / "result"

from project2_pi.homework_bridge import solve_project2


def ensure_result_dir() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pchip_endpoint_slope(h1: float, h2: float, delta1: float, delta2: float) -> float:
    slope = ((2.0 * h1 + h2) * delta1 - h1 * delta2) / (h1 + h2)
    if slope == 0.0 or np.sign(slope) != np.sign(delta1):
        return 0.0
    if np.sign(delta1) != np.sign(delta2) and abs(slope) > abs(3.0 * delta1):
        return 3.0 * delta1
    return slope


def pchip_coefficients(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    m = np.zeros(n)

    for i in range(1, n - 1):
        if delta[i - 1] == 0.0 or delta[i] == 0.0 or np.sign(delta[i - 1]) != np.sign(delta[i]):
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    m[0] = pchip_endpoint_slope(h[0], h[1], delta[0], delta[1])
    m[-1] = pchip_endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])

    a = y[:-1].copy()
    b = m[:-1].copy()
    c = (3.0 * delta - 2.0 * m[:-1] - m[1:]) / h
    d = (m[:-1] + m[1:] - 2.0 * delta) / (h**2)
    return a, b, c, d


def evaluate_piecewise_cubic(
    x_nodes: np.ndarray,
    coeffs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    x_eval: np.ndarray,
) -> np.ndarray:
    a, b, c, d = coeffs
    n = len(x_nodes) - 1
    interval_ids = np.searchsorted(x_nodes, x_eval, side="right") - 1
    interval_ids = np.clip(interval_ids, 0, n - 1)
    dx = x_eval - x_nodes[interval_ids]
    return a[interval_ids] + b[interval_ids] * dx + c[interval_ids] * dx**2 + d[interval_ids] * dx**3


def barycentric_weights(x_nodes: np.ndarray) -> np.ndarray:
    diffs = x_nodes[:, None] - x_nodes[None, :]
    np.fill_diagonal(diffs, 1.0)
    return 1.0 / np.prod(diffs, axis=1)


def barycentric_evaluate(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    weights: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    diffs = x_eval[:, None] - x_nodes[None, :]
    exact = np.isclose(diffs, 0.0, atol=1e-14, rtol=0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = weights[None, :] / diffs
        numerator = ratio @ y_nodes
        denominator = np.sum(ratio, axis=1)
        values = numerator / denominator

    if np.any(exact):
        hit_rows, hit_cols = np.where(exact)
        values[hit_rows] = y_nodes[hit_cols]
    return values


def neville_evaluate(x_nodes: np.ndarray, y_nodes: np.ndarray, x_eval: np.ndarray | float) -> np.ndarray | float:
    x_array = np.asarray(x_eval, dtype=float)
    is_scalar = x_array.ndim == 0
    x_flat = np.atleast_1d(x_array)
    n = len(x_nodes)
    table = np.repeat(y_nodes[None, :], x_flat.size, axis=0)
    x_column = x_flat[:, None]

    for level in range(1, n):
        left_nodes = x_nodes[: n - level]
        right_nodes = x_nodes[level:]
        numerator = (x_column - right_nodes) * table[:, : n - level] + (left_nodes - x_column) * table[:, 1 : n - level + 1]
        denominator = left_nodes - right_nodes
        table[:, : n - level] = numerator / denominator

    values = table[:, 0]
    if is_scalar:
        return float(values[0])
    return values


def solve_problem1() -> dict[str, object]:
    voltage = np.array([-1.00, 0.00, 1.27, 2.55, 3.82, 4.92, 5.02], dtype=float)
    current = np.array([-14.58, 0.00, 0.00, 0.00, 0.00, 0.88, 11.17], dtype=float)

    x_dense = np.linspace(voltage.min(), voltage.max(), 1600)
    poly_vals = np.asarray(neville_evaluate(voltage, current, x_dense))
    linear_vals = np.interp(x_dense, voltage, current)
    pchip = pchip_coefficients(voltage, current)
    pchip_vals = evaluate_piecewise_cubic(voltage, pchip, x_dense)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x_dense, poly_vals, label="Global degree-6 interpolation (Neville)", linewidth=2.0, color="#d1495b")
    ax.plot(x_dense, linear_vals, label="Piecewise linear", linewidth=2.0, color="#2e86ab")
    ax.plot(x_dense, pchip_vals, label="Shape-preserving cubic (PCHIP)", linewidth=2.2, color="#2a9d8f")
    ax.scatter(voltage, current, s=55, color="black", zorder=5, label="Data points")
    ax.set_title("Problem 1: Zener diode interpolation comparison")
    ax.set_xlabel("Voltage")
    ax.set_ylabel("Current")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "problem1_zener_interpolation.png", dpi=220)
    plt.close(fig)

    summary_rows: list[dict[str, object]] = []
    for name, values in [
        ("global_degree_6", poly_vals),
        ("piecewise_linear", linear_vals),
        ("shape_preserving_cubic", pchip_vals),
    ]:
        summary_rows.append(
            {
                "curve": name,
                "min_current": float(np.min(values)),
                "max_current": float(np.max(values)),
                "range_width": float(np.max(values) - np.min(values)),
            }
        )
    write_csv(
        RESULT_DIR / "problem1_curve_summary.csv",
        ["curve", "min_current", "max_current", "range_width"],
        summary_rows,
    )

    midpoints = 0.5 * (voltage[:-1] + voltage[1:])
    midpoint_rows: list[dict[str, object]] = []
    midpoint_poly = np.asarray(neville_evaluate(voltage, current, midpoints))
    midpoint_linear = np.interp(midpoints, voltage, current)
    midpoint_pchip = evaluate_piecewise_cubic(voltage, pchip, midpoints)
    for x_mid, p_val, l_val, s_val in zip(midpoints, midpoint_poly, midpoint_linear, midpoint_pchip):
        midpoint_rows.append(
            {
                "midpoint_voltage": float(x_mid),
                "global_degree_6": float(p_val),
                "piecewise_linear": float(l_val),
                "shape_preserving_cubic": float(s_val),
            }
        )
    write_csv(
        RESULT_DIR / "problem1_midpoints.csv",
        ["midpoint_voltage", "global_degree_6", "piecewise_linear", "shape_preserving_cubic"],
        midpoint_rows,
    )

    return {
        "voltage": voltage,
        "current": current,
        "curve_summary": summary_rows,
        "midpoints": midpoint_rows,
    }


@dataclass
class InterpolationSummary:
    node_count: int
    polynomial_degree: int
    max_abs_error: float
    rmse: float
    neville_vs_barycentric_max_diff: float
    neville_vs_barycentric_rmse: float
    neville_seconds: float
    barycentric_seconds: float


def interpolation_experiment(
    func,
    node_counts: list[int],
    figure_path: Path,
    summary_path: Path,
    title_prefix: str,
    y_limits: tuple[float, float] | None = None,
) -> list[InterpolationSummary]:
    x_dense = np.linspace(-1.0, 1.0, 2001)
    y_exact = func(x_dense)
    summaries: list[InterpolationSummary] = []

    ncols = 3
    nrows = math.ceil(len(node_counts) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4.6 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax in axes[len(node_counts) :]:
        ax.axis("off")

    for ax, node_count in zip(axes, node_counts):
        x_nodes = np.linspace(-1.0, 1.0, node_count)
        y_nodes = func(x_nodes)
        neville_start = time.perf_counter()
        y_neville = np.asarray(neville_evaluate(x_nodes, y_nodes, x_dense))
        neville_seconds = time.perf_counter() - neville_start

        barycentric_start = time.perf_counter()
        weights = barycentric_weights(x_nodes)
        y_barycentric = barycentric_evaluate(x_nodes, y_nodes, weights, x_dense)
        barycentric_seconds = time.perf_counter() - barycentric_start

        consistency_diff = y_neville - y_barycentric
        error = y_neville - y_exact
        summary = InterpolationSummary(
            node_count=node_count,
            polynomial_degree=node_count - 1,
            max_abs_error=float(np.max(np.abs(error))),
            rmse=float(np.sqrt(np.mean(error**2))),
            neville_vs_barycentric_max_diff=float(np.max(np.abs(consistency_diff))),
            neville_vs_barycentric_rmse=float(np.sqrt(np.mean(consistency_diff**2))),
            neville_seconds=neville_seconds,
            barycentric_seconds=barycentric_seconds,
        )
        summaries.append(summary)

        ax.plot(x_dense, y_exact, color="#1d3557", linewidth=2.2, label="Exact function")
        ax.plot(x_dense, y_neville, color="#e76f51", linewidth=1.8, label="Neville/Lagrange interpolation")
        ax.scatter(x_nodes, y_nodes, color="black", s=18, zorder=5)
        ax.set_title(
            f"{title_prefix}, {node_count} nodes\n"
            f"deg={summary.polynomial_degree}, max err={summary.max_abs_error:.3e}"
        )
        ax.grid(alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if y_limits is not None:
            ax.set_ylim(*y_limits)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    write_csv(
        summary_path,
        [
            "node_count",
            "polynomial_degree",
            "max_abs_error",
            "rmse",
            "neville_vs_barycentric_max_diff",
            "neville_vs_barycentric_rmse",
            "neville_seconds",
            "barycentric_seconds",
        ],
        [
            {
                "node_count": item.node_count,
                "polynomial_degree": item.polynomial_degree,
                "max_abs_error": item.max_abs_error,
                "rmse": item.rmse,
                "neville_vs_barycentric_max_diff": item.neville_vs_barycentric_max_diff,
                "neville_vs_barycentric_rmse": item.neville_vs_barycentric_rmse,
                "neville_seconds": item.neville_seconds,
                "barycentric_seconds": item.barycentric_seconds,
            }
            for item in summaries
        ],
    )

    return summaries


def solve_problem2() -> list[InterpolationSummary]:
    return interpolation_experiment(
        func=lambda x: 1.0 / (1.0 + 16.0 * x**2),
        node_counts=[5, 7, 9, 17, 19, 21],
        figure_path=RESULT_DIR / "problem2_runge_interpolation.png",
        summary_path=RESULT_DIR / "problem2_runge_error_summary.csv",
        title_prefix="Runge function",
        y_limits=(-0.45, 1.1),
    )


def solve_problem3() -> list[InterpolationSummary]:
    return interpolation_experiment(
        func=lambda x: x * np.sin(2.0 * math.pi * x + 1.0),
        node_counts=[7, 9, 17, 19, 21],
        figure_path=RESULT_DIR / "problem3_sine_interpolation.png",
        summary_path=RESULT_DIR / "problem3_sine_error_summary.csv",
        title_prefix="x sin(2 pi x + 1)",
        y_limits=(-1.35, 0.65),
    )


def write_run_log(
    problem1: dict[str, object],
    problem2: list[InterpolationSummary],
    problem3: list[InterpolationSummary],
    project2: dict[str, object],
) -> None:
    lines: list[str] = []
    lines.append("HW/05 execution log")
    lines.append(f"platform = {platform.platform()}")
    lines.append(f"python = {platform.python_version()}")
    lines.append("")
    lines.append("Problem 1 curve summary:")
    for row in problem1["curve_summary"]:
        lines.append(
            "  {curve}: min={min_current:.6f}, max={max_current:.6f}, range={range_width:.6f}".format(
                **row
            )
        )
    lines.append("")
    lines.append("Problem 2 error summary:")
    for row in problem2:
        lines.append(
            f"  nodes={row.node_count:2d}, degree={row.polynomial_degree:2d}, "
            f"max_abs_error={row.max_abs_error:.6e}, rmse={row.rmse:.6e}, "
            f"|Neville-Bary|_max={row.neville_vs_barycentric_max_diff:.6e}, "
            f"Neville_s={row.neville_seconds:.6e}, Bary_s={row.barycentric_seconds:.6e}"
        )
    lines.append("")
    lines.append("Problem 3 error summary:")
    for row in problem3:
        lines.append(
            f"  nodes={row.node_count:2d}, degree={row.polynomial_degree:2d}, "
            f"max_abs_error={row.max_abs_error:.6e}, rmse={row.rmse:.6e}, "
            f"|Neville-Bary|_max={row.neville_vs_barycentric_max_diff:.6e}, "
            f"Neville_s={row.neville_seconds:.6e}, Bary_s={row.barycentric_seconds:.6e}"
        )
    lines.append("")
    lines.append("Project 2 pi benchmark:")
    project2_leaf_suffix = f", leaf_terms={project2['leaf_terms']}" if "leaf_terms" in project2 else ""
    project2_task_suffix = f", task_terms={project2['task_terms']}" if "task_terms" in project2 else ""
    project2_mode_suffix = f", parallel_mode={project2['parallel_mode']}" if "parallel_mode" in project2 else ""
    lines.append(
        "  backend={backend}, workers={workers}, chunk_terms={chunk_terms}{leaf_suffix}{task_suffix}{mode_suffix}, gpu_used={gpu_used}, "
        "highest_digits={highest_digits}, output_name={output_name}".format(
            **project2,
            leaf_suffix=project2_leaf_suffix,
            task_suffix=project2_task_suffix,
            mode_suffix=project2_mode_suffix,
        )
    )
    for row in project2["benchmark_rows"]:
        row_leaf_suffix = f", leaf_terms={row['leaf_terms']}" if "leaf_terms" in row else ""
        row_task_suffix = f", task_terms={row['task_terms']}" if "task_terms" in row else ""
        row_mode_suffix = f", parallel_mode={row['parallel_mode']}" if "parallel_mode" in row else ""
        lines.append(
            "  digits={digits}, terms={terms}, seconds={seconds:.6f}, digits_per_second={digits_per_second:.2f}, "
            "backend={backend}, workers={workers_used}, chunk_terms={chunk_terms}{leaf_suffix}{task_suffix}{mode_suffix}, gpu_used={gpu_used}, "
            "prefix_ok={prefix_matches_reference}".format(
                **row
                ,
                leaf_suffix=row_leaf_suffix,
                task_suffix=row_task_suffix,
                mode_suffix=row_mode_suffix,
            )
        )
    (RESULT_DIR / "temp-01.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_result_dir()
    problem1 = solve_problem1()
    problem2 = solve_problem2()
    problem3 = solve_problem3()
    project2 = solve_project2()
    write_run_log(problem1, problem2, problem3, project2)


if __name__ == "__main__":
    main()
