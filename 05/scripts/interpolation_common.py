from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

from result_paths import write_csv

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ArrayFunc = Callable[[np.ndarray], np.ndarray]


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
        numerator = (x_column - right_nodes) * table[:, : n - level] + (
            left_nodes - x_column
        ) * table[:, 1 : n - level + 1]
        denominator = left_nodes - right_nodes
        table[:, : n - level] = numerator / denominator

    values = table[:, 0]
    if is_scalar:
        return float(values[0])
    return values


def interpolation_case(
    func: ArrayFunc,
    node_count: int,
    x_dense: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, InterpolationSummary]:
    y_exact = func(x_dense)
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
    return x_nodes, y_nodes, y_neville, y_exact, summary


def plot_interpolation_panels(
    func: ArrayFunc,
    node_counts: list[int],
    figure_path: Path,
    title_prefix: str,
    y_limits: tuple[float, float] | None = None,
    ncols: int | None = None,
) -> None:
    x_dense = np.linspace(-1.0, 1.0, 2001)
    if ncols is None:
        ncols = min(3, len(node_counts))
    nrows = math.ceil(len(node_counts) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.2 * ncols, 4.9 * nrows), squeeze=False)
    axes_flat = axes.reshape(-1)

    for ax in axes_flat[len(node_counts) :]:
        ax.axis("off")

    for ax, node_count in zip(axes_flat, node_counts):
        x_nodes, y_nodes, y_neville, y_exact, summary = interpolation_case(func, node_count, x_dense)
        ax.plot(x_dense, y_exact, color="#1d3557", linewidth=2.2, label="Exact function")
        ax.plot(x_dense, y_neville, color="#e76f51", linewidth=1.8, label="Interpolation")
        ax.scatter(x_nodes, y_nodes, color="black", s=18, zorder=5, label="Nodes")
        ax.set_title(f"{title_prefix}, {node_count} nodes\nmax err={summary.max_abs_error:.3e}")
        ax.grid(alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if y_limits is not None:
            ax.set_ylim(*y_limits)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def plot_error_vs_node_count(summaries: list[InterpolationSummary], figure_path: Path, title: str) -> None:
    node_counts = [item.node_count for item in summaries]
    max_errors = [item.max_abs_error for item in summaries]
    rmses = [item.rmse for item in summaries]

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.semilogy(node_counts, max_errors, marker="o", linewidth=2.0, label="max error")
    ax.semilogy(node_counts, rmses, marker="s", linewidth=2.0, label="RMSE")
    ax.set_title(title)
    ax.set_xlabel("b (number of equispaced nodes)")
    ax.set_ylabel("error")
    ax.set_xticks(node_counts)
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def write_interpolation_summary(path: Path, summaries: list[InterpolationSummary]) -> None:
    write_csv(
        path,
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


def write_log_error_model(path: Path, summaries: list[InterpolationSummary], min_node_count: int) -> None:
    selected = [item for item in summaries if item.node_count >= min_node_count]
    rows = []
    for metric in ["max_abs_error", "rmse"]:
        x = np.array([item.node_count for item in selected], dtype=float)
        y = np.log(np.array([getattr(item, metric) for item in selected], dtype=float))
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        residual_sum = float(np.sum((y - fitted) ** 2))
        total_sum = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 1.0
        rows.append(
            {
                "model": "error = prefactor * exp(slope_per_node * node_count)",
                "metric": metric,
                "min_node_count": min_node_count,
                "sample_count": len(selected),
                "prefactor": float(math.exp(intercept)),
                "slope_per_node": float(slope),
                "per_node_factor": float(math.exp(slope)),
                "r_squared": r_squared,
            }
        )

    write_csv(
        path,
        [
            "model",
            "metric",
            "min_node_count",
            "sample_count",
            "prefactor",
            "slope_per_node",
            "per_node_factor",
            "r_squared",
        ],
        rows,
    )
