from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def fit_line(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = slope * x + intercept
    residuals = y - y_hat
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot
    rmse = math.sqrt(ss_res / (len(x) - 2))
    x_centered = x - np.mean(x)
    leverage = 1.0 / len(x) + (x_centered**2) / np.sum(x_centered**2)
    max_residual_index = int(np.argmax(np.abs(residuals)))
    max_leverage_index = int(np.argmax(leverage))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "rmse": rmse,
        "max_abs_residual": float(np.max(np.abs(residuals))),
        "max_abs_residual_x": float(x[max_residual_index]),
        "max_abs_residual_y": float(y[max_residual_index]),
        "max_leverage": float(np.max(leverage)),
        "max_leverage_x": float(x[max_leverage_index]),
        "max_leverage_y": float(y[max_leverage_index]),
    }


def classify_fit(index: int, x: np.ndarray, y: np.ndarray, stats: dict[str, float]) -> str:
    if index == 0:
        return "散点基本沿直线分布，残差无明显结构，线性模型拟合良好。"
    if index == 1:
        return "整体斜率与截距和第 1 组接近，但残差呈明显曲线趋势，说明线性模型存在系统失配。"
    if index == 2:
        return "大多数点接近一条水平带，仅有一个高残差离群点主导了斜率，因此线性拟合在解释整体结构时并不可靠。"
    return "除一个高杠杆点外其余点的 x 几乎相同，回归直线主要由该高杠杆点决定，属于脆弱且不稳健的拟合。"


def save_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def plot_problem1(data: list[tuple[np.ndarray, np.ndarray]], fit_stats: list[dict[str, float]], out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    x_line = np.linspace(3.5, 19.5, 200)

    for idx, (ax, (x, y), stats) in enumerate(zip(axes.flat, data, fit_stats, strict=True), start=1):
        y_line = stats["slope"] * x_line + stats["intercept"]
        ax.scatter(x, y, s=55, color="#1f77b4", edgecolor="black", linewidth=0.5, zorder=3)
        ax.plot(x_line, y_line, color="#d62728", linewidth=2.0, label="least-squares line")
        ax.set_title(f"Data {idx}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper left", fontsize=9)
        ax.text(
            0.03,
            0.04,
            (
                f"$m={stats['slope']:.4f}$\n"
                f"$b={stats['intercept']:.4f}$\n"
                f"$R^2={stats['r_squared']:.4f}$"
            ),
            transform=ax.transAxes,
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9},
        )
    fig.suptitle("Problem 1: Linear fits for the four data sets", fontsize=15)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_problem1_residuals(
    data: list[tuple[np.ndarray, np.ndarray]], fit_stats: list[dict[str, float]], out_path: Path
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)

    for idx, (ax, (x, y), stats) in enumerate(zip(axes.flat, data, fit_stats, strict=True), start=1):
        residuals = y - (stats["slope"] * x + stats["intercept"])
        ax.axhline(0.0, color="#444444", linewidth=1.2, linestyle="--")
        ax.scatter(x, residuals, s=55, color="#2ca02c", edgecolor="black", linewidth=0.5, zorder=3)
        ax.set_title(f"Data {idx} residuals")
        ax.set_xlabel("x")
        ax.set_ylabel("residual")
    fig.suptitle("Problem 1: Residual diagnostics", fontsize=15)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_problem2(t: np.ndarray, n: np.ndarray, sigma: np.ndarray, params: dict[str, float], out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True)
    t_fine = np.linspace(float(np.min(t)), float(np.max(t)), 400)
    fit_curve = params["A"] * np.exp(-t_fine / params["tau"])
    ax.errorbar(
        t,
        n,
        yerr=sigma,
        fmt="o",
        color="#1f77b4",
        ecolor="#6baed6",
        capsize=3,
        label="observed counts with $\\sigma=\\sqrt{N}$",
    )
    ax.plot(t_fine, fit_curve, color="#d62728", linewidth=2.2, label="weighted nonlinear fit")
    ax.set_xlabel("time")
    ax.set_ylabel("count")
    ax.set_title("Problem 2: Radioactive decay fit")
    ax.legend(loc="upper right")
    ax.text(
        0.60,
        0.78,
        (
            f"$A={params['A']:.2f}\\pm {params['A_err']:.2f}$\n"
            f"$\\tau={params['tau']:.2f}\\pm {params['tau_err']:.2f}$"
        ),
        transform=ax.transAxes,
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.92},
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def decay_model(t: np.ndarray, a: float, tau: float) -> np.ndarray:
    return a * np.exp(-t / tau)


def fit_decay(t: np.ndarray, n: np.ndarray) -> dict[str, float]:
    sigma = np.sqrt(n)
    p0 = np.array([n[0], 80.0], dtype=float)
    popt, pcov = curve_fit(
        decay_model,
        t,
        n,
        p0=p0,
        sigma=sigma,
        absolute_sigma=True,
        maxfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))

    log_n = np.log(n)
    sigma_log = sigma / n
    coeffs, cov = np.polyfit(t, log_n, deg=1, w=1.0 / sigma_log, cov=True)
    slope, intercept = coeffs
    intercept_err = math.sqrt(float(cov[1, 1]))
    slope_err = math.sqrt(float(cov[0, 0]))
    a_lin = math.exp(intercept)
    tau_lin = -1.0 / slope
    a_lin_err = a_lin * intercept_err
    tau_lin_err = slope_err / (slope**2)

    fitted = decay_model(t, *popt)
    chi2 = float(np.sum(((n - fitted) / sigma) ** 2))
    dof = len(t) - len(popt)

    return {
        "A": float(popt[0]),
        "tau": float(popt[1]),
        "A_err": float(perr[0]),
        "tau_err": float(perr[1]),
        "chi2": chi2,
        "chi2_red": chi2 / dof,
        "A_linearized": float(a_lin),
        "tau_linearized": float(tau_lin),
        "A_linearized_err": float(a_lin_err),
        "tau_linearized_err": float(tau_lin_err),
    }


def problem1(result_dir: Path, log_buffer: io.StringIO) -> list[dict[str, float]]:
    x_common = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=float)
    datasets = [
        (x_common, np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68], dtype=float)),
        (x_common, np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74], dtype=float)),
        (x_common, np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73], dtype=float)),
        (np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], dtype=float), np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89], dtype=float)),
    ]

    problem1_rows: list[list[object]] = []
    fit_stats: list[dict[str, float]] = []
    print("Problem 1: linear least-squares fits", file=log_buffer)
    for idx, (x, y) in enumerate(datasets, start=1):
        stats = fit_line(x, y)
        stats["dataset"] = idx
        stats["assessment"] = classify_fit(idx - 1, x, y, stats)
        fit_stats.append(stats)
        problem1_rows.append(
            [
                idx,
                f"{stats['slope']:.10f}",
                f"{stats['intercept']:.10f}",
                f"{stats['r_squared']:.10f}",
                f"{stats['rmse']:.10f}",
                f"{stats['max_abs_residual']:.10f}",
                f"{stats['max_leverage']:.10f}",
                stats["assessment"],
            ]
        )
        print(
            (
                f"  Data {idx}: slope={stats['slope']:.10f}, intercept={stats['intercept']:.10f}, "
                f"R^2={stats['r_squared']:.10f}, rmse={stats['rmse']:.10f}"
            ),
            file=log_buffer,
        )
        print(f"    assessment: {stats['assessment']}", file=log_buffer)

    save_csv(
        result_dir / "problem1_fit_summary.csv",
        [
            "dataset",
            "slope",
            "intercept",
            "r_squared",
            "rmse",
            "max_abs_residual",
            "max_leverage",
            "assessment",
        ],
        problem1_rows,
    )
    plot_problem1(datasets, fit_stats, result_dir / "problem1_linear_fits.png")
    plot_problem1_residuals(datasets, fit_stats, result_dir / "problem1_residuals.png")
    return fit_stats


def problem2(result_dir: Path, log_buffer: io.StringIO) -> dict[str, float]:
    t = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165], dtype=float)
    n = np.array([106, 80, 98, 75, 74, 73, 49, 38, 37, 22, 20, 19], dtype=float)
    sigma = np.sqrt(n)
    decay_params = fit_decay(t, n)
    plot_problem2(t, n, sigma, decay_params, result_dir / "problem2_decay_fit.png")

    save_csv(
        result_dir / "problem2_decay_fit.csv",
        ["parameter", "value"],
        [
            ["A", f"{decay_params['A']:.10f}"],
            ["A_err", f"{decay_params['A_err']:.10f}"],
            ["tau", f"{decay_params['tau']:.10f}"],
            ["tau_err", f"{decay_params['tau_err']:.10f}"],
            ["chi2", f"{decay_params['chi2']:.10f}"],
            ["chi2_red", f"{decay_params['chi2_red']:.10f}"],
            ["A_linearized", f"{decay_params['A_linearized']:.10f}"],
            ["A_linearized_err", f"{decay_params['A_linearized_err']:.10f}"],
            ["tau_linearized", f"{decay_params['tau_linearized']:.10f}"],
            ["tau_linearized_err", f"{decay_params['tau_linearized_err']:.10f}"],
        ],
    )

    observed_vs_fit_rows = []
    fitted_counts = decay_model(t, decay_params["A"], decay_params["tau"])
    for ti, ni, si, fi in zip(t, n, sigma, fitted_counts, strict=True):
        observed_vs_fit_rows.append([f"{ti:.0f}", f"{ni:.0f}", f"{si:.6f}", f"{fi:.6f}"])
    save_csv(
        result_dir / "problem2_observed_vs_fit.csv",
        ["t", "N", "sigma", "fit"],
        observed_vs_fit_rows,
    )

    print("\nProblem 2: weighted exponential decay fit", file=log_buffer)
    print(
        (
            f"  nonlinear weighted fit: A={decay_params['A']:.10f} +/- {decay_params['A_err']:.10f}, "
            f"tau={decay_params['tau']:.10f} +/- {decay_params['tau_err']:.10f}"
        ),
        file=log_buffer,
    )
    print(
        (
            f"  linearized weighted check: A={decay_params['A_linearized']:.10f} +/- "
            f"{decay_params['A_linearized_err']:.10f}, tau={decay_params['tau_linearized']:.10f} +/- "
            f"{decay_params['tau_linearized_err']:.10f}"
        ),
        file=log_buffer,
    )
    print(
        f"  chi^2={decay_params['chi2']:.10f}, reduced chi^2={decay_params['chi2_red']:.10f}",
        file=log_buffer,
    )
    return decay_params


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result_dir = root / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    log_buffer = io.StringIO()
    fit_stats = problem1(result_dir, log_buffer)
    decay_params = problem2(result_dir, log_buffer)
    summary = {
        "problem1": fit_stats,
        "problem2": decay_params,
    }
    (result_dir / "hw08_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (result_dir / "temp-01.log").write_text(log_buffer.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
