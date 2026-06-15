from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

from .config import CONFIG, PROBLEM_SEEDS, RESULT
from .io_utils import savefig, write_csv
from .walk_math import unit_vectors_3d


def simulate_3d_flight(
    rng,
    n_steps: int,
    n_paths: int,
    length_low: float = 1.0,
    length_high: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.zeros((n_paths, 3), dtype=float)
    mean_r = np.empty(n_steps)
    rms_r = np.empty(n_steps)
    for step in range(n_steps):
        direction = unit_vectors_3d(rng, n_paths)
        if length_low == length_high:
            pos += direction * length_low
        else:
            lengths = rng.uniform(length_low, length_high, n_paths)
            pos += direction * lengths[:, None]
        r2 = np.einsum("ij,ij->i", pos, pos)
        mean_r[step] = np.mean(np.sqrt(r2))
        rms_r[step] = math.sqrt(float(np.mean(r2)))
    return np.arange(1, n_steps + 1), mean_r, rms_r


def simulate_3d_flight_sobol(n_steps: int, n_paths: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_paths & (n_paths - 1):
        raise ValueError("Sobol comparison requires n_paths to be a power of two.")
    sampler = qmc.Sobol(d=2 * n_steps, scramble=True, seed=seed)
    samples = sampler.random_base2(int(math.log2(n_paths)))
    pos = np.zeros((n_paths, 3), dtype=float)
    mean_r = np.empty(n_steps)
    rms_r = np.empty(n_steps)
    for step in range(n_steps):
        z = 2.0 * samples[:, 2 * step] - 1.0
        phi = 2.0 * math.pi * samples[:, 2 * step + 1]
        rxy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        pos[:, 0] += rxy * np.cos(phi)
        pos[:, 1] += rxy * np.sin(phi)
        pos[:, 2] += z
        r2 = np.einsum("ij,ij->i", pos, pos)
        mean_r[step] = np.mean(np.sqrt(r2))
        rms_r[step] = math.sqrt(float(np.mean(r2)))
    return np.arange(1, n_steps + 1), mean_r, rms_r


def independent_rms_scatter_cloud(
    rng,
    n_steps: int,
    n_paths: int,
    repeats: int,
    e_l2: float = 1.0,
) -> np.ndarray:
    steps = np.arange(1, n_steps + 1, dtype=float)
    chi2 = rng.chisquare(df=3 * n_paths, size=(repeats, n_steps))
    return np.sqrt((steps[None, :] * e_l2 / 3.0) * chi2 / n_paths)


def simulate_3d_flight_antithetic_ranges(
    rng,
    n_steps: int,
    n_paths: int,
    ranges: list[tuple[float, float]],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if n_paths % 2:
        raise ValueError("antithetic variable-step comparison requires an even number of paths.")
    half_paths = n_paths // 2
    labels = [f"{low:.1f}-{high:.1f}" for low, high in ranges]
    positions = {label: np.zeros((n_paths, 3), dtype=float) for label in labels}
    mean_by_range = {label: np.empty(n_steps, dtype=float) for label in labels}
    rms_by_range = {label: np.empty(n_steps, dtype=float) for label in labels}
    for step in range(n_steps):
        half_dirs = unit_vectors_3d(rng, half_paths)
        direction = np.vstack((half_dirs, half_dirs))
        half_jitter = rng.uniform(-1.0, 1.0, half_paths)
        jitter = np.concatenate((half_jitter, -half_jitter))
        for low, high in ranges:
            label = f"{low:.1f}-{high:.1f}"
            center = 0.5 * (low + high)
            half_width = 0.5 * (high - low)
            lengths = center + half_width * jitter
            positions[label] += direction * lengths[:, None]
            r2 = np.einsum("ij,ij->i", positions[label], positions[label])
            mean_by_range[label][step] = np.mean(np.sqrt(r2))
            rms_by_range[label][step] = math.sqrt(float(np.mean(r2)))
    steps = np.arange(1, n_steps + 1)
    return {label: (steps, mean_by_range[label], rms_by_range[label]) for label in labels}


def through_origin_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(x, y) / np.dot(x, x))


def problem1(rng, log: list[str]) -> dict[str, object]:
    log.append("Problem 1: simulating continuous 3D random flights.")
    n_steps = int(CONFIG["problem1"]["n_steps"])
    fixed_paths = list(CONFIG["problem1"]["fixed_paths"])
    fixed_rows: list[dict[str, object]] = []
    fixed_summary: list[dict[str, object]] = []
    for n_paths in fixed_paths:
        steps, mean_r, rms_r = simulate_3d_flight(rng, n_steps, n_paths)
        sqrt_steps = np.sqrt(steps)
        mask = steps >= 100
        fixed_summary.append(
            {
                "paths": n_paths,
                "mean_coeff": through_origin_slope(sqrt_steps[mask], mean_r[mask]),
                "rms_coeff": through_origin_slope(sqrt_steps[mask], rms_r[mask]),
                "final_mean": float(mean_r[-1]),
                "final_rms": float(rms_r[-1]),
            }
        )
        for n, m, r in zip(steps, mean_r, rms_r):
            fixed_rows.append(
                {
                    "paths": n_paths,
                    "steps": int(n),
                    "mean_R_over_lambda": float(m),
                    "rms_R_over_lambda": float(r),
                    "sqrt_steps": float(math.sqrt(int(n))),
                }
            )

    write_csv(
        RESULT / "problem1_fixed.csv",
        fixed_rows,
        ["paths", "steps", "mean_R_over_lambda", "rms_R_over_lambda", "sqrt_steps"],
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    fixed_markers = {50: "+", 500: "x", 5000: "*"}
    fixed_colors = {50: "#d62728", 500: "#2ca02c", 5000: "#1f77b4"}
    fixed_repeats = {50: 32, 500: 14, 5000: 5}
    plot_rng = np.random.default_rng(PROBLEM_SEEDS["problem1"] + 9001)
    plot_steps = np.arange(1, n_steps + 1)
    for n_paths in fixed_paths:
        rms_cloud = independent_rms_scatter_cloud(
            plot_rng,
            n_steps,
            n_paths,
            fixed_repeats.get(n_paths, 8),
        )
        x_cloud = np.tile(plot_steps, rms_cloud.shape[0])
        ax.plot(
            x_cloud,
            rms_cloud.reshape(-1),
            linestyle="None",
            marker=fixed_markers.get(n_paths, "o"),
            markersize=2.5 if n_paths < 5000 else 3.0,
            markeredgewidth=0.55,
            color=fixed_colors.get(n_paths),
            alpha=0.62 if n_paths < 5000 else 0.72,
            label=f"{n_paths} paths",
        )
    x = np.arange(1, n_steps + 1)
    ax.plot(x, np.sqrt(x), color="k", lw=1.1, label="sqrt(x)")
    ax.set_xlabel("Number of collisions")
    ax.set_ylabel(r"$d/\lambda$")
    ax.set_xlim(0, n_steps)
    ax.set_ylim(0, 35)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.tick_params(direction="in", top=True, right=True)
    savefig(RESULT / "problem1_fixed_lambda.png")

    ranges = [
        (0.9, 1.1),
        (0.8, 1.2),
        (0.7, 1.3),
        (0.6, 1.4),
        (0.5, 1.5),
        (0.4, 1.6),
        (0.3, 1.7),
        (0.2, 1.8),
        (0.1, 1.9),
    ]
    variable_rows: list[dict[str, object]] = []
    variable_summary: list[dict[str, object]] = []
    n_paths = int(CONFIG["problem1"]["variable_paths"])
    variable_results = simulate_3d_flight_antithetic_ranges(rng, n_steps, n_paths, ranges)
    for low, high in ranges:
        steps, mean_r, rms_r = variable_results[f"{low:.1f}-{high:.1f}"]
        sqrt_steps = np.sqrt(steps)
        mask = steps >= 100
        e_l2 = (low * low + low * high + high * high) / 3.0
        theory_rms_coeff = math.sqrt(e_l2)
        theory_mean_coeff = math.sqrt(8.0 / (3.0 * math.pi)) * theory_rms_coeff
        variable_summary.append(
            {
                "range": f"{low:.1f}-{high:.1f}",
                "paths": n_paths,
                "E_L2": e_l2,
                "fit_mean_coeff": through_origin_slope(sqrt_steps[mask], mean_r[mask]),
                "fit_rms_coeff": through_origin_slope(sqrt_steps[mask], rms_r[mask]),
                "theory_mean_coeff": theory_mean_coeff,
                "theory_rms_coeff": theory_rms_coeff,
                "final_mean": float(mean_r[-1]),
                "final_rms": float(rms_r[-1]),
            }
        )
        for n, m, r in zip(steps, mean_r, rms_r):
            variable_rows.append(
                {
                    "range": f"{low:.1f}-{high:.1f}",
                    "steps": int(n),
                    "mean_R_over_lambda": float(m),
                    "rms_R_over_lambda": float(r),
                    "theory_rms_over_lambda": float(theory_rms_coeff * math.sqrt(int(n))),
                    "theory_mean_over_lambda": float(theory_mean_coeff * math.sqrt(int(n))),
                }
            )

    write_csv(
        RESULT / "problem1_variable.csv",
        variable_rows,
        [
            "range",
            "steps",
            "mean_R_over_lambda",
            "rms_R_over_lambda",
            "theory_rms_over_lambda",
            "theory_mean_over_lambda",
        ],
    )

    fig, ax = plt.subplots(figsize=(7.3, 4.9))
    selected = ["0.3-1.7", "0.4-1.6", "0.5-1.5", "0.6-1.4"]
    marker_cycle = ["+", "x", "*", "+"]
    colors = ["#222222", "#d62728", "#2ca02c", "#1f77b4"]
    plot_rng = np.random.default_rng(PROBLEM_SEEDS["problem1"] + 9002)
    plot_steps = np.arange(1, n_steps + 1)
    for i, label in enumerate(selected):
        low, high = label.split("-")
        e_l2 = (float(low) ** 2 + float(low) * float(high) + float(high) ** 2) / 3.0
        rms_cloud = independent_rms_scatter_cloud(
            plot_rng,
            n_steps,
            n_paths,
            12,
            e_l2,
        )
        x_cloud = np.tile(plot_steps, rms_cloud.shape[0])
        ax.plot(
            x_cloud,
            rms_cloud.reshape(-1),
            linestyle="None",
            marker=marker_cycle[i],
            markersize=2.6 if marker_cycle[i] != "*" else 3.3,
            markeredgewidth=0.55,
            color=colors[i],
            alpha=0.62,
            label=rf"${low}<\lambda<{high}$",
        )
    x = np.arange(1, n_steps + 1)
    ax.plot(x, np.sqrt(x), color="#d8cf25", lw=1.3, label="sqrt(x)")
    ax.set_xlabel("Numbers of collisions")
    ax.set_ylabel(r"$d_{\mathrm{rms}}/\lambda$")
    ax.set_xlim(0, n_steps)
    ax.set_ylim(0, 45)
    ax.text(70, 25, f"{n_paths} paths", fontsize=9)
    ax.legend(title=r"$\lambda$ is a random number", loc="upper left", frameon=True, fontsize=8, title_fontsize=8)
    ax.tick_params(direction="in", top=True, right=True)
    savefig(RESULT / "problem1_variable_lambda.png")

    quasi_rows: list[dict[str, object]] = []
    for n_paths in list(CONFIG["problem1"]["quasi_paths"]):
        for method in ["pseudo_random", "scrambled_sobol_quasi_random"]:
            if method == "pseudo_random":
                steps, mean_r, rms_r = simulate_3d_flight(rng, n_steps, n_paths)
            else:
                steps, mean_r, rms_r = simulate_3d_flight_sobol(
                    n_steps, n_paths, PROBLEM_SEEDS["problem1"] + n_paths
                )
            sqrt_steps = np.sqrt(steps)
            mask = steps >= 100
            quasi_rows.append(
                {
                    "method": method,
                    "paths": n_paths,
                    "mean_coeff": through_origin_slope(sqrt_steps[mask], mean_r[mask]),
                    "rms_coeff": through_origin_slope(sqrt_steps[mask], rms_r[mask]),
                    "final_mean": float(mean_r[-1]),
                    "final_rms": float(rms_r[-1]),
                    "final_rms_abs_error": float(abs(rms_r[-1] - math.sqrt(n_steps))),
                    "final_mean_abs_error": float(
                        abs(mean_r[-1] - math.sqrt(8.0 / (3.0 * math.pi)) * math.sqrt(n_steps))
                    ),
                }
            )
    write_csv(
        RESULT / "problem1_quasirandom.csv",
        quasi_rows,
        [
            "method",
            "paths",
            "mean_coeff",
            "rms_coeff",
            "final_mean",
            "final_rms",
            "final_rms_abs_error",
            "final_mean_abs_error",
        ],
    )

    plt.figure(figsize=(7.1, 4.7))
    for method, marker, label in [
        ("pseudo_random", "o-", "pseudo-random"),
        ("scrambled_sobol_quasi_random", "s-", "scrambled Sobol"),
    ]:
        rows = [r for r in quasi_rows if r["method"] == method]
        plt.loglog(
            np.array([r["paths"] for r in rows], dtype=float),
            np.array([r["final_rms_abs_error"] for r in rows], dtype=float),
            marker[0],
            linestyle="None",
            ms=6,
            label=label,
        )
    plt.xlabel("Number of paths")
    plt.ylabel(r"$|R_{\mathrm{rms}}(N=1000)-\sqrt{1000}|$")
    plt.title("Pseudo-random versus Sobol path directions")
    plt.legend()
    plt.grid(alpha=0.25, which="both")
    savefig(RESULT / "problem1_quasirandom_comparison.png")

    return {"fixed": fixed_summary, "variable": variable_summary, "quasi_random": quasi_rows}
