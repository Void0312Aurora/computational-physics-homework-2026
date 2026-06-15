from __future__ import annotations

import itertools
import math

import matplotlib.pyplot as plt
import numpy as np

from .config import CONFIG, RESULT
from .io_utils import savefig, write_csv


def unit_vectors_3d(rng, n: int) -> np.ndarray:
    z = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * math.pi, n)
    rxy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((rxy * np.cos(phi), rxy * np.sin(phi), z))


def chain_sample_count(n: int, rotating: bool = False) -> int:
    if rotating:
        if n <= 100:
            return 15000
        if n <= 500:
            return 6000
        if n <= 1000:
            return 3500
        if n <= 2000:
            return 2200
        return 1200
    if n <= 100:
        return 30000
    if n <= 500:
        return 15000
    if n <= 1000:
        return 8000
    if n <= 2000:
        return 5000
    if n <= 5000:
        return 2500
    if n <= 10000:
        return int(CONFIG["problem2"]["fjc_long_chain_samples"])
    return 1500


def chain_observables(pos: np.ndarray, sum_pos: np.ndarray, sum_norm2: np.ndarray, n: int) -> tuple[float, float]:
    re2 = np.einsum("ij,ij->i", pos, pos)
    rg2 = sum_norm2 / (n + 1) - np.einsum("ij,ij->i", sum_pos, sum_pos) / ((n + 1) ** 2)
    return float(np.mean(re2)), float(np.mean(rg2))


def frc_next_dirs(rng, prev: np.ndarray, theta: float) -> np.ndarray:
    samples = prev.shape[0]
    alpha = rng.uniform(0.0, 2.0 * math.pi, samples)
    base = np.zeros_like(prev)
    use_z = np.abs(prev[:, 2]) < 0.9
    base[use_z, 2] = 1.0
    base[~use_z, 1] = 1.0
    e1 = np.cross(prev, base)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(prev, e1)
    return (
        math.cos(theta) * prev
        + math.sin(theta) * (np.cos(alpha)[:, None] * e1 + np.sin(alpha)[:, None] * e2)
    )


def simulate_walk_stats(
    rng,
    dim: int,
    n_values: list[int],
    samples: int,
    mode: str,
) -> list[dict[str, object]]:
    max_n = max(n_values)
    targets = set(n_values)
    pos = np.zeros((samples, dim), dtype=float)
    sum_pos = np.zeros((samples, dim), dtype=float)
    sum_norm2 = np.zeros(samples, dtype=float)
    rows: list[dict[str, object]] = []
    for step in range(1, max_n + 1):
        if mode == "2d_random_direction":
            angle = rng.uniform(0.0, 2.0 * math.pi, samples)
            delta = np.column_stack((np.cos(angle), np.sin(angle)))
        else:
            axis = rng.integers(0, dim, samples)
            sign = rng.choice(np.array([-1.0, 1.0]), samples)
            delta = np.zeros((samples, dim), dtype=float)
            delta[np.arange(samples), axis] = sign
        pos += delta
        sum_pos += pos
        sum_norm2 += np.einsum("ij,ij->i", pos, pos)
        if step in targets:
            re2, rg2 = chain_observables(pos, sum_pos, sum_norm2, step)
            rows.append(
                {
                    "model": mode,
                    "N": step,
                    "samples": samples,
                    "mean_Re2": re2,
                    "mean_Rg2": rg2,
                    "Rg2_over_Re2": rg2 / re2,
                    "acceptance_rate": 1.0,
                    "effective_samples": samples,
                    "tau_Re2": 1.0,
                    "tau_Rg2": 1.0,
                }
            )
    return rows


def signed_permutation_matrices(dim: int) -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    identity = np.eye(dim, dtype=np.int32)
    for perm in itertools.permutations(range(dim)):
        for signs in itertools.product((-1, 1), repeat=dim):
            matrix = np.zeros((dim, dim), dtype=np.int32)
            for row, col in enumerate(perm):
                matrix[row, col] = signs[row]
            if not np.array_equal(matrix, identity):
                matrices.append(matrix)
    return matrices


def pivot_parameters(n: int) -> tuple[int, int, int]:
    if n <= 64:
        return 1600, 8 * n, 8
    if n <= 128:
        return 1400, 10 * n, 10
    if n <= 256:
        return 1200, 12 * n, 14
    if n <= 512:
        return 1000, 14 * n, 18
    return 800, 16 * n, 24


def integrated_autocorrelation_time(values: np.ndarray, max_lag: int | None = None) -> float:
    centered = values - np.mean(values)
    var = float(np.dot(centered, centered) / len(centered))
    if var <= 0.0:
        return 1.0
    if max_lag is None:
        max_lag = min(200, max(1, len(values) // 4))
    tau = 1.0
    for lag in range(1, max_lag + 1):
        rho = float(np.dot(centered[:-lag], centered[lag:]) / ((len(values) - lag) * var))
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    return max(1.0, tau)


def pivot_saw_stat(rng, dim: int, n: int, label: str) -> dict[str, object]:
    samples, burn_in, stride = pivot_parameters(n)
    transforms = signed_permutation_matrices(dim)
    coords = np.zeros((n + 1, dim), dtype=np.int32)
    coords[:, 0] = np.arange(n + 1, dtype=np.int32)
    proposals = 0
    accepted = 0

    def propose() -> None:
        nonlocal coords, proposals, accepted
        proposals += 1
        pivot_index = int(rng.integers(0, n))
        matrix = transforms[int(rng.integers(0, len(transforms)))]
        pivot = coords[pivot_index].copy()
        new_tail = (coords[pivot_index + 1 :] - pivot) @ matrix.T + pivot
        head = {tuple(row) for row in coords[: pivot_index + 1]}
        for row in new_tail:
            if tuple(row) in head:
                return
        coords[pivot_index + 1 :] = new_tail
        accepted += 1

    for _ in range(burn_in):
        propose()

    re2_values = np.empty(samples, dtype=float)
    rg2_values = np.empty(samples, dtype=float)
    for sample in range(samples):
        for _ in range(stride):
            propose()
        rel = coords - coords[0]
        re = rel[-1]
        re2_values[sample] = float(np.dot(re, re))
        sum_pos = np.sum(rel, axis=0, dtype=float)
        sum_norm2 = float(np.sum(rel.astype(float) * rel.astype(float)))
        rg2_values[sample] = sum_norm2 / (n + 1) - float(np.dot(sum_pos, sum_pos)) / ((n + 1) ** 2)

    re2 = float(np.mean(re2_values))
    rg2 = float(np.mean(rg2_values))
    tau_re2 = integrated_autocorrelation_time(re2_values)
    tau_rg2 = integrated_autocorrelation_time(rg2_values)
    return {
        "model": label,
        "N": n,
        "samples": samples,
        "mean_Re2": re2,
        "mean_Rg2": rg2,
        "Rg2_over_Re2": rg2 / re2,
        "acceptance_rate": accepted / proposals if proposals else 0.0,
        "effective_samples": min(samples / tau_re2, samples / tau_rg2),
        "tau_Re2": tau_re2,
        "tau_Rg2": tau_rg2,
    }


def fit_power_law(n: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(np.log(n), np.log(y), 1)
    return float(slope), float(intercept)


def bootstrap_power_fit(rng, n: np.ndarray, y: np.ndarray, n_boot: int) -> tuple[float, float, float, float]:
    slopes: list[float] = []
    attempts = 0
    while len(slopes) < n_boot and attempts < n_boot * 20:
        attempts += 1
        idx = rng.integers(0, len(n), len(n))
        if len(np.unique(n[idx])) < 2:
            continue
        slope, _ = fit_power_law(n[idx], y[idx])
        if math.isfinite(slope):
            slopes.append(slope)
    if not slopes:
        return float("nan"), float("nan"), float("nan"), float("nan")
    arr = np.array(slopes)
    return (
        float(np.mean(arr)),
        float(np.quantile(arr, 0.025)),
        float(np.quantile(arr, 0.975)),
        float(np.std(arr, ddof=1)),
    )


def fit_rows_for_windows(
    rows: list[dict[str, object]], rng, min_windows: list[int], n_boot: int
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for model in sorted({str(r["model"]) for r in rows}):
        model_rows = [r for r in rows if r["model"] == model]
        for observable, key in [("Re2", "mean_Re2"), ("Rg2", "mean_Rg2")]:
            for min_n in min_windows:
                data = [
                    r
                    for r in model_rows
                    if int(r["N"]) >= min_n and math.isfinite(float(r[key])) and float(r[key]) > 0.0
                ]
                if len(data) < 3:
                    continue
                n = np.array([r["N"] for r in data], dtype=float)
                y = np.array([r[key] for r in data], dtype=float)
                slope, intercept = fit_power_law(n, y)
                _, slope_low, slope_high, slope_std = bootstrap_power_fit(rng, n, y, n_boot)
                out.append(
                    {
                        "model": model,
                        "observable": observable,
                        "min_N": min_n,
                        "points": len(data),
                        "slope": slope,
                        "nu": slope / 2.0,
                        "nu_ci_low": slope_low / 2.0,
                        "nu_ci_high": slope_high / 2.0,
                        "nu_bootstrap_sd": slope_std / 2.0,
                        "prefactor": math.exp(intercept),
                    }
                )
    return out


def add_power_fits(rows: list[dict[str, object]], rng, min_n: int = 64, n_boot: int = 800) -> dict[str, dict[str, float]]:
    fits: dict[str, dict[str, float]] = {}
    for model in sorted({str(r["model"]) for r in rows}):
        data = [
            r
            for r in rows
            if r["model"] == model
            and r["N"] >= min_n
            and math.isfinite(float(r["mean_Re2"]))
            and math.isfinite(float(r["mean_Rg2"]))
            and float(r["mean_Re2"]) > 0.0
            and float(r["mean_Rg2"]) > 0.0
        ]
        n = np.array([r["N"] for r in data], dtype=float)
        if len(n) < 3:
            continue
        re2 = np.array([r["mean_Re2"] for r in data], dtype=float)
        rg2 = np.array([r["mean_Rg2"] for r in data], dtype=float)
        slope_re, intercept_re = fit_power_law(n, re2)
        slope_rg, intercept_rg = fit_power_law(n, rg2)
        _, re_low, re_high, re_sd = bootstrap_power_fit(rng, n, re2, n_boot)
        _, rg_low, rg_high, rg_sd = bootstrap_power_fit(rng, n, rg2, n_boot)
        fits[model] = {
            "Re2_slope": slope_re,
            "Re2_nu": slope_re / 2.0,
            "Re2_nu_ci_low": re_low / 2.0,
            "Re2_nu_ci_high": re_high / 2.0,
            "Re2_nu_bootstrap_sd": re_sd / 2.0,
            "Rg2_slope": slope_rg,
            "Rg2_nu": slope_rg / 2.0,
            "Rg2_nu_ci_low": rg_low / 2.0,
            "Rg2_nu_ci_high": rg_high / 2.0,
            "Rg2_nu_bootstrap_sd": rg_sd / 2.0,
            "Re2_prefactor": math.exp(intercept_re),
            "Rg2_prefactor": math.exp(intercept_rg),
        }
    return fits


def write_fit_diagnostics(rows: list[dict[str, object]], rng, stem: str) -> list[dict[str, object]]:
    fit_rows = fit_rows_for_windows(
        rows,
        rng,
        list(CONFIG["polymer_fit"]["sensitivity_min_N"]),
        int(CONFIG["polymer_fit"]["bootstrap_replicates"]),
    )
    fields = [
        "model",
        "observable",
        "min_N",
        "points",
        "slope",
        "nu",
        "nu_ci_low",
        "nu_ci_high",
        "nu_bootstrap_sd",
        "prefactor",
    ]
    write_csv(RESULT / f"{stem}_fit_windows.csv", fit_rows, fields)

    plt.figure(figsize=(7.4, 4.9))
    markers = {"Re2": "o", "Rg2": "s"}
    for model in sorted({str(r["model"]) for r in fit_rows}):
        for observable, marker in markers.items():
            data = [r for r in fit_rows if r["model"] == model and r["observable"] == observable]
            if not data:
                continue
            x = np.array([r["min_N"] for r in data], dtype=float)
            y = np.array([r["nu"] for r in data], dtype=float)
            y_low = np.array([r["nu_ci_low"] for r in data], dtype=float)
            y_high = np.array([r["nu_ci_high"] for r in data], dtype=float)
            err = np.vstack((y - y_low, y_high - y))
            plt.errorbar(
                x,
                y,
                yerr=err,
                fmt=marker,
                linestyle="None",
                capsize=3,
                label=f"{model} {observable}",
            )
    plt.xscale("log", base=2)
    plt.xlabel("Minimum N included in fit")
    plt.ylabel(r"Fitted $\nu$")
    plt.title("Flory exponent fit-window sensitivity")
    plt.legend(fontsize=7, ncol=2)
    plt.grid(alpha=0.25, which="both")
    savefig(RESULT / f"{stem}_fit_window_sensitivity.png")
    return fit_rows


def simulate_3d_return_probability(rng, n_walkers: int = 30000, n_steps: int = 5000) -> list[dict[str, object]]:
    pos = np.zeros((n_walkers, 3), dtype=np.int32)
    ever = np.zeros(n_walkers, dtype=bool)
    records: list[dict[str, object]] = []
    record_steps = set([10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
    record_steps.update(range(100, n_steps + 1, 100))
    for step in range(1, n_steps + 1):
        axis = rng.integers(0, 3, n_walkers)
        sign = rng.choice(np.array([-1, 1], dtype=np.int32), n_walkers)
        pos[np.arange(n_walkers), axis] += sign
        at_origin = np.all(pos == 0, axis=1)
        ever |= at_origin
        if step in record_steps:
            records.append(
                {
                    "steps": step,
                    "walkers": n_walkers,
                    "ever_return_fraction": float(np.mean(ever)),
                    "origin_fraction_at_step": float(np.mean(at_origin)),
                }
            )
    return records
