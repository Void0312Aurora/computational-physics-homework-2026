from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.special import erf


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "result"
RESULT.mkdir(exist_ok=True)

SEED = 20260527
PROBLEM1_BATCH_DRAWS = 100_000_000
EXACT_EXP = math.e - 1.0
EXACT_SINGULAR = 2.0 * math.log(2.0)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def float_fmt(value: float, digits: int = 12) -> float:
    return float(f"{value:.{digits}g}")


def log_spaced_ints(start: int, stop: int, count: int) -> np.ndarray:
    values = np.unique(np.rint(np.geomspace(start, stop, count)).astype(int))
    values[0] = start
    values[-1] = stop
    return values


def smooth_positive(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return values.copy()
    log_values = np.log(np.maximum(values, 1e-300))
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(log_values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window) / window
    return np.exp(np.convolve(padded, kernel, mode="valid"))


def smooth_series(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return values.copy()
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def consume_random_draws(rng: np.random.Generator, count: int, chunk: int = 5_000_000) -> None:
    remaining = count
    while remaining > 0:
        n = min(chunk, remaining)
        rng.random(n)
        remaining -= n


class ParkMiller:
    def __init__(self, seed: int = 1) -> None:
        self.state = seed % 2147483647
        if self.state <= 0:
            self.state += 2147483646

    def random(self, n: int) -> np.ndarray:
        values = np.empty(n, dtype=np.float64)
        state = self.state
        for i in range(n):
            state = (16807 * state) % 2147483647
            values[i] = state / 2147483647.0
        self.state = state
        return values


def halton_sequence(size: int, base: int, start: int = 1) -> np.ndarray:
    seq = np.empty(size, dtype=np.float64)
    for idx, n in enumerate(range(start, start + size)):
        f = 1.0 / base
        r = 0.0
        m = n
        while m > 0:
            r += f * (m % base)
            m //= base
            f /= base
        seq[idx] = r
    return seq


def singular_f(x: np.ndarray) -> np.ndarray:
    return 1.0 / (np.sqrt(x) + x)


def append_log(lines: Iterable[str]) -> None:
    with (RESULT / "temp-01.log").open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line.rstrip() + "\n")


def problem1(rng: np.random.Generator) -> dict[str, object]:
    legacy_samples_per_estimate = 1000
    legacy_repeats = [10_000, 100_000]
    sample_sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    extra_repeats = {10_000: 10_000, 100_000: 10_000, 1_000_000: 1_000, 10_000_000: 200}
    rows: list[dict[str, object]] = []
    dist_by_n: dict[int, tuple[int, np.ndarray]] = {}

    variance = (math.e**2 - 1.0) / 2.0 - EXACT_EXP**2
    colors = {
        1_000: "#37b24d",
        10_000: "#f59f00",
        100_000: "#f03e3e",
        1_000_000: "#7048e8",
        10_000_000: "#1c7ed6",
    }
    markers = {1_000: "x", 10_000: "o", 100_000: "+", 1_000_000: "d", 10_000_000: "^"}

    def normal_bin_probabilities(bin_edges: np.ndarray, std: float) -> np.ndarray:
        z = (bin_edges - EXACT_EXP) / (std * math.sqrt(2.0))
        cdf = 0.5 * (1.0 + erf(z))
        return np.diff(cdf)

    def batch_size(samples_per_estimate: int, repeat: int) -> int:
        return min(repeat, max(1, PROBLEM1_BATCH_DRAWS // samples_per_estimate))

    def exp_mean_rows(u: np.ndarray) -> np.ndarray:
        np.exp(u, out=u)
        return u.mean(axis=1)

    for repeat in legacy_repeats:
        estimates = np.empty(repeat)
        batch = batch_size(legacy_samples_per_estimate, repeat)
        pos = 0
        while pos < repeat:
            b = min(batch, repeat - pos)
            u = rng.random((b, legacy_samples_per_estimate))
            estimates[pos : pos + b] = exp_mean_rows(u)
            pos += b
        if repeat == max(legacy_repeats):
            dist_by_n[legacy_samples_per_estimate] = (repeat, estimates)
            empirical_std = float(np.std(estimates, ddof=1))
            theory_std = math.sqrt(variance / legacy_samples_per_estimate)
            rows.append(
                {
                    "N": legacy_samples_per_estimate,
                    "repeat": repeat,
                    "mean_estimate": float_fmt(float(np.mean(estimates))),
                    "empirical_std": float_fmt(empirical_std),
                    "theory_std": float_fmt(theory_std),
                    "exact": float_fmt(EXACT_EXP),
                    "mean_abs_error": float_fmt(abs(float(np.mean(estimates)) - EXACT_EXP)),
                }
            )

    extra_rng = np.random.default_rng(SEED + 10001)
    for n, repeat in extra_repeats.items():
        estimates = np.empty(repeat)
        batch = batch_size(n, repeat)
        pos = 0
        while pos < repeat:
            b = min(batch, repeat - pos)
            u = extra_rng.random((b, n))
            estimates[pos : pos + b] = exp_mean_rows(u)
            pos += b
        dist_by_n[n] = (repeat, estimates)
        empirical_std = float(np.std(estimates, ddof=1))
        theory_std = math.sqrt(variance / n)
        rows.append(
            {
                "N": n,
                "repeat": repeat,
                "mean_estimate": float_fmt(float(np.mean(estimates))),
                "empirical_std": float_fmt(empirical_std),
                "theory_std": float_fmt(theory_std),
                "exact": float_fmt(EXACT_EXP),
                "mean_abs_error": float_fmt(abs(float(np.mean(estimates)) - EXACT_EXP)),
            }
        )

    final_n = 1_000_000
    u = rng.random(final_n)
    np.exp(u, out=u)
    final_est = float(np.mean(u))
    final_se = float(np.std(u, ddof=1) / math.sqrt(final_n))
    rows.append(
        {
            "N": final_n,
            "repeat": 1,
            "mean_estimate": float_fmt(final_est),
            "empirical_std": "",
            "theory_std": float_fmt(math.sqrt(variance / final_n)),
            "exact": float_fmt(EXACT_EXP),
            "mean_abs_error": float_fmt(abs(final_est - EXACT_EXP)),
        }
    )

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    x_min, x_max = 1.62, 1.82
    bins = np.linspace(x_min, x_max, 201)
    centers = 0.5 * (bins[:-1] + bins[1:])

    for n in sample_sizes:
        repeat, estimates = dist_by_n[n]
        counts, _ = np.histogram(estimates, bins=bins)
        y = counts / repeat
        label = f"N={n:,}"
        ax.plot(
            centers,
            y,
            color=colors[n],
            marker=markers[n],
            markersize=3.2,
            linewidth=1.0,
            markevery=3,
            label=label,
        )

        theory_std = math.sqrt(variance / n)
        normal_y = normal_bin_probabilities(bins, theory_std)
        ax.plot(centers, normal_y, color=colors[n], linestyle=":", linewidth=1.1, alpha=0.68)

    ax.axvline(EXACT_EXP, color="black", linewidth=1.2, linestyle="--", label=r"$e-1$ / Dirac limit")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("MC estimate of integral")
    ax.set_ylabel("Bin probability, fixed bin width 0.001")
    ax.set_title(r"MC estimate distribution contracts toward a Dirac delta")
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULT / "problem1_mc_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    zoom_min = EXACT_EXP - 0.004
    zoom_max = EXACT_EXP + 0.004
    zoom_bins = np.arange(zoom_min, zoom_max + 0.00005, 0.0001)
    zoom_centers = 0.5 * (zoom_bins[:-1] + zoom_bins[1:])
    zoom_max_y = 0.0
    for n in [10_000, 100_000, 1_000_000, 10_000_000]:
        repeat, estimates = dist_by_n[n]
        counts, _ = np.histogram(estimates, bins=zoom_bins)
        y = counts / repeat
        theory_std = math.sqrt(variance / n)
        normal_y = normal_bin_probabilities(zoom_bins, theory_std)
        zoom_max_y = max(zoom_max_y, float(np.max(y)), float(np.max(normal_y)))
        ax.plot(
            zoom_centers,
            y,
            color=colors[n],
            marker=markers[n],
            markersize=2.6,
            linewidth=0.8,
            alpha=0.38,
            markevery=4,
        )
        ax.plot(zoom_centers, normal_y, color=colors[n], linewidth=1.8, label=f"N={n:,}")

    ax.axvline(EXACT_EXP, color="black", linewidth=1.2, linestyle="--", label=r"$e-1$")
    ax.set_xlim(zoom_min, zoom_max)
    ax.set_ylim(0.0, zoom_max_y * 1.14)
    ax.set_xlabel("MC estimate of integral")
    ax.set_ylabel("Bin probability, fixed bin width 0.0001")
    ax.set_title(r"Local view of contraction near $e-1$")
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULT / "problem1_mc_distribution_zoom.png", dpi=180)
    plt.close(fig)
    write_csv(RESULT / "problem1_distribution.csv", rows)

    return {
        "exact": EXACT_EXP,
        "variance_integrand": variance,
        "distribution_rows": rows,
        "final": {
            "N": final_n,
            "estimate": final_est,
            "standard_error": final_se,
            "absolute_error": abs(final_est - EXACT_EXP),
            "z_error": abs(final_est - EXACT_EXP) / final_se,
        },
    }


def problem2(rng: np.random.Generator) -> dict[str, object]:
    n = 1_000_000
    u = rng.random(n)
    x = u * u
    weights = 2.0 / (1.0 + np.sqrt(x))
    estimate = float(np.mean(weights))
    sd = float(np.std(weights, ddof=1))
    se = sd / math.sqrt(n)

    rows = [
        {
            "method": "importance_g_alpha_over_sqrt_x",
            "alpha": 0.5,
            "N": n,
            "estimate": float_fmt(estimate),
            "standard_deviation_of_weight": float_fmt(sd),
            "standard_error": float_fmt(se),
            "exact": float_fmt(EXACT_SINGULAR),
            "absolute_error": float_fmt(abs(estimate - EXACT_SINGULAR)),
        }
    ]
    write_csv(RESULT / "problem2_importance.csv", rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(weights, bins=80, density=True, color="#2f9e44", alpha=0.78)
    ax.axvline(EXACT_SINGULAR, color="black", linewidth=1.2, label=r"exact $2\ln 2$")
    ax.axvline(estimate, color="#c92a2a", linewidth=1.2, label="MC mean")
    ax.set_xlabel(r"importance weight $2/(1+\sqrt{x})$")
    ax.set_ylabel("Density")
    ax.set_title("Finite-variance importance weights")
    ax.grid(alpha=0.24)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT / "problem2_weight_distribution.png", dpi=180)
    plt.close(fig)

    ns = log_spaced_ints(500, 200_000, 28)
    repeats = 240
    convergence_rows: list[dict[str, object]] = []
    uniform_errors = []
    importance_errors = []
    conv_rng = np.random.default_rng(SEED + 20002)
    draws_for_convergence = 0
    for sample_count in ns:
        uniform_est = np.empty(repeats)
        importance_est = np.empty(repeats)
        for r in range(repeats):
            uu = conv_rng.random(sample_count)
            uu = np.maximum(uu, np.finfo(float).tiny)
            uniform_est[r] = float(np.mean(singular_f(uu)))
            ui = conv_rng.random(sample_count)
            xi = ui * ui
            importance_est[r] = float(np.mean(2.0 / (1.0 + np.sqrt(xi))))
            draws_for_convergence += 2 * int(sample_count)
        uniform_rms = float(np.sqrt(np.mean((uniform_est - EXACT_SINGULAR) ** 2)))
        importance_rms = float(np.sqrt(np.mean((importance_est - EXACT_SINGULAR) ** 2)))
        uniform_errors.append(uniform_rms)
        importance_errors.append(importance_rms)
        convergence_rows.append(
            {
                "method": "uniform_simple_mc",
                "N": int(sample_count),
                "repeat": repeats,
                "rms_error": float_fmt(uniform_rms),
                "median_abs_error": float_fmt(float(np.median(np.abs(uniform_est - EXACT_SINGULAR)))),
            }
        )
        convergence_rows.append(
            {
                "method": "importance_sampling",
                "N": int(sample_count),
                "repeat": repeats,
                "rms_error": float_fmt(importance_rms),
                "median_abs_error": float_fmt(float(np.median(np.abs(importance_est - EXACT_SINGULAR)))),
            }
        )
    write_csv(RESULT / "problem2_convergence.csv", convergence_rows)

    uniform_errors_arr = np.array(uniform_errors)
    importance_errors_arr = np.array(importance_errors)
    uniform_smooth = smooth_positive(uniform_errors_arr, 5)
    importance_smooth = smooth_positive(importance_errors_arr, 5)
    fig, ax = plt.subplots(figsize=(7.5, 4.9))
    ax.loglog(ns, uniform_errors_arr, "o", color="#c92a2a", alpha=0.35, markersize=4, label="uniform RMS samples")
    ax.loglog(ns, importance_errors_arr, "s", color="#2f9e44", alpha=0.35, markersize=4, label="importance RMS samples")
    ax.loglog(ns, uniform_smooth, "-", color="#c92a2a", linewidth=2.0, label="uniform smoothed trend")
    ax.loglog(ns, importance_smooth, "-", color="#2f9e44", linewidth=2.0, label=r"importance trend")
    ref = importance_smooth[-1] * (ns / ns[-1]) ** (-0.5)
    ax.loglog(ns, ref, ":", color="black", label=r"reference $N^{-1/2}$")
    ax.set_xlabel("number of samples N")
    ax.set_ylabel("RMS absolute error")
    ax.set_title(r"Importance sampling restores finite-variance convergence")
    ax.grid(alpha=0.24, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT / "problem2_convergence.png", dpi=180)
    plt.close(fig)

    consume_random_draws(rng, draws_for_convergence)

    return {
        "exact": EXACT_SINGULAR,
        "estimate": estimate,
        "standard_deviation_of_weight": sd,
        "standard_error": se,
        "absolute_error": abs(estimate - EXACT_SINGULAR),
        "convergence_rows": convergence_rows,
    }


@dataclass
class StratifiedResult:
    method: str
    estimate: float
    repeat_std: float
    repeat_var: float
    mean_abs_error: float
    mean_samples: float
    min_n: int
    max_n: int


def fixed_stratified_once(rng: np.random.Generator, k: int, n_each: int) -> float:
    u = rng.random((k, n_each))
    left = np.arange(k, dtype=np.float64)[:, None] / k
    x = left + u / k
    values = singular_f(x)
    return float(values.mean(axis=1).mean())


def optimal_allocation(k: int, total: int, pilot_n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    u = rng.random((k, pilot_n))
    left = np.arange(k, dtype=np.float64)[:, None] / k
    x = left + u / k
    sigma = np.std(singular_f(x), axis=1, ddof=1)
    weights = sigma + 1e-15
    raw = weights / np.sum(weights) * total
    alloc = np.maximum(2, np.floor(raw).astype(int))
    delta = int(total - np.sum(alloc))
    frac = raw - np.floor(raw)
    if delta > 0:
        order = np.argsort(-frac)
        alloc[order[:delta]] += 1
    elif delta < 0:
        order = np.argsort(frac)
        idx = 0
        while delta < 0 and idx < len(order):
            j = order[idx]
            removable = min(alloc[j] - 2, -delta)
            if removable > 0:
                alloc[j] -= removable
                delta += removable
            idx += 1
    return alloc, sigma


def stratum_integral(a: float, b: float) -> float:
    return 2.0 * math.log((1.0 + math.sqrt(b)) / (1.0 + math.sqrt(a)))


def stratum_second_moment_integral(a: float, b: float) -> float:
    ta = math.sqrt(a)
    tb = math.sqrt(b)

    def primitive(t: float) -> float:
        return 2.0 * math.log(t) - 2.0 * math.log1p(t) + 2.0 / (1.0 + t)

    return primitive(tb) - primitive(ta)


def exact_stratum_sigma(a: float, b: float) -> float:
    h = b - a
    mean = stratum_integral(a, b) / h
    second = stratum_second_moment_integral(a, b) / h
    return math.sqrt(max(second - mean * mean, 0.0))


def exact_neyman_allocation_regularized(k: int, total: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    h = 1.0 / k
    first_exact = stratum_integral(0.0, h)
    sigmas = np.empty(k - 1, dtype=np.float64)
    for j in range(k - 1):
        a = (j + 1) / k
        b = (j + 2) / k
        sigmas[j] = exact_stratum_sigma(a, b)

    raw = sigmas / np.sum(sigmas) * total
    alloc = np.maximum(1, np.floor(raw).astype(int))
    delta = int(total - np.sum(alloc))
    frac = raw - np.floor(raw)
    if delta > 0:
        alloc[np.argsort(-frac)[:delta]] += 1
    elif delta < 0:
        order = np.argsort(frac)
        idx = 0
        while delta < 0 and idx < len(order):
            j = order[idx]
            removable = min(alloc[j] - 1, -delta)
            if removable > 0:
                alloc[j] -= removable
                delta += removable
            idx += 1

    predicted_var = float(np.sum((h * sigmas) ** 2 / alloc))
    return alloc, sigmas, first_exact, predicted_var


def allocated_stratified_once(
    rng: np.random.Generator, alloc: np.ndarray, *, offset: int = 0, base: float = 0.0
) -> float:
    full_k = len(alloc) + offset
    total = base
    for i, ni in enumerate(alloc):
        u = rng.random(int(ni))
        stratum_index = i + offset
        x = (stratum_index + u) / full_k
        total += float(np.mean(singular_f(x))) / full_k
    return total


def problem3(rng: np.random.Generator) -> dict[str, object]:
    k = 5000
    n_each = 10
    total = k * n_each
    repeats = 140

    simple = np.empty(repeats)
    fixed = np.empty(repeats)
    for r in range(repeats):
        u = rng.random(total)
        u = np.maximum(u, np.finfo(float).tiny)
        simple[r] = np.mean(singular_f(u))
        fixed[r] = fixed_stratified_once(rng, k, n_each)

    alloc, sigma, first_exact, predicted_var = exact_neyman_allocation_regularized(k, total)
    optimal = np.empty(repeats)
    for r in range(repeats):
        optimal[r] = allocated_stratified_once(rng, alloc, offset=1, base=first_exact)

    method_arrays = {
        "simple_uniform_mc": (simple, total, total, total),
        "stratified_fixed_ni_10": (fixed, total, n_each, n_each),
        "stratified_regularized_neyman": (optimal, total, int(np.min(alloc)), int(np.max(alloc))),
    }
    rows: list[dict[str, object]] = []
    results: dict[str, dict[str, object]] = {}
    for name, (arr, samples, min_n, max_n) in method_arrays.items():
        repeat_std = float(np.std(arr, ddof=1))
        row = {
            "method": name,
            "repeat": repeats,
            "mean_estimate": float_fmt(float(np.mean(arr))),
            "repeat_std": float_fmt(repeat_std),
            "repeat_variance": float_fmt(repeat_std**2),
            "predicted_std": float_fmt(math.sqrt(predicted_var))
            if name == "stratified_regularized_neyman"
            else "",
            "exact": float_fmt(EXACT_SINGULAR),
            "mean_abs_error": float_fmt(abs(float(np.mean(arr)) - EXACT_SINGULAR)),
            "total_samples": samples,
            "min_ni": min_n,
            "max_ni": max_n,
        }
        rows.append(row)
        results[name] = row
    write_csv(RESULT / "problem3_stratified.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    labels = ["Simple MC", "Fixed stratified", "Regularized Neyman"]
    data = [simple, fixed, optimal]
    axes[0].boxplot(data, tick_labels=labels, showfliers=False)
    axes[0].axhline(EXACT_SINGULAR, color="black", linewidth=1.2, linestyle="--")
    axes[0].set_ylabel("Integral estimate")
    axes[0].set_title("Repeated estimates")
    axes[0].grid(axis="y", alpha=0.24)

    plotted_strata = np.arange(2, 122)
    continuous_alloc = sigma / np.sum(sigma) * total
    axes[1].scatter(
        plotted_strata,
        alloc[:120],
        color="#5f3dc4",
        s=14,
        alpha=0.38,
        linewidths=0,
        label="integer allocation",
    )
    axes[1].plot(
        plotted_strata,
        continuous_alloc[:120],
        color="#5f3dc4",
        linewidth=2.0,
        label="continuous Neyman trend",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Stratum index near x=0")
    axes[1].set_ylabel("allocated samples")
    axes[1].set_title("Regularized Neyman allocation")
    axes[1].grid(alpha=0.24, which="both")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULT / "problem3_variance_comparison.png", dpi=180)
    plt.close(fig)

    return {
        "exact": EXACT_SINGULAR,
        "k": k,
        "fixed_n_i": n_each,
        "total_samples": total,
        "repeats": repeats,
        "rows": rows,
        "optimal_allocation": {
            "first_stratum_exact": first_exact,
            "predicted_std": math.sqrt(predicted_var),
            "min": int(np.min(alloc)),
            "max": int(np.max(alloc)),
            "strata_2_to_11": [int(x) for x in alloc[:10]],
            "last_10": [int(x) for x in alloc[-10:]],
            "sigma_strata_2_to_6": [float(x) for x in sigma[:5]],
        },
    }


def estimate_singular_from_u(u: np.ndarray) -> float:
    x = np.maximum(u, np.finfo(float).tiny)
    return float(np.mean(singular_f(x)))


def running_lcg_estimates(ns: list[int], seed: int) -> list[float]:
    gen = ParkMiller(seed)
    max_n = max(ns)
    u = gen.random(max_n)
    values = singular_f(np.maximum(u, np.finfo(float).tiny))
    csum = np.cumsum(values)
    return [float(csum[n - 1] / n) for n in ns]


def running_halton_estimates(ns: list[int]) -> list[float]:
    max_n = max(ns)
    u = halton_sequence(max_n, 2, start=1)
    values = singular_f(np.maximum(u, np.finfo(float).tiny))
    csum = np.cumsum(values)
    return [float(csum[n - 1] / n) for n in ns]


def fit_exponent(ns: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    mask = errors > 1e-15
    coeff = np.polyfit(np.log(ns[mask]), np.log(errors[mask]), 1)
    return float(coeff[0]), float(coeff[1])


def problem4() -> dict[str, object]:
    ns = log_spaced_ints(256, 262_144, 26)
    ns_list = [int(x) for x in ns]

    pseudo_repeats = 96
    # Average pseudo-random errors over independent LCG streams before fitting.
    lcg_errors_all = []
    lcg_estimates_all = []
    for rep in range(pseudo_repeats):
        estimates = running_lcg_estimates(ns_list, seed=13579 + 97 * rep)
        lcg_estimates_all.append(estimates)
        lcg_errors_all.append([abs(v - EXACT_SINGULAR) for v in estimates])
    lcg_est_mean = np.mean(np.array(lcg_estimates_all), axis=0)
    lcg_err_rms = np.sqrt(np.mean(np.array(lcg_errors_all) ** 2, axis=0))

    halton_est = np.array(running_halton_estimates(ns_list))
    halton_err = np.abs(halton_est - EXACT_SINGULAR)

    lcg_slope, lcg_intercept = fit_exponent(ns.astype(float), lcg_err_rms)
    halton_slope, halton_intercept = fit_exponent(ns.astype(float), halton_err)

    rows: list[dict[str, object]] = []
    for i, n in enumerate(ns_list):
        rows.append(
            {
                "method": f"park_miller_lcg_rms_{pseudo_repeats}_runs",
                "N": n,
                "estimate": float_fmt(float(lcg_est_mean[i])),
                "absolute_error": float_fmt(float(lcg_err_rms[i])),
                "fit_exponent": float_fmt(lcg_slope),
            }
        )
        rows.append(
            {
                "method": "halton_base2",
                "N": n,
                "estimate": float_fmt(float(halton_est[i])),
                "absolute_error": float_fmt(float(halton_err[i])),
                "fit_exponent": float_fmt(halton_slope),
            }
        )
    write_csv(RESULT / "problem4_convergence.csv", rows)

    lcg_frac = lcg_err_rms / EXACT_SINGULAR
    halton_frac = halton_err / EXACT_SINGULAR
    lcg_fit_frac = math.exp(lcg_intercept) * ns**lcg_slope / EXACT_SINGULAR
    halton_fit_frac = math.exp(halton_intercept) * ns**halton_slope / EXACT_SINGULAR
    fig, ax = plt.subplots(figsize=(7.6, 5.1))
    ax.loglog(ns, lcg_frac, "o", color="#c92a2a", alpha=0.28, markersize=4, label="pseudo-random RMS samples")
    ax.loglog(ns, halton_frac, "s", color="#1971c2", alpha=0.28, markersize=4, label="Halton samples")
    ax.loglog(ns, lcg_fit_frac, "-", color="#c92a2a", linewidth=2.25, label=f"pseudo-random fit, slope {lcg_slope:.3f}")
    ax.loglog(ns, halton_fit_frac, "-", color="#1971c2", linewidth=2.25, label=f"Halton fit, slope {halton_slope:.3f}")
    ref_half = lcg_frac[-1] * (ns / ns[-1]) ** (-0.5)
    ref_two_thirds = halton_frac[-1] * (ns / ns[-1]) ** (-2.0 / 3.0)
    ref_one = halton_frac[-1] * (ns / ns[-1]) ** (-1.0)
    ax.loglog(ns, ref_half, ":", color="black", label=r"$N^{-1/2}$")
    ax.loglog(ns, ref_two_thirds, ":", color="#868e96", label=r"$N^{-2/3}$")
    ax.loglog(ns, ref_one, "-.", color="#868e96", linewidth=1.0, label=r"$N^{-1}$")
    ax.set_xlabel("number of points N")
    ax.set_ylabel("fractional accuracy of integral")
    ax.set_title(r"Pseudo-random and quasi-random convergence")
    ax.grid(which="both", alpha=0.24)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT / "problem4_convergence.png", dpi=180)
    plt.close(fig)

    return {
        "exact": EXACT_SINGULAR,
        "lcg_exponent": lcg_slope,
        "halton_exponent": halton_slope,
        "rows": rows,
        "final_lcg": rows[-2],
        "final_halton": rows[-1],
    }


def oscillating_h(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.sin(2.0 * np.sqrt(np.maximum(math.pi**2 - x * x, 0.0)))


def problem5(rng: np.random.Generator) -> dict[str, object]:
    exact, exact_err = quad(
        lambda t: 2.0 * math.sin(2.0 * math.sqrt(max(math.pi**2 - t * t, 0.0))),
        0.0,
        math.pi,
        epsabs=1e-12,
        limit=300,
    )
    shift = 2.0
    height = 4.0
    ns = log_spaced_ints(1000, 5_000_000, 72)
    max_n = int(ns[-1])
    hm_rng = np.random.default_rng(SEED + 50005)
    x = hm_rng.random(max_n) * math.pi
    y = hm_rng.random(max_n) * height
    positive = oscillating_h(x) + shift
    hits = y <= positive
    csum = np.cumsum(hits)

    rows: list[dict[str, object]] = []
    estimates = []
    for n in ns:
        area_positive = math.pi * height * float(csum[n - 1] / n)
        estimate = area_positive - shift * math.pi
        p_hat = float(csum[n - 1] / n)
        se = math.pi * height * math.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / n)
        estimates.append(estimate)
        rows.append(
            {
                "N": int(n),
                "estimate": float_fmt(estimate),
                "exact_quad": float_fmt(float(exact)),
                "standard_error": float_fmt(se),
                "absolute_error": float_fmt(abs(estimate - exact)),
                "hit_fraction": float_fmt(p_hat),
            }
        )
    write_csv(RESULT / "problem5_hitmiss.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    show = 18000
    colors = np.where(hits[:show], "#1c7ed6", "#e03131")
    axes[0].scatter(x[:show], y[:show], c=colors, s=2.2, alpha=0.72, linewidths=0)
    grid = np.linspace(0.0, math.pi, 600)
    axes[0].plot(grid, oscillating_h(grid) + shift, color="black", linewidth=1.2)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("shifted hit-and-miss points")
    axes[0].set_xlim(0.0, math.pi)
    axes[0].set_ylim(0.0, height)
    axes[0].grid(alpha=0.2)

    estimates_arr = np.array(estimates)
    estimates_smooth = smooth_series(estimates_arr, 9)

    trace_ns = np.arange(100, 60_001, 100)
    trace_estimates = math.pi * height * (csum[trace_ns - 1] / trace_ns) - shift * math.pi
    trace_smooth = smooth_series(trace_estimates, 21)
    axes[1].plot(trace_ns, trace_estimates, color="#1c7ed6", linewidth=0.65, alpha=0.30)
    axes[1].plot(trace_ns, trace_smooth, color="#1c7ed6", linewidth=1.55)
    axes[1].axhline(exact, color="black", linestyle="--", linewidth=1.1, label="reference")
    axes[1].set_xlabel("N trials")
    axes[1].set_ylabel("I")
    axes[1].set_title("running estimate, first 60000 trials")
    axes[1].grid(alpha=0.24)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULT / "problem5_hitmiss_points.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(ns, estimates_arr, "o", color="#5f3dc4", alpha=0.28, markersize=3.2, label="running samples")
    ax.plot(ns, estimates_smooth, "-", color="#5f3dc4", linewidth=2.0, label="smoothed trend")
    se_arr = np.array([float(row["standard_error"]) for row in rows])
    ax.fill_between(
        ns,
        exact - se_arr,
        exact + se_arr,
        color="#5f3dc4",
        alpha=0.12,
        linewidth=0,
        label=r"reference $\pm 1$ SE",
    )
    ax.axhline(exact, color="black", linestyle="--", linewidth=1.1, label="quadrature reference")
    ax.set_xscale("log")
    ax.set_xlabel("number of trials N")
    ax.set_ylabel("integral estimate")
    ax.set_title("Convergence of hit-and-miss integration")
    ax.grid(alpha=0.24)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT / "problem5_convergence.png", dpi=180)
    plt.close(fig)

    consume_random_draws(rng, 2 * max_n)

    return {
        "reference_quad": exact,
        "quad_error_estimate": exact_err,
        "shift": shift,
        "height": height,
        "rows": rows,
        "final": rows[-1],
    }


def sphere_volume_exact(dim: int, radius: float = 1.0) -> float:
    return math.pi ** (dim / 2.0) * radius**dim / math.gamma(dim / 2.0 + 1.0)


def sphere_surface(dim: int, radius: float = 1.0) -> float:
    return 2.0 * math.pi ** (dim / 2.0) * radius ** (dim - 1.0) / math.gamma(dim / 2.0)


def radial_mc_volume(dim: int, n: int, rng: np.random.Generator, radius: float = 1.0) -> tuple[float, float]:
    r = rng.random(n) * radius
    values = sphere_surface(dim, 1.0) * r ** (dim - 1)
    estimate = radius * float(np.mean(values))
    se = radius * float(np.std(values, ddof=1) / math.sqrt(n))
    return estimate, se


def cube_mc_volume(dim: int, n: int, rng: np.random.Generator, radius: float = 1.0, batch: int = 50_000) -> tuple[float, float, int]:
    hits = 0
    done = 0
    while done < n:
        m = min(batch, n - done)
        pts = rng.uniform(-radius, radius, size=(m, dim))
        hits += int(np.count_nonzero(np.sum(pts * pts, axis=1) <= radius * radius))
        done += m
    p = hits / n
    volume = (2.0 * radius) ** dim * p
    se = (2.0 * radius) ** dim * math.sqrt(max(p * (1.0 - p), 0.0) / n)
    return volume, se, hits


def problem6(rng: np.random.Generator) -> dict[str, object]:
    dims = [2, 3, 5, 10, 15, 20, 25, 30]
    cube_n = {2: 200_000, 3: 200_000, 5: 300_000, 10: 500_000, 15: 700_000, 20: 1_000_000, 25: 1_000_000, 30: 1_000_000}
    radial_n = 300_000
    rows: list[dict[str, object]] = []

    for dim in dims:
        exact = sphere_volume_exact(dim)
        cube_est, cube_se, hits = cube_mc_volume(dim, cube_n[dim], rng)
        radial_est, radial_se = radial_mc_volume(dim, radial_n, rng)
        cube_probability = exact / (2.0**dim)
        expected_hits = cube_n[dim] * cube_probability
        cube_relative_se_theory = (
            math.sqrt((1.0 - cube_probability) / (cube_n[dim] * cube_probability))
            if cube_probability > 0.0
            else float("inf")
        )
        rows.append(
            {
                "dimension": dim,
                "radius": 1,
                "exact_volume": float_fmt(exact),
                "cube_mc_N": cube_n[dim],
                "cube_expected_hits": float_fmt(expected_hits),
                "cube_hits": hits,
                "cube_estimate": float_fmt(cube_est),
                "cube_standard_error": float_fmt(cube_se),
                "cube_relative_se_theory": float_fmt(cube_relative_se_theory),
                "cube_relative_error": float_fmt(abs(cube_est - exact) / exact if exact else float("nan")),
                "radial_mc_N": radial_n,
                "radial_estimate": float_fmt(radial_est),
                "radial_standard_error": float_fmt(radial_se),
                "radial_relative_standard_error": float_fmt(radial_se / exact if exact else float("nan")),
                "radial_relative_error": float_fmt(abs(radial_est - exact) / exact if exact else float("nan")),
            }
        )
    write_csv(RESULT / "problem6_sphere.csv", rows)

    dims_arr = np.array(dims)
    exact_arr = np.array([sphere_volume_exact(d) for d in dims])
    cube_arr = np.array([float(r["cube_estimate"]) for r in rows])
    radial_arr = np.array([float(r["radial_estimate"]) for r in rows])

    expected_hits = np.array([float(r["cube_expected_hits"]) for r in rows])
    radial_rel_err = np.array([float(r["radial_relative_error"]) for r in rows])
    radial_rel_se = np.array([float(r["radial_relative_standard_error"]) for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    positive_cube = np.where(cube_arr > 0.0, cube_arr, np.nan)
    zero_hit_dims = dims_arr[cube_arr == 0.0]
    zero_marker_y = np.full_like(zero_hit_dims, np.min(exact_arr) * 0.45, dtype=np.float64)
    axes[0].semilogy(dims_arr, exact_arr, "o-", color="black", label="exact volume")
    axes[0].semilogy(dims_arr, positive_cube, "s--", color="#c92a2a", label="cube hit-or-miss")
    if len(zero_hit_dims) > 0:
        axes[0].semilogy(
            zero_hit_dims,
            zero_marker_y,
            "v",
            color="#c92a2a",
            markersize=7,
            label="cube zero-hit",
        )
    axes[0].semilogy(dims_arr, radial_arr, "d-", color="#1971c2", label="radial MC")
    axes[0].set_xlabel("dimension")
    axes[0].set_ylabel("unit sphere volume")
    axes[0].set_title("volume estimate")
    axes[0].set_ylim(np.min(exact_arr) * 0.25, np.max(exact_arr) * 2.5)
    axes[0].grid(alpha=0.24, which="both")
    axes[0].legend()

    axes[1].semilogy(dims_arr, expected_hits, "o-", color="#c92a2a", label="expected cube hits")
    axes[1].semilogy(dims_arr, radial_rel_err, "s-", color="#1971c2", label="radial relative error")
    axes[1].semilogy(dims_arr, radial_rel_se, ":", color="#1971c2", label="radial relative SE")
    axes[1].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xlabel("dimension")
    axes[1].set_title("why cube hit-or-miss fails")
    axes[1].grid(alpha=0.24, which="both")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULT / "problem6_sphere_volumes.png", dpi=180)
    plt.close(fig)

    return {"rows": rows, "dim20": next(r for r in rows if r["dimension"] == 20), "dim25": next(r for r in rows if r["dimension"] == 25)}


def main() -> None:
    (RESULT / "temp-01.log").write_text("", encoding="utf-8")
    rng = np.random.default_rng(SEED)
    append_log(["HW/11 Monte Carlo experiment run", f"seed={SEED}"])

    summary: dict[str, object] = {}
    summary["problem1"] = problem1(rng)
    append_log(["Problem 1 completed"])
    summary["problem2"] = problem2(rng)
    append_log(["Problem 2 completed"])
    summary["problem3"] = problem3(rng)
    append_log(["Problem 3 completed"])
    summary["problem4"] = problem4()
    append_log(["Problem 4 completed"])
    summary["problem5"] = problem5(rng)
    append_log(["Problem 5 completed"])
    summary["problem6"] = problem6(rng)
    append_log(["Problem 6 completed"])

    serializable = json.loads(json.dumps(summary, default=float))
    (RESULT / "hw11_summary.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    append_log(["Summary written to result/hw11_summary.json"])


if __name__ == "__main__":
    main()
