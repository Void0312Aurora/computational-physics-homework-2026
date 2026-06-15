from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare

from .config import CONFIG, RESULT
from .io_utils import savefig, write_csv
from .rngs import make_rng, rng_label


def lag1_corr(u: np.ndarray) -> float:
    x = u[:-1] - float(np.mean(u[:-1]))
    y = u[1:] - float(np.mean(u[1:]))
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def plot_single_metric_bars(rows: list[dict[str, object]]) -> None:
    labels = [r["generator"] for r in rows]
    x = np.arange(len(rows))
    metrics = [
        ("mean_abs_error", "Mean absolute error", "rng_uniform_mean_abs_error.png"),
        ("variance_abs_error", "Variance absolute error", "rng_uniform_variance_abs_error.png"),
        ("lag1_correlation", "Lag-1 correlation", "rng_uniform_lag1_correlation.png"),
        ("chi_square_pvalue", r"$\chi^2$ p-value", "rng_uniform_chi_square_pvalue.png"),
    ]
    for key, ylabel, filename in metrics:
        plt.figure(figsize=(7.0, 4.6))
        values = [r[key] for r in rows]
        colors = ["#2f80ed" if v >= 0 else "#d9480f" for v in values]
        plt.bar(x, values, color=colors)
        if key in {"lag1_correlation", "chi_square_pvalue"}:
            plt.axhline(0.0, color="k", lw=0.8)
        if key == "chi_square_pvalue":
            plt.axhline(0.05, color="r", ls="--", lw=1.0, label="0.05")
            plt.legend()
        plt.xticks(x, labels, rotation=15, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"Uniform(0,1) {ylabel}")
        plt.grid(axis="y", alpha=0.25)
        savefig(RESULT / filename)


def plot_per_generator_uniform(kind: str, label: str, u: np.ndarray, bins: int) -> None:
    plt.figure(figsize=(7.0, 4.6))
    plt.hist(u, bins=bins, range=(0.0, 1.0), density=True, alpha=0.72)
    plt.axhline(1.0, color="r", ls="--", lw=1.2, label="ideal density")
    plt.xlabel("u")
    plt.ylabel("Density")
    plt.title(f"Uniform histogram: {label}")
    plt.legend()
    plt.grid(alpha=0.25)
    savefig(RESULT / f"rng_uniform_hist_{kind}.png")

    points = min(5000, len(u) - 1)
    plt.figure(figsize=(5.6, 5.2))
    plt.scatter(u[:points], u[1 : points + 1], s=3, alpha=0.25)
    plt.xlabel(r"$u_i$")
    plt.ylabel(r"$u_{i+1}$")
    plt.title(f"Lag-1 scatter: {label}")
    plt.grid(alpha=0.25)
    savefig(RESULT / f"rng_uniform_lag1_scatter_{kind}.png")


def uniform_diagnostics(seed: int) -> list[dict[str, object]]:
    cfg = CONFIG["rng_diagnostics"]
    n = int(cfg["uniform_n"])
    bins = int(cfg["hist_bins"])
    rows: list[dict[str, object]] = []
    for offset, kind in enumerate(cfg["generators"]):
        rng = make_rng(kind, seed + 1000 * offset)
        u = np.asarray(rng.random(n), dtype=float)
        counts, _ = np.histogram(u, bins=bins, range=(0.0, 1.0))
        chi = chisquare(counts, np.full(bins, n / bins))
        label = rng_label(kind)
        plot_per_generator_uniform(kind, label, u, bins)
        rows.append(
            {
                "generator": label,
                "kind": kind,
                "samples": n,
                "mean": float(np.mean(u)),
                "mean_abs_error": float(abs(np.mean(u) - 0.5)),
                "variance": float(np.var(u)),
                "variance_abs_error": float(abs(np.var(u) - 1.0 / 12.0)),
                "lag1_correlation": lag1_corr(u),
                "chi_square": float(chi.statistic),
                "chi_square_pvalue": float(chi.pvalue),
            }
        )
    fields = [
        "generator",
        "kind",
        "samples",
        "mean",
        "mean_abs_error",
        "variance",
        "variance_abs_error",
        "lag1_correlation",
        "chi_square",
        "chi_square_pvalue",
    ]
    write_csv(RESULT / "rng_uniform_diagnostics.csv", rows, fields)

    plt.figure(figsize=(7.2, 4.7))
    labels = [r["generator"] for r in rows]
    x = np.arange(len(rows))
    plt.bar(x - 0.2, [r["mean_abs_error"] for r in rows], width=0.4, label="mean abs error")
    plt.bar(x + 0.2, [r["variance_abs_error"] for r in rows], width=0.4, label="variance abs error")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Absolute error")
    plt.title("Uniform(0,1) moment diagnostics")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    savefig(RESULT / "rng_uniform_diagnostics.png")
    plot_single_metric_bars(rows)
    return rows


def plot_integer_diagnostics(rows: list[dict[str, object]], categories: np.ndarray, expected: float) -> None:
    labels = [r["generator"] for r in rows]
    x = np.arange(len(rows))

    plt.figure(figsize=(7.0, 4.6))
    plt.bar(x, [r["chi_square_pvalue"] for r in rows], color="#2f80ed")
    plt.axhline(0.05, color="r", ls="--", lw=1.0, label="0.05")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel(r"$\chi^2$ p-value")
    plt.title("Discrete integer uniformity test")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    savefig(RESULT / "rng_integer_chi_square_pvalue.png")

    plt.figure(figsize=(7.0, 4.6))
    plt.bar(x, [r["max_relative_count_error"] for r in rows], color="#12a37f")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Maximum relative count error")
    plt.title("Discrete integer count error")
    plt.grid(axis="y", alpha=0.25)
    savefig(RESULT / "rng_integer_count_error.png")

    offsets = np.linspace(-0.28, 0.28, len(rows))
    bar_width = min(0.8 / len(rows), 0.24)
    plt.figure(figsize=(7.4, 4.7))
    for offset, row in zip(offsets, rows):
        counts = [row[f"count_{int(value)}"] for value in categories]
        plt.bar(categories + offset, counts, width=bar_width, label=row["generator"])
    plt.axhline(expected, color="k", ls="--", lw=1.0, label="expected")
    plt.xticks(categories)
    plt.xlabel("Integer value")
    plt.ylabel("Count")
    plt.title("Discrete integer counts by generator")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    savefig(RESULT / "rng_integer_counts_comparison.png")

    for row in rows:
        kind = str(row["kind"])
        label = str(row["generator"])
        counts = [row[f"count_{int(value)}"] for value in categories]
        plt.figure(figsize=(6.6, 4.4))
        plt.bar(categories, counts, color="#2f80ed")
        plt.axhline(expected, color="r", ls="--", lw=1.1, label="expected")
        plt.xticks(categories)
        plt.xlabel("Integer value")
        plt.ylabel("Count")
        plt.title(f"Discrete integer counts: {label}")
        plt.legend()
        plt.grid(axis="y", alpha=0.25)
        savefig(RESULT / f"rng_integer_counts_{kind}.png")


def integer_diagnostics(seed: int) -> list[dict[str, object]]:
    cfg = CONFIG["rng_diagnostics"]
    n = int(cfg["integer_n"])
    low = int(cfg["integer_low"])
    high = int(cfg["integer_high"])
    width = high - low
    if width <= 0:
        raise ValueError("integer_high must be greater than integer_low")
    categories = np.arange(low, high)
    expected_counts = np.full(width, n / width)
    rows: list[dict[str, object]] = []
    for offset, kind in enumerate(cfg["generators"]):
        rng = make_rng(kind, seed + 10000 * offset)
        draws = np.asarray(rng.integers(low, high, n), dtype=np.int64)
        counts = np.bincount(draws - low, minlength=width)[:width]
        chi = chisquare(counts, expected_counts)
        row: dict[str, object] = {
            "generator": rng_label(kind),
            "kind": kind,
            "samples": n,
            "low": low,
            "high": high,
            "chi_square": float(chi.statistic),
            "chi_square_pvalue": float(chi.pvalue),
            "max_relative_count_error": float(np.max(np.abs(counts - expected_counts) / expected_counts)),
        }
        for value, count in zip(categories, counts):
            row[f"count_{int(value)}"] = int(count)
        rows.append(row)
    fields = [
        "generator",
        "kind",
        "samples",
        "low",
        "high",
        "chi_square",
        "chi_square_pvalue",
        "max_relative_count_error",
    ] + [f"count_{int(value)}" for value in categories]
    write_csv(RESULT / "rng_integer_diagnostics.csv", rows, fields)
    plot_integer_diagnostics(rows, categories, float(n / width))
    return rows


def plot_integral_per_generator(rows: list[dict[str, object]], ns: list[int], kind: str) -> None:
    label = rng_label(kind)
    problem_info = [
        ("hw11_exp_integral", r"HW11 $\int_0^1 e^x dx$", "exp"),
        ("hw11_singular_importance", r"HW11 importance sampling", "importance"),
    ]

    plt.figure(figsize=(7.2, 4.7))
    for problem_name, title, _suffix in problem_info:
        data = [r for r in rows if r["kind"] == kind and r["problem"] == problem_name]
        plt.loglog([r["N"] for r in data], [r["rms_abs_error"] for r in data], "o-", label=title)
    ref_n = np.array(ns, dtype=float)
    first = [r for r in rows if r["kind"] == kind][0]
    plt.loglog(ref_n, first["rms_abs_error"] * (ref_n[0] / ref_n) ** 0.5, "k--", lw=1.0, label=r"$N^{-1/2}$")
    plt.xlabel("N")
    plt.ylabel("RMS absolute error")
    plt.title(f"HW11-style integral tests: {label}")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25, which="both")
    savefig(RESULT / f"rng_hw11_integrals_{kind}.png")

    for problem_name, title, suffix in problem_info:
        data = [r for r in rows if r["kind"] == kind and r["problem"] == problem_name]
        plt.figure(figsize=(7.2, 4.7))
        plt.loglog([r["N"] for r in data], [r["rms_abs_error"] for r in data], "o-", label=label)
        ref_n = np.array(ns, dtype=float)
        plt.loglog(ref_n, data[0]["rms_abs_error"] * (ref_n[0] / ref_n) ** 0.5, "k--", lw=1.0, label=r"$N^{-1/2}$")
        plt.xlabel("N")
        plt.ylabel("RMS absolute error")
        plt.title(f"{title}: {label}")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.25, which="both")
        savefig(RESULT / f"rng_hw11_{suffix}_{kind}.png")


def hw11_integral_diagnostics(seed: int) -> list[dict[str, object]]:
    cfg = CONFIG["rng_diagnostics"]
    ns = list(cfg["hw11_integral_samples"])
    repeats = int(cfg["hw11_integral_repeats"])
    rows: list[dict[str, object]] = []
    exact_exp = math.e - 1.0
    exact_importance = 2.0 * math.log(2.0)
    problems = [
        ("hw11_exp_integral", exact_exp, lambda u: np.exp(u)),
        ("hw11_singular_importance", exact_importance, lambda u: 2.0 / (1.0 + u)),
    ]
    for gen_index, kind in enumerate(cfg["generators"]):
        for n in ns:
            for problem_name, exact, transform in problems:
                estimates = np.empty(repeats, dtype=float)
                for rep in range(repeats):
                    rng = make_rng(kind, seed + 100000 * gen_index + 1000 * rep + n)
                    u = np.asarray(rng.random(n), dtype=float)
                    estimates[rep] = float(np.mean(transform(u)))
                errors = estimates - exact
                rows.append(
                    {
                        "generator": rng_label(kind),
                        "kind": kind,
                        "problem": problem_name,
                        "N": n,
                        "repeats": repeats,
                        "mean_estimate": float(np.mean(estimates)),
                        "exact": exact,
                        "bias": float(np.mean(estimates) - exact),
                        "mean_abs_error": float(np.mean(np.abs(errors))),
                        "rms_abs_error": float(math.sqrt(np.mean(errors * errors))),
                        "std_estimate": float(np.std(estimates, ddof=1)),
                    }
                )
    fields = [
        "generator",
        "kind",
        "problem",
        "N",
        "repeats",
        "mean_estimate",
        "exact",
        "bias",
        "mean_abs_error",
        "rms_abs_error",
        "std_estimate",
    ]
    write_csv(RESULT / "rng_hw11_integral_comparison.csv", rows, fields)

    for problem_name, title in [
        ("hw11_exp_integral", r"HW11 $\int_0^1 e^x dx$"),
        ("hw11_singular_importance", r"HW11 importance sampling for $1/(\sqrt{x}+x)$"),
    ]:
        plt.figure(figsize=(7.2, 4.7))
        for kind in cfg["generators"]:
            data = [r for r in rows if r["kind"] == kind and r["problem"] == problem_name]
            plt.loglog(
                [r["N"] for r in data],
                [r["rms_abs_error"] for r in data],
                "o-",
                label=rng_label(kind),
            )
        ref_n = np.array(ns, dtype=float)
        first = [r for r in rows if r["problem"] == problem_name and r["kind"] == cfg["generators"][0]][0]
        plt.loglog(ref_n, first["rms_abs_error"] * (ref_n[0] / ref_n) ** 0.5, "k--", lw=1.0, label=r"$N^{-1/2}$")
        plt.xlabel("N")
        plt.ylabel("RMS absolute error")
        plt.title(title)
        plt.legend(fontsize=8)
        plt.grid(alpha=0.25, which="both")
        suffix = "exp" if problem_name == "hw11_exp_integral" else "importance"
        savefig(RESULT / f"rng_hw11_{suffix}_comparison.png")
    for kind in cfg["generators"]:
        plot_integral_per_generator(rows, ns, kind)
    return rows


def rng_diagnostics(seed: int, log: list[str]) -> dict[str, object]:
    log.append("RNG diagnostics: comparing NumPy PCG64, custom xoshiro256**, and Park-Miller.")
    uniform_rows = uniform_diagnostics(seed)
    integer_rows = integer_diagnostics(seed + 333)
    integral_rows = hw11_integral_diagnostics(seed + 777)
    return {"uniform": uniform_rows, "integers": integer_rows, "hw11_integrals": integral_rows}
