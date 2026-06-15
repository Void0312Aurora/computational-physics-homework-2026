from __future__ import annotations

import csv
import io
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


SEED = 20260520
TRUE_PI = math.pi


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"


def save_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def format_float(value: float, digits: int = 10) -> str:
    return f"{value:.{digits}g}"


class LCG:
    def __init__(self, modulus: int, multiplier: int, increment: int, seed: int = 1):
        self.m = int(modulus)
        self.a = int(multiplier)
        self.c = int(increment)
        self.state = seed % self.m

    def random(self, size: int | tuple[int, ...]) -> np.ndarray:
        total = int(np.prod(size)) if isinstance(size, tuple) else int(size)
        out = np.empty(total, dtype=np.float64)
        state = self.state
        for i in range(total):
            state = (self.a * state + self.c) % self.m
            out[i] = state / self.m
        self.state = state
        if isinstance(size, tuple):
            return out.reshape(size)
        return out


class NumpyRNG:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def random(self, size: int | tuple[int, ...]) -> np.ndarray:
        return self.rng.random(size)


def radical_inverse(indices: np.ndarray, base: int) -> np.ndarray:
    n = indices.astype(np.int64).copy()
    inv = 1.0 / base
    factor = inv
    result = np.zeros_like(n, dtype=np.float64)
    while np.any(n > 0):
        digit = n % base
        result += digit * factor
        n //= base
        factor *= inv
    return result


class HaltonSequence:
    def __init__(self, bases: tuple[int, ...] = (2,), start_index: int = 1):
        self.bases = bases
        self.index = int(start_index)

    def random(self, size: int | tuple[int, ...]) -> np.ndarray:
        if isinstance(size, tuple):
            if len(size) != 2 or size[1] != len(self.bases):
                raise ValueError("HaltonSequence expects size=(n, dimension).")
            n = int(size[0])
            indices = np.arange(self.index, self.index + n, dtype=np.int64)
            self.index += n
            cols = [radical_inverse(indices, base) for base in self.bases]
            return np.column_stack(cols)
        if len(self.bases) != 1:
            raise ValueError("One-dimensional output requires one base.")
        n = int(size)
        indices = np.arange(self.index, self.index + n, dtype=np.int64)
        self.index += n
        return radical_inverse(indices, self.bases[0])


@dataclass(frozen=True)
class GeneratorSpec:
    name: str
    label: str
    factory: Callable[[int], object]


GENERATOR_SPECS: list[GeneratorSpec] = [
    GeneratorSpec("numpy_pcg64", "Built-in NumPy PCG64", lambda seed: NumpyRNG(seed)),
    GeneratorSpec(
        "lcg_minstd",
        "Own pseudo-random LCG: m=2^31-1, a=48271, c=0",
        lambda seed: LCG(2**31 - 1, 48271, 0, seed),
    ),
    GeneratorSpec(
        "lcg_slide_large_m",
        "LCG from slide: m=2^31-1, a=4, c=1",
        lambda seed: LCG(2**31 - 1, 4, 1, seed),
    ),
    GeneratorSpec(
        "lcg_slide_small",
        "LCG from slide: m=482, a=13, c=14",
        lambda seed: LCG(482, 13, 14, seed),
    ),
    GeneratorSpec(
        "lcg_bad27",
        "LCG from slide: m=27, a=26, c=5",
        lambda seed: LCG(27, 26, 5, seed),
    ),
    GeneratorSpec(
        "lcg_bad9",
        "LCG from slide: m=9, a=4, c=1",
        lambda seed: LCG(9, 4, 1, seed),
    ),
]


def chi_square_triplet_test(
    generator: object, n_bins_per_axis: int = 5, n_triplets: int = 10_000
) -> dict[str, object]:
    samples = generator.random((n_triplets, 3))
    samples = np.clip(samples, 0.0, np.nextafter(1.0, 0.0))
    indices = np.floor(samples * n_bins_per_axis).astype(np.int64)
    flat = (
        indices[:, 0] * n_bins_per_axis * n_bins_per_axis
        + indices[:, 1] * n_bins_per_axis
        + indices[:, 2]
    )
    counts = np.bincount(flat, minlength=n_bins_per_axis**3)
    expected = n_triplets / (n_bins_per_axis**3)
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    df = n_bins_per_axis**3 - 1
    p_value = float(stats.chi2.sf(chi2, df))
    lower_tail = float(stats.chi2.cdf(chi2, df))
    central_p = min(1.0, 2.0 * min(p_value, lower_tail))
    standardized = (counts - expected) / math.sqrt(expected)
    return {
        "chi2": chi2,
        "df": df,
        "p_value": p_value,
        "lower_tail": lower_tail,
        "central_p": central_p,
        "expected": expected,
        "min_count": int(counts.min()),
        "max_count": int(counts.max()),
        "std_count": float(counts.std(ddof=1)),
        "max_abs_standardized": float(np.max(np.abs(standardized))),
        "counts": counts,
    }


def repeated_chi_square(
    spec: GeneratorSpec,
    n_repeats: int = 200,
    n_bins_per_axis: int = 5,
    n_triplets: int = 10_000,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for repeat in range(n_repeats):
        generator = spec.factory(SEED + 97 * repeat + 11)
        test = chi_square_triplet_test(generator, n_bins_per_axis, n_triplets)
        rows.append(test | {"repeat": repeat})
    return rows


def problem1(log: io.StringIO) -> dict[str, object]:
    print("Problem 1: chi-squared test of randomness", file=log)
    n_bins_per_axis = 5
    n_bins = n_bins_per_axis**3
    n_triplets = 10_000
    n_repeats = 200
    expected = n_triplets / n_bins
    summary_rows: list[list[object]] = []
    chi2_distributions: dict[str, list[float]] = {}
    pvalue_distributions: dict[str, list[float]] = {}
    example_counts: dict[str, np.ndarray] = {}
    detailed_rows: list[list[object]] = []

    for spec in GENERATOR_SPECS:
        repeated = repeated_chi_square(spec, n_repeats, n_bins_per_axis, n_triplets)
        chi2_values = [float(item["chi2"]) for item in repeated]
        p_values = [float(item["p_value"]) for item in repeated]
        central_p_values = [float(item["central_p"]) for item in repeated]
        chi2_distributions[spec.name] = chi2_values
        pvalue_distributions[spec.name] = p_values

        example = chi_square_triplet_test(spec.factory(SEED + 1), n_bins_per_axis, n_triplets)
        example_counts[spec.name] = np.asarray(example["counts"], dtype=np.int64)
        chi2_critical_low = stats.chi2.ppf(0.005, n_bins - 1)
        chi2_critical_high = stats.chi2.ppf(0.995, n_bins - 1)
        pass_rate = float(np.mean([p >= 0.01 for p in central_p_values]))
        summary_rows.append(
            [
                spec.name,
                spec.label,
                n_bins_per_axis,
                n_bins,
                n_triplets,
                f"{expected:.6f}",
                f"{example['chi2']:.6f}",
                n_bins - 1,
                f"{example['p_value']:.6g}",
                f"{example['lower_tail']:.6g}",
                f"{example['central_p']:.6g}",
                example["min_count"],
                example["max_count"],
                f"{example['std_count']:.6f}",
                f"{example['max_abs_standardized']:.6f}",
                f"{np.mean(chi2_values):.6f}",
                f"{np.std(chi2_values, ddof=1):.6f}",
                f"{np.mean(p_values):.6f}",
                f"{np.mean(central_p_values):.6f}",
                f"{pass_rate:.6f}",
                f"{chi2_critical_low:.6f}",
                f"{chi2_critical_high:.6f}",
            ]
        )
        print(
            f"  {spec.name}: example chi2={example['chi2']:.3f}, upper p={example['p_value']:.3g}, "
            f"central p={example['central_p']:.3g}, "
            f"counts=[{example['min_count']}, {example['max_count']}], repeated mean chi2={np.mean(chi2_values):.3f}",
            file=log,
        )
        for item in repeated:
            detailed_rows.append(
                [
                    spec.name,
                    item["repeat"],
                    f"{item['chi2']:.10f}",
                    item["df"],
                    f"{item['p_value']:.10g}",
                    f"{item['lower_tail']:.10g}",
                    f"{item['central_p']:.10g}",
                    item["min_count"],
                    item["max_count"],
                    f"{item['std_count']:.10f}",
                ]
            )

    save_csv(
        RESULT_DIR / "problem1_chi_square_summary.csv",
        [
            "generator",
            "label",
            "bins_per_axis",
            "total_bins",
            "triplets",
            "expected_per_bin",
            "example_chi2",
            "df",
            "example_p_value",
            "example_lower_tail",
            "example_central_p",
            "example_min_count",
            "example_max_count",
            "example_count_std",
            "example_max_abs_standardized",
            "repeated_chi2_mean",
            "repeated_chi2_std",
            "repeated_p_mean",
            "repeated_central_p_mean",
            "repeated_central_p_pass_rate_ge_0p01",
            "chi2_0p005",
            "chi2_0p995",
        ],
        summary_rows,
    )
    save_csv(
        RESULT_DIR / "problem1_chi_square_repeats.csv",
        [
            "generator",
            "repeat",
            "chi2",
            "df",
            "p_value",
            "lower_tail",
            "central_p",
            "min_count",
            "max_count",
            "count_std",
        ],
        detailed_rows,
    )
    plot_problem1_distributions(RESULT_DIR / "problem1_chi_square_distribution.png", chi2_distributions, n_bins - 1)
    plot_problem1_counts(RESULT_DIR / "problem1_example_bin_counts.png", example_counts, expected)
    return {
        "summary": summary_rows,
        "chi2_distributions": chi2_distributions,
        "pvalue_distributions": pvalue_distributions,
        "n_bins_per_axis": n_bins_per_axis,
        "n_bins": n_bins,
        "n_triplets": n_triplets,
        "n_repeats": n_repeats,
        "expected": expected,
    }


def running_mean(generator: object, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    values = generator.random(n_samples)
    cumulative = np.cumsum(values)
    counts = np.arange(1, n_samples + 1)
    return counts, cumulative / counts


def problem2(log: io.StringIO) -> dict[str, object]:
    print("Problem 2: minimal sample-mean statistical test", file=log)
    n_samples = 20_000
    final_rows: list[list[object]] = []
    running_means: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for spec in GENERATOR_SPECS:
        generator = spec.factory(SEED + 223)
        counts, means = running_mean(generator, n_samples)
        final_mean = float(means[-1])
        z_score = (final_mean - 0.5) / math.sqrt(1.0 / (12.0 * n_samples))
        p_value = 2.0 * stats.norm.sf(abs(z_score))
        final_rows.append(
            [
                spec.name,
                spec.label,
                n_samples,
                f"{final_mean:.10f}",
                f"{final_mean - 0.5:.10f}",
                f"{z_score:.6f}",
                f"{p_value:.6g}",
                f"{means.min():.10f}",
                f"{means.max():.10f}",
            ]
        )
        running_means[spec.name] = (counts, means)
        print(
            f"  {spec.name}: final mean={final_mean:.10f}, z={z_score:.3f}, p={p_value:.3g}",
            file=log,
        )

    save_csv(
        RESULT_DIR / "problem2_sample_mean_summary.csv",
        [
            "generator",
            "label",
            "N",
            "final_mean",
            "final_mean_minus_0p5",
            "z_score",
            "two_sided_p_value",
            "running_min_mean",
            "running_max_mean",
        ],
        final_rows,
    )
    plot_problem2_running_means(RESULT_DIR / "problem2_sample_mean.png", running_means)
    return {"summary": final_rows, "n_samples": n_samples}


def problem3(log: io.StringIO) -> dict[str, object]:
    print("Problem 3: own quasi-random generator tests", file=log)
    quasi_spec = GeneratorSpec(
        "halton_2_3_5",
        "Own quasi-random Halton sequence: bases 2, 3, 5",
        lambda seed: HaltonSequence((2, 3, 5), 1),
    )
    builtin_spec = GeneratorSpec(
        "numpy_pcg64",
        "Built-in NumPy PCG64",
        lambda seed: NumpyRNG(seed),
    )
    n_triplets = 10_000
    n_bins_per_axis = 5
    builtin_triplet = chi_square_triplet_test(builtin_spec.factory(SEED + 1), n_bins_per_axis, n_triplets)
    quasi_triplet = chi_square_triplet_test(quasi_spec.factory(SEED), n_bins_per_axis, n_triplets)

    n_samples = 20_000
    builtin_counts, builtin_means = running_mean(builtin_spec.factory(SEED + 223), n_samples)
    quasi_counts, quasi_means = running_mean(HaltonSequence((2,), 1), n_samples)
    builtin_final_mean = float(builtin_means[-1])
    quasi_final_mean = float(quasi_means[-1])
    builtin_z = (builtin_final_mean - 0.5) / math.sqrt(1.0 / (12.0 * n_samples))
    quasi_z = (quasi_final_mean - 0.5) / math.sqrt(1.0 / (12.0 * n_samples))
    builtin_p = 2.0 * stats.norm.sf(abs(builtin_z))
    quasi_p = 2.0 * stats.norm.sf(abs(quasi_z))

    save_csv(
        RESULT_DIR / "problem3_quasi_tests.csv",
        [
            "test",
            "generator",
            "N",
            "statistic",
            "df_or_z",
            "p_value",
            "lower_tail",
            "central_p",
            "aux_1",
            "aux_2",
        ],
        [
            [
                "chi_square_triplets",
                builtin_spec.name,
                n_triplets,
                f"{builtin_triplet['chi2']:.10f}",
                builtin_triplet["df"],
                f"{builtin_triplet['p_value']:.10g}",
                f"{builtin_triplet['lower_tail']:.10g}",
                f"{builtin_triplet['central_p']:.10g}",
                builtin_triplet["min_count"],
                builtin_triplet["max_count"],
            ],
            [
                "chi_square_triplets",
                quasi_spec.name,
                n_triplets,
                f"{quasi_triplet['chi2']:.10f}",
                quasi_triplet["df"],
                f"{quasi_triplet['p_value']:.10g}",
                f"{quasi_triplet['lower_tail']:.10g}",
                f"{quasi_triplet['central_p']:.10g}",
                quasi_triplet["min_count"],
                quasi_triplet["max_count"],
            ],
            [
                "sample_mean",
                builtin_spec.name,
                n_samples,
                f"{builtin_final_mean:.10f}",
                f"{builtin_z:.10f}",
                f"{builtin_p:.10g}",
                "",
                "",
                f"{builtin_means.min():.10f}",
                f"{builtin_means.max():.10f}",
            ],
            [
                "sample_mean",
                "halton_base2",
                n_samples,
                f"{quasi_final_mean:.10f}",
                f"{quasi_z:.10f}",
                f"{quasi_p:.10g}",
                "",
                "",
                f"{quasi_means.min():.10f}",
                f"{quasi_means.max():.10f}",
            ],
        ],
    )
    plot_problem3_triplet_counts(
        RESULT_DIR / "problem3_halton_counts.png",
        {
            builtin_spec.name: np.asarray(builtin_triplet["counts"], dtype=np.int64),
            quasi_spec.name: np.asarray(quasi_triplet["counts"], dtype=np.int64),
        },
        quasi_triplet["expected"],
    )
    plot_problem3_running_means(
        RESULT_DIR / "problem3_halton_sample_mean.png",
        {
            builtin_spec.name: (builtin_counts, builtin_means),
            "halton_base2": (quasi_counts, quasi_means),
        },
    )
    print(
        f"  NumPy triplets: chi2={builtin_triplet['chi2']:.3f}, upper p={builtin_triplet['p_value']:.3g}, "
        f"central p={builtin_triplet['central_p']:.3g}, counts=[{builtin_triplet['min_count']}, {builtin_triplet['max_count']}]",
        file=log,
    )
    print(
        f"  Halton triplets: chi2={quasi_triplet['chi2']:.3f}, upper p={quasi_triplet['p_value']:.3g}, "
        f"central p={quasi_triplet['central_p']:.3g}, counts=[{quasi_triplet['min_count']}, {quasi_triplet['max_count']}]",
        file=log,
    )
    print(
        f"  NumPy mean: final={builtin_final_mean:.10f}, z={builtin_z:.3f}, p={builtin_p:.3g}",
        file=log,
    )
    print(
        f"  Halton base-2 mean: final={quasi_final_mean:.10f}, z={quasi_z:.3f}, p={quasi_p:.3g}",
        file=log,
    )
    return {
        "chi_square": {
            builtin_spec.name: {
                "n": n_triplets,
                "chi2": builtin_triplet["chi2"],
                "df": builtin_triplet["df"],
                "p_value": builtin_triplet["p_value"],
                "lower_tail": builtin_triplet["lower_tail"],
                "central_p": builtin_triplet["central_p"],
                "min_count": builtin_triplet["min_count"],
                "max_count": builtin_triplet["max_count"],
            },
            quasi_spec.name: {
                "n": n_triplets,
                "chi2": quasi_triplet["chi2"],
                "df": quasi_triplet["df"],
                "p_value": quasi_triplet["p_value"],
                "lower_tail": quasi_triplet["lower_tail"],
                "central_p": quasi_triplet["central_p"],
                "min_count": quasi_triplet["min_count"],
                "max_count": quasi_triplet["max_count"],
            },
        },
        "sample_mean": {
            builtin_spec.name: {
                "n": n_samples,
                "final_mean": builtin_final_mean,
                "z_score": builtin_z,
                "p_value": builtin_p,
            },
            "halton_base2": {
                "n": n_samples,
                "final_mean": quasi_final_mean,
                "z_score": quasi_z,
                "p_value": quasi_p,
            },
        },
    }


def estimate_pi_with_points(points: np.ndarray) -> tuple[int, float, float]:
    radii2 = points[:, 0] ** 2 + points[:, 1] ** 2
    inside = int(np.count_nonzero(radii2 <= 1.0))
    n = len(points)
    pi_est = 4.0 * inside / n
    return inside, pi_est, abs(pi_est - TRUE_PI)


def problem4(log: io.StringIO) -> dict[str, object]:
    print("Problem 4: acceptance-rejection estimates of pi", file=log)
    n_values = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]

    pseudo = LCG(2**31 - 1, 48271, 0, SEED)
    quasi = HaltonSequence((2, 3), 1)
    max_n = max(n_values)
    pseudo_points = pseudo.random((max_n, 2))
    quasi_points = quasi.random((max_n, 2))

    rows: list[list[object]] = []
    method_results: dict[str, list[dict[str, float]]] = {"pseudo_lcg_minstd": [], "quasi_halton_2_3": []}
    for method, points in [("pseudo_lcg_minstd", pseudo_points), ("quasi_halton_2_3", quasi_points)]:
        for n in n_values:
            inside, pi_est, abs_error = estimate_pi_with_points(points[:n])
            if method.startswith("pseudo"):
                p = TRUE_PI / 4.0
                theory_std = 4.0 * math.sqrt(p * (1.0 - p) / n)
            else:
                theory_std = float("nan")
            item = {
                "n": n,
                "inside": inside,
                "pi_est": pi_est,
                "abs_error": abs_error,
                "theory_std": theory_std,
            }
            method_results[method].append(item)
            rows.append(
                [
                    method,
                    n,
                    inside,
                    f"{inside / n:.10f}",
                    f"{pi_est:.10f}",
                    f"{abs_error:.10f}",
                    "" if math.isnan(theory_std) else f"{theory_std:.10f}",
                ]
            )
            print(f"  {method}, N={n}: pi={pi_est:.10f}, abs_error={abs_error:.10f}", file=log)

    save_csv(
        RESULT_DIR / "problem4_pi_acceptance_rejection.csv",
        ["method", "N", "inside", "inside_ratio", "pi_est", "abs_error", "pseudo_theory_std"],
        rows,
    )
    plot_problem4_convergence(RESULT_DIR / "problem4_pi_convergence.png", method_results)
    plot_problem4_points(RESULT_DIR / "problem4_points.png", pseudo_points[:2000], quasi_points[:2000])
    final_pseudo = method_results["pseudo_lcg_minstd"][-1]
    final_quasi = method_results["quasi_halton_2_3"][-1]
    return {
        "n_values": n_values,
        "method_results": method_results,
        "final_pseudo": final_pseudo,
        "final_quasi": final_quasi,
    }


def plot_problem1_distributions(path: Path, chi2_distributions: dict[str, list[float]], df: int) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), constrained_layout=True)
    ax, ax_box = axes
    xs = np.linspace(stats.chi2.ppf(0.0005, df), stats.chi2.ppf(0.9995, df), 500)
    ax.plot(xs, stats.chi2.pdf(xs, df), color="#222222", linewidth=2.0, label=f"chi-square df={df}")
    colors = {
        "numpy_pcg64": "#1f77b4",
        "lcg_minstd": "#2ca02c",
        "lcg_slide_large_m": "#9467bd",
        "lcg_slide_small": "#ff7f0e",
        "lcg_bad27": "#d62728",
        "lcg_bad9": "#8c564b",
    }
    for name in ["numpy_pcg64", "lcg_minstd"]:
        values = chi2_distributions[name]
        ax.hist(
            values,
            bins=18,
            density=True,
            alpha=0.22,
            color=colors.get(name, None),
            label=name,
        )
    ax.axvline(stats.chi2.ppf(0.005, df), color="#555555", linestyle="--", linewidth=1.0)
    ax.axvline(stats.chi2.ppf(0.995, df), color="#555555", linestyle="--", linewidth=1.0)
    ax.set_xlabel("chi-square statistic")
    ax.set_ylabel("density")
    ax.set_title("Generators near the chi-square reference")
    ax.legend(loc="upper right", fontsize=8)

    names = list(chi2_distributions)
    data = [chi2_distributions[name] for name in names]
    short_names = ["PCG64", "MINSTD", "large-m", "m=482", "m=27", "m=9"]
    ax_box.boxplot(data, tick_labels=short_names, showfliers=False)
    ax_box.scatter(
        np.repeat(np.arange(1, len(names) + 1), [len(chi2_distributions[name]) for name in names]),
        np.concatenate([chi2_distributions[name] for name in names]),
        s=5,
        alpha=0.25,
        color="#444444",
        linewidths=0,
    )
    ax_box.axhspan(stats.chi2.ppf(0.005, df), stats.chi2.ppf(0.995, df), color="#2ca02c", alpha=0.12)
    ax_box.set_yscale("log")
    ax_box.set_ylabel("chi-square statistic, log scale")
    ax_box.set_title("All generators on a log scale")
    ax_box.tick_params(axis="x", rotation=25)
    fig.suptitle("Problem 1: repeated chi-square triplet tests")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_problem1_counts(path: Path, example_counts: dict[str, np.ndarray], expected: float) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    selected = ["numpy_pcg64", "lcg_minstd", "lcg_slide_small", "lcg_bad9"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, constrained_layout=True)
    for ax, name in zip(axes.ravel(), selected):
        counts = example_counts[name]
        ax.plot(np.arange(len(counts)), counts, linewidth=1.2)
        ax.axhline(expected, color="#d62728", linestyle="--", linewidth=1.2, label="expected")
        ax.fill_between(
            np.arange(len(counts)),
            expected - math.sqrt(expected),
            expected + math.sqrt(expected),
            color="#d62728",
            alpha=0.12,
            label="expected +/- sqrt(expected)",
        )
        ax.set_title(name)
        ax.set_ylabel("count")
    axes[-1, 0].set_xlabel("flattened 3D bin index")
    axes[-1, 1].set_xlabel("flattened 3D bin index")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Problem 1: example triplet bin counts")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_problem2_running_means(path: Path, running_means: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    colors = {
        "numpy_pcg64": "#1f77b4",
        "lcg_minstd": "#2ca02c",
        "lcg_slide_large_m": "#9467bd",
        "lcg_slide_small": "#ff7f0e",
        "lcg_bad27": "#d62728",
        "lcg_bad9": "#8c564b",
    }
    for name, (counts, means) in running_means.items():
        ax.plot(counts, means, linewidth=1.1, color=colors.get(name, None), label=name)
    ax.axhline(0.5, color="#222222", linestyle="--", linewidth=1.2)
    ax.set_xlim(1, max(next(iter(running_means.values()))[0]))
    ax.set_ylim(0.0, 0.62)
    ax.set_xlabel("number of samples")
    ax.set_ylabel("running sample mean")
    ax.set_title("Problem 2: minimal sample-mean test")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_problem3_triplet_counts(path: Path, counts_map: dict[str, np.ndarray], expected: float) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), sharey=True, constrained_layout=True)
    styles = {
        "numpy_pcg64": ("#1f77b4", "Built-in NumPy PCG64"),
        "halton_2_3_5": ("#2ca02c", "Halton sequence (bases 2,3,5)"),
    }
    for ax, name in zip(axes, ["numpy_pcg64", "halton_2_3_5"]):
        counts = counts_map[name]
        color, title = styles[name]
        ax.plot(np.arange(len(counts)), counts, color=color, linewidth=1.3)
        ax.axhline(expected, color="#d62728", linestyle="--", linewidth=1.2, label="expected")
        ax.fill_between(
            np.arange(len(counts)),
            expected - math.sqrt(expected),
            expected + math.sqrt(expected),
            color="#d62728",
            alpha=0.12,
            label="expected +/- sqrt(expected)",
        )
        ax.set_xlabel("flattened 3D bin index")
        ax.set_title(title)
    axes[0].set_ylabel("count")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Problem 3: triplet bin counts, built-in RNG vs Halton")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_problem3_running_means(path: Path, series_map: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    styles = {
        "numpy_pcg64": ("#1f77b4", "Built-in NumPy PCG64"),
        "halton_base2": ("#2ca02c", "Halton base-2"),
    }
    for name in ["numpy_pcg64", "halton_base2"]:
        counts, means = series_map[name]
        color, label = styles[name]
        ax.plot(counts, means, color=color, linewidth=1.15, label=label)
    ax.axhline(0.5, color="#222222", linestyle="--", linewidth=1.2)
    ax.set_xlabel("number of samples")
    ax.set_ylabel("running sample mean")
    ax.set_title("Problem 3: running sample mean, built-in RNG vs Halton")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_problem4_convergence(path: Path, method_results: dict[str, list[dict[str, float]]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    for method, rows in method_results.items():
        n = np.array([item["n"] for item in rows], dtype=np.float64)
        err = np.array([item["abs_error"] for item in rows], dtype=np.float64)
        ax.loglog(n, err, marker="o", linewidth=1.4, label=method)
    ref_n = np.array([min(method_results["pseudo_lcg_minstd"][0]["n"], method_results["quasi_halton_2_3"][0]["n"]), max(item["n"] for item in method_results["pseudo_lcg_minstd"])])
    reference = 0.75 * (ref_n / ref_n[0]) ** -0.5
    ax.loglog(ref_n, reference, color="#777777", linestyle="--", label="reference slope N^-1/2")
    ax.set_xlabel("N")
    ax.set_ylabel("absolute error of pi estimate")
    ax.set_title("Problem 4: acceptance-rejection convergence")
    ax.legend(loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_problem4_points(path: Path, pseudo_points: np.ndarray, quasi_points: np.ndarray) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    theta = np.linspace(0, math.pi / 2.0, 300)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    for ax, points, title in [
        (axes[0], pseudo_points, "pseudo-random LCG"),
        (axes[1], quasi_points, "quasi-random Halton"),
    ]:
        inside = points[:, 0] ** 2 + points[:, 1] ** 2 <= 1.0
        ax.scatter(points[inside, 0], points[inside, 1], s=5, alpha=0.45, color="#1f77b4", linewidths=0)
        ax.scatter(points[~inside, 0], points[~inside, 1], s=5, alpha=0.45, color="#d62728", linewidths=0)
        ax.plot(circle_x, circle_y, color="#222222", linewidth=1.4)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle("Problem 4: first 2000 points for pi estimation")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log = io.StringIO()
    print("HW10 random-number generator tests", file=log)
    print(f"Python: {platform.python_version()}", file=log)
    print(f"NumPy: {np.__version__}", file=log)
    print(f"SciPy: {stats.__version__ if hasattr(stats, '__version__') else 'available'}", file=log)
    print(f"Seed: {SEED}", file=log)

    summary = {
        "seed": SEED,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "problem1": problem1(log),
        "problem2": problem2(log),
        "problem3": problem3(log),
        "problem4": problem4(log),
    }
    serializable = json.loads(json.dumps(summary, default=lambda obj: None))
    (RESULT_DIR / "hw10_summary.json").write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULT_DIR / "temp-01.log").write_text(log.getvalue(), encoding="utf-8")
    print(log.getvalue())


if __name__ == "__main__":
    main()
