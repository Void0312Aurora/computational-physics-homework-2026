from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln

from .config import CONFIG, RESULT
from .io_utils import savefig, write_csv


def binomial_walk_distribution(n: int, p: float) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(n + 1)
    x = 2 * k - n
    logp = (
        gammaln(n + 1)
        - gammaln(k + 1)
        - gammaln(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log(1.0 - p)
    )
    prob = np.exp(logp)
    prob /= np.sum(prob)
    return x, prob


def distribution_metrics(n: int, p: float) -> dict[str, object]:
    x, prob = binomial_walk_distribution(n, p)
    mean = float(np.sum(x * prob))
    var = float(np.sum((x - mean) ** 2 * prob))
    max_idx = int(np.argmax(prob))
    half = prob[max_idx] / 2.0
    mask = prob >= half
    fwhm = int(x[mask][-1] - x[mask][0]) if np.any(mask) else 0
    return {
        "N": n,
        "p": p,
        "mean_x": mean,
        "theory_mean_x": n * (2.0 * p - 1.0),
        "variance": var,
        "theory_variance": 4.0 * n * p * (1.0 - p),
        "origin_mean_square": var + mean * mean,
        "max_x": int(x[max_idx]),
        "max_probability": float(prob[max_idx]),
        "fwhm_lattice_width": fwhm,
        "sigma": math.sqrt(var),
    }


def problem3(log: list[str]) -> dict[str, object]:
    log.append("Problem 3: computing exact 1D lattice-walk probabilities.")
    ns = list(CONFIG["problem3"]["N_values"])
    unbiased_rows = [distribution_metrics(n, 0.5) for n in ns]
    biased_rows = [distribution_metrics(n, float(CONFIG["problem3"]["biased_p"])) for n in ns]
    fields = [
        "N",
        "p",
        "mean_x",
        "theory_mean_x",
        "variance",
        "theory_variance",
        "origin_mean_square",
        "max_x",
        "max_probability",
        "fwhm_lattice_width",
        "sigma",
    ]
    write_csv(RESULT / "problem3_unbiased_table.csv", unbiased_rows, fields)
    write_csv(RESULT / "problem3_biased_table.csv", biased_rows, fields)

    fig, axes = plt.subplots(2, 4, figsize=(10.0, 5.3), sharey=False)
    axes_flat = axes.ravel()
    for ax, n in zip(axes_flat, ns):
        x, prob = binomial_walk_distribution(n, 0.5)
        ax.bar(x, prob, width=1.8, color="#083dff", edgecolor="black", linewidth=0.2)
        ax.set_title(f"N={n}", fontsize=9)
        ax.set_xlim(-n, n)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.18, axis="y")
    axes_flat[-1].axis("off")
    fig.supxlabel("Position x")
    fig.supylabel(r"$P_N(x)$")
    fig.suptitle("Exact discrete probabilities for unbiased 1D lattice walks", fontsize=11)
    savefig(RESULT / "problem3_probabilities_unbiased.png")

    n = 1024
    x, prob = binomial_walk_distribution(n, 0.5)
    sigma = math.sqrt(n)
    gaussian_mass = 2.0 / (math.sqrt(2.0 * math.pi) * sigma) * np.exp(-(x * x) / (2.0 * sigma * sigma))
    central = np.abs(x) <= 2.0 * sigma
    fit_summary = {
        "N": n,
        "total_L1_error_against_gaussian_mass": float(np.sum(np.abs(prob - gaussian_mass))),
        "central_L1_error_against_gaussian_mass": float(np.sum(np.abs(prob[central] - gaussian_mass[central]))),
        "tail_L1_error_against_gaussian_mass": float(np.sum(np.abs(prob[~central] - gaussian_mass[~central]))),
    }
    plt.figure(figsize=(7.1, 4.8))
    plt.bar(x, prob, width=1.8, color="#083dff", edgecolor="black", linewidth=0.2, label="exact discrete probability")
    plt.plot(x, gaussian_mass, "r-", lw=2.0, label="Gaussian mass approximation")
    plt.xlim(-130, 130)
    plt.xlabel("Position x")
    plt.ylabel(r"$P_N(x)$")
    plt.title("Gaussian approximation for N=1024, p=0.5")
    plt.legend()
    plt.grid(alpha=0.25)
    savefig(RESULT / "problem3_gaussian_fit.png")

    biased_p = float(CONFIG["problem3"]["biased_p"])
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6), sharey=False)
    selected_biased = [64, 256, 1024]
    for ax, n_biased in zip(axes, selected_biased):
        x_biased, prob_biased = binomial_walk_distribution(n_biased, biased_p)
        ax.bar(x_biased, prob_biased, width=1.8, color="#083dff", edgecolor="black", linewidth=0.2)
        mean_biased = n_biased * (2.0 * biased_p - 1.0)
        ax.axvline(mean_biased, color="r", lw=1.0, ls="--", alpha=0.7)
        ax.set_title(f"N={n_biased}", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.18, axis="y")
    fig.supxlabel("Position x")
    fig.supylabel(r"$P_N(x)$")
    fig.suptitle(f"Biased 1D lattice walks, p={biased_p:.1f}", fontsize=11)
    savefig(RESULT / "problem3_probabilities_biased.png")

    plt.figure(figsize=(7.1, 4.8))
    n_arr = np.array(ns, dtype=float)
    var_unbiased = np.array([r["variance"] for r in unbiased_rows], dtype=float)
    var_biased = np.array([r["variance"] for r in biased_rows], dtype=float)
    ms_biased = np.array([r["origin_mean_square"] for r in biased_rows], dtype=float)
    slope_unbiased = float(np.polyfit(np.log(n_arr), np.log(var_unbiased), 1)[0])
    slope_biased = float(np.polyfit(np.log(n_arr), np.log(var_biased), 1)[0])
    slope_biased_origin = float(np.polyfit(np.log(n_arr), np.log(ms_biased), 1)[0])
    plt.loglog(n_arr, var_unbiased, "o", linestyle="None", ms=5.5, label=r"$p=0.5,\ \langle\Delta x^2\rangle$")
    plt.loglog(n_arr, var_biased, "s", linestyle="None", ms=5.5, label=r"$p=0.7,\ \langle\Delta x^2\rangle$")
    plt.loglog(n_arr, ms_biased, "^", linestyle="None", ms=5.5, label=r"$p=0.7,\ \langle x^2\rangle$")
    plt.loglog(n_arr, n_arr, "k--", lw=1.0, label="slope 1")
    plt.loglog(n_arr, 0.16 * n_arr * n_arr, "k:", lw=1.0, label="slope 2")
    plt.xlabel("N")
    plt.ylabel("Mean square scale")
    plt.title("Scaling law for 1D lattice walks")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25, which="both")
    savefig(RESULT / "problem3_loglog_variance.png")

    return {
        "unbiased": unbiased_rows,
        "biased": biased_rows,
        "gaussian_fit": fit_summary,
        "slopes": {
            "variance_p05": slope_unbiased,
            "variance_p07": slope_biased,
            "origin_mean_square_p07": slope_biased_origin,
        },
    }
