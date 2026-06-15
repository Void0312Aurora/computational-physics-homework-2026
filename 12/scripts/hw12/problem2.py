from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from .config import CONFIG, RESULT
from .io_utils import savefig, write_csv
from .walk_math import chain_observables, chain_sample_count, frc_next_dirs, unit_vectors_3d


def simulate_fjc_stats(rng, n_values: list[int]) -> tuple[list[dict[str, object]], np.ndarray]:
    rows: list[dict[str, object]] = []
    distribution_distances: np.ndarray | None = None
    for n in n_values:
        samples = chain_sample_count(n)
        pos = np.zeros((samples, 3), dtype=float)
        sum_pos = np.zeros((samples, 3), dtype=float)
        sum_norm2 = np.zeros(samples, dtype=float)
        for step in range(1, n + 1):
            pos += unit_vectors_3d(rng, samples)
            sum_pos += pos
            sum_norm2 += np.einsum("ij,ij->i", pos, pos)
            if n == 100 and step == n:
                distribution_distances = np.sqrt(np.einsum("ij,ij->i", pos, pos)).copy()
        re2, rg2 = chain_observables(pos, sum_pos, sum_norm2, n)
        factor = 6.0 * (n + 1) / (n + 2)
        rows.append(
            {
                "model": "freely_joined",
                "N": n,
                "samples": samples,
                "mean_Re2": re2,
                "mean_Rg2": rg2,
                "Rg2_over_Re2": rg2 / re2,
                "relation_factor": factor,
                "relative_relation_error": (re2 - factor * rg2) / (factor * rg2),
            }
        )
    assert distribution_distances is not None
    return rows, distribution_distances


def simulate_frc_stats(rng, n_values: list[int], theta_deg: float) -> list[dict[str, object]]:
    theta = math.radians(theta_deg)
    c = math.cos(theta)
    rows: list[dict[str, object]] = []
    for n in n_values:
        samples = chain_sample_count(n, rotating=True)
        pos = np.zeros((samples, 3), dtype=float)
        sum_pos = np.zeros((samples, 3), dtype=float)
        sum_norm2 = np.zeros(samples, dtype=float)
        direction = unit_vectors_3d(rng, samples)
        for step in range(1, n + 1):
            if step > 1:
                direction = frc_next_dirs(rng, direction, theta)
            pos += direction
            sum_pos += pos
            sum_norm2 += np.einsum("ij,ij->i", pos, pos)
        re2, rg2 = chain_observables(pos, sum_pos, sum_norm2, n)
        s = sum((n - k) * (c**k) for k in range(1, n))
        theory_re2 = n + 2.0 * s
        rows.append(
            {
                "model": "freely_rotating_68deg",
                "N": n,
                "samples": samples,
                "mean_Re2": re2,
                "mean_Rg2": rg2,
                "Rg2_over_Re2": rg2 / re2,
                "theory_Re2": theory_re2,
                "relative_Re2_error": (re2 - theory_re2) / theory_re2,
                "theta_deg": theta_deg,
                "cos_theta": c,
            }
        )
    return rows


def problem2(rng, log: list[str]) -> dict[str, object]:
    log.append("Problem 2: simulating polymer chain observables.")
    fjc_n = list(CONFIG["problem2"]["fjc_N"])
    frc_n = list(CONFIG["problem2"]["frc_N"])
    fjc_rows, distances = simulate_fjc_stats(rng, fjc_n)
    frc_rows = simulate_frc_stats(rng, frc_n, float(CONFIG["problem2"]["frc_theta_deg"]))
    rows = fjc_rows + frc_rows
    fields = sorted({key for row in rows for key in row})
    write_csv(RESULT / "problem2_chain_stats.csv", rows, fields)

    fjc = [r for r in rows if r["model"] == "freely_joined"]
    plt.figure(figsize=(7.0, 4.7))
    n = np.array([r["N"] for r in fjc], dtype=float)
    re2 = np.array([r["mean_Re2"] for r in fjc])
    rg2 = np.array([r["mean_Rg2"] for r in fjc])
    plt.loglog(n, re2, "o", linestyle="None", ms=5.5, label=r"simulation $\langle R_e^2\rangle$")
    plt.loglog(n, rg2, "s", linestyle="None", ms=5.5, label=r"simulation $\langle R_g^2\rangle$")
    plt.loglog(n, n, "k--", lw=1.0, label=r"$N$")
    plt.loglog(n, n / 6.0, "k:", lw=1.0, label=r"$N/6$")
    plt.xlabel("Number of segments N")
    plt.ylabel("Mean square distance")
    plt.title("Freely joined random-flight chain: scatter data")
    plt.legend()
    plt.grid(alpha=0.25, which="both")
    savefig(RESULT / "problem2_fjc_relation.png")

    plt.figure(figsize=(7.0, 4.7))
    frc = [r for r in rows if r["model"] == "freely_rotating_68deg"]
    colors = {"FJC": "#2f80ed", "FRC, theta=68 deg": "#d9480f"}
    for label, data, marker in [("FJC", fjc, "o"), ("FRC, theta=68 deg", frc, "s")]:
        n = np.array([r["N"] for r in data], dtype=float)
        re2 = np.array([r["mean_Re2"] for r in data])
        rg2 = np.array([r["mean_Rg2"] for r in data])
        color = colors[label]
        plt.scatter(n, re2, marker=marker, s=32, color=color, label=label + r" $\langle R_e^2\rangle$")
        plt.scatter(
            n,
            rg2,
            marker=marker,
            s=32,
            facecolors="none",
            edgecolors=color,
            label=label + r" $\langle R_g^2\rangle$",
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of segments N")
    plt.ylabel("Mean square distance")
    plt.title("Freely joined versus freely rotating chain: scatter data")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25, which="both")
    savefig(RESULT / "problem2_frc_comparison.png")

    n_dist = 100
    sigma = math.sqrt(n_dist / 3.0)
    x = np.linspace(0.0, np.percentile(distances, 99.8) * 1.05, 400)
    pdf = math.sqrt(2.0 / math.pi) * (x * x / sigma**3) * np.exp(-(x * x) / (2.0 * sigma * sigma))
    plt.figure(figsize=(7.0, 4.7))
    plt.hist(distances, bins=70, density=True, color="#083dff", edgecolor="black", linewidth=0.25, alpha=0.85, label="simulation")
    plt.plot(x, pdf, "r-", lw=2.0, label="Maxwell prediction")
    plt.xlabel(r"End-to-end distance $R$")
    plt.ylabel("Probability density")
    plt.title("End-to-end distance distribution, FJC N=100")
    plt.legend()
    plt.grid(alpha=0.25)
    savefig(RESULT / "problem2_end_to_end_distribution.png")

    dist_summary = {
        "N": n_dist,
        "samples": int(distances.size),
        "mean_R_sim": float(np.mean(distances)),
        "mean_R_theory": float(2.0 * sigma * math.sqrt(2.0 / math.pi)),
        "rms_R_sim": float(math.sqrt(np.mean(distances * distances))),
        "rms_R_theory": float(math.sqrt(n_dist)),
    }
    return {"chain_rows": rows, "distribution": dist_summary}
