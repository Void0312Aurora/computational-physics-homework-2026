from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .config import CONFIG, RESULT
from .io_utils import savefig, write_csv
from .walk_math import (
    add_power_fits,
    pivot_saw_stat,
    simulate_3d_return_probability,
    simulate_walk_stats,
    write_fit_diagnostics,
)


def problem5(rng, log: list[str]) -> dict[str, object]:
    log.append("Problem 5: simulating 3D lattice and self-avoiding walks.")
    n_values = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    saw_values = [16, 32, 64, 128, 256, 512, 1024]
    rows: list[dict[str, object]] = []
    rows += simulate_walk_stats(rng, 3, n_values, 22000, "3d_lattice_traditional")
    for n in saw_values:
        rows.append(pivot_saw_stat(rng, 3, n, "3d_self_avoiding"))
    fields = [
        "model",
        "N",
        "samples",
        "mean_Re2",
        "mean_Rg2",
        "Rg2_over_Re2",
        "acceptance_rate",
        "effective_samples",
        "tau_Re2",
        "tau_Rg2",
    ]
    write_csv(RESULT / "problem5_3d_polymer.csv", rows, fields)
    fits = add_power_fits(
        rows,
        rng,
        min_n=int(CONFIG["polymer_fit"]["primary_min_N"]),
        n_boot=int(CONFIG["polymer_fit"]["bootstrap_replicates"]),
    )
    fit_windows = write_fit_diagnostics(rows, rng, "problem5_3d")

    plt.figure(figsize=(7.3, 4.9))
    markers = {"3d_lattice_traditional": "o", "3d_self_avoiding": "^"}
    labels = {"3d_lattice_traditional": "lattice traditional", "3d_self_avoiding": "self-avoiding"}
    colors = {"3d_lattice_traditional": "#2f80ed", "3d_self_avoiding": "#d9480f"}
    for model in markers:
        data = [r for r in rows if r["model"] == model]
        n = np.array([r["N"] for r in data], dtype=float)
        re2 = np.array([r["mean_Re2"] for r in data], dtype=float)
        rg2 = np.array([r["mean_Rg2"] for r in data], dtype=float)
        color = colors[model]
        plt.loglog(
            n,
            re2,
            markers[model],
            linestyle="None",
            ms=5.5,
            color=color,
            label=labels[model] + r" $\langle R_e^2\rangle$",
        )
        plt.loglog(
            n,
            rg2,
            markers[model],
            linestyle="None",
            ms=5.5,
            mfc="none",
            color=color,
            label=labels[model] + r" $\langle R_g^2\rangle$",
        )
        if model in fits:
            fit_n = np.logspace(np.log10(np.min(n)), np.log10(np.max(n)), 160)
            plt.loglog(
                fit_n,
                fits[model]["Re2_prefactor"] * fit_n ** fits[model]["Re2_slope"],
                color=color,
                lw=1.1,
                alpha=0.85,
            )
            plt.loglog(
                fit_n,
                fits[model]["Rg2_prefactor"] * fit_n ** fits[model]["Rg2_slope"],
                color=color,
                lw=1.1,
                ls="--",
                alpha=0.85,
            )
    ref_n = np.array([min(n_values), max(n_values)], dtype=float)
    plt.loglog(ref_n, ref_n, "k:", lw=1.0, label=r"$N^1$")
    plt.loglog(ref_n, 0.65 * ref_n**1.176, "k--", lw=1.0, label=r"$N^{1.176}$")
    plt.xlabel("N")
    plt.ylabel("Mean square size")
    plt.title("3D polymer scaling: scatter data")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25, which="both")
    savefig(RESULT / "problem5_3d_scaling.png")

    plt.figure(figsize=(7.2, 4.7))
    for model in markers:
        data = [r for r in rows if r["model"] == model]
        n = np.array([r["N"] for r in data], dtype=float)
        ratio = np.array([r["Rg2_over_Re2"] for r in data], dtype=float)
        plt.semilogx(n, ratio, markers[model], linestyle="None", ms=5.5, color=colors[model], label=labels[model])
    plt.axhline(1.0 / 6.0, color="k", ls="--", lw=1.2, label="1/6")
    plt.xlabel("N")
    plt.ylabel(r"$\langle R_g^2\rangle/\langle R_e^2\rangle$")
    plt.title("3D size ratio: scatter data")
    plt.legend()
    plt.grid(alpha=0.25, which="both")
    savefig(RESULT / "problem5_3d_ratio.png")

    transience_rows = simulate_3d_return_probability(rng)
    write_csv(
        RESULT / "problem5_transience.csv",
        transience_rows,
        ["steps", "walkers", "ever_return_fraction", "origin_fraction_at_step"],
    )
    plt.figure(figsize=(7.2, 4.7))
    x = np.array([r["steps"] for r in transience_rows])
    y = np.array([r["ever_return_fraction"] for r in transience_rows])
    plt.scatter(x, y, s=18, alpha=0.72, label="finite-time simulation")
    plt.axhline(0.340537, color="r", ls="--", lw=1.2, label="known eventual return probability")
    plt.xlabel("Maximum simulated steps")
    plt.ylabel("Fraction that has returned to origin")
    plt.title("Finite-time return probability in 3D: scatter data")
    plt.legend()
    plt.grid(alpha=0.25)
    savefig(RESULT / "problem5_transience.png")

    return {"rows": rows, "fits": fits, "fit_windows": fit_windows, "transience": transience_rows[-1]}
