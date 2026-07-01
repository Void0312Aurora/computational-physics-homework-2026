from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
HW_ROOT = ROOT.parents[1]
RESULT_DIR = HW_ROOT / "result"
DIRECT_RESULTS = ROOT / "results" / "q23456_selected_under30_direct_results.csv"
TREND_CSV = RESULT_DIR / "problem5_volume_trend_formula.csv"
TREND_FIGURE = RESULT_DIR / "problem5_volume_trend_formula.png"


def reference_volume(dimension: int) -> float:
    return math.exp(0.5 * dimension * math.log(math.pi) - math.lgamma(0.5 * dimension + 1.0))


def read_direct_rows() -> dict[int, float]:
    rows: dict[int, float] = {}
    if not DIRECT_RESULTS.exists():
        return rows
    with DIRECT_RESULTS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["dimension"])] = float(row["direct_estimate"])
    return rows


def write_trend_csv(direct_rows: dict[int, float]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with TREND_CSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["dimension", "reference_volume", "direct_estimate", "direct_relative_error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dimension in range(0, 101):
            ref = reference_volume(dimension)
            direct = direct_rows.get(dimension)
            writer.writerow(
                {
                    "dimension": dimension,
                    "reference_volume": f"{ref:.18e}",
                    "direct_estimate": "" if direct is None else f"{direct:.18e}",
                    "direct_relative_error": "" if direct is None else f"{abs(direct - ref) / ref:.6e}",
                }
            )


def plot_trend(direct_rows: dict[int, float]) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    dims_linear = list(range(0, 36))
    vols_linear = [reference_volume(n) for n in dims_linear]
    dims_log = list(range(1, 101))
    vols_log = [reference_volume(n) for n in dims_log]

    peak_dimension = max(dims_linear, key=lambda n: reference_volume(n))
    peak_volume = reference_volume(peak_dimension)

    direct_dims = sorted(direct_rows)
    direct_values = [direct_rows[n] for n in direct_dims]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.8))

    axes[0].plot(dims_linear, vols_linear, color="#2563eb", linewidth=2.2, label="formula reference")
    axes[0].scatter([peak_dimension], [peak_volume], color="#dc2626", s=54, zorder=4, label=f"peak n={peak_dimension}")
    if direct_dims:
        axes[0].scatter(direct_dims, direct_values, color="#f97316", s=44, zorder=5, label="direct n=27..30")
    axes[0].set_xlabel("dimension n")
    axes[0].set_ylabel("unit ball volume")
    axes[0].set_title("Low-dimensional peak")
    axes[0].grid(alpha=0.28)
    axes[0].legend(loc="upper right")

    axes[1].semilogy(dims_log, vols_log, color="#2563eb", linewidth=2.2, label="formula reference")
    if direct_dims:
        axes[1].scatter(direct_dims, direct_values, color="#f97316", s=44, zorder=5, label="direct n=27..30")
    axes[1].set_xlabel("dimension n")
    axes[1].set_ylabel("unit ball volume, log scale")
    axes[1].set_title("High-dimensional decay")
    axes[1].grid(alpha=0.28, which="both")
    axes[1].legend(loc="upper right")

    fig.suptitle("Unit hypersphere volume versus dimension", y=0.995, fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.972), h_pad=2.2)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(TREND_FIGURE, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    direct_rows = read_direct_rows()
    write_trend_csv(direct_rows)
    plot_trend(direct_rows)
    peak_dimension = max(range(0, 36), key=lambda n: reference_volume(n))
    print(f"trend_csv={TREND_CSV}")
    print(f"trend_figure={TREND_FIGURE}")
    print(f"peak_dimension_0_35={peak_dimension}")


if __name__ == "__main__":
    main()
