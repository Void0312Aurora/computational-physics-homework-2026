from __future__ import annotations

import csv
import math
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"
getcontext().prec = 80
PI_OVER_4_DECIMAL = Decimal(
    "0.785398163397448309615660845819875721049292349843776455243736148076954101571552"
)


ArrayFunc = Callable[[np.ndarray], np.ndarray]



def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def hypersphere_exact(n: int, radius: float = 1.0) -> float:
    return math.pi ** (0.5 * n) * radius**n / math.gamma(0.5 * n + 1.0)


def hypersphere_log_formula(n: int, radius: float = 1.0) -> float:
    return 0.5 * n * math.log(math.pi) + n * math.log(radius) - math.lgamma(0.5 * n + 1.0)


def advance_square_sum_counts(dp: np.ndarray, weights: np.ndarray, threshold: int) -> np.ndarray:
    updated = np.zeros_like(dp)
    for weight in weights:
        updated[weight:] += dp[: threshold + 1 - weight]
    return updated


def cartesian_cell_bounds(max_dimension: int, subdivisions: int) -> list[dict[str, float | int | str]]:
    threshold = subdivisions * subdivisions
    lower_corner_weights = np.array([k * k for k in range(subdivisions)], dtype=np.int64)
    upper_corner_weights = np.array([k * k for k in range(1, subdivisions + 1)], dtype=np.int64)
    lower_corner_counts = np.zeros(threshold + 1, dtype=np.float64)
    upper_corner_counts = np.zeros(threshold + 1, dtype=np.float64)
    lower_corner_counts[0] = 1.0
    upper_corner_counts[0] = 1.0

    rows: list[dict[str, float | int | str]] = []
    for dimension in range(max_dimension + 1):
        if dimension > 0:
            lower_corner_counts = advance_square_sum_counts(
                lower_corner_counts,
                lower_corner_weights,
                threshold,
            )
            upper_corner_counts = advance_square_sum_counts(
                upper_corner_counts,
                upper_corner_weights,
                threshold,
            )

        cell_scale = 1.0 if dimension == 0 else (2.0 / subdivisions) ** dimension
        lower_bound = float(np.sum(upper_corner_counts) * cell_scale)
        upper_bound = float(np.sum(lower_corner_counts) * cell_scale)
        midpoint = 0.5 * (lower_bound + upper_bound)
        bound_width = upper_bound - lower_bound
        rows.append(
            {
                "dimension": dimension,
                "subdivisions_per_axis": subdivisions,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "midpoint_volume": midpoint,
                "bound_width": bound_width,
                "relative_bound_width": bound_width / midpoint if midpoint != 0.0 else math.inf,
            }
        )
    return rows


def problem5(log: list[str]) -> None:
    max_dimension = 35
    subdivisions = 500
    relative_error_tolerance = 0.10
    rows = cartesian_cell_bounds(max_dimension, subdivisions)
    for row in rows:
        reference_volume = hypersphere_exact(int(row["dimension"]))
        midpoint_volume = float(row["midpoint_volume"])
        absolute_error = abs(midpoint_volume - reference_volume)
        relative_error = absolute_error / abs(reference_volume) if reference_volume != 0.0 else 0.0
        row["reference_volume"] = reference_volume
        row["absolute_error"] = absolute_error
        row["relative_error"] = relative_error
        row["status"] = "accepted" if relative_error <= relative_error_tolerance else "outside_tolerance"
    write_csv(
        RESULT_DIR / "problem5_hypersphere.csv",
        [
            {
                "dimension": row["dimension"],
                "subdivisions_per_axis": row["subdivisions_per_axis"],
                "cartesian_lower_bound": f"{float(row['lower_bound']):.16e}",
                "cartesian_upper_bound": f"{float(row['upper_bound']):.16e}",
                "cartesian_midpoint": f"{float(row['midpoint_volume']):.16e}",
                "reference_volume": f"{float(row['reference_volume']):.16e}",
                "absolute_error": f"{float(row['absolute_error']):.6e}",
                "relative_error": f"{float(row['relative_error']):.6e}",
                "bound_width": f"{float(row['bound_width']):.6e}",
                "relative_bound_width": f"{float(row['relative_bound_width']):.6e}",
                "status": row["status"],
            }
            for row in rows
        ],
        [
            "dimension",
            "subdivisions_per_axis",
            "cartesian_lower_bound",
            "cartesian_upper_bound",
            "cartesian_midpoint",
            "reference_volume",
            "absolute_error",
            "relative_error",
            "bound_width",
            "relative_bound_width",
            "status",
        ],
    )

    formula_rows = []
    for dimension in [28, 30, 32, 35, 40, 50, 100]:
        log_volume = hypersphere_log_formula(dimension)
        formula_rows.append(
            {
                "dimension": dimension,
                "log_volume": f"{log_volume:.16e}",
                "direct_formula_volume": f"{math.exp(log_volume):.16e}",
            }
        )
    write_csv(
        RESULT_DIR / "problem5_formula_extension.csv",
        formula_rows,
        ["dimension", "log_volume", "direct_formula_volume"],
    )

    dims = np.array([int(row["dimension"]) for row in rows])
    reference_values = np.array([float(row["reference_volume"]) for row in rows], dtype=float)
    midpoint_values = np.array([float(row["midpoint_volume"]) for row in rows], dtype=float)
    lower_values = np.array([float(row["lower_bound"]) for row in rows], dtype=float)
    upper_values = np.array([float(row["upper_bound"]) for row in rows], dtype=float)
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(dims, reference_values, "-", color="#2563eb", linewidth=2.0, label="Problem formula reference")
    plt.plot(dims, midpoint_values, "o--", color="#dc2626", linewidth=1.4, markersize=4, label="Cartesian cell midpoint")
    plt.fill_between(
        dims,
        lower_values,
        upper_values,
        color="#f97316",
        alpha=0.16,
        label="Cartesian lower-upper bounds",
    )
    plt.xlabel("Dimension n")
    plt.ylabel("Unit hypersphere volume")
    plt.title("Cartesian cell estimate of a unit n-dimensional hypersphere")
    plt.grid(alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "problem5_hypersphere_volume.png", dpi=200)
    plt.close()

    peak = max(rows, key=lambda row: float(row["reference_volume"]))
    max_relative_row = max(rows, key=lambda row: float(row["relative_error"]))
    validated_rows = [row for row in rows if float(row["relative_error"]) <= relative_error_tolerance]
    max_validated_dimension = max(row["dimension"] for row in validated_rows)
    log.append("Problem 5: unit hypersphere volume")
    log.append(
        f"  target dimension: n={max_dimension}; Cartesian subdivisions per axis={subdivisions}; "
        f"relative error tolerance={relative_error_tolerance:.0%}"
    )
    log.append(
        f"  problem-formula reference volume peaks over n=0..{max_dimension} at n={peak['dimension']}, "
        f"V={float(peak['reference_volume']):.12e}"
    )
    log.append(
        "  Method: deterministic Cartesian cell enclosure on [0,1]^n. "
        "Cells with upper-corner radius <= 1 give a lower bound; cells with lower-corner radius <= 1 "
        "give an upper bound; high-dimensional cell counts are accumulated by square-sum state transitions."
    )
    log.append(
        f"  max midpoint-vs-reference relative error over n=0..{max_dimension} "
        f"occurs at n={max_relative_row['dimension']}, rel={float(max_relative_row['relative_error']):.6e}"
    )
    log.append(f"  largest validated dimension: n={max_validated_dimension}")
    log.append("  Direct log-gamma formula values without volume recurrence or random sampling:")
    for row in formula_rows:
        log.append(
            f"    n={row['dimension']:3d}, "
            f"logV={float(row['log_volume']):.12e}, "
            f"V={float(row['direct_formula_volume']):.12e}"
        )
    log.append("  Cartesian cell bounds and midpoint estimates:")
    for row in rows:
        log.append(
            f"    n={row['dimension']:2d}, "
            f"lower={float(row['lower_bound']):.12e}, "
            f"upper={float(row['upper_bound']):.12e}, "
            f"midpoint={float(row['midpoint_volume']):.12e}, "
            f"reference={float(row['reference_volume']):.12e}, "
            f"rel={float(row['relative_error']):.3e}, "
            f"status={row['status']}"
        )
    log.append("")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    problem5(log)
    print("\n".join(log))


if __name__ == "__main__":
    main()
