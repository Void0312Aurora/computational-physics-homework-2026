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


def composite_trapezoid(f: ArrayFunc, a: float, b: float, n: int) -> float:
    x = np.linspace(a, b, n + 1)
    y = np.asarray(f(x), dtype=float)
    h = (b - a) / n
    return float(h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1]))


def composite_simpson(f: ArrayFunc, a: float, b: float, n: int) -> float:
    if n % 2 != 0:
        raise ValueError("Composite Simpson's rule needs an even number of slices.")
    x = np.linspace(a, b, n + 1)
    y = np.asarray(f(x), dtype=float)
    h = (b - a) / n
    return float(h / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])))


def fit_power_law(
    xs: np.ndarray,
    ys: np.ndarray,
    min_error: float = 1e-14,
    max_error: float = 1e-2,
) -> tuple[float, float]:
    mask = (ys > min_error) & (ys < max_error) & np.isfinite(ys) & np.isfinite(xs)
    log_x = np.log(xs[mask])
    log_y = np.log(ys[mask])
    if len(log_x) < 2:
        return math.nan, math.nan
    slope, intercept = np.polyfit(log_x, log_y, 1)
    return float(slope), float(intercept)


def f_arctan(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + x * x)


def arctan_error(value: float) -> float:
    return float(abs(Decimal.from_float(float(value)) - PI_OVER_4_DECIMAL))


def odd_point_sum(f: ArrayFunc, a: float, b: float, n: int, chunk: int = 1_000_000) -> float:
    h = (b - a) / n
    total = 0.0
    count = n // 2
    for offset in range(0, count, chunk):
        m = min(chunk, count - offset)
        indices = 2.0 * np.arange(offset, offset + m, dtype=float) + 1.0
        total += float(np.sum(f(a + indices * h), dtype=np.float64))
    return total


def trapezoid_sequence(f: ArrayFunc, a: float, b: float, max_k: int) -> list[float]:
    first = 0.5 * (b - a) * float(f(np.array([a], dtype=float))[0] + f(np.array([b], dtype=float))[0])
    values = [first]
    for k in range(1, max_k + 1):
        n = 2**k
        h = (b - a) / n
        values.append(0.5 * values[-1] + h * odd_point_sum(f, a, b, n))
    return values


def romberg_table_from_trapezoids(trapezoids: list[float]) -> list[list[float]]:
    table: list[list[float]] = []
    for k, trap in enumerate(trapezoids):
        row = [trap]
        for j in range(1, k + 1):
            correction = (row[j - 1] - table[k - 1][j - 1]) / (4**j - 1.0)
            row.append(row[j - 1] + correction)
        table.append(row)
    return table


def romberg_table(f: ArrayFunc, a: float, b: float, max_k: int) -> list[list[float]]:
    table: list[list[float]] = []
    for k in range(max_k + 1):
        n = 2**k
        row = [composite_trapezoid(f, a, b, n)]
        for j in range(1, k + 1):
            correction = (row[j - 1] - table[k - 1][j - 1]) / (4**j - 1.0)
            row.append(row[j - 1] + correction)
        table.append(row)
    return table


def first_accuracy_hits() -> tuple[list[dict[str, object]], list[list[float]]]:
    tolerances = [10.0 ** (-p) for p in range(2, 13)]
    rows: list[dict[str, object]] = []

    for tol in tolerances:
        for method in ["trapezoid", "simpson"]:
            hit = None
            for k in range(0 if method == "trapezoid" else 1, 25):
                n = 2**k
                if method == "trapezoid":
                    value = composite_trapezoid(f_arctan, 0.0, 1.0, n)
                else:
                    value = composite_simpson(f_arctan, 0.0, 1.0, n)
                error = arctan_error(value)
                if error < tol:
                    hit = (n, value, error)
                    break
            if hit is None:
                raise RuntimeError(f"{method} did not reach tolerance {tol}")
            rows.append(
                {
                    "method": method,
                    "tolerance": f"{tol:.0e}",
                    "slices": hit[0],
                    "estimate": f"{hit[1]:.16f}",
                    "actual_error": f"{hit[2]:.6e}",
                }
            )

    romberg = romberg_table(f_arctan, 0.0, 1.0, 18)
    for tol in tolerances:
        hit = None
        for k, row in enumerate(romberg):
            value = row[-1]
            error = arctan_error(value)
            if error < tol:
                hit = (2**k, value, error, k)
                break
        if hit is None:
            raise RuntimeError(f"Romberg did not reach tolerance {tol}")
        rows.append(
            {
                "method": "romberg",
                "tolerance": f"{tol:.0e}",
                "slices": hit[0],
                "estimate": f"{hit[1]:.16f}",
                "actual_error": f"{hit[2]:.6e}",
            }
        )
    return rows, romberg


def problem3(log: list[str]) -> None:
    accuracy_rows, romberg = first_accuracy_hits()
    write_csv(
        RESULT_DIR / "problem3_accuracy.csv",
        accuracy_rows,
        ["method", "tolerance", "slices", "estimate", "actual_error"],
    )

    scaling_max_k = 28
    trapezoids = trapezoid_sequence(f_arctan, 0.0, 1.0, scaling_max_k)
    extended_romberg = romberg_table_from_trapezoids(trapezoids)
    scaling_rows: list[dict[str, object]] = []
    for k in range(1, scaling_max_k + 1):
        n = 2**k
        h = 1.0 / n
        trap = trapezoids[k]
        simp = (4.0 * trapezoids[k] - trapezoids[k - 1]) / 3.0
        romb = extended_romberg[k][-1]
        scaling_rows.extend(
            [
                {"method": "trapezoid", "slices": n, "h": h, "estimate": trap, "actual_error": arctan_error(trap)},
                {"method": "simpson", "slices": n, "h": h, "estimate": simp, "actual_error": arctan_error(simp)},
                {"method": "romberg_diagonal", "slices": n, "h": h, "estimate": romb, "actual_error": arctan_error(romb)},
            ]
        )
    write_csv(
        RESULT_DIR / "problem3_error_scaling.csv",
        [
            {
                "method": row["method"],
                "slices": row["slices"],
                "h": f"{float(row['h']):.16e}",
                "estimate": f"{float(row['estimate']):.16f}",
                "actual_error": f"{float(row['actual_error']):.6e}",
            }
            for row in scaling_rows
        ],
        ["method", "slices", "h", "estimate", "actual_error"],
    )

    slopes: dict[str, float] = {}
    plt.figure(figsize=(7.8, 5.0))
    styles = {
        "trapezoid": ("#2563eb", "o", "trapezoid"),
        "simpson": ("#dc2626", "s", "Simpson"),
        "romberg_diagonal": ("#16a34a", "^", "Romberg diagonal"),
    }
    for method, (color, marker, label) in styles.items():
        rows = [row for row in scaling_rows if row["method"] == method and float(row["actual_error"]) > 0.0]
        h = np.array([float(row["h"]) for row in rows])
        err = np.array([float(row["actual_error"]) for row in rows])
        if method == "simpson":
            slope, _ = fit_power_law(h, err, min_error=1e-14, max_error=1e-7)
        elif method == "romberg_diagonal":
            slope, _ = fit_power_law(h, err, min_error=1e-14, max_error=1e-3)
        else:
            slope, _ = fit_power_law(h, err)
        slopes[method] = slope
        plt.loglog(h, err, marker=marker, color=color, linewidth=1.3, markersize=4.2, label=f"{label}, fit p={slope:.2f}")
    plt.axhline(np.finfo(float).eps, color="#6b7280", linewidth=1.0, linestyle=":", label=r"machine epsilon")
    plt.ylim(1e-18, 3e-2)
    plt.gca().invert_xaxis()
    plt.xlabel("Step size h")
    plt.ylabel(r"Actual error $|I_h-\pi/4|$")
    plt.title("Error scaling for three integration methods")
    plt.grid(alpha=0.28, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "problem3_error_scaling.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7.8, 4.8))
    for method, (color, marker, label) in styles.items():
        rows = [
            row
            for row in scaling_rows
            if row["method"] == method and float(row["actual_error"]) > 0.0 and float(row["h"]) <= 2.0**-8
        ]
        h = np.array([float(row["h"]) for row in rows])
        err = np.array([float(row["actual_error"]) for row in rows])
        plt.loglog(h, err, marker=marker, color=color, linewidth=1.25, markersize=4.0, label=label)
    plt.axhline(np.finfo(float).eps, color="#6b7280", linewidth=1.0, linestyle=":", label=r"machine epsilon")
    plt.ylim(1e-18, 1e-10)
    plt.gca().invert_xaxis()
    plt.xlabel("Step size h")
    plt.ylabel(r"Actual error $|I_h-\pi/4|$")
    plt.title("Round-off plateau after further reducing h")
    plt.grid(alpha=0.28, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "problem3_roundoff_zoom.png", dpi=200)
    plt.close()

    observation_rows: list[dict[str, object]] = []
    for method in styles:
        rows = [row for row in scaling_rows if row["method"] == method]
        best = min(rows, key=lambda row: float(row["actual_error"]))
        last = rows[-1]
        observation_rows.append(
            {
                "method": method,
                "best_slices": best["slices"],
                "best_h": f"{float(best['h']):.16e}",
                "best_error": f"{float(best['actual_error']):.6e}",
                "last_slices": last["slices"],
                "last_h": f"{float(last['h']):.16e}",
                "last_error": f"{float(last['actual_error']):.6e}",
            }
        )
    write_csv(
        RESULT_DIR / "problem3_extended_observations.csv",
        observation_rows,
        ["method", "best_slices", "best_h", "best_error", "last_slices", "last_h", "last_error"],
    )

    log.append("Problem 3: accuracy and error scaling for arctan integral")
    for method in ["trapezoid", "simpson", "romberg"]:
        last = [row for row in accuracy_rows if row["method"] == method and row["tolerance"] == "1e-12"][0]
        log.append(
            f"  {method:9s} reaches 1e-12 with N={last['slices']}, "
            f"estimate={last['estimate']}, error={last['actual_error']}"
        )
    log.append(f"  fitted trapezoid order p={slopes['trapezoid']:.4f}")
    log.append(f"  fitted Simpson order p={slopes['simpson']:.4f}")
    log.append(f"  fitted Romberg diagonal effective p={slopes['romberg_diagonal']:.4f}")
    log.append(f"  extended scaling axis: N=2..{2**scaling_max_k}, h=5.000e-01..{2.0 ** (-scaling_max_k):.3e}")
    for row in observation_rows:
        log.append(
            f"  {row['method']} best error={row['best_error']} at N={row['best_slices']}; "
            f"end error={row['last_error']} at N={row['last_slices']}"
        )
    log.append("")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    problem3(log)
    print("\n".join(log))


if __name__ == "__main__":
    main()
