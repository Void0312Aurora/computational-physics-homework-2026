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


def f_problem2(x: np.ndarray) -> np.ndarray:
    return np.sin(np.sqrt(100.0 * x)) ** 2


def adaptive_trapezoid_rows(eps: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    prev = composite_trapezoid(f_problem2, 0.0, 1.0, 1)
    for k in range(1, 32):
        n = 2**k
        estimate = composite_trapezoid(f_problem2, 0.0, 1.0, n)
        error_est = abs(estimate - prev) / 3.0
        rows.append(
            {
                "method": "trapezoid",
                "slices": n,
                "estimate": estimate,
                "estimated_error": error_est,
            }
        )
        if error_est < eps:
            break
        prev = estimate
    return rows


def adaptive_simpson_rows(eps: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    prev_trap = composite_trapezoid(f_problem2, 0.0, 1.0, 1)
    prev_simpson: float | None = None
    for k in range(1, 32):
        n = 2**k
        trap = composite_trapezoid(f_problem2, 0.0, 1.0, n)
        estimate = (4.0 * trap - prev_trap) / 3.0
        error_est = math.nan if prev_simpson is None else abs(estimate - prev_simpson) / 15.0
        rows.append(
            {
                "method": "simpson",
                "slices": n,
                "estimate": estimate,
                "estimated_error": error_est,
            }
        )
        if prev_simpson is not None and error_est < eps:
            break
        prev_trap = trap
        prev_simpson = estimate
    return rows


def problem2(log: list[str]) -> None:
    eps = 1e-10
    trap_rows = adaptive_trapezoid_rows(eps)
    simp_rows = adaptive_simpson_rows(eps)
    # With u = sqrt(100 x), I = (1/50) int_0^10 u sin^2(u) du.
    reference = 0.5 - 0.05 * math.sin(20.0) + (1.0 - math.cos(20.0)) / 400.0

    out_rows: list[dict[str, object]] = []
    for row in trap_rows + simp_rows:
        actual_error = abs(float(row["estimate"]) - reference)
        est = float(row["estimated_error"])
        out_rows.append(
            {
                "method": row["method"],
                "slices": row["slices"],
                "estimate": f"{float(row['estimate']):.15f}",
                "estimated_error": "" if math.isnan(est) else f"{est:.6e}",
                "actual_error_vs_quad": f"{actual_error:.6e}",
            }
        )
    write_csv(
        RESULT_DIR / "problem2_adaptive.csv",
        out_rows,
        ["method", "slices", "estimate", "estimated_error", "actual_error_vs_quad"],
    )

    plt.figure(figsize=(7.2, 4.6))
    for method, rows, color in [
        ("trapezoid", trap_rows, "#2563eb"),
        ("simpson", [row for row in simp_rows if not math.isnan(float(row["estimated_error"]))], "#dc2626"),
    ]:
        slices = np.array([float(row["slices"]) for row in rows])
        errs = np.array([float(row["estimated_error"]) for row in rows])
        plt.loglog(slices, errs, "o-", color=color, label=method)
    plt.axhline(eps, color="black", linewidth=1.2, linestyle="--", label=r"$10^{-10}$ target")
    plt.xlabel("Number of slices N")
    plt.ylabel("Estimated error")
    plt.title("Adaptive integration convergence")
    plt.grid(alpha=0.28, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "problem2_adaptive_convergence.png", dpi=200)
    plt.close()

    final_trap = trap_rows[-1]
    final_simp = simp_rows[-1]
    log.append("Problem 2: adaptive integration")
    log.append(f"  analytic reference: {reference:.15f}")
    log.append(
        "  trapezoid final: "
        f"N={final_trap['slices']}, I={float(final_trap['estimate']):.15f}, "
        f"estimated error={float(final_trap['estimated_error']):.3e}"
    )
    log.append(
        "  Simpson final:   "
        f"N={final_simp['slices']}, I={float(final_simp['estimate']):.15f}, "
        f"estimated error={float(final_simp['estimated_error']):.3e}"
    )
    log.append("")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    problem2(log)
    print("\n".join(log))


if __name__ == "__main__":
    main()
