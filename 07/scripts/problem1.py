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


def composite_simpson(f: ArrayFunc, a: float, b: float, n: int) -> float:
    if n % 2 != 0:
        raise ValueError("Composite Simpson's rule needs an even number of slices.")
    x = np.linspace(a, b, n + 1)
    y = np.asarray(f(x), dtype=float)
    h = (b - a) / n
    return float(h / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])))


def debye_integrand(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-6
    # x^4 exp(x) / (exp(x) - 1)^2 = x^2 - x^4 / 12 + O(x^6)
    out[small] = x[small] ** 2 - x[small] ** 4 / 12.0
    y = x[~small]
    em = np.exp(-y)
    out[~small] = y**4 * em / (1.0 - em) ** 2
    return out


def problem1(log: list[str]) -> None:
    volume = 1000.0e-6
    rho = 6.022e28
    theta_d = 428.0
    k_b = 1.380649e-23
    simpson_slices = 50

    def heat_capacity(temp: float) -> float:
        upper = theta_d / temp
        integral = composite_simpson(debye_integrand, 0.0, upper, simpson_slices)
        return 9.0 * volume * rho * k_b * (temp / theta_d) ** 3 * integral

    temps = np.linspace(5.0, 500.0, 496)
    cvs = np.array([heat_capacity(float(temp)) for temp in temps])
    rows = [
        {"T_K": f"{temp:.6f}", "C_V_J_per_K": f"{cv:.12e}"}
        for temp, cv in zip(temps, cvs, strict=True)
    ]
    write_csv(RESULT_DIR / "problem1_heat_capacity.csv", rows, ["T_K", "C_V_J_per_K"])

    selected_temps = [5.0, 25.0, 50.0, 100.0, 200.0, 300.0, 428.0, 500.0]
    selected_rows = [
        {"T_K": f"{temp:.1f}", "C_V_J_per_K": f"{heat_capacity(temp):.10f}"}
        for temp in selected_temps
    ]
    write_csv(RESULT_DIR / "problem1_selected_values.csv", selected_rows, ["T_K", "C_V_J_per_K"])

    dulong_petit = 3.0 * volume * rho * k_b
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(temps, cvs, color="#2563eb", linewidth=2.2, label="Debye heat capacity")
    plt.axhline(dulong_petit, color="#dc2626", linestyle="--", linewidth=1.4, label="Dulong-Petit limit")
    plt.xlabel("Temperature T (K)")
    plt.ylabel(r"$C_V$ (J/K)")
    plt.title("Debye heat capacity of 1000 cm^3 aluminum")
    plt.grid(alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "problem1_heat_capacity.png", dpi=200)
    plt.close()

    log.append("Problem 1: Debye heat capacity")
    log.append(f"  Simpson slices: {simpson_slices}")
    log.append(f"  Dulong-Petit limit: {dulong_petit:.10f} J/K")
    for row in selected_rows:
        log.append(f"  T={row['T_K']:>5} K, C_V={row['C_V_J_per_K']} J/K")
    log.append("")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    problem1(log)
    print("\n".join(log))


if __name__ == "__main__":
    main()
