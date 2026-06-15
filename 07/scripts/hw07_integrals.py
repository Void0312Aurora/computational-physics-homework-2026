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
from scipy.integrate import cubature


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


def problem4(log: list[str]) -> None:
    n = 1024
    cases = [
        {
            "case": "a",
            "integral": "int_-1^1 sqrt(1-x^2) dx",
            "transformation": "x = sin(theta), theta in [-pi/2, pi/2]",
            "value": composite_simpson(lambda theta: np.cos(theta) ** 2, -math.pi / 2.0, math.pi / 2.0, n),
            "exact": math.pi / 2.0,
        },
        {
            "case": "b",
            "integral": "int_0^pi sin^2(theta) dtheta",
            "transformation": "direct Simpson integration",
            "value": composite_simpson(lambda theta: np.sin(theta) ** 2, 0.0, math.pi, n),
            "exact": math.pi / 2.0,
        },
        {
            "case": "c",
            "integral": "int_0^inf dx / ((1+x) sqrt(x))",
            "transformation": "x = tan^2(theta), theta in [0, pi/2]",
            "value": composite_simpson(lambda theta: np.full_like(theta, 2.0), 0.0, math.pi / 2.0, n),
            "exact": math.pi,
        },
    ]
    rows = []
    for case in cases:
        rows.append(
            {
                "case": case["case"],
                "integral": case["integral"],
                "transformation": case["transformation"],
                "slices": n,
                "estimate": f"{float(case['value']):.16f}",
                "exact": f"{float(case['exact']):.16f}",
                "actual_error": f"{abs(float(case['value']) - float(case['exact'])):.6e}",
            }
        )
    write_csv(
        RESULT_DIR / "problem4_integrals.csv",
        rows,
        ["case", "integral", "transformation", "slices", "estimate", "exact", "actual_error"],
    )

    log.append("Problem 4: transformed integrals")
    for row in rows:
        log.append(
            f"  ({row['case']}) estimate={row['estimate']}, exact={row['exact']}, "
            f"error={row['actual_error']}"
        )
    log.append("")


def hypersphere_exact(n: int, radius: float = 1.0) -> float:
    return math.pi ** (0.5 * n) * radius**n / math.gamma(0.5 * n + 1.0)


PI_DECIMAL = Decimal(
    "3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068"
)


def hypersphere_exact_decimal(n: int) -> Decimal:
    if n == 0:
        return Decimal(1)
    if n % 2 == 0:
        k = n // 2
        return (PI_DECIMAL**k) / Decimal(math.factorial(k))
    k = (n - 1) // 2
    numerator = (Decimal(2) ** Decimal(2 * k + 1)) * Decimal(math.factorial(k)) * (PI_DECIMAL**k)
    denominator = Decimal(math.factorial(2 * k + 1))
    return numerator / denominator


def hypersphere_cubature(
    n: int,
    rtol: float = 1.0e-2,
    atol: float = 1.0e-10,
    max_subdivisions: int = 1000,
) -> tuple[float, float, str, int]:
    lower = np.zeros(n, dtype=float)
    upper = np.ones(n, dtype=float)

    def indicator(points: np.ndarray) -> np.ndarray:
        return (np.sum(points * points, axis=1) <= 1.0).astype(float)

    result = cubature(
        indicator,
        lower,
        upper,
        rule="genz-malik",
        rtol=rtol,
        atol=atol,
        max_subdivisions=max_subdivisions,
        workers=1,
    )
    symmetry_factor = float(2**n)
    estimate = float(np.asarray(result.estimate)) * symmetry_factor
    error = float(np.max(np.atleast_1d(result.error))) * symmetry_factor
    return estimate, error, str(result.status), int(result.subdivisions)


def problem5(log: list[str]) -> None:
    rows = []
    for n in range(0, 10):
        exact = hypersphere_exact_decimal(n)
        if n == 0:
            estimate = 1.0
            error_estimate = 0.0
            status = "exact_seed"
            subdivisions = 0
        elif n == 1:
            estimate = 2.0
            error_estimate = 0.0
            status = "exact_seed"
            subdivisions = 0
        else:
            estimate, error_estimate, status, subdivisions = hypersphere_cubature(n)
        cartesian_value = Decimal(f"{estimate:.16f}")
        abs_difference = abs(exact - cartesian_value)
        relative_difference = abs_difference / abs(exact) if exact != 0 else Decimal(0)
        rows.append(
            {
                "dimension": n,
                "exact_volume": exact,
                "cartesian_volume": cartesian_value,
                "abs_difference": abs_difference,
                "relative_difference": relative_difference,
                "error_estimate": error_estimate,
                "status": status,
                "subdivisions": subdivisions,
            }
        )
    write_csv(
        RESULT_DIR / "problem5_hypersphere.csv",
        [
            {
                "dimension": row["dimension"],
                "cartesian_volume": format(row["cartesian_volume"], "f"),
                "exact_volume": format(row["exact_volume"], "f"),
                "abs_difference": format(row["abs_difference"], "E"),
                "relative_difference": format(row["relative_difference"], "E"),
                "error_estimate": f"{row['error_estimate']:.6E}",
                "status": row["status"],
                "subdivisions": row["subdivisions"],
            }
            for row in rows
        ],
        [
            "dimension",
            "cartesian_volume",
            "exact_volume",
            "abs_difference",
            "relative_difference",
            "error_estimate",
            "status",
            "subdivisions",
        ],
    )

    dims = np.array([int(row["dimension"]) for row in rows])
    exact_values = np.array([float(row["exact_volume"]) for row in rows], dtype=float)
    cartesian_values = np.array([float(row["cartesian_volume"]) for row in rows], dtype=float)
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(dims, exact_values, "o-", color="#2563eb", linewidth=2.0, label="Exact gamma formula")
    plt.plot(
        dims,
        cartesian_values,
        "s--",
        color="#dc2626",
        linewidth=1.6,
        markersize=5,
        label="Cartesian cubature with symmetry",
    )
    plt.xlabel("Dimension n")
    plt.ylabel("Unit hypersphere volume")
    plt.title("Volume of a unit n-dimensional hypersphere")
    plt.grid(alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "problem5_hypersphere_volume.png", dpi=200)
    plt.close()

    peak = max(rows, key=lambda row: float(row["exact_volume"]))
    max_relative_row = max(rows, key=lambda row: row["relative_difference"])
    acceptable_rows = [row for row in rows if row["relative_difference"] <= Decimal("0.1")]
    max_acceptable_dimension = max(row["dimension"] for row in acceptable_rows)
    log.append("Problem 5: unit hypersphere volume")
    log.append(
        f"  exact volume peaks over n=0..9 at n={peak['dimension']}, "
        f"V={format(peak['exact_volume'], 'f')}"
    )
    log.append(f"  Decimal working precision: {getcontext().prec} digits")
    log.append(
        "  True multidimensional Cartesian cubature with symmetry: "
        "integrate over [0,1]^n and multiply by 2^n; rule=genz-malik, rtol=1e-2, max_subdivisions=1000"
    )
    log.append(
        f"  max relative difference over n=0..9 occurs at n={max_relative_row['dimension']}, "
        f"|diff|/|exact|={format(max_relative_row['relative_difference'], 'E')}"
    )
    log.append(f"  largest dimension with relative error <= 10%: n={max_acceptable_dimension}")
    log.append("  exact closed form and Cartesian-integral cross-check:")
    for row in rows:
        log.append(
            f"    n={row['dimension']:2d}, exact={format(row['exact_volume'], 'f')}, "
            f"cartesian={format(row['cartesian_volume'], 'f')}, |diff|={format(row['abs_difference'], 'E')}, "
            f"rel={format(row['relative_difference'], 'E')}, "
            f"err_est={row['error_estimate']:.3E}, status={row['status']}, subs={row['subdivisions']}"
        )
    log.append("")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    log.append("HW/07 numerical integration run")
    log.append("================================")
    log.append("")
    problem1(log)
    problem2(log)
    problem3(log)
    problem4(log)
    problem5(log)
    text = "\n".join(log)
    (RESULT_DIR / "temp-01.log").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
