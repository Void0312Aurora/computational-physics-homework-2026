from __future__ import annotations

import matplotlib
import numpy as np

from interpolation_common import neville_evaluate
from result_paths import PROBLEM1_RESULT_DIR, ensure_result_dir, write_csv

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pchip_endpoint_slope(h1: float, h2: float, delta1: float, delta2: float) -> float:
    slope = ((2.0 * h1 + h2) * delta1 - h1 * delta2) / (h1 + h2)
    if slope == 0.0 or np.sign(slope) != np.sign(delta1):
        return 0.0
    if np.sign(delta1) != np.sign(delta2) and abs(slope) > abs(3.0 * delta1):
        return 3.0 * delta1
    return slope


def pchip_coefficients(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    m = np.zeros(n)

    for i in range(1, n - 1):
        if delta[i - 1] == 0.0 or delta[i] == 0.0 or np.sign(delta[i - 1]) != np.sign(delta[i]):
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    m[0] = pchip_endpoint_slope(h[0], h[1], delta[0], delta[1])
    m[-1] = pchip_endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])

    a = y[:-1].copy()
    b = m[:-1].copy()
    c = (3.0 * delta - 2.0 * m[:-1] - m[1:]) / h
    d = (m[:-1] + m[1:] - 2.0 * delta) / (h**2)
    return a, b, c, d


def evaluate_piecewise_cubic(
    x_nodes: np.ndarray,
    coeffs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    x_eval: np.ndarray,
) -> np.ndarray:
    a, b, c, d = coeffs
    n = len(x_nodes) - 1
    interval_ids = np.searchsorted(x_nodes, x_eval, side="right") - 1
    interval_ids = np.clip(interval_ids, 0, n - 1)
    dx = x_eval - x_nodes[interval_ids]
    return a[interval_ids] + b[interval_ids] * dx + c[interval_ids] * dx**2 + d[interval_ids] * dx**3


def solve_problem1() -> dict[str, object]:
    ensure_result_dir(PROBLEM1_RESULT_DIR)

    voltage = np.array([-1.00, 0.00, 1.27, 2.55, 3.82, 4.92, 5.02], dtype=float)
    current = np.array([-14.58, 0.00, 0.00, 0.00, 0.00, 0.88, 11.17], dtype=float)

    x_dense = np.linspace(voltage.min(), voltage.max(), 1600)
    poly_vals = np.asarray(neville_evaluate(voltage, current, x_dense))
    linear_vals = np.interp(x_dense, voltage, current)
    pchip = pchip_coefficients(voltage, current)
    pchip_vals = evaluate_piecewise_cubic(voltage, pchip, x_dense)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x_dense, poly_vals, label="Global degree-6 interpolation (Neville)", linewidth=2.0, color="#d1495b")
    ax.scatter(voltage, current, s=55, color="black", zorder=5, label="Data points")
    ax.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.5)
    ax.set_title("Problem 1(1): Global degree-6 interpolation")
    ax.set_xlabel("Voltage")
    ax.set_ylabel("Current")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(PROBLEM1_RESULT_DIR / "problem1_global_degree6.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x_dense, linear_vals, label="Piecewise linear", linewidth=2.0, color="#2e86ab")
    ax.plot(x_dense, pchip_vals, label="Shape-preserving cubic (PCHIP)", linewidth=2.2, color="#2a9d8f")
    ax.scatter(voltage, current, s=55, color="black", zorder=5, label="Data points")
    ax.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.5)
    ax.set_title("Problem 1(2): Low-order piecewise interpolation")
    ax.set_xlabel("Voltage")
    ax.set_ylabel("Current")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(PROBLEM1_RESULT_DIR / "problem1_piecewise_low_order.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x_dense, poly_vals, label="Global degree-6 interpolation (Neville)", linewidth=2.0, color="#d1495b")
    ax.plot(x_dense, linear_vals, label="Piecewise linear", linewidth=2.0, color="#2e86ab")
    ax.plot(x_dense, pchip_vals, label="Shape-preserving cubic (PCHIP)", linewidth=2.2, color="#2a9d8f")
    ax.scatter(voltage, current, s=55, color="black", zorder=5, label="Data points")
    ax.set_title("Problem 1: Zener diode interpolation comparison")
    ax.set_xlabel("Voltage")
    ax.set_ylabel("Current")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(PROBLEM1_RESULT_DIR / "problem1_zener_interpolation.png", dpi=220)
    plt.close(fig)

    summary_rows: list[dict[str, object]] = []
    for name, values in [
        ("global_degree_6", poly_vals),
        ("piecewise_linear", linear_vals),
        ("shape_preserving_cubic", pchip_vals),
    ]:
        summary_rows.append(
            {
                "curve": name,
                "min_current": float(np.min(values)),
                "max_current": float(np.max(values)),
                "range_width": float(np.max(values) - np.min(values)),
            }
        )
    write_csv(
        PROBLEM1_RESULT_DIR / "problem1_curve_summary.csv",
        ["curve", "min_current", "max_current", "range_width"],
        summary_rows,
    )

    midpoints = 0.5 * (voltage[:-1] + voltage[1:])
    midpoint_rows: list[dict[str, object]] = []
    midpoint_poly = np.asarray(neville_evaluate(voltage, current, midpoints))
    midpoint_linear = np.interp(midpoints, voltage, current)
    midpoint_pchip = evaluate_piecewise_cubic(voltage, pchip, midpoints)
    for x_mid, p_val, l_val, s_val in zip(midpoints, midpoint_poly, midpoint_linear, midpoint_pchip):
        midpoint_rows.append(
            {
                "midpoint_voltage": float(x_mid),
                "global_degree_6": float(p_val),
                "piecewise_linear": float(l_val),
                "shape_preserving_cubic": float(s_val),
            }
        )
    write_csv(
        PROBLEM1_RESULT_DIR / "problem1_midpoints.csv",
        ["midpoint_voltage", "global_degree_6", "piecewise_linear", "shape_preserving_cubic"],
        midpoint_rows,
    )

    return {
        "voltage": voltage,
        "current": current,
        "curve_summary": summary_rows,
        "midpoints": midpoint_rows,
    }


def main() -> None:
    result = solve_problem1()
    print(f"Problem 1 curve rows: {len(result['curve_summary'])}")


if __name__ == "__main__":
    main()
