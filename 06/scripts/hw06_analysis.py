from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"
MAX_ADJACENT_LOG_JUMP = 1.5
MIN_SEGMENT_POINTS = 4
SMOOTH_WINDOW = 9


def solve_moment_weights(offsets: tuple[int, ...], derivative_order: int) -> tuple[sp.Rational, ...]:
    size = len(offsets)
    matrix = sp.Matrix(
        [[sp.Integer(offset) ** power for offset in offsets] for power in range(size)]
    )
    rhs = sp.Matrix(
        [sp.factorial(derivative_order) if power == derivative_order else 0 for power in range(size)]
    )
    solution = matrix.LUsolve(rhs)
    return tuple(sp.simplify(value) for value in solution)


@dataclass(frozen=True)
class FormulaSpec:
    name: str
    derivative_order: int
    offsets: tuple[int, ...]
    accuracy_order: int
    family: str

    @property
    def weights(self) -> tuple[sp.Rational, ...]:
        return solve_moment_weights(self.offsets, self.derivative_order)

    def evaluate(self, func: Callable[[float], float], x0: float, h: float) -> float:
        total = 0.0
        for weight, offset in zip(self.weights, self.offsets):
            total += float(weight) * func(x0 + offset * h)
        return total / (h ** self.derivative_order)


def ensure_result_dir() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def sympy_float(value: float) -> sp.Float:
    return sp.Float(value, 30)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def split_curve_segments(
    y_values: np.ndarray,
    max_log_jump: float = MAX_ADJACENT_LOG_JUMP,
) -> tuple[np.ndarray, list[np.ndarray]]:
    plot_values = np.asarray(y_values, dtype=float)
    valid = np.isfinite(plot_values) & (plot_values > 0.0)
    valid_indices = np.where(valid)[0]
    if valid_indices.size == 0:
        return valid, []

    if valid_indices.size == 1:
        return valid, [valid_indices]

    log_values = np.log10(plot_values[valid_indices])
    split_points = np.where(np.abs(np.diff(log_values)) > max_log_jump)[0] + 1
    segments = np.split(valid_indices, split_points)
    return valid, [segment for segment in segments if segment.size > 0]


def smooth_log_curve(y_values: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    plot_values = np.asarray(y_values, dtype=float)
    smoothed = np.full_like(plot_values, np.nan, dtype=float)
    valid_indices = np.where(np.isfinite(plot_values) & (plot_values > 0.0))[0]
    if valid_indices.size == 0:
        return smoothed

    if window < 1:
        window = 1
    if window % 2 == 0:
        window += 1
    half_window = window // 2
    log_values = np.log10(plot_values[valid_indices])
    smoothed_log = np.empty_like(log_values)
    for index in range(log_values.size):
        left = max(0, index - half_window)
        right = min(log_values.size, index + half_window + 1)
        smoothed_log[index] = float(np.median(log_values[left:right]))
    smoothed[valid_indices] = np.power(10.0, smoothed_log)
    return smoothed


def plot_error_curves(
    output_path: Path,
    title: str,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    x_label: str = "h",
    y_label: str = "Absolute error",
    x_limits: tuple[float, float] | None = None,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    colors = ["#1b4965", "#5fa8d3", "#ca6702", "#ae2012", "#6a4c93", "#2a9d8f"]
    visible_curve_values: list[np.ndarray] = []
    for index, (name, (x_values, y_values)) in enumerate(curves.items()):
        color = colors[index % len(colors)]
        raw_values = np.asarray(y_values, dtype=float)
        valid_mask = np.isfinite(raw_values) & (raw_values > 0.0)
        smooth_values = smooth_log_curve(raw_values)
        if np.any(valid_mask):
            ax.scatter(
                x_values[valid_mask],
                raw_values[valid_mask],
                s=11,
                alpha=0.24,
                color=color,
                edgecolors="none",
                zorder=1,
            )

        smooth_mask = np.isfinite(smooth_values) & (smooth_values > 0.0)
        if np.any(smooth_mask):
            ax.loglog(
                x_values[smooth_mask],
                smooth_values[smooth_mask],
                label=name,
                linewidth=2.2,
                color=color,
                zorder=2,
            )

        visible_mask = valid_mask.copy()
        if x_limits is not None:
            visible_mask &= (x_values >= x_limits[0]) & (x_values <= x_limits[1])
        if np.any(visible_mask):
            visible_curve_values.append(np.asarray(y_values, dtype=float)[visible_mask])
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if visible_curve_values:
        combined_values = np.concatenate(visible_curve_values)
        ymin = float(np.nanmin(combined_values))
        ymax = float(np.nanmax(combined_values))
        ax.set_ylim(max(ymin / 3.0, 1.0e-16), ymax * 3.0)
    ax.legend(frameon=True)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_problem4_model(output_path: Path, hs: np.ndarray, errors: np.ndarray, model: np.ndarray) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    ax.plot(
        np.log10(hs),
        np.log10(errors),
        "o",
        markersize=3.4,
        color="#3d348b",
        label="Measured error",
        alpha=0.8,
    )
    ax.plot(
        np.log10(hs),
        np.log10(model),
        linewidth=2.4,
        color="#1d3557",
        label="Model g(h)",
    )
    ax.set_xlabel(r"$\log_{10}(h)$")
    ax.set_ylabel(r"$\log_{10}(|\mathrm{error}|)$")
    ax.set_title(r"Forward-difference error for $f(x)=\sin x$ at $x=0.5$")
    ax.legend(frameon=True)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def summarize_best_error_per_formula(
    formulas: list[FormulaSpec],
    func: Callable[[float], float],
    x0: float,
    exact_value: float,
    hs: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    records: list[dict[str, object]] = []
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for spec in formulas:
        errors = np.array([abs(spec.evaluate(func, x0, float(h)) - exact_value) for h in hs], dtype=float)
        curves[spec.name] = (hs, errors)
        best_index = int(errors.argmin())
        records.append(
            {
                "formula": spec.name,
                "accuracy_order": spec.accuracy_order,
                "best_h": float(hs[best_index]),
                "best_abs_error": float(errors[best_index]),
            }
        )
    records.sort(key=lambda row: float(row["best_abs_error"]))
    return records, curves


def analyze_derivative_problem(
    *,
    problem_label: str,
    func: Callable[[float], float],
    exact_derivative: Callable[[float], float],
    x0: float,
    table_values: dict[float, float],
    formulas: list[FormulaSpec],
    exact_table_output: str,
    sweep_output: str,
    plot_output: str,
) -> dict[str, object]:
    exact_value = exact_derivative(x0)
    table_rows: list[dict[str, object]] = []
    for spec in formulas:
        approx = spec.evaluate(lambda x: table_values[round(x, 1)], x0, 0.1)
        error = approx - exact_value
        table_rows.append(
            {
                "formula": spec.name,
                "family": spec.family,
                "accuracy_order": spec.accuracy_order,
                "approximation": approx,
                "abs_error": abs(error),
                "signed_error": error,
            }
        )

    write_csv(
        RESULT_DIR / exact_table_output,
        table_rows,
        ["formula", "family", "accuracy_order", "approximation", "abs_error", "signed_error"],
    )

    hs_dense = np.logspace(-16, -1, 400)
    hs_report = np.array([10.0 ** (-k) for k in range(1, 13)], dtype=float)
    sweep_rows: list[dict[str, object]] = []
    plot_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    best_per_formula: list[dict[str, object]] = []

    overall_best_error = float("inf")
    overall_best_formula = ""
    overall_best_h = 0.0

    for spec in formulas:
        errors_dense = np.array(
            [abs(spec.evaluate(func, x0, float(h)) - exact_value) for h in hs_dense],
            dtype=float,
        )
        plot_curves[spec.name] = (hs_dense, errors_dense)
        dense_best_index = int(errors_dense.argmin())
        dense_best_error = float(errors_dense[dense_best_index])
        dense_best_h = float(hs_dense[dense_best_index])
        best_per_formula.append(
            {
                "formula": spec.name,
                "best_h": dense_best_h,
                "best_abs_error": dense_best_error,
                "accuracy_order": spec.accuracy_order,
            }
        )
        if dense_best_error < overall_best_error:
            overall_best_error = dense_best_error
            overall_best_formula = spec.name
            overall_best_h = dense_best_h

        for h in hs_report:
            approx = spec.evaluate(func, x0, float(h))
            sweep_rows.append(
                {
                    "formula": spec.name,
                    "h": float(h),
                    "approximation": approx,
                    "abs_error": abs(approx - exact_value),
                }
            )

    write_csv(
        RESULT_DIR / sweep_output,
        sweep_rows,
        ["formula", "h", "approximation", "abs_error"],
    )
    plot_error_curves(
        RESULT_DIR / plot_output,
        title=f"{problem_label}: absolute error versus h",
        curves=plot_curves,
        x_limits=(1.0e-10, 1.0e-1),
    )

    best_from_table = min(table_rows, key=lambda row: float(row["abs_error"]))
    return {
        "exact_value": exact_value,
        "table_results": table_rows,
        "table_best_formula": best_from_table["formula"],
        "table_best_abs_error": float(best_from_table["abs_error"]),
        "overall_best_formula": overall_best_formula,
        "overall_best_h": overall_best_h,
        "overall_best_abs_error": overall_best_error,
        "best_per_formula": best_per_formula,
        "table_csv": str((RESULT_DIR / exact_table_output).relative_to(ROOT)),
        "sweep_csv": str((RESULT_DIR / sweep_output).relative_to(ROOT)),
        "plot": str((RESULT_DIR / plot_output).relative_to(ROOT)),
    }


def estimate_total_error_model(
    *,
    formulas: list[FormulaSpec],
    func_expr: sp.Expr,
    x_symbol: sp.Symbol,
    x0: float,
    empirical_best_rows: list[dict[str, object]],
    output_name: str,
) -> list[dict[str, object]]:
    unit_roundoff = float(np.finfo(float).eps / 2.0)
    f0_abs = abs(float(func_expr.subs(x_symbol, x0)))
    empirical_by_formula = {str(row["formula"]): row for row in empirical_best_rows}
    rows: list[dict[str, object]] = []

    for spec in formulas:
        leading_derivative_order = spec.derivative_order + spec.accuracy_order
        leading_moment = sum(
            float(weight) * (float(offset) ** leading_derivative_order)
            for weight, offset in zip(spec.weights, spec.offsets)
        ) / math.factorial(leading_derivative_order)
        leading_derivative_value = float(
            sp.diff(func_expr, x_symbol, leading_derivative_order).subs(x_symbol, x0)
        )
        truncation_constant = abs(leading_moment * leading_derivative_value)
        roundoff_constant = f0_abs * sum(abs(float(weight)) for weight in spec.weights)

        model_best_h = (
            (spec.derivative_order * roundoff_constant * unit_roundoff)
            / (spec.accuracy_order * truncation_constant)
        ) ** (1.0 / (spec.accuracy_order + spec.derivative_order))
        model_min_error = truncation_constant * (model_best_h ** spec.accuracy_order) + (
            roundoff_constant * unit_roundoff / (model_best_h ** spec.derivative_order)
        )
        empirical = empirical_by_formula[spec.name]
        empirical_best_h = float(empirical["best_h"])
        empirical_min_error = float(empirical["best_abs_error"])
        rows.append(
            {
                "formula": spec.name,
                "accuracy_order": spec.accuracy_order,
                "truncation_constant": truncation_constant,
                "roundoff_constant_estimate": roundoff_constant,
                "model_best_h": model_best_h,
                "empirical_best_h": empirical_best_h,
                "best_h_ratio_empirical_over_model": empirical_best_h / model_best_h,
                "model_min_error": model_min_error,
                "empirical_min_error": empirical_min_error,
            }
        )

    write_csv(
        RESULT_DIR / output_name,
        rows,
        [
            "formula",
            "accuracy_order",
            "truncation_constant",
            "roundoff_constant_estimate",
            "model_best_h",
            "empirical_best_h",
            "best_h_ratio_empirical_over_model",
            "model_min_error",
            "empirical_min_error",
        ],
    )
    return rows


def analyze_problem4() -> dict[str, object]:
    def forward_difference(func: Callable[[float], float], x: float, delta: float) -> float:
        return (func(x + delta) - func(x)) / delta

    part_a_function = lambda x: x * (x - 1.0)
    part_a_exact = 1.0
    part_a_delta = 1e-2
    part_a_approx = forward_difference(part_a_function, 1.0, part_a_delta)

    deltas = np.array([10.0 ** (-k) for k in range(2, 19, 2)], dtype=float)
    rows_b: list[dict[str, object]] = []
    errors_b = []
    for delta in deltas:
        approximation = forward_difference(part_a_function, 1.0, float(delta))
        abs_error = abs(approximation - part_a_exact)
        rows_b.append(
            {
                "delta": float(delta),
                "approximation": approximation,
                "abs_error": abs_error,
            }
        )
        errors_b.append(abs_error)

    write_csv(
        RESULT_DIR / "problem4_partb_forward_difference.csv",
        rows_b,
        ["delta", "approximation", "abs_error"],
    )
    plot_error_curves(
        RESULT_DIR / "problem4_partb_error.png",
        title=r"Problem 4(b): forward-difference error for $f(x)=x(x-1)$ at $x=1$",
        curves={"Forward difference": (deltas, np.array(errors_b, dtype=float))},
        x_label=r"$\delta$",
        y_label="Absolute error",
    )

    hs = np.logspace(-20, 0, 201)
    x0 = 0.5
    exact = math.cos(x0)
    measured_errors = np.array(
        [abs(forward_difference(math.sin, x0, float(h)) - exact) for h in hs],
        dtype=float,
    )
    epsilon_tilde = 7.0e-17
    model_errors = (np.sin(x0) * hs / 2.0) + (2.0 * epsilon_tilde * np.sin(x0) / hs)

    write_csv(
        RESULT_DIR / "problem4_partc_error_model.csv",
        [
            {
                "h": float(h),
                "log10_h": math.log10(float(h)),
                "measured_abs_error": float(err),
                "model_abs_error": float(model),
            }
            for h, err, model in zip(hs, measured_errors, model_errors)
        ],
        ["h", "log10_h", "measured_abs_error", "model_abs_error"],
    )
    plot_problem4_model(
        RESULT_DIR / "problem4_partc_error_model.png",
        hs=hs,
        errors=measured_errors,
        model=model_errors,
    )

    part_c_formulas = [
        FormulaSpec("Forward O(h)", 1, (0, 1), 1, "forward"),
        FormulaSpec("Backward O(h)", 1, (-1, 0), 1, "backward"),
        FormulaSpec("Forward O(h^2)", 1, (0, 1, 2), 2, "forward"),
        FormulaSpec("Backward O(h^2)", 1, (-2, -1, 0), 2, "backward"),
        FormulaSpec("Central O(h^2)", 1, (-1, 0, 1), 2, "central"),
        FormulaSpec("Central O(h^4)", 1, (-2, -1, 0, 1, 2), 4, "central"),
    ]
    template_best_rows, template_curves = summarize_best_error_per_formula(
        formulas=part_c_formulas,
        func=math.sin,
        x0=x0,
        exact_value=exact,
        hs=hs,
    )
    template_curve_rows: list[dict[str, object]] = []
    for spec in part_c_formulas:
        curve_hs, curve_errors = template_curves[spec.name]
        for h, error in zip(curve_hs, curve_errors):
            template_curve_rows.append(
                {
                    "formula": spec.name,
                    "h": float(h),
                    "log10_h": math.log10(float(h)),
                    "abs_error": float(error),
                    "log10_abs_error": math.log10(float(error)),
                }
            )

    write_csv(
        RESULT_DIR / "problem4_partc_template_comparison.csv",
        template_curve_rows,
        ["formula", "h", "log10_h", "abs_error", "log10_abs_error"],
    )
    write_csv(
        RESULT_DIR / "problem4_partc_template_best.csv",
        template_best_rows,
        ["formula", "accuracy_order", "best_h", "best_abs_error"],
    )
    plot_error_curves(
        RESULT_DIR / "problem4_partc_template_comparison.png",
        title=r"Problem 4(c): error comparison for $f(x)=\sin x$ at $x=0.5$",
        curves=template_curves,
        x_limits=(1.0e-20, 1.0e0),
    )

    empirical_index = int(measured_errors.argmin())
    empirical_best_h = float(hs[empirical_index])
    empirical_best_error = float(measured_errors[empirical_index])
    theoretical_best_h = float(math.sqrt(4.0 * epsilon_tilde))

    return {
        "part_a": {
            "delta": part_a_delta,
            "approximation": part_a_approx,
            "exact_value": part_a_exact,
            "abs_error": abs(part_a_approx - part_a_exact),
        },
        "part_b_csv": str((RESULT_DIR / "problem4_partb_forward_difference.csv").relative_to(ROOT)),
        "part_b_plot": str((RESULT_DIR / "problem4_partb_error.png").relative_to(ROOT)),
        "part_b_rows": rows_b,
        "part_c_csv": str((RESULT_DIR / "problem4_partc_error_model.csv").relative_to(ROOT)),
        "part_c_plot": str((RESULT_DIR / "problem4_partc_error_model.png").relative_to(ROOT)),
        "part_c_empirical_best_h": empirical_best_h,
        "part_c_empirical_best_error": empirical_best_error,
        "part_c_theoretical_best_h": theoretical_best_h,
        "part_c_template_csv": str((RESULT_DIR / "problem4_partc_template_comparison.csv").relative_to(ROOT)),
        "part_c_template_best_csv": str((RESULT_DIR / "problem4_partc_template_best.csv").relative_to(ROOT)),
        "part_c_template_plot": str((RESULT_DIR / "problem4_partc_template_comparison.png").relative_to(ROOT)),
        "part_c_template_best_rows": template_best_rows,
        "epsilon_tilde": epsilon_tilde,
        "machine_epsilon": float(np.finfo(float).eps),
        "unit_roundoff": float(np.finfo(float).eps / 2.0),
    }


def dump_summary(summary: dict[str, object]) -> None:
    (RESULT_DIR / "hw06_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_problem2() -> dict[str, object]:
    problem2_formulas = [
        FormulaSpec("Forward O(h)", 1, (0, 1), 1, "forward"),
        FormulaSpec("Backward O(h)", 1, (-1, 0), 1, "backward"),
        FormulaSpec("Forward O(h^2)", 1, (0, 1, 2), 2, "forward"),
        FormulaSpec("Backward O(h^2)", 1, (-2, -1, 0), 2, "backward"),
        FormulaSpec("Central O(h^2)", 1, (-1, 0, 1), 2, "central"),
        FormulaSpec("Central O(h^4)", 1, (-2, -1, 0, 1, 2), 4, "central"),
    ]
    problem2 = analyze_derivative_problem(
        problem_label="Problem 2",
        func=lambda x: x * math.exp(x),
        exact_derivative=lambda x: (x + 1.0) * math.exp(x),
        x0=2.0,
        table_values={
            1.8: 10.889365,
            1.9: 12.703199,
            2.0: 14.778112,
            2.1: 17.148957,
            2.2: 19.855030,
        },
        formulas=problem2_formulas,
        exact_table_output="problem2_table_h0p1.csv",
        sweep_output="problem2_error_sweep.csv",
        plot_output="problem2_error_curves.png",
    )
    x = sp.symbols("x")
    total_error_model_rows = estimate_total_error_model(
        formulas=problem2_formulas,
        func_expr=x * sp.exp(x),
        x_symbol=x,
        x0=2.0,
        empirical_best_rows=problem2["best_per_formula"],  # type: ignore[arg-type]
        output_name="problem2_total_error_model.csv",
    )
    problem2["total_error_model_rows"] = total_error_model_rows
    problem2["total_error_model_csv"] = str(
        (RESULT_DIR / "problem2_total_error_model.csv").relative_to(ROOT)
    )
    problem2["unit_roundoff"] = float(np.finfo(float).eps / 2.0)
    return problem2


def run_problem3() -> dict[str, object]:
    problem3_formulas = [
        FormulaSpec("Forward O(h)", 2, (0, 1, 2), 1, "forward"),
        FormulaSpec("Backward O(h)", 2, (-2, -1, 0), 1, "backward"),
        FormulaSpec("Central O(h^2)", 2, (-1, 0, 1), 2, "central"),
        FormulaSpec("Central O(h^4)", 2, (-2, -1, 0, 1, 2), 4, "central"),
    ]
    problem3 = analyze_derivative_problem(
        problem_label="Problem 3",
        func=lambda x: x * math.exp(x * x),
        exact_derivative=lambda x: 2.0 * x * (3.0 + 2.0 * x * x) * math.exp(x * x),
        x0=2.0,
        table_values={
            1.8: 45.960699,
            1.9: 70.235500,
            2.0: 109.196300,
            2.1: 172.765873,
            2.2: 278.232574,
        },
        formulas=problem3_formulas,
        exact_table_output="problem3_table_h0p1.csv",
        sweep_output="problem3_error_sweep.csv",
        plot_output="problem3_error_curves.png",
    )
    return problem3


def run_problem4() -> dict[str, object]:
    return analyze_problem4()


def main() -> None:
    ensure_result_dir()

    print("Running Homework 7 analysis in", ROOT)

    problem2 = run_problem2()
    print("Problem 2 best table formula:", problem2["table_best_formula"])
    print("Problem 2 best dense-scan formula:", problem2["overall_best_formula"])

    problem3 = run_problem3()
    print("Problem 3 best table formula:", problem3["table_best_formula"])
    print("Problem 3 best dense-scan formula:", problem3["overall_best_formula"])

    problem4 = run_problem4()
    print("Problem 4 empirical best h:", problem4["part_c_empirical_best_h"])

    summary = {
        "problem2": problem2,
        "problem3": problem3,
        "problem4": problem4,
    }
    dump_summary(summary)
    print("Summary written to", (RESULT_DIR / "hw06_summary.json").relative_to(ROOT))
