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


def term_text(offset: int) -> str:
    if offset == 0:
        return "f(x)"
    if offset == 1:
        return "f(x+h)"
    if offset == -1:
        return "f(x-h)"
    if offset > 0:
        return f"f(x+{offset}h)"
    return f"f(x{offset}h)"


def rational_latex(value: sp.Rational) -> str:
    return sp.latex(sp.simplify(value))


def format_linear_combination(offsets: tuple[int, ...], weights: tuple[sp.Rational, ...]) -> str:
    pieces: list[str] = []
    for weight, offset in zip(weights, offsets):
        if sp.simplify(weight) == 0:
            continue
        sign = "-" if weight < 0 else "+"
        mag = sp.simplify(abs(weight))
        term = term_text(offset)
        if mag == 1:
            body = term
        else:
            body = rf"{rational_latex(mag)}\,{term}"
        pieces.append((sign, body))

    if not pieces:
        return "0"

    first_sign, first_body = pieces[0]
    text = first_body if first_sign == "+" else f"-{first_body}"
    for sign, body in pieces[1:]:
        text += f" {sign} {body}"
    return text


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_symbolic_linear_combination(
    coefficients: tuple[sp.Expr, ...],
    symbols: tuple[str, ...],
) -> str:
    pieces: list[tuple[str, str]] = []
    for coefficient, symbol in zip(coefficients, symbols):
        coefficient = sp.simplify(coefficient)
        if coefficient == 0:
            continue
        sign = "-" if coefficient < 0 else "+"
        magnitude = sp.simplify(abs(coefficient))
        body = symbol if magnitude == 1 else rf"{sp.latex(magnitude)}\,{symbol}"
        pieces.append((sign, body))

    if not pieces:
        return "0"

    first_sign, first_body = pieces[0]
    text = first_body if first_sign == "+" else f"-{first_body}"
    for sign, body in pieces[1:]:
        text += f" {sign} {body}"
    return text


def moment_equations_latex(offsets: tuple[int, ...], derivative_order: int) -> list[str]:
    symbols = tuple(rf"c_{{{index}}}" for index in range(len(offsets)))
    equations: list[str] = []
    for power in range(len(offsets)):
        coefficients = tuple(sp.Integer(offset) ** power for offset in offsets)
        lhs = format_symbolic_linear_combination(coefficients, symbols)
        rhs = sp.factorial(derivative_order) if power == derivative_order else 0
        equations.append(rf"{lhs} = {sp.latex(rhs)}")
    return equations


def symbolic_template_latex(offsets: tuple[int, ...]) -> str:
    return " + ".join(
        rf"c_{{{index}}}\,{term_text(offset)}"
        for index, offset in enumerate(offsets)
    )


def ordinal_text(value: int) -> str:
    mapping = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}
    return mapping.get(value, f"{value}th")


def leading_error_data(
    offsets: tuple[int, ...],
    weights: tuple[sp.Rational, ...],
    derivative_order: int,
) -> tuple[int, int, sp.Rational]:
    power = len(offsets)
    while True:
        moment = sp.simplify(
            sum(weight * (sp.Integer(offset) ** power) for weight, offset in zip(weights, offsets))
        )
        if moment != 0:
            coefficient = sp.simplify(moment / sp.factorial(power))
            return power - derivative_order, power, coefficient
        power += 1


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


def build_problem1_formula_report() -> dict[str, object]:
    cases = [
        (2, "forward", (0, 1, 2), 1),
        (2, "backward", (-2, -1, 0), 1),
        (2, "central", (-1, 0, 1), 2),
        (3, "forward", (0, 1, 2, 3), 1),
        (3, "backward", (-3, -2, -1, 0), 1),
        (3, "central", (-2, -1, 0, 1, 2), 2),
        (4, "forward", (0, 1, 2, 3, 4), 1),
        (4, "backward", (-4, -3, -2, -1, 0), 1),
        (4, "central", (-2, -1, 0, 1, 2), 2),
        (5, "forward", (0, 1, 2, 3, 4, 5), 1),
        (5, "backward", (-5, -4, -3, -2, -1, 0), 1),
        (5, "central", (-3, -2, -1, 0, 1, 2, 3), 2),
    ]

    records: list[dict[str, object]] = []
    lines = [
        "# Problem 1 Formula Derivation",
        "",
        "For stencil offsets $s_j$ and coefficients $c_j$, Taylor expansion gives",
        "",
        r"$$",
        r"\sum_j c_j f(x+s_j h) = \sum_{k=0}^{\infty} \frac{h^k}{k!}\left(\sum_j c_j s_j^k\right) f^{(k)}(x).",
        r"$$",
        "",
        "Choose the coefficients so that",
        "",
        r"$$",
        r"\sum_j c_j s_j^k = 0 \quad (0 \le k < m), \qquad \sum_j c_j s_j^m = m!,",
        r"$$",
        "",
        "then",
        "",
        r"$$",
        r"f^{(m)}(x) = \frac{1}{h^m}\sum_j c_j f(x+s_j h) + O(h^p),",
        r"$$",
        "",
        "where $p=1$ for the minimum forward/backward stencil and $p=2$ for the symmetric central stencil used here.",
        "",
        "The coefficients below are obtained by solving these moment equations directly rather than calling a black-box finite-difference template.",
        "",
    ]

    for derivative_order, family, offsets, truncation_order in cases:
        spec = FormulaSpec(
            name=f"{family}_d{derivative_order}",
            derivative_order=derivative_order,
            offsets=offsets,
            accuracy_order=truncation_order,
            family=family,
        )
        weights = spec.weights
        formula = format_linear_combination(offsets, weights)
        symbols = tuple(rf"c_{{{index}}}" for index in range(len(offsets)))
        equations = moment_equations_latex(offsets, derivative_order)
        error_order, next_power, error_coefficient = leading_error_data(
            offsets,
            weights,
            derivative_order,
        )
        latex = (
            rf"f^{{({derivative_order})}}(x)"
            rf" = \frac{{{formula}}}{{h^{{{derivative_order}}}}}"
            rf" + O(h^{{{truncation_order}}})"
        )
        lines.append(f"## {family.title()} difference for the {ordinal_text(derivative_order)} derivative")
        lines.append("")
        lines.append(f"Offsets: `{offsets}`. Assume")
        lines.append("")
        lines.append(r"$$")
        lines.append(rf"D(x) = \frac{{{symbolic_template_latex(offsets)}}}{{h^{{{derivative_order}}}}}")
        lines.append(r"$$")
        lines.append("")
        lines.append("The Taylor-moment system is")
        lines.append("")
        lines.append(r"$$")
        lines.append(r"\begin{cases}")
        for equation_index, equation in enumerate(equations):
            suffix = r"\\" if equation_index < len(equations) - 1 else ""
            lines.append(equation + suffix)
        lines.append(r"\end{cases}")
        lines.append(r"$$")
        lines.append("")
        lines.append("Solving gives")
        lines.append("")
        lines.append(r"$$")
        lines.append(
            r",\quad ".join(
                rf"{symbol} = {sp.latex(weight)}"
                for symbol, weight in zip(symbols, weights)
            )
        )
        lines.append(r"$$")
        lines.append("")
        lines.append("Hence")
        lines.append("")
        lines.append(r"$$")
        lines.append(latex)
        lines.append(r"$$")
        lines.append("")
        lines.append("The first retained truncation term is")
        lines.append("")
        lines.append(r"$$")
        lines.append(rf"{sp.latex(error_coefficient)}\,h^{{{error_order}}} f^{{({next_power})}}(x)")
        lines.append(r"$$")
        lines.append("")
        records.append(
            {
                "derivative_order": derivative_order,
                "family": family,
                "offsets": list(offsets),
                "weights": [str(weight) for weight in weights],
                "formula_latex": latex,
                "truncation_order": truncation_order,
                "moment_equations": equations,
                "leading_error_order": error_order,
                "leading_error_coefficient": str(error_coefficient),
            }
        )

    output_path = RESULT_DIR / "problem1_formulas.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    write_csv(
        RESULT_DIR / "problem1_coefficients.csv",
        [
            {
                "derivative_order": record["derivative_order"],
                "family": record["family"],
                "offsets": " ".join(map(str, record["offsets"])),
                "weights": " ".join(record["weights"]),
                "truncation_order": record["truncation_order"],
            }
            for record in records
        ],
        ["derivative_order", "family", "offsets", "weights", "truncation_order"],
    )
    return {
        "report": str(output_path.relative_to(ROOT)),
        "coefficient_table": str((RESULT_DIR / "problem1_coefficients.csv").relative_to(ROOT)),
        "cases": records,
    }


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

    problem1 = build_problem1_formula_report()
    print("Problem 1 formulas written to", problem1["report"])

    problem2 = run_problem2()
    print("Problem 2 best table formula:", problem2["table_best_formula"])
    print("Problem 2 best dense-scan formula:", problem2["overall_best_formula"])

    problem3 = run_problem3()
    print("Problem 3 best table formula:", problem3["table_best_formula"])
    print("Problem 3 best dense-scan formula:", problem3["overall_best_formula"])

    problem4 = run_problem4()
    print("Problem 4 empirical best h:", problem4["part_c_empirical_best_h"])

    summary = {
        "problem1": problem1,
        "problem2": problem2,
        "problem3": problem3,
        "problem4": problem4,
    }
    dump_summary(summary)
    print("Summary written to", (RESULT_DIR / "hw06_summary.json").relative_to(ROOT))
