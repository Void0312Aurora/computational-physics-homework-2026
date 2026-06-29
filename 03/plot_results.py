from __future__ import annotations

import csv
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path
from statistics import median
from textwrap import dedent, fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

getcontext().prec = 80

ROOT = Path(__file__).resolve().parent
RESULT_DIR = Path(os.environ.get("HW03_RESULT_DIR", ROOT / "result")).resolve()
ANSWER_DIR = Path(os.environ.get("HW03_ANSWER_DIR", ROOT / "docs" / "answer")).resolve()
ASSET_DIR = ANSWER_DIR / "assets"
SHARED_REF_DIR = ROOT.parent / "docs" / "ref"
PROFILE_SOURCE = SHARED_REF_DIR / "pic.jpg"
SKIP_ANSWER = os.environ.get("HW03_SKIP_ANSWER", "").lower() in {"1", "true", "yes"}

PRECISION_ORDER = ["float", "double", "quad"]
PRECISION_TITLES = {"float": "Single Precision", "double": "Double Precision", "quad": "Quad Precision"}
METHOD_ORDER = ["direct", "expanded", "horner"]
METHOD_LABELS = {"direct": "Direct", "expanded": "Expanded", "horner": "Horner"}
METHOD_COLORS = {"direct": "#24577a", "expanded": "#c75b39", "horner": "#5b8f45"}
PROBLEM1_COLORS = {"standard": "#c75b39", "rationalized": "#2c7a7b"}


@dataclass
class Problem1Summary:
    b: str
    precision: str
    standard_error: Decimal
    rationalized_error: Decimal
    improvement: Decimal


@dataclass
class Problem2Stats:
    precision: str
    method: str
    max_rel_error: Decimal
    x_at_max: Decimal
    median_rel_error: Decimal


def decimal_sqrt(value: Decimal) -> Decimal:
    return value.sqrt()


def format_decimal(value: Decimal, digits: int = 6) -> str:
    return f"{value:.{digits}E}"


def format_compact_float(value: Decimal, digits: int = 8) -> str:
    return f"{float(value):.{digits}g}"


def ensure_answer_assets() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    if PROFILE_SOURCE.exists():
        shutil.copyfile(PROFILE_SOURCE, ASSET_DIR / "profile.jpg")


def draw_flow_diagram(output: Path, title: str, steps: list[str], accent: str) -> None:
    figure_height = max(4.4, 1.18 * len(steps) + 0.9)
    fig, ax = plt.subplots(figsize=(8.4, figure_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(steps))
    ax.axis("off")

    y_positions = [len(steps) - idx - 0.5 for idx in range(len(steps))]
    for idx, (y, step) in enumerate(zip(y_positions, steps)):
        box = FancyBboxPatch(
            (0.10, y - 0.24),
            0.80,
            0.42,
            boxstyle="round,pad=0.018,rounding_size=0.03",
            linewidth=1.6,
            edgecolor=accent,
            facecolor="#f8fbff",
        )
        ax.add_patch(box)
        ax.text(
            0.5,
            y - 0.03,
            fill(step, width=26),
            ha="center",
            va="center",
            fontsize=10.0,
            color="#1b1b1b",
        )
        if idx < len(steps) - 1:
            next_y = y_positions[idx + 1]
            ax.annotate(
                "",
                xy=(0.5, next_y + 0.21),
                xytext=(0.5, y - 0.27),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": accent},
            )

    fig.suptitle(title, fontsize=14.0, fontweight="bold", color="#1f1f1f", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=220)
    plt.close(fig)


def create_hw_report_assets() -> None:
    ensure_answer_assets()
    draw_flow_diagram(
        ASSET_DIR / "problem1_workflow.png",
        "Problem 1 Workflow",
        [
            "Select large-b cases and test precisions.",
            "Compute the small root with two formulas.",
            "Use a high-precision reference root.",
            "Compare relative errors and growth trends.",
        ],
        accent="#c75b39",
    )
    draw_flow_diagram(
        ASSET_DIR / "problem2_workflow.png",
        "Problem 2 Workflow",
        [
            "Sample x on [0.7, 1.3].",
            "Evaluate direct, expanded, and Horner forms.",
            "Build high-precision reference values.",
            "Locate error peaks near x = 1.",
        ],
        accent="#24577a",
    )
    draw_flow_diagram(
        ASSET_DIR / "problem3_workflow.png",
        "Problem 3 Workflow",
        [
            "Probe binary16 limits on _Float16.",
            "Evaluate ((1+t)^2 - 1) / t in half precision.",
            "Compare with the rearranged form 2 + t.",
            "Explain the induced roundoff amplification.",
        ],
        accent="#5b8f45",
    )


def load_problem1_rows() -> list[dict[str, str]]:
    with (RESULT_DIR / "problem1_roots.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_problem2_rows() -> list[dict[str, str]]:
    with (RESULT_DIR / "problem2_values.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_problem3_metrics() -> list[dict[str, str]]:
    with (RESULT_DIR / "problem3_half_metrics.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_problem3_roundoff() -> list[dict[str, str]]:
    with (RESULT_DIR / "problem3_roundoff.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def analyze_problem1(rows: list[dict[str, str]]) -> list[Problem1Summary]:
    grouped: dict[tuple[str, str], dict[str, Decimal]] = defaultdict(dict)

    for row in rows:
        b = Decimal(row["b"])
        reference = Decimal(2) / (b + decimal_sqrt(b * b - Decimal(4)))
        approx = Decimal(row["root"])
        rel_error = abs(approx - reference) / abs(reference)
        grouped[(row["b"], row["precision"])][row["method"]] = rel_error

    summaries: list[Problem1Summary] = []
    labels_by_value = {Decimal(row["b"]): row["b"] for row in rows}
    for precision in PRECISION_ORDER:
        b_values = sorted({Decimal(item["b"]) for item in rows}, key=lambda value: value)
        for b_value in b_values:
            matched = labels_by_value[b_value]
            methods = grouped[(matched, precision)]
            standard_error = methods["standard"]
            rationalized_error = methods["rationalized"]
            floor = Decimal("1e-70")
            summaries.append(
                Problem1Summary(
                    b=matched,
                    precision=precision,
                    standard_error=standard_error,
                    rationalized_error=rationalized_error,
                    improvement=standard_error / max(rationalized_error, floor),
                )
            )

    with (RESULT_DIR / "problem1_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["b", "precision", "standard_rel_error", "rationalized_rel_error", "improvement_factor"])
        for item in summaries:
            writer.writerow(
                [
                    item.b,
                    item.precision,
                    format_decimal(item.standard_error, 8),
                    format_decimal(item.rationalized_error, 8),
                    format_decimal(item.improvement, 8),
                ]
            )

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for axis, precision in zip(axes, PRECISION_ORDER):
        xs = [float(item.b) for item in summaries if item.precision == precision]
        standard = [float(item.standard_error) for item in summaries if item.precision == precision]
        rationalized = [float(item.rationalized_error) for item in summaries if item.precision == precision]
        axis.loglog(xs, standard, marker="o", color=PROBLEM1_COLORS["standard"], label="Standard")
        axis.loglog(xs, rationalized, marker="s", color=PROBLEM1_COLORS["rationalized"], label="Rationalized")
        axis.set_title(PRECISION_TITLES[precision])
        axis.set_ylabel("Relative error")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("b")
    fig.suptitle("Problem 1: Relative Error of the Small Root", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(RESULT_DIR / "problem1_relative_error.png", dpi=220)
    plt.close(fig)

    return summaries


def analyze_problem2(rows: list[dict[str, str]]) -> list[Problem2Stats]:
    by_precision_method: dict[tuple[str, str], list[tuple[Decimal, Decimal]]] = defaultdict(list)
    references_by_precision: dict[str, dict[Decimal, Decimal]] = defaultdict(dict)
    plot_cache: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    error_cache: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    stats: list[Problem2Stats] = []

    for row in rows:
        x = Decimal(row["x"])
        precision = row["precision"]
        if x not in references_by_precision[precision]:
            references_by_precision[precision][x] = (x - Decimal(1)) ** 10
        value = Decimal(row["value"])
        by_precision_method[(precision, row["method"])].append((x, value))
        plot_cache[precision][row["method"]].append((float(x), float(value)))

    for precision in PRECISION_ORDER:
        for method in METHOD_ORDER:
            pairs = by_precision_method[(precision, method)]
            rel_errors: list[tuple[Decimal, Decimal]] = []
            for x, value in pairs:
                reference = references_by_precision[precision][x]
                if reference == 0:
                    continue
                rel_error = abs(value - reference) / abs(reference)
                rel_errors.append((x, rel_error))
                error_cache[precision][method].append((float(x), max(float(rel_error), 1e-40)))

            max_x, max_err = max(rel_errors, key=lambda item: item[1])
            stats.append(
                Problem2Stats(
                    precision=precision,
                    method=method,
                    max_rel_error=max_err,
                    x_at_max=max_x,
                    median_rel_error=median([item[1] for item in rel_errors]),
                )
            )

    with (RESULT_DIR / "problem2_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["precision", "method", "max_rel_error", "x_at_max", "median_rel_error"])
        for item in stats:
            writer.writerow(
                [
                    item.precision,
                    item.method,
                    format_decimal(item.max_rel_error, 8),
                    f"{item.x_at_max}",
                    format_decimal(item.median_rel_error, 8),
                ]
            )

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    for axis, precision in zip(axes, PRECISION_ORDER):
        sorted_x = sorted(references_by_precision[precision])
        x_ref = [float(x) for x in sorted_x]
        y_ref = [float(references_by_precision[precision][x]) for x in sorted_x]
        axis.plot(x_ref, y_ref, color="#111111", linewidth=2.2, label="Reference")
        for method in METHOD_ORDER:
            x_values = [item[0] for item in plot_cache[precision][method]]
            y_values = [item[1] for item in plot_cache[precision][method]]
            axis.plot(x_values, y_values, linewidth=1.6, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axis.set_ylabel("Value")
        axis.set_title(PRECISION_TITLES[precision])
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=4, fontsize=9)
    axes[-1].set_xlabel("x")
    fig.suptitle("Problem 2: Zoomed Values on x in [0.7, 1.3]", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(RESULT_DIR / "problem2_zoom.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 9.4), sharex=True, sharey=True)
    for axis, method in zip(axes, ["expanded", "horner"]):
        for precision in PRECISION_ORDER:
            x_values = [item[0] for item in error_cache[precision][method]]
            y_values = [item[1] for item in error_cache[precision][method]]
            axis.semilogy(
                x_values,
                y_values,
                linestyle="None",
                marker=".",
                markersize=2.8,
                alpha=0.72,
                label=PRECISION_TITLES[precision],
            )
        axis.axvline(1.0, color="#111111", linestyle="--", linewidth=0.9, alpha=0.55)
        axis.set_title(f"{METHOD_LABELS[method]} Relative Error (sample points)")
        axis.set_ylabel("Relative error")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("x")
    fig.suptitle("Problem 2: Relative-Error Distribution", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(RESULT_DIR / "problem2_relative_error.png", dpi=220)
    plt.close(fig)

    return stats


def analyze_problem3(
    metrics_rows: list[dict[str, str]], roundoff_rows: list[dict[str, str]]
) -> tuple[dict[str, str], list[dict[str, Decimal]]]:
    metrics = {row["metric"]: row["value"] for row in metrics_rows}
    analyzed_rows: list[dict[str, Decimal]] = []

    for row in roundoff_rows:
        t = Decimal(row["t"])
        exact = Decimal(2) + t
        direct = Decimal(row["direct"])
        rearranged = Decimal(row["rearranged"])
        analyzed_rows.append(
            {
                "t": t,
                "exact": exact,
                "direct": direct,
                "rearranged": rearranged,
                "direct_rel_error": abs(direct - exact) / abs(exact),
                "rearranged_rel_error": abs(rearranged - exact) / abs(exact),
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x_values = [float(item["t"]) for item in analyzed_rows]
    axes[0].plot(x_values, [float(item["exact"]) for item in analyzed_rows], color="#111111", linewidth=2.2, label="Exact 2+t")
    axes[0].plot(x_values, [float(item["direct"]) for item in analyzed_rows], color="#c75b39", marker="o", label="Direct")
    axes[0].plot(
        x_values,
        [float(item["rearranged"]) for item in analyzed_rows],
        color="#2c7a7b",
        marker="s",
        label="Rearranged",
    )
    axes[0].set_xscale("log")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("Computed value")
    axes[0].set_title("Half-Precision Computed Value")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].loglog(
        x_values,
        [max(float(item["direct_rel_error"]), 1e-16) for item in analyzed_rows],
        color="#c75b39",
        marker="o",
        label="Direct error",
    )
    axes[1].loglog(
        x_values,
        [max(float(item["rearranged_rel_error"]), 1e-16) for item in analyzed_rows],
        color="#2c7a7b",
        marker="s",
        label="Rearranged error",
    )
    axes[1].invert_xaxis()
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("Relative error")
    axes[1].set_title("Half-Precision Relative Error")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.suptitle("Problem 3: Roundoff Error in Half Precision", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RESULT_DIR / "problem3_roundoff.png", dpi=220)
    plt.close(fig)

    return metrics, analyzed_rows


def build_answer(
    problem1: list[Problem1Summary], problem2: list[Problem2Stats], metrics: dict[str, str], roundoff: list[dict[str, Decimal]]
) -> None:
    ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    answer_path = ANSWER_DIR / "answer.md"
    if answer_path.exists():
        existing = answer_path.read_text(encoding="utf-8")
        if "\\tableofcontents" in existing and "## Problem 1(1)" in existing:
            return

    p1_lookup = {(item.precision, item.b): item for item in problem1}
    p2_lookup = {(item.precision, item.method): item for item in problem2}
    first_zero_case = next((item for item in roundoff if item["direct"] == 0), roundoff[0])
    slide_rel_error = abs(Decimal("0.01050") - Decimal("0.01031")) / Decimal("0.01031")
    float_100 = p1_lookup[("float", "100")]
    float_10000 = p1_lookup[("float", "10000")]
    double_1e10 = p1_lookup[("double", "10000000000")]
    quad_1e10 = p1_lookup[("quad", "10000000000")]
    p2_float_direct = p2_lookup[("float", "direct")]
    p2_float_expanded = p2_lookup[("float", "expanded")]
    p2_float_horner = p2_lookup[("float", "horner")]
    p2_quad_expanded = p2_lookup[("quad", "expanded")]

    problem1_rows = [
        ("float", "100", p1_lookup[("float", "100")]),
        ("float", "1000", p1_lookup[("float", "1000")]),
        ("float", "10000", p1_lookup[("float", "10000")]),
        ("double", "100000", p1_lookup[("double", "100000")]),
        ("double", "10000000000", p1_lookup[("double", "10000000000")]),
        ("quad", "10000000000", p1_lookup[("quad", "10000000000")]),
    ]
    problem1_table = "\n".join(
        (
            f"| {precision} | {b_value} | {format_decimal(item.standard_error, 4)} | "
            f"{format_decimal(item.rationalized_error, 4)} | {format_decimal(item.improvement, 4)} |"
        )
        for precision, b_value, item in problem1_rows
    )
    problem1_standard_table = "\n".join(
        f"| {precision} | {b_value} | {format_decimal(item.standard_error, 4)} |"
        for precision, b_value, item in problem1_rows
    )

    problem2_table = "\n".join(
        (
            f"| {precision} | {method} | {format_decimal(item.max_rel_error, 4)} | "
            f"{format_compact_float(item.x_at_max)} | {format_decimal(item.median_rel_error, 4)} |"
        )
        for precision in PRECISION_ORDER
        for method in METHOD_ORDER
        for item in [p2_lookup[(precision, method)]]
    )
    problem2_horner_table = "\n".join(
        (
            f"| {precision} | {format_decimal(p2_lookup[(precision, 'direct')].max_rel_error, 4)} | "
            f"{format_decimal(p2_lookup[(precision, 'expanded')].max_rel_error, 4)} | "
            f"{format_decimal(p2_lookup[(precision, 'horner')].max_rel_error, 4)} | "
            f"{format_decimal(p2_lookup[(precision, 'expanded')].max_rel_error / p2_lookup[(precision, 'horner')].max_rel_error, 4)} | "
            f"{format_decimal(p2_lookup[(precision, 'expanded')].median_rel_error / p2_lookup[(precision, 'horner')].median_rel_error, 4)} |"
        )
        for precision in PRECISION_ORDER
    )

    answer = dedent(
        f"""\
        ---
        title: "Homework 4"
        subtitle: "数值稳定性分析报告"
        author: "姜玥晟"
        date: "2026-04-21"
        geometry: margin=1in
        fontsize: 11pt
        header-includes:
          - \\renewcommand{{\\figurename}}{{图}}
          - \\renewcommand{{\\tablename}}{{表}}
        ---

        | ![Portrait](assets/profile.jpg){{ width=20% }} |
        |:--:|

        | 项目 | 内容 |
        |:--|:--|
        | 作业目录 | `HW/03` |
        | 报告主题 | 二次方程求根稳定性、多项式求值稳定性与 half precision 舍入误差 |
        | 实验环境 | `C` 数值程序与 `Python` 统计绘图程序 |
        | 文档内容 | 正文给出题意、实验设计、结果分析与扩展讨论；完整实现保留于 `scripts/` |

        \\newpage

        # 第一题 二次方程小根的相消误差分析

        ## 题目陈述

        考察二次方程

        $$
        x^2 - b x + 1 = 0,
        $$

        其两根可写为

        $$
        x_1 = \\frac{{b+r}}{{2}}, \\qquad
        x_2 = \\frac{{b-r}}{{2}}, \\qquad
        r = \\sqrt{{b^2 - 4}}.
        $$

        当 $b$ 足够大时，$r$ 与 $b$ 的数值非常接近，小根

        $$
        x_2 = \\frac{{b-r}}{{2}}
        $$

        的计算将涉及两个接近量之间的相减。

        ## Problem 1(1)：$b=100$ 附近的误差考察

        相关脚本：

        - [本地 scripts/problem1.c](../../scripts/problem1.c)
        - [GitHub scripts/problem1.c](https://github.com/Void0312Aurora/computational-physics-homework-2026/blob/main/03/scripts/problem1.c)

        ### 待求问题

        首先考察该表达式在 `b=100` 附近的相对误差。

        ### 解决方式

        为使实验步骤表达得更规范，本文以伪代码方式给出求解过程：

        ```text
        Input : precisions P, b = 100
        Output: standard-form error at b = 100

        for each p in P do
            r <- sqrt(b^2 - 4)
            x_std <- (b - r) / 2
            x_ref <- ref_root(b)
            err_std <- relerr(x_std, x_ref)
        end for
        ```

        其中 `ref_root(b)` 表示高精度参考解，`relerr` 表示相对误差计算函数。

        ### 问题答案

        对于题目指定的 `b=100`，本机 IEEE `float` 运算给出

        $$
        x_2^{{\\text{{standard}}}} = 0.0100021362,
        $$

        高精度参考值约为 $0.010001000200050014$。标准公式相对误差为 {format_decimal(float_100.standard_error, 4)}。

        题图第二页另给出 `b=97` 的三位有效数字教学示例：

        $$
        x_2^{{\\text{{exact}}}} = 0.01031, \\qquad
        x_2^{{\\text{{standard}}}} = 0.01050,
        $$

        该示例的相对误差为 {format_decimal(slide_rel_error, 4)}，约为 `1.84%`，用于说明低精度三位有效数字运算中的相消误差；它不是 `b=100` 的主计算结果。

        ### 分析

        标准公式的不稳定性来源于相消误差。由近似关系

        $$
        \\sqrt{{b^2-4}} = b\\sqrt{{1-4/b^2}} \\approx b - \\frac{{2}}{{b}}
        $$

        可知，当 $b$ 很大时，表达式

        $$
        b - \\sqrt{{b^2 - 4}}
        $$

        仅保留数量级约为 $2/b$ 的小量，高位有效数字会在减法阶段被提前消耗。因此，本问只能说明标准公式已经出现相消误差；如何通过代数改写避开该相减过程，留到 Problem 1(3) 再处理。

        ## Problem 1(2)：大参数区间的误差比较

        相关脚本：

        - [本地 scripts/problem1.c](../../scripts/problem1.c)
        - [GitHub scripts/problem1.c](https://github.com/Void0312Aurora/computational-physics-homework-2026/blob/main/03/scripts/problem1.c)

        ### 待求问题

        随后增大 $b$ 取值并列表比较误差变化。

        ### 解决方式

        为使实验步骤表达得更规范，本文以伪代码方式给出求解过程：

        ```text
        Input : precisions P, test values B
        Output: standard-form error table

        for each p in P do
            for each b in B do
                r <- sqrt(b^2 - 4)
                x_std <- (b - r) / 2
                x_ref <- ref_root(b)
                err_std <- relerr(x_std, x_ref)
            end for
        end for
        ```

        其中 `ref_root(b)` 表示高精度参考解，`relerr` 表示相对误差计算函数。

        ### 问题答案

        继续增大 $b$ 并仍然只使用标准公式 $x_2=(b-r)/2$ 时，代表性相对误差如下。

        | 精度 | b | 标准公式相对误差 |
        |:--|--:|--:|
        {problem1_standard_table}

        对于 `float`，当 `b=10000` 时，标准公式的相对误差已达到 {format_decimal(float_10000.standard_error, 4)}，说明小根信息几乎完全丢失。对于 `double`，在 `b=10^10` 时也出现同类退化，而 `__float128` 在同一测试点仍维持 {format_decimal(quad_1e10.standard_error, 4)} 的误差量级。

        ### 分析

        标准公式的不稳定性来源于相消误差。由近似关系

        $$
        \\sqrt{{b^2-4}} = b\\sqrt{{1-4/b^2}} \\approx b - \\frac{{2}}{{b}}
        $$

        可知，当 $b$ 很大时，表达式

        $$
        b - \\sqrt{{b^2 - 4}}
        $$

        仅保留数量级约为 $2/b$ 的小量，高位有效数字会在减法阶段被提前消耗。表中的退化正是这一相消过程随 $b$ 增大而加剧的结果。下一问再对这个表达式进行代数改写，并用同一组 $b$ 值复现实验。

        ## Problem 1(3)：有理化改写与再实验

        相关脚本：

        - [本地 scripts/problem1.c](../../scripts/problem1.c)
        - [GitHub scripts/problem1.c](https://github.com/Void0312Aurora/computational-physics-homework-2026/blob/main/03/scripts/problem1.c)

        ### 待求问题

        最后将小根公式改写为

        $$
        x_2 = \\frac{{(b-r)(b+r)}}{{2(b+r)}} = \\frac{{2}}{{b+r}},
        $$

        并重新进行同样的实验，以比较两种实现的稳定性差异。

        ### 解决方式

        为使实验步骤表达得更规范，本文以伪代码方式给出求解过程：

        ```text
        Input : precisions P, test values B
        Output: error table and error curve

        for each p in P do
            for each b in B do
                r <- sqrt(b^2 - 4)
                x_std <- (b - r) / 2
                x_rat <- 2 / (b + r)
                x_ref <- ref_root(b)
                err_std <- relerr(x_std, x_ref)
                err_rat <- relerr(x_rat, x_ref)
            end for
        end for
        ```

        其中 `ref_root(b)` 表示高精度参考解，`relerr` 表示相对误差计算函数。

        ### 问题答案

        对同一组 $b$ 值重新实验后，代表性结果如下。

        ![Problem 1 relative-error comparison after rationalization](../../result/problem1_relative_error.png){{ width=88% }}

        | 精度 | b | 标准公式相对误差 | 有理化公式相对误差 | 改进倍数 |
        |:--|--:|--:|--:|--:|
        {problem1_table}

        这些数据说明，有理化后未再出现与标准公式同量级的误差爆炸。实验范围内，有理化公式始终显著优于标准公式；例如在 `double` 且 $b=10^{{10}}$ 条件下，标准公式误差为 {format_decimal(double_1e10.standard_error, 4)}，而有理化公式误差为 {format_decimal(double_1e10.rationalized_error, 4)}。

        ### 分析

        标准公式的不稳定性来源于相消误差。由近似关系

        $$
        \\sqrt{{b^2-4}} = b\\sqrt{{1-4/b^2}} \\approx b - \\frac{{2}}{{b}}
        $$

        可知，当 $b$ 很大时，表达式

        $$
        b - \\sqrt{{b^2 - 4}}
        $$

        仅保留数量级约为 $2/b$ 的小量，高位有效数字会在减法阶段被提前消耗。有理化公式

        $$
        x_2 = \\frac{{2}}{{b + \\sqrt{{b^2 - 4}}}}
        $$

        将这一相减过程替换为除法运算，因此显著抑制了误差放大。进一步地，由根的乘积关系 $x_1 x_2 = 1$ 还可得到另一种稳定实现，即先稳定计算较大的根 $x_1$，再由 $x_2 = 1/x_1$ 得到小根。

        # 第二题 多项式写法对数值稳定性的影响

        ## 题目陈述

        作业要求在 `single`、`double` 与 `quad` 三种精度下计算

        $$
        (x-1)^{{10}}
        $$

        及其多项式展开式，并在区间 $x\\in[0.7, 1.3]$ 上观察数值结果。除直接形式与朴素展开形式外，还需使用 Horner 方法重新组织多项式计算，比较三种实现的误差分布，并解释在 $x=1$ 邻域出现差异的原因。

        ## 解决方案

        本题的实验组织可形式化表示为：

        ```text
        Input : grid X, methods M, precisions P
        Output: zoomed plots and error statistics

        for each p in P do
            for each x in X do
                y_ref <- high_precision((x - 1)^10)
                for each m in M do
                    y <- eval(m, x, p)
                    if y_ref != 0 then
                        err[m, x, p] <- relerr(y, y_ref)
                    end if
                end for
            end for
        end for
        ```

        其中 `M = {{direct, expanded, horner}}`，并基于 `err` 进一步统计最大误差、峰值位置与中位误差。

        ## 问题答案

        1. 局部放大结果表明，直接形式在三个精度下均最接近参考曲线，而展开形式与 Horner 形式在 $x=1$ 附近出现明显偏离。

        ![Problem 2 zoomed-value comparison](../../result/problem2_zoom.png){{ width=88% }}

        造成该现象的直接原因在于：当 $x$ 接近 `1` 时，真值 $(x-1)^{{10}}$ 极小；若先将多项式展开为多个数量级接近的项再相加减，就会显著放大舍入误差。

        2. 相对误差统计结果如下。图中使用离散采样点而非折线连接；靠近 $x=1$ 的针状峰来自相对误差分母 $|(x-1)^{{10}}|$ 极小，少量绝对舍入误差被放大。相邻采样点误差可差多个数量级，若用折线连接会产生视觉上的竖直突变线，因此这里展示采样分布而不把它解释成连续曲线突变。

        ![Problem 2 relative-error distribution](../../result/problem2_relative_error.png){{ width=96% }}

        | 精度 | 形式 | 最大相对误差 | 峰值位置 x | 中位相对误差 |
        |:--|:--|--:|--:|--:|
        {problem2_table}

        其中最显著的退化出现在 `float + expanded` 组合，其最大相对误差达到 {format_decimal(p2_float_expanded.max_rel_error, 4)}，峰值出现在 `x={format_compact_float(p2_float_expanded.x_at_max)}` 附近；相比之下，同为 `float` 的直接形式最大相对误差仅为 {format_decimal(p2_float_direct.max_rel_error, 4)}。即使在 `quad` 精度下，展开形式的最大相对误差仍达到 {format_decimal(p2_quad_expanded.max_rel_error, 4)}，说明表达式结构对稳定性的影响并不会因精度提升而完全消失。

        3. Horner 方法的独立对比结果如下。表中“最大误差改善倍数”和“中位误差改善倍数”均定义为 `expanded` 误差除以 `horner` 误差；数值大于 `1` 表示 Horner 优于朴素展开。

        | 精度 | direct 最大相对误差 | expanded 最大相对误差 | horner 最大相对误差 | 最大误差改善倍数 | 中位误差改善倍数 |
        |:--|--:|--:|--:|--:|--:|
        {problem2_horner_table}

        结果表明，Horner 方法在三种精度下都比朴素展开稳定；以 `float` 为例，最大相对误差从 {format_decimal(p2_float_expanded.max_rel_error, 4)} 降到 {format_decimal(p2_float_horner.max_rel_error, 4)}，改善约 {format_decimal(p2_float_expanded.max_rel_error / p2_float_horner.max_rel_error, 4)} 倍。但它仍显著劣于直接形式：同一精度下 `direct` 的最大相对误差只有 {format_decimal(p2_float_direct.max_rel_error, 4)}。结论是 Horner 改善了展开式的运算顺序，但没有恢复直接形式保留的关键小量结构 $(x-1)$。

        ## 讨论和扩展

        本题显示出“问题条件数”与“算法稳定性”之间的区别。在 $x=1$ 附近，真值本身非常接近零，因此任何微小的绝对误差都可能被转换为极大的相对误差。直接形式由于始终保留 $(x-1)$ 这一小量结构，运算路径最短，因而最稳定；朴素展开形式则会反复累积和抵消数量级接近的项，最容易触发相消误差；Horner 形式虽然改善了运算顺序，但并未改变展开后计算的本质。若需进一步提高稳定性，可结合高精度算术、补偿求和或围绕 $x=1$ 的局部重参数化方案。

        # 第三题 Half Precision 的机器精度与舍入误差

        ## 题目陈述

        作业要求给出 half precision（binary16）的 machine precision、数值范围等基本指标，并构造一个具体算例，说明在半精度环境下 roundoff error 可以严重到何种程度。

        ## 解决方案

        本题采用的实验步骤可归纳为如下伪代码：

        ```text
        Input : half type h, parameter set T
        Output: metric table and roundoff plot

        probe epsilon(h), min_normal(h)
        probe true_min(h), max_finite(h)

        for each t in T do
            y_dir <- ((1 + t)^2 - 1) / t
            y_alt <- 2 + t
            y_ref <- high_precision(2 + t)
            if y_ref != 0 then
                err_dir <- relerr(y_dir, y_ref)
                err_alt <- relerr(y_alt, y_ref)
            end if
        end for
        ```

        通过比较 `y_dir` 与 `y_alt` 的误差变化，可直接观察 half precision 下的舍入误差放大现象。

        ## 问题答案

        1. `_Float16` 的主要数值指标如表所示。

        | 指标 | 数值 |
        |:--|--:|
        | `_Float16` size | {metrics['sizeof_half']} bytes |
        | bit pattern of `1.0` | `{metrics['bits_of_one']}` |
        | machine epsilon | {metrics['machine_epsilon']} |
        | min normal | {metrics['min_normal']} |
        | true min | {metrics['true_min']} |
        | max finite | {metrics['max_finite']} |

        上述结果与 binary16 的 `1` 位符号位、`5` 位指数位和 `10` 位尾数位结构一致，说明实验环境中的 half precision 实现符合常见 IEEE 754 binary16 格式。

        2. 为说明舍入误差的严重程度，选取表达式

        $$
        \\frac{{(1+t)^2 - 1}}{{t}},
        $$

        其精确值为 $2+t$。当 `t={format_compact_float(first_zero_case['t'], 12)}` 时，半精度直接计算结果已退化为 `{first_zero_case['direct']}`，而精确值仍为 `{first_zero_case['exact']}`；此时相对误差达到 {format_decimal(first_zero_case['direct_rel_error'], 4)}。若改写为 `2+t`，同一点的相对误差仅为 {format_decimal(first_zero_case['rearranged_rel_error'], 4)}。

        ![Problem 3 roundoff comparison in half precision](../../result/problem3_roundoff.png){{ width=88% }}

        3. 因此，half precision 的 machine epsilon 约为 `2^(-10)`，有效数字极为有限；一旦表达式包含相消运算，舍入误差完全可能将本应处于 `2` 量级的结果直接压缩为 `0`。

        ## 讨论和扩展

        half precision 在 `1` 附近的间隔约为 `2^(-10)`。当 $t$ 小于或接近这一阈值时，`1+t` 将首先被舍入回 `1`，进而使 `(1+t)^2-1` 被舍入为 `0`。本题中的首个失真点正好出现在 `t={format_compact_float(first_zero_case['t'], 12)}`，即 machine epsilon 的一半附近，这与 binary16 的舍入行为相一致。由此可见，在低精度格式中，算法结构往往比“理论等价”更重要；对含有相消项的表达式进行代数改写，是避免灾难性误差的关键手段。
        """
    ).replace("\n        ", "\n").lstrip()

    answer_path.write_text(answer, encoding="utf-8")


def main() -> None:
    problem1_rows = load_problem1_rows()
    problem2_rows = load_problem2_rows()
    problem3_metrics = load_problem3_metrics()
    problem3_roundoff = load_problem3_roundoff()

    problem1_summary = analyze_problem1(problem1_rows)
    problem2_summary = analyze_problem2(problem2_rows)
    metrics, roundoff = analyze_problem3(problem3_metrics, problem3_roundoff)
    if not SKIP_ANSWER:
        create_hw_report_assets()
        build_answer(problem1_summary, problem2_summary, metrics, roundoff)


if __name__ == "__main__":
    main()
