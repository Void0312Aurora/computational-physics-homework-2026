from __future__ import annotations

import platform

from interpolation_common import InterpolationSummary
from problem1 import solve_problem1
from problem2 import solve_problem2
from problem3 import solve_problem3
from project2_pi.homework_bridge import solve_project2
from result_paths import RUN_RESULT_DIR, ensure_result_dir


def write_run_log(
    problem1: dict[str, object],
    problem2: list[InterpolationSummary],
    problem3: list[InterpolationSummary],
    project2: dict[str, object],
) -> None:
    lines: list[str] = []
    lines.append("HW/05 execution log")
    lines.append(f"platform = {platform.platform()}")
    lines.append(f"python = {platform.python_version()}")
    lines.append("")
    lines.append("Problem 1 curve summary:")
    for row in problem1["curve_summary"]:
        lines.append(
            "  {curve}: min={min_current:.6f}, max={max_current:.6f}, range={range_width:.6f}".format(
                **row
            )
        )
    lines.append("")
    lines.append("Problem 2 error summary:")
    for row in problem2:
        lines.append(
            f"  nodes={row.node_count:2d}, degree={row.polynomial_degree:2d}, "
            f"max_abs_error={row.max_abs_error:.6e}, rmse={row.rmse:.6e}, "
            f"|Neville-Bary|_max={row.neville_vs_barycentric_max_diff:.6e}, "
            f"Neville_s={row.neville_seconds:.6e}, Bary_s={row.barycentric_seconds:.6e}"
        )
    lines.append("")
    lines.append("Problem 3 error summary:")
    for row in problem3:
        lines.append(
            f"  nodes={row.node_count:2d}, degree={row.polynomial_degree:2d}, "
            f"max_abs_error={row.max_abs_error:.6e}, rmse={row.rmse:.6e}, "
            f"|Neville-Bary|_max={row.neville_vs_barycentric_max_diff:.6e}, "
            f"Neville_s={row.neville_seconds:.6e}, Bary_s={row.barycentric_seconds:.6e}"
        )
    lines.append("")
    lines.append("Project 2 pi benchmark:")
    project2_leaf_suffix = f", leaf_terms={project2['leaf_terms']}" if "leaf_terms" in project2 else ""
    project2_task_suffix = f", task_terms={project2['task_terms']}" if "task_terms" in project2 else ""
    project2_mode_suffix = f", parallel_mode={project2['parallel_mode']}" if "parallel_mode" in project2 else ""
    lines.append(
        "  backend={backend}, workers={workers}, chunk_terms={chunk_terms}{leaf_suffix}{task_suffix}{mode_suffix}, gpu_used={gpu_used}, "
        "highest_digits={highest_digits}, output_name={output_name}".format(
            **project2,
            leaf_suffix=project2_leaf_suffix,
            task_suffix=project2_task_suffix,
            mode_suffix=project2_mode_suffix,
        )
    )
    for row in project2["benchmark_rows"]:
        row_leaf_suffix = f", leaf_terms={row['leaf_terms']}" if "leaf_terms" in row else ""
        row_task_suffix = f", task_terms={row['task_terms']}" if "task_terms" in row else ""
        row_mode_suffix = f", parallel_mode={row['parallel_mode']}" if "parallel_mode" in row else ""
        lines.append(
            "  digits={digits}, terms={terms}, seconds={seconds:.6f}, digits_per_second={digits_per_second:.2f}, "
            "backend={backend}, workers={workers_used}, chunk_terms={chunk_terms}{leaf_suffix}{task_suffix}{mode_suffix}, gpu_used={gpu_used}, "
            "prefix_ok={prefix_matches_reference}".format(
                **row,
                leaf_suffix=row_leaf_suffix,
                task_suffix=row_task_suffix,
                mode_suffix=row_mode_suffix,
            )
        )
    ensure_result_dir(RUN_RESULT_DIR)
    (RUN_RESULT_DIR / "temp-01.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_result_dir()
    problem1 = solve_problem1()
    problem2 = solve_problem2()
    problem3 = solve_problem3()
    project2 = solve_project2()
    write_run_log(problem1, problem2, problem3, project2)


if __name__ == "__main__":
    main()
