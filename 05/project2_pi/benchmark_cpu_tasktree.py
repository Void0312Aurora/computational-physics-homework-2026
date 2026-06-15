from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from . import benchmark_matrix
from . import homework_bridge


RESULT_DIR = Path(__file__).resolve().parent.parent / "result"


@dataclass
class TaskTreeSummaryRow:
    route: str
    parallel_mode: str
    target_digits: int
    repeats: int
    avg_seconds: float
    best_seconds: float
    avg_digits_per_second: float
    best_digits_per_second: float
    threads: int
    chunk_terms: int
    leaf_terms: int
    task_terms: int
    prefix_ok: bool
    status: str
    seconds_samples: str
    digits_per_second_samples: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_digits_list(text: str) -> list[int]:
    values = [int(token) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("digits list must not be empty")
    return values


def parse_modes_list(text: str) -> list[str]:
    modes = [homework_bridge.normalize_project2_cpp_parallel_mode(token) for token in text.split(",") if token.strip()]
    if not modes:
        raise ValueError("modes list must not be empty")
    return modes


def write_csv(path: Path, rows: list[TaskTreeSummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].as_dict().keys()))
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)


def write_markdown(path: Path, rows: list[TaskTreeSummaryRow]) -> None:
    lines = [
        "# Project 2 CPU Parallel Modes Summary",
        "",
        "| route | mode | digits | repeats | avg_seconds | best_seconds | avg_digits_per_second | best_digits_per_second | threads | chunk_terms | leaf_terms | task_terms | prefix_ok | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {route} | {parallel_mode} | {target_digits} | {repeats} | {avg_seconds:.6f} | {best_seconds:.6f} | "
            "{avg_digits_per_second:.2f} | {best_digits_per_second:.2f} | {threads} | {chunk_terms} | "
            "{leaf_terms} | {task_terms} | {prefix_ok} | {status} |".format(**row.as_dict())
        )
        lines.append(
            f"  samples_seconds={row.seconds_samples}; samples_digits_per_second={row.digits_per_second_samples}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_mode(
    digits: int,
    parallel_mode: str,
    repeats: int,
    *,
    threads: int | None,
    chunk_terms: int | None,
    leaf_terms: int | None,
    task_terms: int | None,
) -> TaskTreeSummaryRow:
    base_config = homework_bridge.resolve_project2_cpp_config(digits)
    actual_threads = threads if threads is not None else base_config.threads
    actual_chunk_terms = chunk_terms if chunk_terms is not None else base_config.chunk_terms
    actual_leaf_terms = leaf_terms if leaf_terms is not None else base_config.leaf_terms
    actual_task_terms = task_terms if task_terms is not None else base_config.task_terms

    runs = [
        benchmark_matrix.benchmark_cpp_gmp(
            digits,
            threads=actual_threads,
            chunk_terms=actual_chunk_terms,
            leaf_terms=actual_leaf_terms,
            task_terms=actual_task_terms,
            parallel_mode=parallel_mode,
        )
        for _ in range(repeats)
    ]

    seconds_samples = [run.seconds for run in runs]
    digits_per_second_samples = [run.digits_per_second for run in runs]
    route = runs[0].route if runs else {
        "chunked": "cpp_gmp_openmp",
        "tasks": "cpp_gmp_openmp_tasks",
        "frontier": "cpp_gmp_openmp_frontier",
    }[parallel_mode]
    prefix_ok = all(run.prefix_ok for run in runs)
    status = "ok" if all(run.status == "ok" for run in runs) else "failed"
    return TaskTreeSummaryRow(
        route=route,
        parallel_mode=parallel_mode,
        target_digits=digits,
        repeats=repeats,
        avg_seconds=mean(seconds_samples),
        best_seconds=min(seconds_samples),
        avg_digits_per_second=mean(digits_per_second_samples),
        best_digits_per_second=max(digits_per_second_samples),
        threads=actual_threads,
        chunk_terms=actual_chunk_terms,
        leaf_terms=actual_leaf_terms,
        task_terms=actual_task_terms,
        prefix_ok=prefix_ok,
        status=status,
        seconds_samples=";".join(f"{value:.6f}" for value in seconds_samples),
        digits_per_second_samples=";".join(f"{value:.2f}" for value in digits_per_second_samples),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CPU C++ Project 2 parallel modes with repeated runs.")
    parser.add_argument(
        "--digits-list",
        default="10000000,50000000,100000000",
        help="Comma-separated digit sizes to probe in ascending order.",
    )
    parser.add_argument(
        "--modes",
        default="chunked,tasks,frontier",
        help="Comma-separated parallel modes to compare: chunked,tasks,frontier",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Number of sequential runs per mode and digit size.",
    )
    parser.add_argument("--threads", type=int, default=None, help="Override OpenMP thread count.")
    parser.add_argument("--chunk-terms", type=int, default=None, help="Override chunk size for chunked mode.")
    parser.add_argument("--leaf-terms", type=int, default=None, help="Override leaf block size.")
    parser.add_argument("--task-terms", type=int, default=None, help="Override task spawn threshold for tasks mode.")
    parser.add_argument(
        "--csv",
        default="result/project2_cpu_cpp_parallel_modes_summary.csv",
        help="CSV output path relative to HW/05.",
    )
    parser.add_argument(
        "--md",
        default="result/project2_cpu_cpp_parallel_modes_summary.md",
        help="Markdown output path relative to HW/05.",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    digits_list = parse_digits_list(args.digits_list)
    modes = parse_modes_list(args.modes)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[TaskTreeSummaryRow] = []
    for digits in digits_list:
        for parallel_mode in modes:
            rows.append(
                run_mode(
                    digits,
                    parallel_mode,
                    args.repeats,
                    threads=args.threads,
                    chunk_terms=args.chunk_terms,
                    leaf_terms=args.leaf_terms,
                    task_terms=args.task_terms,
                )
            )

    csv_path = Path(args.csv)
    md_path = Path(args.md)
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    for row in rows:
        print(
            f"{row.route}: digits={row.target_digits} repeats={row.repeats} "
            f"avg_seconds={row.avg_seconds:.6f} avg_digits_per_second={row.avg_digits_per_second:.2f} "
            f"best_digits_per_second={row.best_digits_per_second:.2f} status={row.status}"
        )
    print(f"wrote_csv={csv_path}")
    print(f"wrote_md={md_path}")


if __name__ == "__main__":
    main()
