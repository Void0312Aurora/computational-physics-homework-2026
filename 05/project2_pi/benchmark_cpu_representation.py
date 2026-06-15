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
class CpuRepresentationRow:
    route: str
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
    notes: str
    seconds_samples: str
    digits_per_second_samples: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_digits_list(text: str) -> list[int]:
    values = [int(token) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("digits list must not be empty")
    return values


def write_csv(path: Path, rows: list[CpuRepresentationRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].as_dict().keys()))
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)


def write_markdown(path: Path, rows: list[CpuRepresentationRow]) -> None:
    lines = [
        "# Project 2 CPU Representation Refactor Summary",
        "",
        "| route | digits | repeats | avg_seconds | best_seconds | avg_digits_per_second | best_digits_per_second | threads | chunk_terms | leaf_terms | task_terms | prefix_ok | status | notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {route} | {target_digits} | {repeats} | {avg_seconds:.6f} | {best_seconds:.6f} | "
            "{avg_digits_per_second:.2f} | {best_digits_per_second:.2f} | {threads} | {chunk_terms} | "
            "{leaf_terms} | {task_terms} | {prefix_ok} | {status} | {notes} |".format(**row.as_dict())
        )
        lines.append(
            f"  samples_seconds={row.seconds_samples}; samples_digits_per_second={row.digits_per_second_samples}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_runs(
    route: str,
    digits: int,
    repeats: int,
    runs: list[benchmark_matrix.RouteBenchmark],
    *,
    threads: int,
    chunk_terms: int,
    leaf_terms: int,
    task_terms: int,
) -> CpuRepresentationRow:
    seconds_samples = [run.seconds for run in runs]
    digits_per_second_samples = [run.digits_per_second for run in runs]
    prefix_ok = all(run.prefix_ok for run in runs)
    status = "ok" if all(run.status == "ok" for run in runs) else "failed"
    notes = runs[0].notes if runs else ""
    return CpuRepresentationRow(
        route=route,
        target_digits=digits,
        repeats=repeats,
        avg_seconds=mean(seconds_samples),
        best_seconds=min(seconds_samples),
        avg_digits_per_second=mean(digits_per_second_samples),
        best_digits_per_second=max(digits_per_second_samples),
        threads=threads,
        chunk_terms=chunk_terms,
        leaf_terms=leaf_terms,
        task_terms=task_terms,
        prefix_ok=prefix_ok,
        status=status,
        notes=notes,
        seconds_samples=";".join(f"{value:.6f}" for value in seconds_samples),
        digits_per_second_samples=";".join(f"{value:.2f}" for value in digits_per_second_samples),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the current CPU GMP object-tree backend against the new level-pooled representation backend.")
    parser.add_argument(
        "--digits-list",
        default="10000000,50000000,100000000",
        help="Comma-separated digit sizes to probe in ascending order.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of sequential runs per route and digit size.",
    )
    parser.add_argument("--threads", type=int, default=None, help="Override OpenMP thread count.")
    parser.add_argument("--chunk-terms", type=int, default=None, help="Override chunk size.")
    parser.add_argument("--leaf-terms", type=int, default=None, help="Override leaf block size.")
    parser.add_argument(
        "--csv",
        default="result/project2_cpu_cpp_representation_summary.csv",
        help="CSV output path relative to HW/05.",
    )
    parser.add_argument(
        "--md",
        default="result/project2_cpu_cpp_representation_summary.md",
        help="Markdown output path relative to HW/05.",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    digits_list = parse_digits_list(args.digits_list)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if not homework_bridge.has_project2_cpp_levelpool_backend():
        raise RuntimeError("project2_gmp_levelpool_backend is unavailable; run `make cpp_levelpool_backend` first")

    rows: list[CpuRepresentationRow] = []
    for digits in digits_list:
        config = homework_bridge.resolve_project2_cpp_config(digits)
        threads = args.threads if args.threads is not None else config.threads
        chunk_terms = args.chunk_terms if args.chunk_terms is not None else config.chunk_terms
        leaf_terms = args.leaf_terms if args.leaf_terms is not None else config.leaf_terms
        task_terms = chunk_terms

        chunked_runs = [
            benchmark_matrix.benchmark_cpp_gmp(
                digits,
                threads=threads,
                chunk_terms=chunk_terms,
                leaf_terms=leaf_terms,
                task_terms=task_terms,
                parallel_mode="chunked",
            )
            for _ in range(args.repeats)
        ]
        rows.append(
            aggregate_runs(
                "cpp_gmp_openmp",
                digits,
                args.repeats,
                chunked_runs,
                threads=threads,
                chunk_terms=chunk_terms,
                leaf_terms=leaf_terms,
                task_terms=task_terms,
            )
        )

        tasks_runs = [
            benchmark_matrix.benchmark_cpp_gmp(
                digits,
                threads=threads,
                chunk_terms=chunk_terms,
                leaf_terms=leaf_terms,
                task_terms=task_terms,
                parallel_mode="tasks",
            )
            for _ in range(args.repeats)
        ]
        rows.append(
            aggregate_runs(
                "cpp_gmp_openmp_tasks",
                digits,
                args.repeats,
                tasks_runs,
                threads=threads,
                chunk_terms=chunk_terms,
                leaf_terms=leaf_terms,
                task_terms=task_terms,
            )
        )

        levelpool_runs = [
            benchmark_matrix.benchmark_cpp_levelpool(
                digits,
                threads=threads,
                chunk_terms=chunk_terms,
                leaf_terms=leaf_terms,
            )
            for _ in range(args.repeats)
        ]
        rows.append(
            aggregate_runs(
                "cpp_gmp_levelpool",
                digits,
                args.repeats,
                levelpool_runs,
                threads=threads,
                chunk_terms=chunk_terms,
                leaf_terms=leaf_terms,
                task_terms=task_terms,
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
