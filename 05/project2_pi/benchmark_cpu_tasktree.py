from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from .benchmark_runner_utils import (
    CpuRouteBenchmark,
    benchmark_cpp_gmp,
    collect_system_context,
    format_samples,
    iso_timestamp,
    median_iqr,
    order_items,
    resolve_order_strategy,
    write_metadata_json,
)
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
    median_seconds: float
    iqr_seconds: float
    avg_digits_per_second: float
    best_digits_per_second: float
    median_digits_per_second: float
    iqr_digits_per_second: float
    threads: int
    chunk_terms: int
    leaf_terms: int
    task_terms: int
    prefix_ok: bool
    status: str
    seconds_samples: str
    digits_per_second_samples: str
    run_index_samples: str
    execution_order: str
    order_strategy: str
    seed: int
    loadavg_start: str
    loadavg_end: str
    omp_proc_bind: str
    omp_places: str
    process_cpu_affinity: str
    cpu_governor: str
    cpu_frequency_mhz: str
    nvidia_smi: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TaskTreeJob:
    digits: int
    parallel_mode: str
    sample_index: int


@dataclass(frozen=True)
class TaskTreeRunRecord:
    run_index: int
    sample_index: int
    result: CpuRouteBenchmark


def parse_digits_list(text: str) -> list[int]:
    values = [int(token) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("digits list must not be empty")
    return values


def parse_modes_list(text: str) -> list[str]:
    modes = [
        homework_bridge.normalize_project2_cpp_parallel_mode(token)
        for token in text.split(",")
        if token.strip()
    ]
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
        "| route | mode | digits | repeats | avg_seconds | median_seconds | iqr_seconds | best_seconds | avg_digits_per_second | median_digits_per_second | iqr_digits_per_second | best_digits_per_second | threads | chunk_terms | leaf_terms | task_terms | prefix_ok | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {route} | {parallel_mode} | {target_digits} | {repeats} | {avg_seconds:.6f} | "
            "{median_seconds:.6f} | {iqr_seconds:.6f} | {best_seconds:.6f} | "
            "{avg_digits_per_second:.2f} | {median_digits_per_second:.2f} | "
            "{iqr_digits_per_second:.2f} | {best_digits_per_second:.2f} | {threads} | {chunk_terms} | "
            "{leaf_terms} | {task_terms} | {prefix_ok} | {status} |".format(
                **row.as_dict()
            )
        )
        lines.append(
            f"  samples_seconds={row.seconds_samples}; samples_digits_per_second={row.digits_per_second_samples}; "
            f"run_index_samples={row.run_index_samples}; execution_order={row.execution_order}; "
            f"order_strategy={row.order_strategy}; loadavg={row.loadavg_start}->{row.loadavg_end}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_jobs(
    digits_list: list[int], modes: list[str], repeats: int, *, strategy: str, seed: int
) -> list[TaskTreeJob]:
    if strategy == "grouped":
        jobs = [
            TaskTreeJob(
                digits=digits, parallel_mode=parallel_mode, sample_index=sample_index
            )
            for digits in digits_list
            for parallel_mode in modes
            for sample_index in range(1, repeats + 1)
        ]
    else:
        jobs = [
            TaskTreeJob(
                digits=digits, parallel_mode=parallel_mode, sample_index=sample_index
            )
            for digits in digits_list
            for sample_index in range(1, repeats + 1)
            for parallel_mode in modes
        ]
    return order_items(jobs, strategy=strategy, seed=seed)


def run_mode_once(
    digits: int,
    parallel_mode: str,
    *,
    threads: int | None,
    chunk_terms: int | None,
    leaf_terms: int | None,
    task_terms: int | None,
) -> CpuRouteBenchmark:
    base_config = homework_bridge.resolve_project2_cpp_config(digits)
    actual_threads = threads if threads is not None else base_config.threads
    actual_chunk_terms = (
        chunk_terms if chunk_terms is not None else base_config.chunk_terms
    )
    actual_leaf_terms = leaf_terms if leaf_terms is not None else base_config.leaf_terms
    actual_task_terms = task_terms if task_terms is not None else base_config.task_terms

    return benchmark_cpp_gmp(
        digits,
        threads=actual_threads,
        chunk_terms=actual_chunk_terms,
        leaf_terms=actual_leaf_terms,
        task_terms=actual_task_terms,
        parallel_mode=parallel_mode,
    )


def aggregate_mode_records(
    digits: int,
    parallel_mode: str,
    records: list[TaskTreeRunRecord],
    *,
    threads: int | None,
    chunk_terms: int | None,
    leaf_terms: int | None,
    task_terms: int | None,
    order_strategy: str,
    seed: int,
    system_start: dict[str, str],
    system_end: dict[str, str],
) -> TaskTreeSummaryRow:
    base_config = homework_bridge.resolve_project2_cpp_config(digits)
    actual_threads = threads if threads is not None else base_config.threads
    actual_chunk_terms = (
        chunk_terms if chunk_terms is not None else base_config.chunk_terms
    )
    actual_leaf_terms = leaf_terms if leaf_terms is not None else base_config.leaf_terms
    actual_task_terms = task_terms if task_terms is not None else base_config.task_terms

    runs = [record.result for record in records]
    seconds_samples = [run.seconds for run in runs]
    digits_per_second_samples = [run.digits_per_second for run in runs]
    median_seconds, iqr_seconds = median_iqr(seconds_samples)
    median_digits_per_second, iqr_digits_per_second = median_iqr(
        digits_per_second_samples
    )
    route = (
        runs[0].route
        if runs
        else {
            "chunked": "cpp_gmp_openmp",
            "tasks": "cpp_gmp_openmp_tasks",
            "frontier": "cpp_gmp_openmp_frontier",
        }[parallel_mode]
    )
    prefix_ok = all(run.prefix_ok for run in runs)
    status = "ok" if all(run.status == "ok" for run in runs) else "failed"
    run_indices = [record.run_index for record in records]
    sample_indices = [record.sample_index for record in records]
    return TaskTreeSummaryRow(
        route=route,
        parallel_mode=parallel_mode,
        target_digits=digits,
        repeats=len(records),
        avg_seconds=mean(seconds_samples),
        best_seconds=min(seconds_samples),
        median_seconds=median_seconds,
        iqr_seconds=iqr_seconds,
        avg_digits_per_second=mean(digits_per_second_samples),
        best_digits_per_second=max(digits_per_second_samples),
        median_digits_per_second=median_digits_per_second,
        iqr_digits_per_second=iqr_digits_per_second,
        threads=actual_threads,
        chunk_terms=actual_chunk_terms,
        leaf_terms=actual_leaf_terms,
        task_terms=actual_task_terms,
        prefix_ok=prefix_ok,
        status=status,
        seconds_samples=format_samples(seconds_samples, 6),
        digits_per_second_samples=format_samples(digits_per_second_samples, 2),
        run_index_samples=";".join(str(value) for value in run_indices),
        execution_order=";".join(
            f"{run_index}:sample{sample_index}"
            for run_index, sample_index in zip(run_indices, sample_indices)
        ),
        order_strategy=order_strategy,
        seed=seed,
        loadavg_start=system_start.get("loadavg", ""),
        loadavg_end=system_end.get("loadavg", ""),
        omp_proc_bind=system_start.get("omp_proc_bind", ""),
        omp_places=system_start.get("omp_places", ""),
        process_cpu_affinity=system_start.get("process_cpu_affinity", ""),
        cpu_governor=system_start.get("cpu_governor", ""),
        cpu_frequency_mhz=system_start.get("cpu_frequency_mhz", ""),
        nvidia_smi=system_start.get("nvidia_smi", ""),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CPU C++ Project 2 parallel modes with repeated runs."
    )
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
        default=5,
        help="Number of runs per mode and digit size. Lower explicitly for long runs.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle individual run order using --seed.",
    )
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="Use the default interleaved run order explicitly.",
    )
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="Use legacy grouped order with each mode repeated together.",
    )
    parser.add_argument(
        "--seed", type=int, default=20260615, help="Seed used by --shuffle."
    )
    parser.add_argument(
        "--threads", type=int, default=None, help="Override OpenMP thread count."
    )
    parser.add_argument(
        "--chunk-terms",
        type=int,
        default=None,
        help="Override chunk size for chunked mode.",
    )
    parser.add_argument(
        "--leaf-terms", type=int, default=None, help="Override leaf block size."
    )
    parser.add_argument(
        "--task-terms",
        type=int,
        default=None,
        help="Override task spawn threshold for tasks mode.",
    )
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
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Metadata JSON sidecar path. Defaults to <csv>.metadata.json.",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if sum(bool(value) for value in (args.shuffle, args.interleave, args.grouped)) > 1:
        raise ValueError(
            "--shuffle, --interleave, and --grouped are mutually exclusive"
        )

    digits_list = parse_digits_list(args.digits_list)
    modes = parse_modes_list(args.modes)
    order_strategy = resolve_order_strategy(shuffle=args.shuffle, grouped=args.grouped)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    system_start = collect_system_context()
    started_at = iso_timestamp()
    records_by_key: dict[tuple[int, str], list[TaskTreeRunRecord]] = {}
    metadata_records: list[dict[str, object]] = []
    jobs = build_jobs(
        digits_list, modes, args.repeats, strategy=order_strategy, seed=args.seed
    )
    for run_index, job in enumerate(jobs, start=1):
        result = run_mode_once(
            job.digits,
            job.parallel_mode,
            threads=args.threads,
            chunk_terms=args.chunk_terms,
            leaf_terms=args.leaf_terms,
            task_terms=args.task_terms,
        )
        record = TaskTreeRunRecord(
            run_index=run_index, sample_index=job.sample_index, result=result
        )
        records_by_key.setdefault((job.digits, job.parallel_mode), []).append(record)
        metadata_records.append(
            {
                "run_index": run_index,
                "sample_index": job.sample_index,
                "target_digits": job.digits,
                "parallel_mode": job.parallel_mode,
                "route": result.route,
                "seconds": result.seconds,
                "digits_per_second": result.digits_per_second,
                "prefix_ok": result.prefix_ok,
                "status": result.status,
                "notes": result.notes,
            }
        )

    system_end = collect_system_context()
    finished_at = iso_timestamp()

    rows = []
    for digits in digits_list:
        for parallel_mode in modes:
            rows.append(
                aggregate_mode_records(
                    digits,
                    parallel_mode,
                    records_by_key[(digits, parallel_mode)],
                    threads=args.threads,
                    chunk_terms=args.chunk_terms,
                    leaf_terms=args.leaf_terms,
                    task_terms=args.task_terms,
                    order_strategy=order_strategy,
                    seed=args.seed,
                    system_start=system_start,
                    system_end=system_end,
                )
            )

    csv_path = Path(args.csv)
    md_path = Path(args.md)
    metadata_path = (
        Path(args.metadata_json)
        if args.metadata_json
        else csv_path.with_suffix(csv_path.suffix + ".metadata.json")
    )
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    write_metadata_json(
        metadata_path,
        {
            "script": "benchmark_cpu_tasktree.py",
            "started_at": started_at,
            "finished_at": finished_at,
            "digits_list": digits_list,
            "modes": modes,
            "repeats": args.repeats,
            "order_strategy": order_strategy,
            "seed": args.seed,
            "system_context_start": system_start,
            "system_context_end": system_end,
            "runs": metadata_records,
        },
    )

    for row in rows:
        print(
            f"{row.route}: digits={row.target_digits} repeats={row.repeats} "
            f"median_seconds={row.median_seconds:.6f} iqr_seconds={row.iqr_seconds:.6f} "
            f"avg_seconds={row.avg_seconds:.6f} avg_digits_per_second={row.avg_digits_per_second:.2f} "
            f"best_digits_per_second={row.best_digits_per_second:.2f} status={row.status}"
        )
    print(f"wrote_csv={csv_path}")
    print(f"wrote_md={md_path}")
    print(f"wrote_metadata={metadata_path}")


if __name__ == "__main__":
    main()
