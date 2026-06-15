from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from . import benchmark_matrix


RESULT_DIR = Path(__file__).resolve().parent.parent / "result"


@dataclass
class ExtremeBenchmarkRow:
    route: str
    target_digits: int
    seconds: float
    digits_per_second: float
    prefix_ok: bool
    status: str
    stop_reason: str
    notes: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


ROUTE_LABELS = {
    "cpu-cpp": "cpp_gmp_openmp",
    "cpu-cpp-tasks": "cpp_gmp_openmp_tasks",
    "cpu-cpp-frontier": "cpp_gmp_openmp_frontier",
    "cpu-cpp-levelpool": "cpp_gmp_levelpool",
    "cpu-python": "python_gmp_bridge",
    "hybrid-fast-auto": "gpu_hybrid_fast_auto",
    "hybrid-fast-python": "gpu_hybrid_fast_python",
    "hybrid-legacy": "gpu_hybrid_legacy_default",
    "hybrid-high-digits-auto": "gpu_hybrid_high_digits_auto",
}


def parse_digits_list(text: str) -> list[int]:
    values = [int(token) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("digits list must not be empty")
    return values


def write_csv(path: Path, rows: list[ExtremeBenchmarkRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].as_dict().keys()))
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)


def write_markdown(path: Path, rows: list[ExtremeBenchmarkRow]) -> None:
    lines = [
        "# CPU / Hybrid Extreme Exploration",
        "",
        "| route | target_digits | seconds | digits_per_second | prefix_ok | status | stop_reason | notes |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {route} | {target_digits} | {seconds:.6f} | {digits_per_second:.2f} | {prefix_ok} | {status} | {stop_reason} | {notes} |".format(
                **row.as_dict()
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def benchmark_route(route: str, digits: int, cpu_threads: int | None) -> benchmark_matrix.RouteBenchmark:
    if route == "cpu-cpp":
        config = benchmark_matrix.homework_bridge.resolve_project2_cpp_config(digits)
        if cpu_threads is not None:
            config = benchmark_matrix.homework_bridge.Project2CppConfig(
                threads=cpu_threads,
                chunk_terms=config.chunk_terms,
                leaf_terms=config.leaf_terms,
                task_terms=config.task_terms,
                parallel_mode=config.parallel_mode,
            )
        return benchmark_matrix.benchmark_cpp_gmp(
            digits,
            threads=config.threads,
            chunk_terms=config.chunk_terms,
            leaf_terms=config.leaf_terms,
            task_terms=config.task_terms,
            parallel_mode="chunked",
        )
    if route == "cpu-cpp-tasks":
        config = benchmark_matrix.homework_bridge.resolve_project2_cpp_config(digits)
        if cpu_threads is not None:
            config = benchmark_matrix.homework_bridge.Project2CppConfig(
                threads=cpu_threads,
                chunk_terms=config.chunk_terms,
                leaf_terms=config.leaf_terms,
                task_terms=config.task_terms,
                parallel_mode=config.parallel_mode,
            )
        return benchmark_matrix.benchmark_cpp_gmp(
            digits,
            threads=config.threads,
            chunk_terms=config.chunk_terms,
            leaf_terms=config.leaf_terms,
            task_terms=config.task_terms,
            parallel_mode="tasks",
        )
    if route == "cpu-cpp-frontier":
        config = benchmark_matrix.homework_bridge.resolve_project2_cpp_config(digits)
        if cpu_threads is not None:
            config = benchmark_matrix.homework_bridge.Project2CppConfig(
                threads=cpu_threads,
                chunk_terms=config.chunk_terms,
                leaf_terms=config.leaf_terms,
                task_terms=config.task_terms,
                parallel_mode=config.parallel_mode,
            )
        return benchmark_matrix.benchmark_cpp_gmp(
            digits,
            threads=config.threads,
            chunk_terms=config.chunk_terms,
            leaf_terms=config.leaf_terms,
            task_terms=config.task_terms,
            parallel_mode="frontier",
        )
    if route == "cpu-cpp-levelpool":
        config = benchmark_matrix.homework_bridge.resolve_project2_cpp_config(digits)
        if cpu_threads is not None:
            config = benchmark_matrix.homework_bridge.Project2CppConfig(
                threads=cpu_threads,
                chunk_terms=config.chunk_terms,
                leaf_terms=config.leaf_terms,
                task_terms=config.task_terms,
                parallel_mode=config.parallel_mode,
            )
        return benchmark_matrix.benchmark_cpp_levelpool(
            digits,
            threads=config.threads,
            chunk_terms=config.chunk_terms,
            leaf_terms=config.leaf_terms,
        )
    if route == "cpu-python":
        return benchmark_matrix.benchmark_python_gmp_bridge(digits)
    if route == "hybrid-fast-auto":
        return benchmark_matrix.benchmark_hybrid(
            ROUTE_LABELS[route],
            digits,
            gpu_stages="merge-and-final",
            gpu_chunk_format="binary16",
            sqrt_mode="chunk-gpu-rsqrt-prototype",
            division_mode="auto",
            chunk_arith_backend="auto",
        )
    if route == "hybrid-fast-python":
        return benchmark_matrix.benchmark_hybrid(
            ROUTE_LABELS[route],
            digits,
            gpu_stages="merge-and-final",
            gpu_chunk_format="binary16",
            sqrt_mode="chunk-gpu-rsqrt-prototype",
            division_mode="auto",
            chunk_arith_backend="python",
        )
    if route == "hybrid-high-digits-auto":
        return benchmark_matrix.benchmark_hybrid(
            ROUTE_LABELS[route],
            digits,
            gpu_stages="merge-and-final",
            gpu_chunk_format="binary16",
            sqrt_mode="chunk-gpu-rsqrt-prototype",
            division_mode="auto",
            chunk_arith_backend="auto",
            gpu_memory_budget_gb=12.0,
        )
    if route == "hybrid-legacy":
        return benchmark_matrix.benchmark_hybrid(
            ROUTE_LABELS[route],
            digits,
            gpu_stages="final-only",
            gpu_chunk_format="binary16",
            sqrt_mode="mpz-isqrt",
            division_mode="mpz-div",
            chunk_arith_backend="python",
        )
    raise ValueError(f"unknown route: {route}")


def summarize_route(rows: list[ExtremeBenchmarkRow], route: str) -> str:
    route_rows = [row for row in rows if row.route == route]
    if not route_rows:
        return f"{route}: no_data"
    last = route_rows[-1]
    best = max(route_rows, key=lambda row: row.digits_per_second)
    max_ok = max((row.target_digits for row in route_rows if row.status == "ok"), default=0)
    return (
        f"{route}: max_ok_digits={max_ok} "
        f"best_dps={best.digits_per_second:.2f}@{best.target_digits} "
        f"last_status={last.status} stop_reason={last.stop_reason}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore digit/s and successful size limits on CPU and hybrid Project 2 routes.")
    parser.add_argument(
        "--routes",
        default="cpu-cpp,cpu-python,hybrid-fast-auto",
        help="Comma-separated route ids: cpu-cpp,cpu-cpp-tasks,cpu-cpp-frontier,cpu-cpp-levelpool,cpu-python,hybrid-fast-auto,hybrid-fast-python,hybrid-high-digits-auto,hybrid-legacy",
    )
    parser.add_argument(
        "--digits-list",
        default="20000000,50000000,100000000",
        help="Comma-separated digit sizes to probe in ascending order.",
    )
    parser.add_argument(
        "--soft-limit-seconds",
        type=float,
        default=90.0,
        help="After a successful point exceeds this wall time, larger points for the same route are skipped.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=44,
        help="Thread count for the C++ CPU baseline.",
    )
    parser.add_argument(
        "--csv",
        default="result/project2_cpu_hybrid_extremes.csv",
    )
    parser.add_argument(
        "--md",
        default="result/project2_cpu_hybrid_extremes.md",
    )
    args = parser.parse_args()

    routes = [token.strip() for token in args.routes.split(",") if token.strip()]
    digits_list = parse_digits_list(args.digits_list)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[ExtremeBenchmarkRow] = []
    stopped_routes: dict[str, str] = {}

    for route in routes:
        for digits in digits_list:
            if route in stopped_routes:
                break
            try:
                result = benchmark_route(route, digits, cpu_threads=args.cpu_threads)
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    ExtremeBenchmarkRow(
                        route=route,
                        target_digits=digits,
                        seconds=0.0,
                        digits_per_second=0.0,
                        prefix_ok=False,
                        status="failed",
                        stop_reason="exception",
                        notes=str(exc),
                    )
                )
                stopped_routes[route] = "exception"
                break

            stop_reason = ""
            if result.status != "ok":
                stop_reason = "failed"
                stopped_routes[route] = stop_reason
            elif result.seconds >= args.soft_limit_seconds:
                stop_reason = "soft_limit_reached"
                stopped_routes[route] = stop_reason

            rows.append(
                ExtremeBenchmarkRow(
                    route=route,
                    target_digits=digits,
                    seconds=result.seconds,
                    digits_per_second=result.digits_per_second,
                    prefix_ok=result.prefix_ok,
                    status=result.status,
                    stop_reason=stop_reason,
                    notes=result.notes,
                )
            )

    csv_path = Path(args.csv)
    md_path = Path(args.md)
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    for route in routes:
        print(summarize_route(rows, route))
    print(f"wrote_csv={csv_path}")
    print(f"wrote_md={md_path}")


if __name__ == "__main__":
    main()
