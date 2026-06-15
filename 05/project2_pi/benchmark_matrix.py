from __future__ import annotations

import csv
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gmpy2

from . import gpu_pi_full_cuda as full_cuda
from . import gpu_pi_hybrid as hybrid
from . import homework_bridge


RESULT_DIR = Path(__file__).resolve().parent.parent / "result"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent.parent
KEY_VALUE_PATTERN = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


@dataclass
class RouteBenchmark:
    route: str
    group: str
    target_digits: int
    seconds: float
    digits_per_second: float
    prefix_ok: bool
    status: str
    notes: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for match in KEY_VALUE_PATTERN.finditer(text):
        values[match.group(1)] = match.group(2)
    return values


def write_csv(path: Path, rows: list[RouteBenchmark]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].as_dict().keys()))
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)


def write_markdown(path: Path, rows: list[RouteBenchmark]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Project2 Pi Benchmark Matrix",
        "",
        "| route | group | target_digits | seconds | digits_per_second | prefix_ok | status | notes |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {route} | {group} | {target_digits} | {seconds:.6f} | {digits_per_second:.2f} | {prefix_ok} | {status} | {notes} |".format(
                **row.as_dict()
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def benchmark_python_gmp_bridge(digits: int) -> RouteBenchmark:
    workers, chunk_terms = homework_bridge.resolve_project2_parallel_config(digits)
    start = time.perf_counter()
    pi_digits, _ = homework_bridge.compute_pi_digits_integer_gmpy2(digits, workers=workers, chunk_terms=chunk_terms)
    seconds = time.perf_counter() - start
    prefix_ok = homework_bridge.pi_prefix_matches_reference(pi_digits, digits)
    return RouteBenchmark(
        route="python_gmp_bridge",
        group="full_scale",
        target_digits=digits,
        seconds=seconds,
        digits_per_second=digits / seconds,
        prefix_ok=prefix_ok,
        status="ok" if prefix_ok else "prefix_mismatch",
        notes=f"workers={workers},chunk_terms={chunk_terms}",
    )


def benchmark_cpp_gmp(
    digits: int,
    threads: int | None = None,
    chunk_terms: int | None = None,
    leaf_terms: int | None = None,
    task_terms: int | None = None,
    parallel_mode: str | None = None,
) -> RouteBenchmark:
    binary = PROJECT_ROOT / "project2_pi" / "bin" / "project2_gmp_backend"
    auto_config = homework_bridge.resolve_project2_cpp_config(digits)
    if threads is None:
        threads = auto_config.threads
    if chunk_terms is None:
        chunk_terms = auto_config.chunk_terms
    if leaf_terms is None:
        leaf_terms = auto_config.leaf_terms
    if task_terms is None:
        task_terms = auto_config.task_terms
    if parallel_mode is None:
        parallel_mode = auto_config.parallel_mode
    parallel_mode = homework_bridge.normalize_project2_cpp_parallel_mode(parallel_mode)
    route = {
        "chunked": "cpp_gmp_openmp",
        "tasks": "cpp_gmp_openmp_tasks",
        "frontier": "cpp_gmp_openmp_frontier",
    }[parallel_mode]
    proc = run_command(
        [
            str(binary),
            "--digits",
            str(digits),
            "--threads",
            str(threads),
            "--chunk-terms",
            str(chunk_terms),
            "--leaf-terms",
            str(leaf_terms),
            "--task-terms",
            str(task_terms),
            "--parallel-mode",
            parallel_mode,
        ],
        cwd=PROJECT_ROOT,
    )
    if proc.returncode != 0:
        return RouteBenchmark(
            route=route,
            group="full_scale",
            target_digits=digits,
            seconds=0.0,
            digits_per_second=0.0,
            prefix_ok=False,
            status="failed",
            notes=proc.stderr.strip() or "cpp backend failed",
        )
    values = parse_key_values(proc.stdout)
    prefix_ok = values.get("prefix_ok", "False") == "True"
    return RouteBenchmark(
        route=route,
        group="full_scale",
        target_digits=digits,
        seconds=float(values["seconds"]),
        digits_per_second=float(values["digits_per_second"]),
        prefix_ok=prefix_ok,
        status="ok" if prefix_ok else "prefix_mismatch",
        notes=(
            f"threads={threads},chunk_terms={chunk_terms},leaf_terms={leaf_terms},"
            f"task_terms={task_terms},parallel_mode={parallel_mode}"
        ),
    )


def benchmark_cpp_levelpool(
    digits: int,
    threads: int | None = None,
    chunk_terms: int | None = None,
    leaf_terms: int | None = None,
) -> RouteBenchmark:
    binary = homework_bridge.resolve_project2_cpp_levelpool_binary()
    if not binary.is_file():
        raise RuntimeError("project2_gmp_levelpool_backend is unavailable; run `make cpp_levelpool_backend` first")

    auto_config = homework_bridge.resolve_project2_cpp_config(digits)
    if threads is None:
        threads = auto_config.threads
    if chunk_terms is None:
        chunk_terms = auto_config.chunk_terms
    if leaf_terms is None:
        leaf_terms = auto_config.leaf_terms

    proc = run_command(
        [
            str(binary),
            "--digits",
            str(digits),
            "--threads",
            str(threads),
            "--chunk-terms",
            str(chunk_terms),
            "--leaf-terms",
            str(leaf_terms),
        ],
        cwd=PROJECT_ROOT,
    )
    if proc.returncode != 0:
        return RouteBenchmark(
            route="cpp_gmp_levelpool",
            group="full_scale",
            target_digits=digits,
            seconds=0.0,
            digits_per_second=0.0,
            prefix_ok=False,
            status="failed",
            notes=proc.stderr.strip() or "levelpool backend failed",
        )
    values = parse_key_values(proc.stdout)
    prefix_ok = values.get("prefix_ok", "False") == "True"
    return RouteBenchmark(
        route="cpp_gmp_levelpool",
        group="full_scale",
        target_digits=digits,
        seconds=float(values["seconds"]),
        digits_per_second=float(values["digits_per_second"]),
        prefix_ok=prefix_ok,
        status="ok" if prefix_ok else "prefix_mismatch",
        notes=f"threads={threads},chunk_terms={chunk_terms},leaf_terms={leaf_terms},representation=levelpool",
    )


def benchmark_hybrid(
    route: str,
    digits: int,
    *,
    gpu_stages: str,
    gpu_chunk_format: str,
    sqrt_mode: str,
    division_mode: str,
    chunk_arith_backend: str,
    gpu_memory_budget_gb: float | None = None,
) -> RouteBenchmark:
    workers, chunk_terms = hybrid.resolve_parallel_config(digits)
    warmup_digits = min(10_000, digits)
    hybrid.set_chunk_arith_backend(chunk_arith_backend)
    if gpu_stages == "merge-and-final" and gpu_chunk_format == "binary16" and chunk_arith_backend != "python":
        hybrid.ensure_cuda_chunk_ops_ready(required=(chunk_arith_backend == "cuda-ext"))
    if warmup_digits < digits:
        hybrid.compute_pi_digits_integer_gpu_hybrid(
            digits=warmup_digits,
            workers=workers,
            chunk_terms=chunk_terms,
            bs_leaf_terms=hybrid.PROJECT2_BS_LEAF_TERMS,
            gpu_threshold_digits=1,
            gpu_stages=gpu_stages,
            gpu_chunk_format=gpu_chunk_format,
            sqrt_mode=sqrt_mode,
            division_mode=division_mode,
            gpu_memory_budget_gb=gpu_memory_budget_gb,
        )
    start = time.perf_counter()
    pi_digits, _, _, _ = hybrid.compute_pi_digits_integer_gpu_hybrid(
        digits=digits,
        workers=workers,
        chunk_terms=chunk_terms,
        bs_leaf_terms=hybrid.PROJECT2_BS_LEAF_TERMS,
        gpu_threshold_digits=50_000,
        gpu_stages=gpu_stages,
        gpu_chunk_format=gpu_chunk_format,
        sqrt_mode=sqrt_mode,
        division_mode=division_mode,
        gpu_memory_budget_gb=gpu_memory_budget_gb,
    )
    seconds = time.perf_counter() - start
    prefix_ok = hybrid.pi_prefix_matches_reference(pi_digits, digits)
    return RouteBenchmark(
        route=route,
        group="full_scale",
        target_digits=digits,
        seconds=seconds,
        digits_per_second=digits / seconds,
        prefix_ok=prefix_ok,
        status="ok" if prefix_ok else "prefix_mismatch",
        notes=(
            f"workers={workers},chunk_terms={chunk_terms},gpu_stages={gpu_stages},"
            f"sqrt={sqrt_mode},div={division_mode},chunk_backend={chunk_arith_backend},"
            f"gpu_budget_gb={gpu_memory_budget_gb}"
        ),
    )


def benchmark_full_cuda_pipeline(digits: int) -> RouteBenchmark:
    runtime = full_cuda.FullCudaRuntime()
    warmup_digits = min(10_000, digits)
    if warmup_digits < digits:
        runtime.benchmark_pipeline(
            digits=warmup_digits,
            workers=None,
            chunk_terms=None,
            bs_leaf_terms=hybrid.PROJECT2_BS_LEAF_TERMS,
            output_path=None,
        )
    stats = runtime.benchmark_pipeline(
        digits=digits,
        workers=None,
        chunk_terms=None,
        bs_leaf_terms=hybrid.PROJECT2_BS_LEAF_TERMS,
        output_path=None,
    )
    return RouteBenchmark(
        route="full_cuda_pipeline",
        group="prototype_full_pi",
        target_digits=digits,
        seconds=stats.total_seconds,
        digits_per_second=stats.digits_per_second,
        prefix_ok=stats.prefix_ok,
        status="ok" if stats.prefix_ok else "prefix_mismatch",
        notes=f"workers={stats.workers},chunk_terms={stats.chunk_terms},gpu_calls={stats.gpu_calls}",
    )


def benchmark_native_rns(digits: int, iterations: int) -> RouteBenchmark:
    binary = PROJECT_ROOT / "project2_gpu_native_rns" / "bin" / "project2_gpu_native_rns_smoke"
    proc = run_command(
        [
            str(binary),
            "--pi-end-to-end-benchmark",
            "--pi-digits",
            str(digits),
            "--iterations",
            str(iterations),
            "--output-tag",
            "codex_matrix",
        ],
        cwd=PROJECT_ROOT / "project2_gpu_native_rns",
    )
    values = parse_key_values(proc.stdout)
    ok = proc.returncode == 0 and values.get("pi_end_to_end_benchmark_status") == "ok"
    digits_per_second = float(values.get("avg_digits_per_second_e2e", "0"))
    seconds = float(values.get("avg_end_to_end_ms", "0")) / 1000.0
    return RouteBenchmark(
        route="gpu_native_rns_end_to_end",
        group="prototype_full_pi",
        target_digits=digits,
        seconds=seconds,
        digits_per_second=digits_per_second,
        prefix_ok=values.get("prefix_match", "0") in {"1", "True", "true"},
        status="ok" if ok else "failed",
        notes="native-rns benchmark ceiling is still sub-1k digits",
    )


def benchmark_throughput_mainline(digits: int) -> RouteBenchmark:
    binary = PROJECT_ROOT / "project2_gpu_throughput_mainline" / "bin" / "project2_gpu_throughput_mainline"
    proc = run_command(
        [
            str(binary),
            "--pi-end-to-end-smoke",
            "--term-count",
            "256",
            "--slot-count",
            "2048",
            "--modulus-count",
            "10",
            "--target-digits",
            str(digits),
            "--report-decimal-digits",
            "32",
            "--warmup",
            "1",
            "--iterations",
            "3",
            "--output-tag",
            "codex_matrix",
        ],
        cwd=PROJECT_ROOT / "project2_gpu_throughput_mainline",
    )
    values = parse_key_values(proc.stdout)
    ok = proc.returncode == 0 and values.get("pi_end_to_end_smoke_status") == "ok"
    digits_per_second = float(values.get("steady_state_pi_digits_per_second", "0"))
    seconds = float(values.get("steady_state_pi_result_ms", "0")) / 1000.0
    return RouteBenchmark(
        route="gpu_throughput_mainline_end_to_end",
        group="prototype_full_pi",
        target_digits=digits,
        seconds=seconds,
        digits_per_second=digits_per_second,
        prefix_ok=values.get("prefix_match", "0") in {"1", "True", "true"},
        status="ok" if ok else "failed",
        notes="throughput-mainline is frozen and only validated on low-thousands digits",
    )


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        benchmark_python_gmp_bridge(10_000_000),
        benchmark_cpp_gmp(10_000_000),
        benchmark_cpp_levelpool(10_000_000),
        benchmark_cpp_gmp(10_000_000, parallel_mode="frontier"),
        benchmark_hybrid(
            "gpu_hybrid_legacy_default",
            10_000_000,
            gpu_stages="final-only",
            gpu_chunk_format="binary16",
            sqrt_mode="mpz-isqrt",
            division_mode="mpz-div",
            chunk_arith_backend="python",
        ),
        benchmark_hybrid(
            "gpu_hybrid_merge_fast_python",
            10_000_000,
            gpu_stages="merge-and-final",
            gpu_chunk_format="binary16",
            sqrt_mode="chunk-gpu-rsqrt-prototype",
            division_mode="auto",
            chunk_arith_backend="python",
        ),
        benchmark_hybrid(
            "gpu_hybrid_merge_fast_auto",
            10_000_000,
            gpu_stages="merge-and-final",
            gpu_chunk_format="binary16",
            sqrt_mode="chunk-gpu-rsqrt-prototype",
            division_mode="auto",
            chunk_arith_backend="auto",
        ),
        benchmark_full_cuda_pipeline(100_000),
        benchmark_native_rns(870, iterations=3),
        benchmark_throughput_mainline(2_500),
    ]
    csv_path = RESULT_DIR / "project2_route_benchmark_current.csv"
    md_path = RESULT_DIR / "project2_route_benchmark_current.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    for row in rows:
        print(
            f"{row.route}: group={row.group} digits={row.target_digits} "
            f"seconds={row.seconds:.6f} digits_per_second={row.digits_per_second:.2f} "
            f"prefix_ok={row.prefix_ok} status={row.status}"
        )
    print(f"wrote_csv={csv_path}")
    print(f"wrote_md={md_path}")


if __name__ == "__main__":
    main()
