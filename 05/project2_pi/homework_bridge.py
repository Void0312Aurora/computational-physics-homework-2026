from __future__ import annotations

import csv
import multiprocessing as mp
import os
import re
import subprocess
import time
from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path

try:
    import gmpy2
except ImportError:
    gmpy2 = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if PROJECT_ROOT.name == "scripts":
    PROJECT_ROOT = PROJECT_ROOT.parent
RESULT_DIR = PROJECT_ROOT / "result"
C3_OVER_24 = 640320**3 // 24
PI_PREFIX_REFERENCE = "3.14159265358979323846264338327950288419716939937510"
PI_PREFIX_REFERENCE_NO_DOT = PI_PREFIX_REFERENCE.replace(".", "")
PROJECT2_GUARD_DIGITS = 48
PROJECT2_DEFAULT_CHUNK_TERMS = 8192
PROJECT2_DEFAULT_BENCHMARK_DIGITS = [100000, 1000000, 10000000]
PROJECT2_CPP_DEFAULT_LEAF_TERMS = 8
PROJECT2_CPP_DEFAULT_PARALLEL_MODE = "chunked"
KEY_VALUE_PATTERN = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


@dataclass(frozen=True)
class Project2CppConfig:
    threads: int
    chunk_terms: int
    leaf_terms: int
    task_terms: int
    parallel_mode: str


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for match in KEY_VALUE_PATTERN.finditer(text):
        values[match.group(1)] = match.group(2)
    return values


def chudnovsky_bs(a: int, b: int) -> tuple[int, int, int]:
    if b - a == 1:
        if a == 0:
            return 1, 1, 13591409
        p = (6 * a - 5) * (2 * a - 1) * (6 * a - 1)
        q = a * a * a * C3_OVER_24
        t = p * (13591409 + 545140134 * a)
        if a % 2:
            t = -t
        return p, q, t

    m = (a + b) // 2
    p1, q1, t1 = chudnovsky_bs(a, m)
    p2, q2, t2 = chudnovsky_bs(m, b)
    return p1 * p2, q1 * q2, t1 * q2 + p1 * t2


def compute_pi_string_decimal(digits: int) -> tuple[str, int]:
    getcontext().prec = digits + 50
    terms = digits // 14 + 1
    _, q, t = chudnovsky_bs(0, terms)
    pi_value = (Decimal(426880) * Decimal(10005).sqrt() * Decimal(q)) / Decimal(t)
    return format(pi_value, f".{digits}f"), terms


def estimate_chudnovsky_term_count(digits: int, guard_digits: int = PROJECT2_GUARD_DIGITS) -> int:
    return (digits + guard_digits) // 14 + 1


def chudnovsky_bs_mpz(a: int, b: int) -> tuple[object, object, object]:
    if gmpy2 is None:
        raise RuntimeError("gmpy2 backend is unavailable")
    if b - a == 1:
        if a == 0:
            one = gmpy2.mpz(1)
            return one, one, gmpy2.mpz(13591409)
        a_mpz = gmpy2.mpz(a)
        p = (6 * a_mpz - 5) * (2 * a_mpz - 1) * (6 * a_mpz - 1)
        q = a_mpz * a_mpz * a_mpz * C3_OVER_24
        t = p * (13591409 + 545140134 * a_mpz)
        if a % 2:
            t = -t
        return p, q, t

    m = (a + b) // 2
    p1, q1, t1 = chudnovsky_bs_mpz(a, m)
    p2, q2, t2 = chudnovsky_bs_mpz(m, b)
    return p1 * p2, q1 * q2, t1 * q2 + p1 * t2


def format_pi_digits_integer(pi_digits: object, digits: int) -> str:
    raw = pi_digits.digits(10) if hasattr(pi_digits, "digits") else str(pi_digits)
    expected = digits + 1
    if len(raw) < expected:
        raw = "0" * (expected - len(raw)) + raw
    return raw[0] + "." + raw[1:]


def pi_prefix_matches_reference(pi_digits: object, digits: int) -> bool:
    if digits < len(PI_PREFIX_REFERENCE_NO_DOT) - 1:
        return format_pi_digits_integer(pi_digits, digits).startswith(PI_PREFIX_REFERENCE)
    trim_power = digits + 1 - len(PI_PREFIX_REFERENCE_NO_DOT)
    leading = pi_digits // (gmpy2.mpz(10) ** trim_power) if trim_power > 0 else pi_digits
    return str(leading) == PI_PREFIX_REFERENCE_NO_DOT


def merge_chudnovsky_partials(
    left: tuple[object, object, object],
    right: tuple[object, object, object],
) -> tuple[object, object, object]:
    p1, q1, t1 = left
    p2, q2, t2 = right
    return p1 * p2, q1 * q2, t1 * q2 + p1 * t2


def project2_bs_worker(work_range: tuple[int, int]) -> tuple[object, object, object]:
    start, end = work_range
    return chudnovsky_bs_mpz(start, end)


def chudnovsky_bs_parallel_mpz(term_count: int, workers: int, chunk_terms: int) -> tuple[object, object, object]:
    if workers <= 1 or term_count <= chunk_terms:
        return chudnovsky_bs_mpz(0, term_count)

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < term_count:
        end = min(start + chunk_terms, term_count)
        ranges.append((start, end))
        start = end

    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    with ctx.Pool(processes=min(workers, len(ranges))) as pool:
        partials = pool.map(project2_bs_worker, ranges)

    while len(partials) > 1:
        merged: list[tuple[object, object, object]] = []
        for index in range(0, len(partials), 2):
            if index + 1 == len(partials):
                merged.append(partials[index])
            else:
                merged.append(merge_chudnovsky_partials(partials[index], partials[index + 1]))
        partials = merged
    return partials[0]


def compute_pi_digits_integer_gmpy2(
    digits: int,
    workers: int,
    chunk_terms: int,
    guard_digits: int = PROJECT2_GUARD_DIGITS,
) -> tuple[object, int]:
    if gmpy2 is None:
        raise RuntimeError("gmpy2 backend is unavailable")

    total_digits = digits + guard_digits
    terms = estimate_chudnovsky_term_count(digits, guard_digits=guard_digits)
    _, q, t = chudnovsky_bs_parallel_mpz(terms, workers, chunk_terms)
    scale = gmpy2.mpz(10) ** total_digits
    sqrt_term = gmpy2.isqrt(gmpy2.mpz(10005) * scale * scale)
    pi_scaled = (gmpy2.mpz(426880) * sqrt_term * q) // t
    if guard_digits:
        pi_scaled //= gmpy2.mpz(10) ** guard_digits
    return pi_scaled, terms


def resolve_project2_workers() -> int:
    env_value = os.environ.get("PROJECT2_WORKERS")
    if env_value:
        return max(1, int(env_value))
    logical_cpus = os.cpu_count() or 1
    return min(24, max(1, logical_cpus // 2))


def resolve_project2_chunk_terms() -> int:
    env_value = os.environ.get("PROJECT2_CHUNK_TERMS")
    if env_value:
        return max(256, int(env_value))
    return PROJECT2_DEFAULT_CHUNK_TERMS


def resolve_project2_parallel_config(digits: int) -> tuple[int, int]:
    env_workers = os.environ.get("PROJECT2_WORKERS")
    env_chunk = os.environ.get("PROJECT2_CHUNK_TERMS")
    if env_workers or env_chunk:
        workers = max(1, int(env_workers)) if env_workers else resolve_project2_workers()
        chunk_terms = max(256, int(env_chunk)) if env_chunk else resolve_project2_chunk_terms()
        return workers, chunk_terms

    logical_cpus = os.cpu_count() or 1
    large_workers = min(32, logical_cpus)
    if digits >= 50_000_000:
        return large_workers, 524_288
    if digits >= 20_000_000:
        return large_workers, 262_144
    if digits >= 10_000_000:
        return large_workers, 131_072
    if digits >= 5_000_000:
        return large_workers, 65_536
    return min(24, max(1, logical_cpus // 2)), PROJECT2_DEFAULT_CHUNK_TERMS


def resolve_project2_benchmark_digits(use_gmpy2: bool) -> list[int]:
    env_value = os.environ.get("PROJECT2_BENCHMARK_DIGITS")
    if env_value:
        return [int(token.strip()) for token in env_value.split(",") if token.strip()]
    return PROJECT2_DEFAULT_BENCHMARK_DIGITS if use_gmpy2 else [10000, 50000, 100000]


def resolve_project2_homework_backend() -> str:
    env_value = os.environ.get("PROJECT2_HOMEWORK_BACKEND", "cpu").strip().lower()
    if env_value not in {"auto", "cpu", "gpu"}:
        raise ValueError("PROJECT2_HOMEWORK_BACKEND must be one of: auto, cpu, gpu")
    return env_value


def resolve_project2_cpp_binary() -> Path:
    return PROJECT_ROOT / "project2_pi" / "bin" / "project2_gmp_backend"


def resolve_project2_cpp_levelpool_binary() -> Path:
    return PROJECT_ROOT / "project2_pi" / "bin" / "project2_gmp_levelpool_backend"


def has_project2_cpp_backend() -> bool:
    return resolve_project2_cpp_binary().is_file()


def has_project2_cpp_levelpool_backend() -> bool:
    return resolve_project2_cpp_levelpool_binary().is_file()


def resolve_project2_cpp_threads() -> int:
    env_value = os.environ.get("PROJECT2_CPP_THREADS")
    if env_value:
        return max(1, int(env_value))
    logical_cpus = os.cpu_count() or 1
    return min(44, max(1, logical_cpus // 2))


def normalize_project2_cpp_parallel_mode(mode: str | None) -> str:
    normalized = (mode or PROJECT2_CPP_DEFAULT_PARALLEL_MODE).strip().lower()
    if normalized not in {"chunked", "tasks", "frontier"}:
        raise ValueError("PROJECT2_CPP_PARALLEL_MODE must be one of: chunked, tasks, frontier")
    return normalized


def default_project2_cpp_chunk_terms(digits: int) -> int:
    if digits >= 150_000_000:
        return 262_144
    if digits >= 10_000_000:
        return 131_072
    if digits >= 5_000_000:
        return 65_536
    return PROJECT2_DEFAULT_CHUNK_TERMS


def resolve_project2_cpp_config(digits: int) -> Project2CppConfig:
    threads = resolve_project2_cpp_threads()
    env_chunk = os.environ.get("PROJECT2_CPP_CHUNK_TERMS")
    env_leaf = os.environ.get("PROJECT2_CPP_LEAF_TERMS")
    env_task = os.environ.get("PROJECT2_CPP_TASK_TERMS")
    env_mode = os.environ.get("PROJECT2_CPP_PARALLEL_MODE")

    default_chunk_terms = default_project2_cpp_chunk_terms(digits)
    chunk_terms = max(256, int(env_chunk)) if env_chunk else default_chunk_terms
    leaf_terms = max(1, int(env_leaf)) if env_leaf else PROJECT2_CPP_DEFAULT_LEAF_TERMS
    task_terms = max(256, int(env_task)) if env_task else chunk_terms
    parallel_mode = normalize_project2_cpp_parallel_mode(env_mode)

    return Project2CppConfig(
        threads=threads,
        chunk_terms=chunk_terms,
        leaf_terms=leaf_terms,
        task_terms=task_terms,
        parallel_mode=parallel_mode,
    )


def can_use_gpu_hybrid() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def run_project2_cpp_backend(
    digits: int,
    threads: int,
    chunk_terms: int,
    leaf_terms: int,
    task_terms: int,
    parallel_mode: str,
    output_path: Path | None = None,
) -> dict[str, object]:
    binary = resolve_project2_cpp_binary()
    if not binary.is_file():
        raise RuntimeError("project2 C++ backend binary is unavailable; run `make cpp_backend` first")

    cmd = [
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
    ]
    if output_path is not None:
        cmd.extend(["--output", str(output_path)])

    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "project2 C++ backend failed")

    values = parse_key_values(proc.stdout)
    return {
        "digits": digits,
        "terms": int(values["terms"]),
        "seconds": float(values["seconds"]),
        "digits_per_second": float(values["digits_per_second"]),
        "threads": int(values["threads"]),
        "chunk_terms": int(values["chunk_terms"]),
        "leaf_terms": int(values.get("leaf_terms", leaf_terms)),
        "task_terms": int(values.get("task_terms", task_terms)),
        "parallel_mode": values.get("parallel_mode", parallel_mode),
        "prefix_matches_reference": values.get("prefix_ok", "False") == "True",
    }


def solve_project2_gpu_hybrid(digits_list: list[int]) -> dict[str, object]:
    from . import gpu_pi_hybrid as hybrid

    profile = hybrid.PROJECT2_HYBRID_PROFILES["fast-auto"]
    highest_digits = max(digits_list)
    output_name = f"project2_pi_gpu_hybrid_mainline_{highest_digits}_digits.txt"
    for stale in RESULT_DIR.glob("project2_pi_gpu_hybrid_mainline_*_digits.txt"):
        if stale.name != output_name:
            stale.unlink()
    rows, _, _ = hybrid.benchmark_digits(
        digits_list=digits_list,
        gpu_threshold_digits=int(profile["gpu_threshold_digits"]),
        gpu_stages=str(profile["gpu_stages"]),
        gpu_chunk_format=str(profile["gpu_chunk_format"]),
        csv_path=RESULT_DIR / "project2_gpu_hybrid_mainline_benchmark.csv",
        output_path=RESULT_DIR / output_name,
        sqrt_mode=str(profile["sqrt_mode"]),
        division_mode=str(profile["division_mode"]),
        chunk_arith_backend=str(profile["chunk_arith_backend"]),
    )
    benchmark_rows = [
        {
            "digits": int(row["digits"]),
            "terms": int(row["terms"]),
            "seconds": float(row["seconds"]),
            "digits_per_second": float(row["digits_per_second"]),
            "backend": "gpu_hybrid_fast_auto",
            "workers_used": int(row["workers_used"]),
            "chunk_terms": int(row["chunk_terms"]),
            "gpu_used": bool(int(row["gpu_calls"]) > 0),
            "prefix_matches_reference": bool(row["prefix_matches_reference"]),
        }
        for row in rows
    ]
    highest_row = benchmark_rows[-1]
    return {
        "benchmark_rows": benchmark_rows,
        "backend": "gpu_hybrid_fast_auto",
        "workers": highest_row["workers_used"],
        "chunk_terms": highest_row["chunk_terms"],
        "highest_digits": highest_digits,
        "output_name": output_name,
        "gpu_used": True,
    }


def solve_project2_cpp_backend(digits_list: list[int]) -> dict[str, object]:
    highest_digits = max(digits_list)
    output_name = f"project2_pi_{highest_digits}_digits.txt"

    benchmark_rows: list[dict[str, object]] = []
    highest_threads = 1
    highest_chunk_terms = 0
    highest_leaf_terms = PROJECT2_CPP_DEFAULT_LEAF_TERMS
    highest_task_terms = 0
    highest_parallel_mode = PROJECT2_CPP_DEFAULT_PARALLEL_MODE
    output_path = RESULT_DIR / output_name

    for digits in digits_list:
        config = resolve_project2_cpp_config(digits)
        row = run_project2_cpp_backend(
            digits,
            threads=config.threads,
            chunk_terms=config.chunk_terms,
            leaf_terms=config.leaf_terms,
            task_terms=config.task_terms,
            parallel_mode=config.parallel_mode,
            output_path=(output_path if digits == highest_digits else None),
        )
        benchmark_rows.append(
            {
                "digits": digits,
                "terms": int(row["terms"]),
                "seconds": float(row["seconds"]),
                "digits_per_second": float(row["digits_per_second"]),
                "backend": {
                    "chunked": "cpp_gmp_openmp_optimized",
                    "tasks": "cpp_gmp_openmp_tasks_optimized",
                    "frontier": "cpp_gmp_openmp_frontier_optimized",
                }[str(row["parallel_mode"])],
                "workers_used": int(row["threads"]),
                "chunk_terms": int(row["chunk_terms"]),
                "leaf_terms": int(row["leaf_terms"]),
                "task_terms": int(row["task_terms"]),
                "parallel_mode": str(row["parallel_mode"]),
                "gpu_used": False,
                "prefix_matches_reference": bool(row["prefix_matches_reference"]),
            }
        )
        if digits == highest_digits:
            highest_threads = int(row["threads"])
            highest_chunk_terms = int(row["chunk_terms"])
            highest_leaf_terms = int(row["leaf_terms"])
            highest_task_terms = int(row["task_terms"])
            highest_parallel_mode = str(row["parallel_mode"])

    write_csv(RESULT_DIR / "project2_pi_benchmark.csv", benchmark_rows)
    return {
        "benchmark_rows": benchmark_rows,
        "backend": {
            "chunked": "cpp_gmp_openmp_optimized",
            "tasks": "cpp_gmp_openmp_tasks_optimized",
            "frontier": "cpp_gmp_openmp_frontier_optimized",
        }[highest_parallel_mode],
        "workers": highest_threads,
        "chunk_terms": highest_chunk_terms,
        "leaf_terms": highest_leaf_terms,
        "task_terms": highest_task_terms,
        "parallel_mode": highest_parallel_mode,
        "highest_digits": highest_digits,
        "output_name": output_name,
        "gpu_used": False,
    }


def solve_project2_python_backend(use_gmpy2: bool, digits_list: list[int]) -> dict[str, object]:
    backend = "gmpy2_parallel_mpz" if use_gmpy2 else "decimal_binary_splitting"
    benchmark_rows: list[dict[str, object]] = []
    highest_digits_string = ""
    highest_digits = 0
    highest_workers = 1
    highest_chunk_terms = 0

    max_benchmark_digits = max(digits_list)
    for digits in digits_list:
        workers, chunk_terms = resolve_project2_parallel_config(digits) if use_gmpy2 else (1, 0)
        start = time.perf_counter()
        if use_gmpy2:
            pi_digits, terms = compute_pi_digits_integer_gmpy2(digits, workers=workers, chunk_terms=chunk_terms)
            prefix_matches_reference = pi_prefix_matches_reference(pi_digits, digits)
            pi_string = format_pi_digits_integer(pi_digits, digits) if digits == max_benchmark_digits else ""
        else:
            pi_string, terms = compute_pi_string_decimal(digits)
            prefix_matches_reference = pi_string.startswith(PI_PREFIX_REFERENCE)
        elapsed = time.perf_counter() - start
        benchmark_rows.append(
            {
                "digits": digits,
                "terms": terms,
                "seconds": elapsed,
                "digits_per_second": digits / elapsed,
                "backend": backend,
                "workers_used": workers,
                "chunk_terms": chunk_terms,
                "gpu_used": False,
                "prefix_matches_reference": prefix_matches_reference,
            }
        )
        if digits == max_benchmark_digits:
            highest_digits_string = pi_string
            highest_digits = digits
            highest_workers = workers
            highest_chunk_terms = chunk_terms

    write_csv(RESULT_DIR / "project2_pi_benchmark.csv", benchmark_rows)

    output_name = f"project2_pi_{highest_digits}_digits.txt"
    with (RESULT_DIR / output_name).open("w", encoding="utf-8") as handle:
        handle.write(highest_digits_string)
        handle.write("\n")

    return {
        "benchmark_rows": benchmark_rows,
        "backend": backend,
        "workers": highest_workers,
        "chunk_terms": highest_chunk_terms,
        "highest_digits": highest_digits,
        "output_name": output_name,
        "gpu_used": False,
    }


def solve_project2() -> dict[str, object]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    use_gmpy2 = gmpy2 is not None
    benchmark_digits = resolve_project2_benchmark_digits(use_gmpy2)
    backend_mode = resolve_project2_homework_backend()
    cpp_available = has_project2_cpp_backend()

    if backend_mode == "gpu":
        if can_use_gpu_hybrid():
            return solve_project2_gpu_hybrid(benchmark_digits)
        raise RuntimeError("PROJECT2_HOMEWORK_BACKEND=gpu was requested, but CUDA is not available")

    if backend_mode == "cpu":
        if cpp_available:
            return solve_project2_cpp_backend(benchmark_digits)
        return solve_project2_python_backend(use_gmpy2, benchmark_digits)

    if cpp_available:
        return solve_project2_cpp_backend(benchmark_digits)
    if use_gmpy2:
        return solve_project2_python_backend(use_gmpy2, benchmark_digits)
    if can_use_gpu_hybrid():
        return solve_project2_gpu_hybrid(benchmark_digits)
    return solve_project2_python_backend(use_gmpy2, benchmark_digits)
