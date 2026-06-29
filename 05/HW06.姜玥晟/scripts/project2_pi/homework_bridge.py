from __future__ import annotations

import os
import time
from decimal import Decimal, getcontext
from multiprocessing import get_all_start_methods, get_context

from result_paths import (
    PROJECT2_CPU_BENCHMARK_DIR,
    PROJECT2_OUTPUT_DIR,
    ensure_result_dir,
    write_csv,
)

try:
    import gmpy2
except ImportError:
    gmpy2 = None


C3_OVER_24 = 640320**3 // 24
PI_PREFIX_REFERENCE = "3.14159265358979323846264338327950288419716939937510"
PI_PREFIX_REFERENCE_NO_DOT = PI_PREFIX_REFERENCE.replace(".", "")
PROJECT2_GUARD_DIGITS = 48
PROJECT2_DEFAULT_CHUNK_TERMS = 8192
PROJECT2_DEFAULT_BENCHMARK_DIGITS = [100000, 1000000, 10000000]
PROJECT2_CSV_FIELDS = [
    "digits",
    "terms",
    "seconds",
    "digits_per_second",
    "backend",
    "workers_used",
    "chunk_terms",
    "gpu_used",
    "prefix_matches_reference",
]


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


def estimate_chudnovsky_term_count(
    digits: int,
    guard_digits: int = PROJECT2_GUARD_DIGITS,
) -> int:
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


def chudnovsky_bs_parallel_mpz(
    term_count: int,
    workers: int,
    chunk_terms: int,
) -> tuple[object, object, object]:
    if workers <= 1 or term_count <= chunk_terms:
        return chudnovsky_bs_mpz(0, term_count)

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < term_count:
        end = min(start + chunk_terms, term_count)
        ranges.append((start, end))
        start = end

    ctx = get_context("fork") if "fork" in get_all_start_methods() else get_context()
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


def format_pi_digits_integer(pi_digits: object, digits: int) -> str:
    raw = pi_digits.digits(10) if hasattr(pi_digits, "digits") else str(pi_digits)
    expected = digits + 1
    if len(raw) < expected:
        raw = "0" * (expected - len(raw)) + raw
    return raw[0] + "." + raw[1:]


def pi_prefix_matches_reference(pi_digits: object, digits: int) -> bool:
    candidate = format_pi_digits_integer(pi_digits, digits)
    if digits < len(PI_PREFIX_REFERENCE_NO_DOT) - 1:
        return candidate == PI_PREFIX_REFERENCE[: len(candidate)]
    trim_power = digits + 1 - len(PI_PREFIX_REFERENCE_NO_DOT)
    leading = pi_digits // (gmpy2.mpz(10) ** trim_power) if trim_power > 0 else pi_digits
    return str(leading) == PI_PREFIX_REFERENCE_NO_DOT


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


def validate_project2_backend_mode() -> None:
    backend_mode = os.environ.get("PROJECT2_HOMEWORK_BACKEND", "cpu").strip().lower()
    if backend_mode not in {"auto", "cpu"}:
        raise RuntimeError(
            "This cleaned submission package keeps only the CPU Python/gmpy2 Project2 runner. "
            "Historical C++/GPU benchmark artifacts remain under results/project2."
        )


def solve_project2() -> dict[str, object]:
    validate_project2_backend_mode()
    use_gmpy2 = gmpy2 is not None
    digits_list = resolve_project2_benchmark_digits(use_gmpy2)
    backend = "gmpy2_parallel_mpz" if use_gmpy2 else "decimal_binary_splitting"
    highest_digits = max(digits_list)
    highest_digits_string = ""
    highest_workers = 1
    highest_chunk_terms = 0
    benchmark_rows: list[dict[str, object]] = []

    ensure_result_dir(PROJECT2_OUTPUT_DIR, PROJECT2_CPU_BENCHMARK_DIR)

    for digits in digits_list:
        workers, chunk_terms = resolve_project2_parallel_config(digits) if use_gmpy2 else (1, 0)
        start = time.perf_counter()
        if use_gmpy2:
            pi_digits, terms = compute_pi_digits_integer_gmpy2(
                digits,
                workers=workers,
                chunk_terms=chunk_terms,
            )
            prefix_matches_reference = pi_prefix_matches_reference(pi_digits, digits)
            pi_string = format_pi_digits_integer(pi_digits, digits) if digits == highest_digits else ""
        else:
            pi_string, terms = compute_pi_string_decimal(digits)
            prefix_matches_reference = pi_string == PI_PREFIX_REFERENCE[: len(pi_string)]
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

        if digits == highest_digits:
            highest_digits_string = pi_string
            highest_workers = workers
            highest_chunk_terms = chunk_terms

    write_csv(
        PROJECT2_CPU_BENCHMARK_DIR / "project2_pi_benchmark.csv",
        PROJECT2_CSV_FIELDS,
        benchmark_rows,
    )

    output_name = f"project2_pi_{highest_digits}_digits.txt"
    with (PROJECT2_OUTPUT_DIR / output_name).open("w", encoding="utf-8") as handle:
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
