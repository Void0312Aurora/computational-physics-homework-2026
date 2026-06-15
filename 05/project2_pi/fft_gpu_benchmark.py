from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import gmpy2

from .gpu_fft_backend import (
    GpuFftMultiplier,
    chunks_to_decimal_string,
    parse_digits_list,
    random_chunks_for_digits,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def benchmark_case(multiplier: GpuFftMultiplier, digits: int, repeats: int, seed: int) -> dict[str, object]:
    setup_start = time.perf_counter()
    lhs_chunks = random_chunks_for_digits(digits, seed)
    rhs_chunks = random_chunks_for_digits(digits, seed * 3 + 11)
    setup_seconds = time.perf_counter() - setup_start

    if repeats <= 0:
        _, probe_stats = multiplier.multiply_chunks(lhs_chunks, rhs_chunks)
        repeats = max(1, min(4, math.ceil(0.40 / max(probe_stats.total_seconds, 1e-6))))

    totals: dict[str, float] = {
        "split_seconds": 0.0,
        "upload_seconds": 0.0,
        "kernel_seconds": 0.0,
        "download_seconds": 0.0,
        "combine_seconds": 0.0,
        "normalize_seconds": 0.0,
        "total_seconds": 0.0,
        "carry_passes": 0.0,
        "peak_gpu_gb": 0.0,
    }
    fft_length = 0
    valid_length = 0
    device = ""

    for _ in range(repeats):
        _, stats = multiplier.multiply_chunks(lhs_chunks, rhs_chunks)
        fft_length = stats.fft_length
        valid_length = stats.valid_length
        device = stats.device
        totals["split_seconds"] += stats.split_seconds
        totals["upload_seconds"] += stats.upload_seconds
        totals["kernel_seconds"] += stats.kernel_seconds
        totals["download_seconds"] += stats.download_seconds
        totals["combine_seconds"] += stats.combine_seconds
        totals["normalize_seconds"] += stats.normalize_seconds
        totals["total_seconds"] += stats.total_seconds
        totals["carry_passes"] += stats.carry_passes
        totals["peak_gpu_gb"] = max(totals["peak_gpu_gb"], stats.peak_gpu_gb)

    averages = {key: value / repeats for key, value in totals.items()}
    return {
        "decimal_digits": digits,
        "chunks": lhs_chunks.numel(),
        "valid_length": valid_length,
        "fft_length": fft_length,
        "repeats": repeats,
        "setup_seconds": setup_seconds,
        "split_seconds": averages["split_seconds"],
        "upload_seconds": averages["upload_seconds"],
        "kernel_seconds": averages["kernel_seconds"],
        "kernel_digits_per_second": digits / averages["kernel_seconds"],
        "download_seconds": averages["download_seconds"],
        "combine_seconds": averages["combine_seconds"],
        "normalize_seconds": averages["normalize_seconds"],
        "total_seconds": averages["total_seconds"],
        "end_to_end_digits_per_second": digits / averages["total_seconds"],
        "carry_passes": averages["carry_passes"],
        "peak_gpu_gb": totals["peak_gpu_gb"],
        "decimal_base": multiplier.decimal_base,
        "split_base": multiplier.split_base,
        "device": device,
    }


def verify_case(multiplier: GpuFftMultiplier, digits: int, seed: int) -> dict[str, object]:
    lhs_chunks = random_chunks_for_digits(digits, seed)
    rhs_chunks = random_chunks_for_digits(digits, seed * 5 + 17)

    result_chunks, backend_stats = multiplier.multiply_chunks(lhs_chunks, rhs_chunks)

    lhs_text = chunks_to_decimal_string(lhs_chunks)
    rhs_text = chunks_to_decimal_string(rhs_chunks)
    gmp_start = time.perf_counter()
    exact = gmpy2.mpz(lhs_text) * gmpy2.mpz(rhs_text)
    gmp_seconds = time.perf_counter() - gmp_start

    result_text = chunks_to_decimal_string(result_chunks)
    got = gmpy2.mpz(result_text)
    return {
        "decimal_digits": digits,
        "chunks": lhs_chunks.numel(),
        "fft_length": backend_stats.fft_length,
        "backend_total_seconds": backend_stats.total_seconds,
        "gmp_reference_seconds": gmp_seconds,
        "carry_passes": backend_stats.carry_passes,
        "exact_match": exact == got,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the formal exact GPU FFT multiplication backend.")
    parser.add_argument("--digits-list", default="1000000,10000000,50000000,100000000")
    parser.add_argument("--verify-digits-list", default="10000,100000,500000,1000000,5000000")
    parser.add_argument("--repeats", type=int, default=0)
    parser.add_argument("--csv", default="result/project2_fft_gpu_benchmark.csv")
    parser.add_argument("--verify-csv", default="result/project2_fft_gpu_verify.csv")
    parser.add_argument("--seed", type=int, default=20260415)
    args = parser.parse_args()

    multiplier = GpuFftMultiplier()
    warmup_a = random_chunks_for_digits(1000, args.seed)
    warmup_b = random_chunks_for_digits(1000, args.seed + 1)
    multiplier.multiply_chunks(warmup_a, warmup_b)

    benchmark_rows = []
    for digits in parse_digits_list(args.digits_list):
        row = benchmark_case(multiplier, digits, args.repeats, args.seed + digits)
        benchmark_rows.append(row)
        print(
            "benchmark",
            f"digits={row['decimal_digits']}",
            f"chunks={row['chunks']}",
            f"fft_length={row['fft_length']}",
            f"repeats={row['repeats']}",
            f"kernel_seconds={row['kernel_seconds']:.6f}",
            f"kernel_digits_per_second={row['kernel_digits_per_second']:.6f}",
            f"normalize_seconds={row['normalize_seconds']:.6f}",
            f"total_seconds={row['total_seconds']:.6f}",
            f"end_to_end_digits_per_second={row['end_to_end_digits_per_second']:.6f}",
            f"carry_passes={row['carry_passes']:.2f}",
            f"peak_gpu_gb={row['peak_gpu_gb']:.6f}",
            f"device={row['device']}",
        )

    verify_rows = []
    for digits in parse_digits_list(args.verify_digits_list):
        row = verify_case(multiplier, digits, args.seed + digits * 13)
        verify_rows.append(row)
        print(
            "verify",
            f"digits={row['decimal_digits']}",
            f"chunks={row['chunks']}",
            f"fft_length={row['fft_length']}",
            f"backend_total_seconds={row['backend_total_seconds']:.6f}",
            f"gmp_reference_seconds={row['gmp_reference_seconds']:.6f}",
            f"exact_match={row['exact_match']}",
        )

    write_csv(Path(args.csv), benchmark_rows)
    write_csv(Path(args.verify_csv), verify_rows)


if __name__ == "__main__":
    main()
