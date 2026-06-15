from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .cuda_chunk_ops import (
    add_abs_base65536,
    compare_abs_base65536,
    cuda_chunk_ops_available,
    div2_base65536,
    mul_small_base65536,
    sub_abs_base65536,
    trim_chunks_base65536,
)
from .gpu_fft_backend import GpuFftMultiplier


RESULT_DIR = Path(__file__).resolve().parent.parent / "result"
BINARY_CHUNK_BASE = 1 << 16


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def random_normalized_chunks(length: int, *, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    chunks = torch.randint(0, BINARY_CHUNK_BASE, (length,), dtype=torch.int64, generator=generator)
    chunks[-1] = max(1, int(chunks[-1].item()))
    return chunks.to(device=device)


@dataclass
class FullCudaPrototypeStats:
    chunk_length: int
    repeats: int
    add_seconds: float
    sub_seconds: float
    mul_small_seconds: float
    div2_seconds: float
    compare_seconds: float
    fft_mul_seconds: float
    total_seconds: float
    extension_loaded: bool
    device: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class FullCudaPipelineStats:
    digits: int
    workers: int
    chunk_terms: int
    bs_leaf_terms: int
    terms: int
    total_seconds: float
    digits_per_second: float
    partial_generation_seconds: float
    chunk_conversion_seconds: float
    merge_tree_seconds: float
    sqrt_compute_seconds: float
    sqrt_scale_seconds: float
    sqrt_conversion_seconds: float
    final_division_seconds: float
    final_chunk_to_mpz_seconds: float
    newton_reciprocal_seconds: float
    newton_qd_multiply_seconds: float
    newton_correction_seconds: float
    newton_reciprocal_iterations: int
    newton_correction_digits: int
    newton_cpu_correction_used: bool
    newton_chunk_fastpath_used: bool
    gpu_calls: int
    merge_gpu_calls: int
    final_gpu_calls: int
    gpu_prepare_seconds: float
    gpu_backend_total_seconds: float
    gpu_kernel_seconds: float
    gpu_finalize_seconds: float
    gpu_peak_gb: float
    prefix_ok: bool
    extension_loaded: bool
    device: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class FullCudaRuntime:
    def __init__(self, device: str = "cuda") -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        self.device = torch.device(device)
        self._multiplier: GpuFftMultiplier | None = None

    def _ensure_multiplier(self) -> GpuFftMultiplier:
        if self._multiplier is None:
            self._multiplier = GpuFftMultiplier(decimal_base=BINARY_CHUNK_BASE, split_base=1 << 8)
        return self._multiplier

    def benchmark_arith(self, chunk_length: int, repeats: int, seed: int) -> FullCudaPrototypeStats:
        extension_loaded = cuda_chunk_ops_available()
        if not extension_loaded:
            raise RuntimeError("project2 CUDA chunk ops extension failed to load")

        left = random_normalized_chunks(chunk_length, seed=seed, device=self.device)
        right = random_normalized_chunks(chunk_length, seed=seed + 1, device=self.device)

        add_elapsed = 0.0
        sub_elapsed = 0.0
        mul_small_elapsed = 0.0
        div2_elapsed = 0.0
        compare_elapsed = 0.0
        fft_mul_elapsed = 0.0

        add_result = None
        for _ in range(repeats):
            torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            add_result = add_abs_base65536(left, right)
            torch.cuda.synchronize(self.device)
            add_elapsed += time.perf_counter() - start

        larger = add_result if add_result is not None else add_abs_base65536(left, right)
        smaller = trim_chunks_base65536(left)

        for _ in range(repeats):
            torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            sub_abs_base65536(larger, smaller)
            torch.cuda.synchronize(self.device)
            sub_elapsed += time.perf_counter() - start

        for _ in range(repeats):
            torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            mul_small_base65536(left, 426880)
            torch.cuda.synchronize(self.device)
            mul_small_elapsed += time.perf_counter() - start

        for _ in range(repeats):
            torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            div2_base65536(left)
            torch.cuda.synchronize(self.device)
            div2_elapsed += time.perf_counter() - start

        for _ in range(repeats):
            torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            compare_abs_base65536(left, right)
            torch.cuda.synchronize(self.device)
            compare_elapsed += time.perf_counter() - start

        for _ in range(repeats):
            torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            self._ensure_multiplier().multiply_chunks(
                left,
                right,
                output_device="cuda",
                assume_nonzero_trimmed=True,
                a_length=chunk_length,
                b_length=chunk_length,
            )
            torch.cuda.synchronize(self.device)
            fft_mul_elapsed += time.perf_counter() - start

        total_seconds = add_elapsed + sub_elapsed + mul_small_elapsed + div2_elapsed + compare_elapsed + fft_mul_elapsed
        return FullCudaPrototypeStats(
            chunk_length=chunk_length,
            repeats=repeats,
            add_seconds=add_elapsed,
            sub_seconds=sub_elapsed,
            mul_small_seconds=mul_small_elapsed,
            div2_seconds=div2_elapsed,
            compare_seconds=compare_elapsed,
            fft_mul_seconds=fft_mul_elapsed,
            total_seconds=total_seconds,
            extension_loaded=extension_loaded,
            device=torch.cuda.get_device_name(self.device),
        )

    def benchmark_pipeline(
        self,
        digits: int,
        workers: int | None,
        chunk_terms: int | None,
        bs_leaf_terms: int,
        output_path: Path | None,
    ) -> FullCudaPipelineStats:
        from . import gpu_pi_hybrid as hybrid

        resolved_workers, resolved_chunk_terms = hybrid.resolve_parallel_config(digits)
        if workers is not None:
            resolved_workers = workers
        if chunk_terms is not None:
            resolved_chunk_terms = chunk_terms

        hybrid.set_chunk_arith_backend("cuda-ext")
        extension_loaded = True
        if resolved_workers <= 1:
            extension_loaded = cuda_chunk_ops_available()
            if not extension_loaded:
                raise RuntimeError("project2 CUDA chunk ops extension failed to load")

        start = time.perf_counter()
        pi_digits, terms, hybrid_stats, phase_stats = hybrid.compute_pi_digits_integer_gpu_hybrid(
            digits=digits,
            workers=resolved_workers,
            chunk_terms=resolved_chunk_terms,
            bs_leaf_terms=bs_leaf_terms,
            gpu_threshold_digits=1,
            gpu_stages="merge-and-final",
            gpu_chunk_format="binary16",
            sqrt_mode="chunk-gpu-rsqrt-prototype",
            division_mode="newton-chunk-gpu-seed-prototype",
        )
        total_seconds = time.perf_counter() - start

        prefix_ok = hybrid.pi_prefix_matches_reference(pi_digits, digits)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(hybrid.format_pi_digits_integer(pi_digits, digits) + "\n", encoding="utf-8")

        digits_per_second = digits / total_seconds if total_seconds > 0 else 0.0
        return FullCudaPipelineStats(
            digits=digits,
            workers=resolved_workers,
            chunk_terms=resolved_chunk_terms,
            bs_leaf_terms=bs_leaf_terms,
            terms=terms,
            total_seconds=total_seconds,
            digits_per_second=digits_per_second,
            partial_generation_seconds=phase_stats.partial_generation_seconds,
            chunk_conversion_seconds=phase_stats.chunk_conversion_seconds,
            merge_tree_seconds=phase_stats.merge_tree_seconds,
            sqrt_compute_seconds=phase_stats.sqrt_compute_seconds,
            sqrt_scale_seconds=phase_stats.sqrt_scale_seconds,
            sqrt_conversion_seconds=phase_stats.sqrt_conversion_seconds,
            final_division_seconds=phase_stats.final_division_seconds,
            final_chunk_to_mpz_seconds=phase_stats.final_chunk_to_mpz_seconds,
            newton_reciprocal_seconds=phase_stats.newton_reciprocal_seconds,
            newton_qd_multiply_seconds=phase_stats.newton_qd_multiply_seconds,
            newton_correction_seconds=phase_stats.newton_correction_seconds,
            newton_reciprocal_iterations=phase_stats.newton_reciprocal_iterations,
            newton_correction_digits=phase_stats.newton_correction_digits,
            newton_cpu_correction_used=phase_stats.newton_cpu_correction_used,
            newton_chunk_fastpath_used=phase_stats.newton_chunk_fastpath_used,
            gpu_calls=hybrid_stats.gpu_calls,
            merge_gpu_calls=hybrid_stats.merge_gpu_calls,
            final_gpu_calls=hybrid_stats.final_gpu_calls,
            gpu_prepare_seconds=hybrid_stats.gpu_prepare_seconds,
            gpu_backend_total_seconds=hybrid_stats.gpu_backend_total_seconds,
            gpu_kernel_seconds=hybrid_stats.gpu_kernel_seconds,
            gpu_finalize_seconds=hybrid_stats.gpu_finalize_seconds,
            gpu_peak_gb=hybrid_stats.gpu_peak_gb,
            prefix_ok=prefix_ok,
            extension_loaded=extension_loaded,
            device=torch.cuda.get_device_name(self.device),
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full-CUDA prototype runtime and stage-2 Pi smoke pipeline for Project 2.")
    parser.add_argument("--mode", choices=["microbench", "pipeline"], default="microbench")
    parser.add_argument("--chunk-length", type=int, default=1 << 18)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--digits", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunk-terms", type=int, default=None)
    parser.add_argument("--bs-leaf-terms", type=int, default=8)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    runtime = FullCudaRuntime()

    if args.mode == "microbench":
        stats = runtime.benchmark_arith(
            chunk_length=args.chunk_length,
            repeats=args.repeats,
            seed=args.seed,
        )
        csv_path = Path(args.csv) if args.csv else RESULT_DIR / "project2_full_cuda_prototype.csv"
        write_csv(csv_path, [stats.as_dict()])
        print(
            "full_cuda_prototype",
            f"chunk_length={stats.chunk_length}",
            f"repeats={stats.repeats}",
            f"add_seconds={stats.add_seconds:.6f}",
            f"sub_seconds={stats.sub_seconds:.6f}",
            f"mul_small_seconds={stats.mul_small_seconds:.6f}",
            f"div2_seconds={stats.div2_seconds:.6f}",
            f"compare_seconds={stats.compare_seconds:.6f}",
            f"fft_mul_seconds={stats.fft_mul_seconds:.6f}",
            f"total_seconds={stats.total_seconds:.6f}",
            f"device={stats.device}",
        )
        return

    output_path = Path(args.output) if args.output else None
    stats = runtime.benchmark_pipeline(
        digits=args.digits,
        workers=args.workers,
        chunk_terms=args.chunk_terms,
        bs_leaf_terms=args.bs_leaf_terms,
        output_path=output_path,
    )
    csv_path = Path(args.csv) if args.csv else RESULT_DIR / "project2_full_cuda_pipeline_smoke.csv"
    write_csv(csv_path, [stats.as_dict()])
    print(
        "full_cuda_pipeline",
        f"digits={stats.digits}",
        f"workers={stats.workers}",
        f"chunk_terms={stats.chunk_terms}",
        f"terms={stats.terms}",
        f"total_seconds={stats.total_seconds:.6f}",
        f"digits_per_second={stats.digits_per_second:.6f}",
        f"partial_generation_seconds={stats.partial_generation_seconds:.6f}",
        f"merge_tree_seconds={stats.merge_tree_seconds:.6f}",
        f"sqrt_compute_seconds={stats.sqrt_compute_seconds:.6f}",
        f"sqrt_scale_seconds={stats.sqrt_scale_seconds:.6f}",
        f"final_division_seconds={stats.final_division_seconds:.6f}",
        f"gpu_backend_total_seconds={stats.gpu_backend_total_seconds:.6f}",
        f"gpu_kernel_seconds={stats.gpu_kernel_seconds:.6f}",
        f"prefix_ok={stats.prefix_ok}",
        f"device={stats.device}",
    )


if __name__ == "__main__":
    main()
