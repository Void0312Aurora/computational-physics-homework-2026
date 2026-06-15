from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, TypeVar

import gmpy2
import numpy as np
import torch

from .gpu_fft_backend import (
    GpuFftMultiplier,
    chunks_to_decimal_string,
    decimal_string_to_chunks,
    estimate_fft_working_set_gb,
)


C3_OVER_24 = 640320**3 // 24
PI_PREFIX_REFERENCE = "3.14159265358979323846264338327950288419716939937510"
PI_PREFIX_REFERENCE_NO_DOT = PI_PREFIX_REFERENCE.replace(".", "")
PROJECT2_GUARD_DIGITS = 48
PROJECT2_BS_LEAF_TERMS = 8
PROJECT2_PARTIAL_TARGET_TASKS = 24
PROJECT2_MIN_CHUNK_TERMS = 8192
LOG10_2 = math.log10(2.0)
LOG2_10 = math.log2(10.0)
RESULT_DIR = Path(__file__).resolve().parent.parent / "result"
BINARY_CHUNK_BASE = 1 << 16
BINARY_CHUNK_SPLIT_BASE = 1 << 8
PROJECT2_CHUNK_ARITH_BACKEND = os.environ.get("PROJECT2_CHUNK_ARITH_BACKEND", "auto")
PROJECT2_CUDA_CHUNK_ARITH_SMALL_LEN = 65_536
PROJECT2_CUDA_CHUNK_COMPARE_LEN = 16_777_216
PROJECT2_GPU_SEED_SIGNIFICANT_CHUNKS = 3
PROJECT2_GPU_SEED_TOP_SOURCE_CHUNKS = 4
_PROJECT2_CUDA_CHUNK_OPS_MODULE = None
_PROJECT2_CUDA_CHUNK_OPS_READY = None
PROJECT2_HYBRID_PROFILES: dict[str, dict[str, object]] = {
    "legacy-default": {
        "gpu_threshold_digits": 50_000,
        "gpu_stages": "final-only",
        "gpu_chunk_format": "binary16",
        "sqrt_mode": "mpz-isqrt",
        "division_mode": "mpz-div",
        "chunk_arith_backend": "python",
        "gpu_memory_budget_gb": None,
    },
    "fast-python": {
        "gpu_threshold_digits": 50_000,
        "gpu_stages": "merge-and-final",
        "gpu_chunk_format": "binary16",
        "sqrt_mode": "chunk-gpu-rsqrt-prototype",
        "division_mode": "auto",
        "chunk_arith_backend": "python",
        "gpu_memory_budget_gb": None,
    },
    "fast-auto": {
        "gpu_threshold_digits": 50_000,
        "gpu_stages": "merge-and-final",
        "gpu_chunk_format": "binary16",
        "sqrt_mode": "chunk-gpu-rsqrt-prototype",
        "division_mode": "auto",
        "chunk_arith_backend": "auto",
        "gpu_memory_budget_gb": None,
    },
    "high-digits-auto": {
        "gpu_threshold_digits": 50_000,
        "gpu_stages": "merge-and-final",
        "gpu_chunk_format": "binary16",
        "sqrt_mode": "chunk-gpu-rsqrt-prototype",
        "division_mode": "auto",
        "chunk_arith_backend": "auto",
        "gpu_memory_budget_gb": 12.0,
        "stream_partials": False,
    },
    "bounded-stream-auto": {
        "gpu_threshold_digits": 50_000,
        "gpu_stages": "merge-and-final",
        "gpu_chunk_format": "binary16",
        "sqrt_mode": "chunk-gpu-rsqrt-prototype",
        "division_mode": "auto",
        "chunk_arith_backend": "auto",
        "gpu_memory_budget_gb": 10.0,
        "stream_partials": True,
    },
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_digits_list(text: str) -> list[int]:
    values = [int(token) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("digits list must not be empty")
    return values


def set_chunk_arith_backend(backend: str) -> None:
    global PROJECT2_CHUNK_ARITH_BACKEND
    if backend not in {"python", "cuda-ext", "auto"}:
        raise ValueError(f"unsupported chunk arithmetic backend: {backend}")
    PROJECT2_CHUNK_ARITH_BACKEND = backend


def load_cuda_chunk_ops_module():
    global _PROJECT2_CUDA_CHUNK_OPS_MODULE, _PROJECT2_CUDA_CHUNK_OPS_READY
    if _PROJECT2_CUDA_CHUNK_OPS_READY is True:
        return _PROJECT2_CUDA_CHUNK_OPS_MODULE
    if _PROJECT2_CUDA_CHUNK_OPS_READY is False:
        return None
    try:
        from . import cuda_chunk_ops as chunk_ops

        chunk_ops.load_project2_cuda_chunk_ops()
    except Exception:
        _PROJECT2_CUDA_CHUNK_OPS_MODULE = None
        _PROJECT2_CUDA_CHUNK_OPS_READY = False
        return None

    _PROJECT2_CUDA_CHUNK_OPS_MODULE = chunk_ops
    _PROJECT2_CUDA_CHUNK_OPS_READY = True
    return chunk_ops


def ensure_cuda_chunk_ops_ready(required: bool = False) -> bool:
    module = load_cuda_chunk_ops_module()
    if module is None and required:
        raise RuntimeError("failed to load project2 CUDA chunk ops extension")
    return module is not None


def should_use_cuda_chunk_ops(op_name: str, base: int, *tensors: torch.Tensor) -> bool:
    if PROJECT2_CHUNK_ARITH_BACKEND == "python":
        return False
    if base != BINARY_CHUNK_BASE:
        return False
    if not tensors or any(tensor.device.type != "cuda" for tensor in tensors):
        return False
    if load_cuda_chunk_ops_module() is None:
        return False
    if PROJECT2_CHUNK_ARITH_BACKEND == "cuda-ext":
        return True
    if PROJECT2_CHUNK_ARITH_BACKEND != "auto":
        return False

    max_len = max(int(tensor.numel()) for tensor in tensors)
    if op_name == "compare":
        return max_len <= PROJECT2_CUDA_CHUNK_COMPARE_LEN
    return max_len <= PROJECT2_CUDA_CHUNK_ARITH_SMALL_LEN


def estimate_chudnovsky_term_count(digits: int, guard_digits: int = PROJECT2_GUARD_DIGITS) -> int:
    return (digits + guard_digits) // 14 + 1


def nearest_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    lower = 1 << (value.bit_length() - 1)
    upper = lower << 1
    if value == lower:
        return lower
    return lower if value - lower <= upper - value else upper


def estimate_decimal_digits(value: gmpy2.mpz) -> int:
    if value == 0:
        return 1
    return int((value.bit_length() - 1) * LOG10_2) + 1


def decimal_digits_to_bits(decimal_digits: int, extra_bits: int = 0) -> int:
    return max(128, int(math.ceil(decimal_digits * LOG2_10)) + extra_bits)


def format_pi_digits_integer(pi_digits: gmpy2.mpz, digits: int) -> str:
    raw = pi_digits.digits(10)
    expected = digits + 1
    if len(raw) < expected:
        raw = "0" * (expected - len(raw)) + raw
    return raw[0] + "." + raw[1:]


def pi_prefix_matches_reference(pi_digits: gmpy2.mpz, digits: int) -> bool:
    if digits < len(PI_PREFIX_REFERENCE_NO_DOT) - 1:
        return format_pi_digits_integer(pi_digits, digits).startswith(PI_PREFIX_REFERENCE)
    trim_power = digits + 1 - len(PI_PREFIX_REFERENCE_NO_DOT)
    leading = pi_digits // (gmpy2.mpz(10) ** trim_power) if trim_power > 0 else pi_digits
    return str(leading) == PI_PREFIX_REFERENCE_NO_DOT


def mpz_to_binary16_chunks(
    value: gmpy2.mpz,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if value == 0:
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        return torch.tensor([0], dtype=torch.int64, device=target_device)

    byte_length = max(1, (value.bit_length() + 7) // 8)
    raw = value.to_bytes(byte_length, "little")
    if len(raw) & 1:
        raw += b"\x00"

    int64_view = np.frombuffer(raw, dtype="<u2").astype(np.int64)
    chunks = torch.from_numpy(int64_view)
    if device is not None:
        chunks = chunks.to(device=torch.device(device), dtype=torch.int64)
    return chunks


def binary16_chunks_to_mpz(chunks: torch.Tensor, sign: int = 1) -> gmpy2.mpz:
    normalized = trim_chunks(chunks, device="cpu")
    if normalized.numel() == 1 and int(normalized[0].item()) == 0:
        return gmpy2.mpz(0)
    uint16_values = normalized.numpy().astype("<u2", copy=False)
    raw = uint16_values.tobytes().rstrip(b"\x00") or b"\x00"
    value = gmpy2.mpz.from_bytes(raw, "little")
    return -value if sign < 0 else value


def chudnovsky_single_term_mpz(a: int) -> tuple[gmpy2.mpz, gmpy2.mpz, gmpy2.mpz]:
    if a == 0:
        one = gmpy2.mpz(1)
        return one, one, gmpy2.mpz(13591409)
    a_mpz = gmpy2.mpz(a)
    p = (6 * a_mpz - 5) * (2 * a_mpz - 1) * (6 * a_mpz - 1)
    q = a_mpz * a_mpz * a_mpz * C3_OVER_24
    t = p * (13591409 + 545140134 * a_mpz)
    if a & 1:
        t = -t
    return p, q, t


def chudnovsky_bs_mpz_block(a: int, b: int) -> tuple[gmpy2.mpz, gmpy2.mpz, gmpy2.mpz]:
    p_total = gmpy2.mpz(1)
    q_total = gmpy2.mpz(1)
    t_total = gmpy2.mpz(0)
    for index in range(a, b):
        p_term, q_term, t_term = chudnovsky_single_term_mpz(index)
        t_total = t_total * q_term + p_total * t_term
        p_total *= p_term
        q_total *= q_term
    return p_total, q_total, t_total


def chudnovsky_bs_mpz(
    a: int,
    b: int,
    leaf_terms: int = PROJECT2_BS_LEAF_TERMS,
) -> tuple[gmpy2.mpz, gmpy2.mpz, gmpy2.mpz]:
    if b - a <= leaf_terms:
        return chudnovsky_bs_mpz_block(a, b)

    m = (a + b) // 2
    p1, q1, t1 = chudnovsky_bs_mpz(a, m, leaf_terms=leaf_terms)
    p2, q2, t2 = chudnovsky_bs_mpz(m, b, leaf_terms=leaf_terms)
    return p1 * p2, q1 * q2, t1 * q2 + p1 * t2


def project2_bs_worker(work_range: tuple[int, int, int]) -> tuple[gmpy2.mpz, gmpy2.mpz, gmpy2.mpz]:
    start, end, leaf_terms = work_range
    return chudnovsky_bs_mpz(start, end, leaf_terms=leaf_terms)


def build_partial_ranges(term_count: int, chunk_terms: int, leaf_terms: int) -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    start = 0
    while start < term_count:
        end = min(start + chunk_terms, term_count)
        ranges.append((start, end, leaf_terms))
        start = end
    return ranges


def compute_sqrt_term_mpfr_exact(total_digits: int, extra_bits: int = 256) -> gmpy2.mpz:
    precision_bits = decimal_digits_to_bits(total_digits + 8, extra_bits=extra_bits)
    ctx = gmpy2.get_context().copy()
    ctx.precision = precision_bits
    ctx.emax = gmpy2.get_emax_max()
    ctx.emin = gmpy2.get_emin_min()
    ten_power = gmpy2.mpz(10) ** total_digits
    with gmpy2.context(ctx):
        sqrt_scaled = gmpy2.sqrt(gmpy2.mpfr(10005)) * gmpy2.mpfr(ten_power)
        candidate = gmpy2.mpz(gmpy2.floor(sqrt_scaled))

    target = gmpy2.mpz(10005) * ten_power * ten_power
    while (candidate + 1) * (candidate + 1) <= target:
        candidate += 1
    while candidate * candidate > target:
        candidate -= 1
    return candidate


def exact_division_via_mpfr(
    numerator: gmpy2.mpz,
    denominator: gmpy2.mpz,
    quotient_digits_hint: int,
    extra_bits: int = 256,
) -> tuple[gmpy2.mpz, bool]:
    precision_bits = decimal_digits_to_bits(quotient_digits_hint + 8, extra_bits=extra_bits)
    ctx = gmpy2.get_context().copy()
    ctx.precision = precision_bits
    ctx.emax = gmpy2.get_emax_max()
    ctx.emin = gmpy2.get_emin_min()
    try:
        with gmpy2.context(ctx):
            quotient = gmpy2.mpz(gmpy2.floor(gmpy2.mpfr(numerator) / gmpy2.mpfr(denominator)))
    except OverflowError:
        return numerator // denominator, True

    product = quotient * denominator
    while product > numerator:
        quotient -= 1
        product -= denominator
    while product + denominator <= numerator:
        quotient += 1
        product += denominator
    return quotient, False


@dataclass
class HybridMulStats:
    gpu_calls: int = 0
    cpu_calls: int = 0
    gpu_prepare_seconds: float = 0.0
    gpu_backend_total_seconds: float = 0.0
    gpu_kernel_seconds: float = 0.0
    gpu_finalize_seconds: float = 0.0
    gpu_peak_gb: float = 0.0
    gpu_max_operand_digits: int = 0
    merge_gpu_calls: int = 0
    final_gpu_calls: int = 0
    budget_cpu_calls: int = 0
    oom_cpu_calls: int = 0
    budget_max_operand_digits: int = 0
    budget_max_estimated_peak_gb: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class PipelinePhaseStats:
    partial_generation_seconds: float = 0.0
    sqrt_compute_seconds: float = 0.0
    sqrt_scale_seconds: float = 0.0
    chunk_conversion_seconds: float = 0.0
    merge_tree_seconds: float = 0.0
    merge_frontier_max_nodes: int = 0
    partial_stream_seconds: float = 0.0
    streaming_partials_used: bool = False
    sqrt_conversion_seconds: float = 0.0
    final_chunk_to_mpz_seconds: float = 0.0
    final_division_seconds: float = 0.0
    resolved_division_mode: str = ""
    division_fallback_used: bool = False
    newton_reciprocal_seconds: float = 0.0
    newton_qd_multiply_seconds: float = 0.0
    newton_correction_seconds: float = 0.0
    newton_reciprocal_iterations: int = 0
    newton_correction_digits: int = 0
    newton_cpu_correction_used: bool = False
    newton_chunk_fastpath_used: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ChunkInteger:
    sign: int
    chunks: torch.Tensor


@dataclass
class NewtonDivisionStats:
    reciprocal_seconds: float = 0.0
    qd_multiply_seconds: float = 0.0
    correction_seconds: float = 0.0
    reciprocal_iterations: int = 0
    correction_digits: int = 0
    cpu_correction_used: bool = False
    chunk_fastpath_used: bool = False


PartialT = TypeVar("PartialT")


def resolve_chunk_device(*tensors: torch.Tensor) -> torch.device:
    for tensor in tensors:
        if tensor.device.type == "cuda":
            return tensor.device
    if tensors:
        return tensors[0].device
    return torch.device("cpu")


def normalize_chunks(chunks: torch.Tensor, device: torch.device | str | None = None) -> torch.Tensor:
    target_device = torch.device(device) if device is not None else chunks.device
    return chunks.to(
        dtype=torch.int64,
        device=target_device,
        copy=chunks.dtype != torch.int64 or chunks.device != target_device,
    )


def zero_chunks(device: torch.device | str) -> torch.Tensor:
    return torch.tensor([0], dtype=torch.int64, device=torch.device(device))


def is_zero_trimmed_chunks(chunks: torch.Tensor) -> bool:
    return chunks.numel() == 1 and int(chunks[0].item()) == 0


def shift_right_chunks(chunks: torch.Tensor, shift_chunks: int) -> torch.Tensor:
    trimmed = trim_chunks(chunks)
    if shift_chunks <= 0:
        return trimmed
    if trimmed.numel() <= shift_chunks:
        return zero_chunks(trimmed.device)
    return trim_chunks(trimmed[shift_chunks:], device=trimmed.device)


def shift_left_chunks(chunks: torch.Tensor, shift_chunks: int) -> torch.Tensor:
    trimmed = trim_chunks(chunks)
    if shift_chunks <= 0:
        return trimmed
    padding = torch.zeros(shift_chunks, dtype=torch.int64, device=trimmed.device)
    return torch.cat((padding, trimmed))


def shift_left_bits_chunks(chunks: torch.Tensor, shift_bits: int, base: int) -> torch.Tensor:
    trimmed = trim_chunks(chunks)
    if shift_bits <= 0 or is_zero_trimmed_chunks(trimmed):
        return trimmed

    chunk_bits = int(math.log2(base))
    whole_chunks, rem_bits = divmod(shift_bits, chunk_bits)
    shifted = shift_left_chunks(trimmed, whole_chunks) if whole_chunks > 0 else trimmed
    if rem_bits == 0:
        return shifted

    low = (shifted << rem_bits) & (base - 1)
    carry = shifted >> (chunk_bits - rem_bits)
    result = torch.zeros(shifted.numel() + 1, dtype=torch.int64, device=shifted.device)
    result[: shifted.numel()] = low
    if shifted.numel() > 0:
        result[1:] += carry
    return trim_chunks(result, device=shifted.device)


def make_scaled_constant(coeff: int, scale_chunks: int, device: torch.device | str) -> torch.Tensor:
    result = torch.zeros(scale_chunks + 1, dtype=torch.int64, device=torch.device(device))
    result[scale_chunks] = coeff
    return result


def sub_scaled_two_minus_chunks(chunks: torch.Tensor, scale_chunks: int, base: int) -> torch.Tensor:
    return sub_scaled_constant_minus_chunks(chunks, scale_chunks, 2, base)


def sub_scaled_constant_minus_chunks(
    chunks: torch.Tensor,
    scale_chunks: int,
    coeff: int,
    base: int,
) -> torch.Tensor:
    product = trim_chunks(chunks)
    target_device = product.device
    constant = make_scaled_constant(coeff, scale_chunks, target_device)
    if compare_abs_chunks(constant, product, assume_trimmed=True) < 0:
        raise ValueError("sub_scaled_constant_minus_chunks underflowed; seed is too coarse")
    return sub_abs_chunks(constant, product, base, assume_trimmed=True)


def div_chunks_by_two(chunks: torch.Tensor, base: int, assume_trimmed: bool = False) -> torch.Tensor:
    trimmed = normalize_chunks(chunks) if assume_trimmed else trim_chunks(chunks)
    if is_zero_trimmed_chunks(trimmed):
        return trimmed
    if should_use_cuda_chunk_ops("div2", base, trimmed):
        return load_cuda_chunk_ops_module().div2_base65536(trimmed)
    result = torch.empty_like(trimmed)
    result[-1] = torch.div(trimmed[-1], 2, rounding_mode="floor")
    if trimmed.numel() > 1:
        half_base = base // 2
        result[:-1] = torch.div(trimmed[:-1], 2, rounding_mode="floor") + torch.remainder(trimmed[1:], 2) * half_base
    return trim_chunks(result, device=trimmed.device)


def trim_chunks(chunks: torch.Tensor, device: torch.device | str | None = None) -> torch.Tensor:
    target_device = torch.device(device) if device is not None else chunks.device
    normalized = normalize_chunks(chunks, device=target_device)
    if normalized.numel() == 0:
        return zero_chunks(target_device)
    if int(normalized[-1].item()) != 0:
        return normalized
    if should_use_cuda_chunk_ops("trim", BINARY_CHUNK_BASE, normalized):
        return load_cuda_chunk_ops_module().trim_chunks_base65536(normalized)
    non_zero = torch.nonzero(normalized, as_tuple=False)
    if non_zero.numel() == 0:
        return zero_chunks(target_device)
    return normalized[: int(non_zero[-1].item()) + 1]


def compare_abs_chunks(left: torch.Tensor, right: torch.Tensor, assume_trimmed: bool = False) -> int:
    target_device = resolve_chunk_device(left, right)
    if assume_trimmed:
        left_trimmed = normalize_chunks(left, device=target_device)
        right_trimmed = normalize_chunks(right, device=target_device)
    else:
        left_trimmed = trim_chunks(left, device=target_device)
        right_trimmed = trim_chunks(right, device=target_device)
    if should_use_cuda_chunk_ops("compare", BINARY_CHUNK_BASE, left_trimmed, right_trimmed):
        return int(load_cuda_chunk_ops_module().compare_abs_base65536(left_trimmed, right_trimmed))
    if left_trimmed.numel() != right_trimmed.numel():
        return 1 if left_trimmed.numel() > right_trimmed.numel() else -1

    reversed_left = torch.flip(left_trimmed, dims=[0])
    reversed_right = torch.flip(right_trimmed, dims=[0])
    diff = torch.nonzero(reversed_left != reversed_right, as_tuple=False)
    if diff.numel() == 0:
        return 0

    first = int(diff[0].item())
    return 1 if int(reversed_left[first].item()) > int(reversed_right[first].item()) else -1


def add_abs_chunks(left: torch.Tensor, right: torch.Tensor, base: int, assume_trimmed: bool = False) -> torch.Tensor:
    target_device = resolve_chunk_device(left, right)
    if assume_trimmed:
        left_trimmed = normalize_chunks(left, device=target_device)
        right_trimmed = normalize_chunks(right, device=target_device)
    else:
        left_trimmed = trim_chunks(left, device=target_device)
        right_trimmed = trim_chunks(right, device=target_device)
    if should_use_cuda_chunk_ops("add", base, left_trimmed, right_trimmed):
        return load_cuda_chunk_ops_module().add_abs_base65536(left_trimmed, right_trimmed)
    max_len = max(left_trimmed.numel(), right_trimmed.numel())
    result = torch.zeros(max_len, dtype=torch.int64, device=target_device)
    result[: left_trimmed.numel()] += left_trimmed
    result[: right_trimmed.numel()] += right_trimmed

    while True:
        carry = torch.div(result, base, rounding_mode="floor")
        if int(torch.count_nonzero(carry).item()) == 0:
            break

        result = result - carry * base
        carry_body = carry[:-1]
        if int(torch.count_nonzero(carry[-1:]).item()) != 0:
            result = torch.cat((result, carry[-1:].clone()))
        if carry_body.numel() > 0:
            result[1 : 1 + carry_body.numel()] += carry_body

    return trim_chunks(result, device=target_device)


def sub_abs_chunks(left: torch.Tensor, right: torch.Tensor, base: int, assume_trimmed: bool = False) -> torch.Tensor:
    if compare_abs_chunks(left, right, assume_trimmed=assume_trimmed) < 0:
        raise ValueError("sub_abs_chunks requires left >= right")

    target_device = resolve_chunk_device(left, right)
    if assume_trimmed:
        left_trimmed = normalize_chunks(left, device=target_device)
        right_trimmed = normalize_chunks(right, device=target_device)
    else:
        left_trimmed = trim_chunks(left, device=target_device)
        right_trimmed = trim_chunks(right, device=target_device)
    if should_use_cuda_chunk_ops("sub", base, left_trimmed, right_trimmed):
        return load_cuda_chunk_ops_module().sub_abs_base65536(left_trimmed, right_trimmed)
    max_len = max(left_trimmed.numel(), right_trimmed.numel())
    result = torch.zeros(max_len, dtype=torch.int64, device=target_device)
    result[: left_trimmed.numel()] += left_trimmed
    result[: right_trimmed.numel()] -= right_trimmed

    while True:
        negative_mask = result < 0
        if int(torch.count_nonzero(negative_mask).item()) == 0:
            break

        borrow = torch.zeros_like(result)
        borrow[negative_mask] = torch.div(-result[negative_mask] + base - 1, base, rounding_mode="floor")
        result[negative_mask] += borrow[negative_mask] * base
        if int(torch.count_nonzero(borrow[-1:]).item()) != 0:
            raise ValueError("unexpected negative tail in sub_abs_chunks")
        if borrow.numel() > 1:
            result[1:] -= borrow[:-1]

    return trim_chunks(result, device=target_device)


def add_chunk_integers(left: ChunkInteger, right: ChunkInteger, base: int) -> ChunkInteger:
    target_device = resolve_chunk_device(left.chunks, right.chunks)
    left_chunks = trim_chunks(left.chunks, device=target_device)
    right_chunks = trim_chunks(right.chunks, device=target_device)
    left_is_zero = is_zero_trimmed_chunks(left_chunks)
    right_is_zero = is_zero_trimmed_chunks(right_chunks)

    if left_is_zero:
        return ChunkInteger(right.sign, right_chunks)
    if right_is_zero:
        return ChunkInteger(left.sign, left_chunks)

    if left.sign == right.sign:
        return ChunkInteger(left.sign, add_abs_chunks(left_chunks, right_chunks, base))

    compare = compare_abs_chunks(left_chunks, right_chunks)
    if compare == 0:
        return ChunkInteger(1, zero_chunks(target_device))
    if compare > 0:
        return ChunkInteger(left.sign, sub_abs_chunks(left_chunks, right_chunks, base))
    return ChunkInteger(right.sign, sub_abs_chunks(right_chunks, left_chunks, base))


def mul_abs_chunks_small(
    chunks: torch.Tensor,
    multiplier: int,
    base: int,
    assume_trimmed: bool = False,
) -> torch.Tensor:
    trimmed = normalize_chunks(chunks) if assume_trimmed else trim_chunks(chunks)
    if multiplier == 0 or is_zero_trimmed_chunks(trimmed):
        return zero_chunks(trimmed.device)
    if multiplier == 1:
        return trimmed
    if should_use_cuda_chunk_ops("mul_small", base, trimmed):
        return load_cuda_chunk_ops_module().mul_small_base65536(trimmed, int(multiplier))

    result = trimmed * multiplier
    while True:
        carry = torch.div(result, base, rounding_mode="floor")
        if int(torch.count_nonzero(carry).item()) == 0:
            break

        result = result - carry * base
        carry_body = carry[:-1]
        if int(torch.count_nonzero(carry[-1:]).item()) != 0:
            result = torch.cat((result, carry[-1:].clone()))
        if carry_body.numel() > 0:
            result[1 : 1 + carry_body.numel()] += carry_body

    return trim_chunks(result, device=trimmed.device)


def mul_chunk_integer_small(value: ChunkInteger, multiplier: int, base: int) -> ChunkInteger:
    if multiplier < 0:
        return ChunkInteger(-value.sign, mul_abs_chunks_small(value.chunks, -multiplier, base))
    return ChunkInteger(value.sign, mul_abs_chunks_small(value.chunks, multiplier, base))


def small_chunk_integer(
    value: int,
    chunk_format: str,
    device: torch.device | str | None = None,
) -> ChunkInteger:
    if value == 0:
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        return ChunkInteger(1, zero_chunks(target_device))

    sign = 1
    magnitude = value
    if magnitude < 0:
        sign = -1
        magnitude = -magnitude

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    if chunk_format == "binary16":
        if magnitude < BINARY_CHUNK_BASE:
            chunks = torch.tensor([magnitude], dtype=torch.int64, device=target_device)
            return ChunkInteger(sign, chunks)
        return mpz_to_chunk_integer(gmpy2.mpz(sign * magnitude), chunk_format, device=target_device)

    return mpz_to_chunk_integer(gmpy2.mpz(sign * magnitude), chunk_format, device=target_device)


def pow_small_integer_chunks_gpu(
    base_value: int,
    exponent: int,
    hybrid_multiplier: "HybridGpuMultiplier",
    device: torch.device | str | None = None,
) -> ChunkInteger:
    if exponent < 0:
        raise ValueError("pow_small_integer_chunks_gpu requires exponent >= 0")
    if base_value < 0:
        raise ValueError("pow_small_integer_chunks_gpu currently requires base_value >= 0")

    target_device = torch.device(device) if device is not None else torch.device(
        "cuda" if hybrid_multiplier.chunk_format == "binary16" and torch.cuda.is_available() else "cpu"
    )
    result = small_chunk_integer(1, hybrid_multiplier.chunk_format, device=target_device)
    if exponent == 0:
        return result

    factor = small_chunk_integer(base_value, hybrid_multiplier.chunk_format, device=target_device)
    remaining = exponent
    while remaining > 0:
        if remaining & 1:
            result = hybrid_multiplier.mul_chunks(result, factor, stage="merge")
        remaining >>= 1
        if remaining != 0:
            factor = hybrid_multiplier.mul_chunks(factor, factor, stage="merge")
    return result


def build_decimal_scale_chunks_gpu(
    decimal_digits: int,
    hybrid_multiplier: "HybridGpuMultiplier",
    device: torch.device | str | None = None,
) -> ChunkInteger:
    if decimal_digits < 0:
        raise ValueError("decimal_digits must be non-negative")
    target_device = torch.device(device) if device is not None else torch.device(
        "cuda" if hybrid_multiplier.chunk_format == "binary16" and torch.cuda.is_available() else "cpu"
    )
    # 10^n = 5^n * 2^n; in base 2^16 limbs, the 2^n factor is just a bit-shift.
    pow5 = pow_small_integer_chunks_gpu(5, decimal_digits, hybrid_multiplier, device=target_device)
    return ChunkInteger(1, shift_left_bits_chunks(pow5.chunks, decimal_digits, hybrid_multiplier.chunk_base))


def mpz_to_chunk_integer(
    value: gmpy2.mpz,
    chunk_format: str,
    device: torch.device | str | None = None,
) -> ChunkInteger:
    sign = 1
    magnitude = value
    if magnitude < 0:
        sign = -1
        magnitude = -magnitude

    if chunk_format == "binary16":
        return ChunkInteger(sign, mpz_to_binary16_chunks(magnitude, device=device))

    text = magnitude.digits(10)
    _, chunks = decimal_string_to_chunks(text)
    if device is not None:
        chunks = chunks.to(device=torch.device(device), dtype=torch.int64)
    return ChunkInteger(sign, chunks)


def chunk_integer_to_mpz(value: ChunkInteger, chunk_format: str) -> gmpy2.mpz:
    if chunk_format == "binary16":
        return binary16_chunks_to_mpz(value.chunks, sign=value.sign)
    return gmpy2.mpz(chunks_to_decimal_string(value.chunks, sign=value.sign))


def reciprocal_sqrt_constant_bootstrap(constant: int, precision_chunks: int) -> gmpy2.mpz:
    precision_bits = max(128, 16 * precision_chunks + 128)
    ctx = gmpy2.get_context().copy()
    ctx.precision = precision_bits
    ctx.emax = gmpy2.get_emax_max()
    ctx.emin = gmpy2.get_emin_min()
    scale_bits = 16 * precision_chunks
    with gmpy2.context(ctx):
        scaled = gmpy2.mpfr(gmpy2.mpz(1) << scale_bits) / gmpy2.sqrt(gmpy2.mpfr(constant))
        return gmpy2.mpz(gmpy2.floor(scaled))


def normalized_high_chunks_upper_bound(
    chunks: torch.Tensor,
    *,
    significant_chunks: int = PROJECT2_GPU_SEED_SIGNIFICANT_CHUNKS,
    source_chunks: int = PROJECT2_GPU_SEED_TOP_SOURCE_CHUNKS,
    base: int = BINARY_CHUNK_BASE,
) -> torch.Tensor:
    trimmed = trim_chunks(chunks)
    use_chunks = min(int(trimmed.numel()), max(significant_chunks, source_chunks))
    source = trimmed[-use_chunks:].to(dtype=torch.float64, device=trimmed.device)
    flipped = torch.flip(source, dims=[0])
    base_f = torch.tensor(float(base), dtype=torch.float64, device=trimmed.device)
    exponents = torch.arange(use_chunks, dtype=torch.float64, device=trimmed.device)
    normalized = torch.sum(flipped / torch.pow(base_f, exponents))
    tail_bound = torch.pow(base_f, torch.tensor(1 - use_chunks, dtype=torch.float64, device=trimmed.device))
    return normalized + tail_bound


def radix_value_to_chunks(value: torch.Tensor, scale_chunks: int, base: int) -> torch.Tensor:
    if scale_chunks < 0:
        raise ValueError("scale_chunks must be non-negative")

    device = value.device
    if scale_chunks == 0:
        digit = torch.floor(value).to(dtype=torch.int64)
        return trim_chunks(digit.reshape(1), device=device)

    base_f = torch.tensor(float(base), dtype=torch.float64, device=device)
    max_digit_f = torch.tensor(float(base - 1), dtype=torch.float64, device=device)
    out = torch.zeros(scale_chunks + 1, dtype=torch.int64, device=device)
    state = value
    for index in range(scale_chunks, -1, -1):
        digit_f = torch.clamp(torch.floor(state), min=0.0, max=max_digit_f)
        out[index] = digit_f.to(dtype=torch.int64)
        state = (state - digit_f) * base_f
    return trim_chunks(out, device=device)


def radix_fraction_to_chunks(value: torch.Tensor, scale_chunks: int, base: int) -> torch.Tensor:
    if scale_chunks <= 0:
        return zero_chunks(value.device)

    base_f = torch.tensor(float(base), dtype=torch.float64, device=value.device)
    return radix_value_to_chunks(value * base_f, scale_chunks - 1, base)


def approximate_reciprocal_seed_chunks_gpu(
    denominator_chunks: torch.Tensor,
    precision_chunks: int,
    base: int,
    *,
    significant_chunks: int = PROJECT2_GPU_SEED_SIGNIFICANT_CHUNKS,
    source_chunks: int = PROJECT2_GPU_SEED_TOP_SOURCE_CHUNKS,
) -> torch.Tensor:
    if precision_chunks <= 0:
        return zero_chunks(denominator_chunks.device)

    upper_bound = normalized_high_chunks_upper_bound(
        denominator_chunks,
        significant_chunks=significant_chunks,
        source_chunks=source_chunks,
        base=base,
    )
    base_f = torch.tensor(float(base), dtype=torch.float64, device=denominator_chunks.device)
    small_precision = min(precision_chunks, significant_chunks)
    seed_small = radix_value_to_chunks(base_f / upper_bound, small_precision, base)
    if precision_chunks > small_precision:
        seed_small = shift_left_chunks(seed_small, precision_chunks - small_precision)
    return trim_chunks(seed_small, device=denominator_chunks.device)


def approximate_reciprocal_sqrt_seed_chunks_gpu(
    constant: int,
    precision_chunks: int,
    base: int,
    device: torch.device | str,
    *,
    significant_chunks: int = PROJECT2_GPU_SEED_SIGNIFICANT_CHUNKS,
) -> torch.Tensor:
    if precision_chunks <= 0:
        return zero_chunks(device)

    target_device = torch.device(device)
    small_precision = min(precision_chunks, significant_chunks)
    constant_tensor = torch.tensor(float(constant), dtype=torch.float64, device=target_device)
    rsqrt_value = torch.rsqrt(constant_tensor)
    if hasattr(torch, "nextafter"):
        rsqrt_value = torch.nextafter(rsqrt_value, torch.zeros_like(rsqrt_value))
    else:
        rsqrt_value = rsqrt_value * (1.0 - 2.0**-52)
    seed_small = radix_fraction_to_chunks(rsqrt_value, small_precision, base)
    if precision_chunks > small_precision:
        seed_small = shift_left_chunks(seed_small, precision_chunks - small_precision)
    return trim_chunks(seed_small, device=target_device)


class HybridGpuMultiplier:
    def __init__(
        self,
        threshold_digits: int,
        enable_gpu: bool = True,
        use_gpu_for_merge: bool = True,
        use_gpu_for_final: bool = True,
        chunk_format: str = "binary16",
        gpu_memory_budget_gb: float | None = None,
    ) -> None:
        self.threshold_digits = threshold_digits
        self.enable_gpu = enable_gpu
        self.use_gpu_for_merge = use_gpu_for_merge
        self.use_gpu_for_final = use_gpu_for_final
        self.chunk_format = chunk_format
        self.gpu_memory_budget_gb = gpu_memory_budget_gb
        self.chunk_base = BINARY_CHUNK_BASE if chunk_format == "binary16" else 10_000
        self.chunk_log10_base = math.log10(self.chunk_base)
        self.stats = HybridMulStats()
        if chunk_format not in {"decimal4", "binary16"}:
            raise ValueError(f"unsupported chunk format: {chunk_format}")
        self.multiplier = None

    def _ensure_multiplier(self) -> GpuFftMultiplier:
        if self.multiplier is not None:
            return self.multiplier
        if not self.enable_gpu:
            raise RuntimeError("GPU backend was requested while GPU support is disabled")

        if self.chunk_format == "decimal4":
            self.multiplier = GpuFftMultiplier()
        else:
            self.multiplier = GpuFftMultiplier(
                decimal_base=BINARY_CHUNK_BASE,
                split_base=BINARY_CHUNK_SPLIT_BASE,
            )
        return self.multiplier

    def _record_gpu(self, backend_stats, prepare_seconds: float, finalize_seconds: float, operand_digits: int, stage: str) -> None:
        self.stats.gpu_calls += 1
        self.stats.gpu_prepare_seconds += prepare_seconds
        self.stats.gpu_backend_total_seconds += backend_stats.total_seconds
        self.stats.gpu_kernel_seconds += backend_stats.kernel_seconds
        self.stats.gpu_finalize_seconds += finalize_seconds
        self.stats.gpu_peak_gb = max(self.stats.gpu_peak_gb, backend_stats.peak_gpu_gb)
        self.stats.gpu_max_operand_digits = max(self.stats.gpu_max_operand_digits, operand_digits)
        if stage == "merge":
            self.stats.merge_gpu_calls += 1
        elif stage == "final":
            self.stats.final_gpu_calls += 1

    def _estimate_digits_from_chunks(self, chunks: torch.Tensor) -> int:
        trimmed = trim_chunks(chunks)
        if is_zero_trimmed_chunks(trimmed):
            return 1
        return max(1, int(trimmed.numel() * self.chunk_log10_base) + 1)

    def _estimate_chunks_from_digits(self, digits: int) -> int:
        return max(1, int(math.ceil(max(1, digits) / self.chunk_log10_base)))

    def _estimate_gpu_working_set_gb(self, left_chunks: int, right_chunks: int) -> float:
        return estimate_fft_working_set_gb(left_chunks, right_chunks)

    def _budget_requires_cpu(self, operand_digits: int, left_chunks: int, right_chunks: int) -> tuple[bool, float]:
        if self.gpu_memory_budget_gb is None:
            return False, 0.0
        estimated_peak_gb = self._estimate_gpu_working_set_gb(left_chunks, right_chunks)
        return estimated_peak_gb > self.gpu_memory_budget_gb, estimated_peak_gb

    def _record_budget_fallback(self, operand_digits: int, estimated_peak_gb: float) -> None:
        self.stats.budget_cpu_calls += 1
        self.stats.budget_max_operand_digits = max(self.stats.budget_max_operand_digits, operand_digits)
        self.stats.budget_max_estimated_peak_gb = max(self.stats.budget_max_estimated_peak_gb, estimated_peak_gb)

    def _record_oom_fallback(self) -> None:
        self.stats.oom_cpu_calls += 1

    def _mul_mpz_cpu(self, left: gmpy2.mpz, right: gmpy2.mpz) -> gmpy2.mpz:
        self.stats.cpu_calls += 1
        return left * right

    def _mul_chunks_cpu(
        self,
        left: ChunkInteger,
        right: ChunkInteger,
        *,
        device: torch.device | str = "cpu",
    ) -> ChunkInteger:
        self.stats.cpu_calls += 1
        product = chunk_integer_to_mpz(left, self.chunk_format)
        product *= chunk_integer_to_mpz(right, self.chunk_format)
        return mpz_to_chunk_integer(product, self.chunk_format, device=device)

    def mul(self, left: gmpy2.mpz, right: gmpy2.mpz, stage: str) -> gmpy2.mpz:
        if left == 0 or right == 0:
            self.stats.cpu_calls += 1
            return gmpy2.mpz(0)

        if stage == "merge" and not self.use_gpu_for_merge:
            return self._mul_mpz_cpu(left, right)
        if stage == "final" and not self.use_gpu_for_final:
            return self._mul_mpz_cpu(left, right)

        operand_digits = max(estimate_decimal_digits(abs(left)), estimate_decimal_digits(abs(right)))
        if not self.enable_gpu or operand_digits < self.threshold_digits:
            return self._mul_mpz_cpu(left, right)

        budget_requires_cpu, estimated_peak_gb = self._budget_requires_cpu(
            operand_digits,
            self._estimate_chunks_from_digits(estimate_decimal_digits(abs(left))),
            self._estimate_chunks_from_digits(estimate_decimal_digits(abs(right))),
        )
        if budget_requires_cpu:
            self._record_budget_fallback(operand_digits, estimated_peak_gb)
            return self._mul_mpz_cpu(left, right)

        sign = 1
        left_abs = left
        right_abs = right
        if left_abs < 0:
            sign = -sign
            left_abs = -left_abs
        if right_abs < 0:
            sign = -sign
            right_abs = -right_abs

        prepare_start = time.perf_counter()
        if self.chunk_format == "decimal4":
            left_text = left_abs.digits(10)
            right_text = right_abs.digits(10)
            _, left_chunks = decimal_string_to_chunks(left_text)
            _, right_chunks = decimal_string_to_chunks(right_text)
        else:
            left_chunks = mpz_to_binary16_chunks(left_abs)
            right_chunks = mpz_to_binary16_chunks(right_abs)
        prepare_seconds = time.perf_counter() - prepare_start

        try:
            result_chunks, backend_stats = self._ensure_multiplier().multiply_chunks(
                left_chunks,
                right_chunks,
                assume_nonzero_trimmed=True,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            self._record_oom_fallback()
            return self._mul_mpz_cpu(left, right)

        finalize_start = time.perf_counter()
        if self.chunk_format == "decimal4":
            result_text = chunks_to_decimal_string(result_chunks, sign=sign)
            result = gmpy2.mpz(result_text)
        else:
            result = binary16_chunks_to_mpz(result_chunks, sign=sign)
        finalize_seconds = time.perf_counter() - finalize_start

        self._record_gpu(backend_stats, prepare_seconds, finalize_seconds, operand_digits, stage)
        return result

    def mul_chunks(self, left: ChunkInteger, right: ChunkInteger, stage: str) -> ChunkInteger:
        target_device = resolve_chunk_device(left.chunks, right.chunks)
        left_chunks = trim_chunks(left.chunks, device=target_device)
        right_chunks = trim_chunks(right.chunks, device=target_device)
        left_is_zero = is_zero_trimmed_chunks(left_chunks)
        right_is_zero = is_zero_trimmed_chunks(right_chunks)
        if left_is_zero or right_is_zero:
            self.stats.cpu_calls += 1
            return ChunkInteger(1, zero_chunks(target_device))

        if stage == "merge" and not self.use_gpu_for_merge:
            return self._mul_chunks_cpu(
                ChunkInteger(left.sign, left_chunks),
                ChunkInteger(right.sign, right_chunks),
                device=target_device,
            )
        if stage == "final" and not self.use_gpu_for_final:
            return self._mul_chunks_cpu(
                ChunkInteger(left.sign, left_chunks),
                ChunkInteger(right.sign, right_chunks),
                device=target_device,
            )

        operand_digits = max(
            self._estimate_digits_from_chunks(left_chunks),
            self._estimate_digits_from_chunks(right_chunks),
        )
        if not self.enable_gpu or operand_digits < self.threshold_digits:
            return self._mul_chunks_cpu(
                ChunkInteger(left.sign, left_chunks),
                ChunkInteger(right.sign, right_chunks),
                device=target_device,
            )

        budget_requires_cpu, estimated_peak_gb = self._budget_requires_cpu(
            operand_digits,
            int(left_chunks.numel()),
            int(right_chunks.numel()),
        )
        if budget_requires_cpu:
            self._record_budget_fallback(operand_digits, estimated_peak_gb)
            return self._mul_chunks_cpu(
                ChunkInteger(left.sign, left_chunks),
                ChunkInteger(right.sign, right_chunks),
                device="cpu",
            )

        output_device = "cuda" if self.chunk_format == "binary16" else ("cuda" if target_device.type == "cuda" else "cpu")
        if output_device == "cuda" and stage == "final":
            torch.cuda.empty_cache()
        try:
            result_chunks, backend_stats = self._ensure_multiplier().multiply_chunks(
                left_chunks,
                right_chunks,
                output_device=output_device,
                assume_nonzero_trimmed=True,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            self._record_oom_fallback()
            return self._mul_chunks_cpu(
                ChunkInteger(left.sign, left_chunks),
                ChunkInteger(right.sign, right_chunks),
                device="cpu",
            )
        self._record_gpu(backend_stats, 0.0, 0.0, operand_digits, stage)
        return ChunkInteger(left.sign * right.sign, result_chunks)

    def reciprocal_chunks_newton_prototype(
        self,
        denominator: ChunkInteger,
        quotient_chunks_estimate: int,
        seed_chunks: int = 131_072,
        extra_scale_chunks: int = 4,
    ) -> tuple[ChunkInteger, int, int]:
        if denominator.sign <= 0:
            raise ValueError("reciprocal_chunks_newton_prototype requires a positive denominator")

        target_device = resolve_chunk_device(denominator.chunks)
        den_chunks = trim_chunks(denominator.chunks, device=target_device)
        m = int(den_chunks.numel())
        quotient_chunks_estimate = max(1, quotient_chunks_estimate)
        scale_chunks = m + quotient_chunks_estimate + extra_scale_chunks
        target_precision_chunks = quotient_chunks_estimate + extra_scale_chunks
        seed = min(max(seed_chunks, target_precision_chunks), m)

        prefix_cpu = den_chunks[-seed:].to(dtype=torch.int64, device="cpu")
        prefix_value = binary16_chunks_to_mpz(prefix_cpu)
        reciprocal_seed = (gmpy2.mpz(1) << (16 * (seed + quotient_chunks_estimate + extra_scale_chunks))) // prefix_value
        current = mpz_to_chunk_integer(reciprocal_seed, self.chunk_format, device=target_device)

        ratio = target_precision_chunks / max(seed, 1)
        # When the seed already covers the requested quotient precision, the exact
        # CPU-side correction step can finish the job without paying for a full
        # Newton refinement round.
        iterations = max(0, math.ceil(math.log2(ratio)))
        if iterations == 0:
            return current, scale_chunks, iterations

        denominator_ci = ChunkInteger(1, den_chunks)
        scale_twice_mpz = gmpy2.mpz(2) * (gmpy2.mpz(self.chunk_base) ** scale_chunks)
        for _ in range(iterations):
            product = self.mul_chunks(denominator_ci, current, stage="merge")
            error_mpz = scale_twice_mpz - chunk_integer_to_mpz(product, self.chunk_format)
            error = mpz_to_chunk_integer(
                error_mpz,
                self.chunk_format,
                device=target_device,
            )
            refined_full = self.mul_chunks(current, error, stage="merge")
            current = ChunkInteger(refined_full.sign, shift_right_chunks(refined_full.chunks, scale_chunks))

        return current, scale_chunks, iterations

    def reciprocal_chunks_gpu_seed_prototype(
        self,
        denominator: ChunkInteger,
        quotient_chunks_estimate: int,
        bootstrap_chunks: int = 1024,
        extra_scale_chunks: int = 4,
    ) -> tuple[ChunkInteger, int, int]:
        if denominator.sign <= 0:
            raise ValueError("reciprocal_chunks_gpu_seed_prototype requires a positive denominator")

        target_device = resolve_chunk_device(denominator.chunks)
        den_chunks = trim_chunks(denominator.chunks, device=target_device)
        m = int(den_chunks.numel())
        target_precision_chunks = min(m, max(1, quotient_chunks_estimate + extra_scale_chunks))
        scale_chunks = m + target_precision_chunks

        current_precision = min(target_precision_chunks, PROJECT2_GPU_SEED_SIGNIFICANT_CHUNKS)
        current = ChunkInteger(
            1,
            approximate_reciprocal_seed_chunks_gpu(
                den_chunks,
                current_precision,
                self.chunk_base,
            ),
        )

        iterations = 0
        while current_precision < target_precision_chunks:
            next_precision = min(target_precision_chunks, current_precision * 2)
            precision_delta = next_precision - current_precision
            promoted = ChunkInteger(1, shift_left_chunks(current.chunks, precision_delta))
            denominator_prefix = ChunkInteger(1, den_chunks[-next_precision:])
            iteration_scale_chunks = 2 * next_precision

            product = self.mul_chunks(denominator_prefix, promoted, stage="merge")
            error_chunks = sub_scaled_two_minus_chunks(
                product.chunks,
                iteration_scale_chunks,
                self.chunk_base,
            )
            refined_full = self.mul_chunks(
                promoted,
                ChunkInteger(1, error_chunks),
                stage="merge",
            )
            current = ChunkInteger(1, shift_right_chunks(refined_full.chunks, iteration_scale_chunks))
            current_precision = next_precision
            iterations += 1

        return current, scale_chunks, iterations

    def reciprocal_sqrt_constant_gpu_seed_prototype(
        self,
        constant: int,
        target_precision_chunks: int,
        bootstrap_chunks: int = 4096,
    ) -> tuple[ChunkInteger, int]:
        if constant <= 0:
            raise ValueError("reciprocal_sqrt_constant_gpu_seed_prototype requires constant > 0")

        target_precision_chunks = max(1, target_precision_chunks)
        current_precision = min(target_precision_chunks, PROJECT2_GPU_SEED_SIGNIFICANT_CHUNKS)
        target_device = torch.device("cuda" if self.chunk_format == "binary16" and torch.cuda.is_available() else "cpu")
        current = ChunkInteger(
            1,
            approximate_reciprocal_sqrt_seed_chunks_gpu(
                constant,
                current_precision,
                self.chunk_base,
                target_device,
            ),
        )

        iterations = 0
        while current_precision < target_precision_chunks:
            next_precision = min(target_precision_chunks, current_precision * 2)
            precision_delta = next_precision - current_precision
            promoted = ChunkInteger(1, shift_left_chunks(current.chunks, precision_delta))
            iteration_scale_chunks = 2 * next_precision

            square = self.mul_chunks(promoted, promoted, stage="merge")
            scaled_square = mul_chunk_integer_small(square, constant, self.chunk_base)
            error_chunks = sub_scaled_constant_minus_chunks(
                scaled_square.chunks,
                iteration_scale_chunks,
                3,
                self.chunk_base,
            )
            refined_full = self.mul_chunks(
                promoted,
                ChunkInteger(1, error_chunks),
                stage="merge",
            )
            shifted = shift_right_chunks(refined_full.chunks, iteration_scale_chunks)
            current = ChunkInteger(1, div_chunks_by_two(shifted, self.chunk_base))
            current_precision = next_precision
            iterations += 1

        return current, iterations

    def divide_chunks_newton_prototype(
        self,
        numerator: ChunkInteger,
        denominator: ChunkInteger,
        seed_chunks: int = 131_072,
        extra_scale_chunks: int = 4,
        seed_mode: str = "mpz-exact",
    ) -> tuple[ChunkInteger, NewtonDivisionStats]:
        if numerator.sign < 0 or denominator.sign <= 0:
            raise ValueError("divide_chunks_newton_prototype requires numerator >= 0 and denominator > 0")

        stats = NewtonDivisionStats()
        target_device = resolve_chunk_device(numerator.chunks, denominator.chunks)
        numerator_ci = ChunkInteger(1, trim_chunks(numerator.chunks, device=target_device))
        denominator_ci = ChunkInteger(1, trim_chunks(denominator.chunks, device=target_device))
        quotient_chunks_estimate = max(
            1,
            int(numerator_ci.chunks.numel()) - int(denominator_ci.chunks.numel()) + 2,
        )
        reciprocal_start = time.perf_counter()
        if seed_mode == "gpu-doubling":
            reciprocal, scale_chunks, iterations = self.reciprocal_chunks_gpu_seed_prototype(
                denominator_ci,
                quotient_chunks_estimate=quotient_chunks_estimate,
                bootstrap_chunks=min(seed_chunks, 4096),
                extra_scale_chunks=extra_scale_chunks,
            )
        elif seed_mode == "mpz-exact":
            reciprocal, scale_chunks, iterations = self.reciprocal_chunks_newton_prototype(
                denominator_ci,
                quotient_chunks_estimate=quotient_chunks_estimate,
                seed_chunks=seed_chunks,
                extra_scale_chunks=extra_scale_chunks,
            )
        else:
            raise ValueError(f"unsupported seed_mode: {seed_mode}")
        stats.reciprocal_seconds = time.perf_counter() - reciprocal_start
        stats.reciprocal_iterations = iterations

        product = self.mul_chunks(numerator_ci, reciprocal, stage="final")
        quotient = ChunkInteger(1, shift_right_chunks(product.chunks, scale_chunks))
        qd_start = time.perf_counter()
        q_times_d = self.mul_chunks(quotient, denominator_ci, stage="merge")
        stats.qd_multiply_seconds = time.perf_counter() - qd_start

        correction_start = time.perf_counter()
        compare_qd = compare_abs_chunks(q_times_d.chunks, numerator_ci.chunks)
        one = mpz_to_chunk_integer(gmpy2.mpz(1), self.chunk_format, device=target_device)

        if compare_qd <= 0:
            remainder_chunks = sub_abs_chunks(
                numerator_ci.chunks,
                q_times_d.chunks,
                self.chunk_base,
            )
            if compare_abs_chunks(remainder_chunks, denominator_ci.chunks) < 0:
                stats.correction_seconds = time.perf_counter() - correction_start
                stats.cpu_correction_used = False
                stats.chunk_fastpath_used = True
                stats.correction_digits = 0
                return quotient, stats

            remainder_after_one = sub_abs_chunks(
                remainder_chunks,
                denominator_ci.chunks,
                self.chunk_base,
            ) if compare_abs_chunks(remainder_chunks, denominator_ci.chunks) >= 0 else None
            if remainder_after_one is not None and compare_abs_chunks(remainder_after_one, denominator_ci.chunks) < 0:
                stats.correction_seconds = time.perf_counter() - correction_start
                stats.cpu_correction_used = False
                stats.chunk_fastpath_used = True
                stats.correction_digits = 1
                return add_chunk_integers(quotient, one, self.chunk_base), stats
        else:
            overshoot_chunks = sub_abs_chunks(
                q_times_d.chunks,
                numerator_ci.chunks,
                self.chunk_base,
            )
            if compare_abs_chunks(overshoot_chunks, denominator_ci.chunks) < 0:
                stats.correction_seconds = time.perf_counter() - correction_start
                stats.cpu_correction_used = False
                stats.chunk_fastpath_used = True
                stats.correction_digits = 1
                return add_chunk_integers(quotient, ChunkInteger(-1, one.chunks), self.chunk_base), stats

        numerator_mpz = chunk_integer_to_mpz(numerator_ci, self.chunk_format)
        denominator_mpz = chunk_integer_to_mpz(denominator_ci, self.chunk_format)
        q_times_d_mpz = chunk_integer_to_mpz(q_times_d, self.chunk_format)

        correction = gmpy2.mpz(0)
        if q_times_d_mpz <= numerator_mpz:
            correction = (numerator_mpz - q_times_d_mpz) // denominator_mpz
        else:
            overshoot = q_times_d_mpz - numerator_mpz
            correction = (overshoot + denominator_mpz - 1) // denominator_mpz

        stats.correction_seconds = time.perf_counter() - correction_start
        stats.cpu_correction_used = True
        stats.chunk_fastpath_used = False
        stats.correction_digits = 0 if correction == 0 else estimate_decimal_digits(correction)
        if correction == 0:
            return quotient, stats

        quotient_mpz = chunk_integer_to_mpz(quotient, self.chunk_format)
        if q_times_d_mpz <= numerator_mpz:
            quotient_mpz += correction
        else:
            quotient_mpz -= correction
        return mpz_to_chunk_integer(quotient_mpz, self.chunk_format, device=target_device), stats


def compute_sqrt_term_chunks_gpu_prototype(
    total_digits: int,
    hybrid_multiplier: HybridGpuMultiplier,
    extra_precision_chunks: int = 2,
) -> tuple[ChunkInteger, float]:
    if hybrid_multiplier.chunk_format != "binary16":
        raise ValueError("compute_sqrt_term_chunks_gpu_prototype currently requires binary16 chunks")

    scale_start = time.perf_counter()
    scale_chunks = pow_small_integer_chunks_gpu(
        10,
        total_digits,
        hybrid_multiplier,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    scale_seconds = time.perf_counter() - scale_start
    precision_chunks = int(scale_chunks.chunks.numel()) + extra_precision_chunks
    inv_sqrt_chunks, _ = hybrid_multiplier.reciprocal_sqrt_constant_gpu_seed_prototype(
        10005,
        precision_chunks,
    )
    sqrt_constant_chunks = mul_chunk_integer_small(
        inv_sqrt_chunks,
        10005,
        hybrid_multiplier.chunk_base,
    )
    full_product = hybrid_multiplier.mul_chunks(scale_chunks, sqrt_constant_chunks, stage="final")
    return ChunkInteger(1, shift_right_chunks(full_product.chunks, precision_chunks)), scale_seconds


def merge_chudnovsky_partials_hybrid(
    left: tuple[gmpy2.mpz, gmpy2.mpz, gmpy2.mpz],
    right: tuple[gmpy2.mpz, gmpy2.mpz, gmpy2.mpz],
    hybrid_multiplier: HybridGpuMultiplier,
) -> tuple[gmpy2.mpz, gmpy2.mpz, gmpy2.mpz]:
    p1, q1, t1 = left
    p2, q2, t2 = right
    p = hybrid_multiplier.mul(p1, p2, stage="merge")
    q = hybrid_multiplier.mul(q1, q2, stage="merge")
    t_left = hybrid_multiplier.mul(t1, q2, stage="merge")
    t_right = hybrid_multiplier.mul(p1, t2, stage="merge")
    return p, q, t_left + t_right


def merge_chudnovsky_partials_chunked(
    left: tuple[ChunkInteger, ChunkInteger, ChunkInteger],
    right: tuple[ChunkInteger, ChunkInteger, ChunkInteger],
    hybrid_multiplier: HybridGpuMultiplier,
) -> tuple[ChunkInteger, ChunkInteger, ChunkInteger]:
    p1, q1, t1 = left
    p2, q2, t2 = right
    p = hybrid_multiplier.mul_chunks(p1, p2, stage="merge")
    q = hybrid_multiplier.mul_chunks(q1, q2, stage="merge")
    t_left = hybrid_multiplier.mul_chunks(t1, q2, stage="merge")
    t_right = hybrid_multiplier.mul_chunks(p1, t2, stage="merge")
    t = add_chunk_integers(t_left, t_right, hybrid_multiplier.chunk_base)
    return p, q, t


def compute_chudnovsky_partials_parallel(
    term_count: int,
    workers: int,
    chunk_terms: int,
    leaf_terms: int = PROJECT2_BS_LEAF_TERMS,
) -> list[tuple[gmpy2.mpz, gmpy2.mpz, gmpy2.mpz]]:
    if workers <= 1 or term_count <= chunk_terms:
        return [chudnovsky_bs_mpz(0, term_count, leaf_terms=leaf_terms)]

    ranges = build_partial_ranges(term_count, chunk_terms, leaf_terms)

    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    with ctx.Pool(processes=min(workers, len(ranges))) as pool:
        return pool.map(project2_bs_worker, ranges)


def iter_chudnovsky_partials_parallel(
    term_count: int,
    workers: int,
    chunk_terms: int,
    leaf_terms: int = PROJECT2_BS_LEAF_TERMS,
) -> Iterator[tuple[gmpy2.mpz, gmpy2.mpz, gmpy2.mpz]]:
    if workers <= 1 or term_count <= chunk_terms:
        yield chudnovsky_bs_mpz(0, term_count, leaf_terms=leaf_terms)
        return

    ranges = build_partial_ranges(term_count, chunk_terms, leaf_terms)
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    with ctx.Pool(processes=min(workers, len(ranges))) as pool:
        yield from pool.imap(project2_bs_worker, ranges, chunksize=1)


def frontier_push(
    frontier: list[PartialT | None],
    partial: PartialT,
    merge_fn: Callable[[PartialT, PartialT], PartialT],
) -> tuple[float, int]:
    merge_seconds = 0.0
    carry = partial
    level = 0
    while True:
        if level >= len(frontier):
            frontier.append(carry)
            break
        if frontier[level] is None:
            frontier[level] = carry
            break

        merge_start = time.perf_counter()
        carry = merge_fn(frontier[level], carry)
        merge_seconds += time.perf_counter() - merge_start
        frontier[level] = None
        level += 1

    live_nodes = sum(entry is not None for entry in frontier)
    return merge_seconds, live_nodes


def frontier_reduce(
    frontier: Iterable[PartialT | None],
    merge_fn: Callable[[PartialT, PartialT], PartialT],
) -> tuple[PartialT, float]:
    result: PartialT | None = None
    merge_seconds = 0.0
    for entry in reversed(list(frontier)):
        if entry is None:
            continue
        if result is None:
            result = entry
            continue

        merge_start = time.perf_counter()
        result = merge_fn(result, entry)
        merge_seconds += time.perf_counter() - merge_start

    if result is None:
        raise ValueError("frontier_reduce requires at least one partial")
    return result, merge_seconds


def merge_partials_hybrid(
    partials: list[tuple[gmpy2.mpz, gmpy2.mpz, gmpy2.mpz]],
    hybrid_multiplier: HybridGpuMultiplier,
) -> tuple[tuple[gmpy2.mpz, gmpy2.mpz, gmpy2.mpz], float, int]:
    merge_seconds = 0.0
    current = partials
    max_nodes = len(current)

    while len(current) > 1:
        merged: list[tuple[gmpy2.mpz, gmpy2.mpz, gmpy2.mpz]] = []
        for index in range(0, len(current), 2):
            merge_start = time.perf_counter()
            if index + 1 == len(current):
                merged.append(current[index])
            else:
                merged.append(
                    merge_chudnovsky_partials_hybrid(
                        current[index],
                        current[index + 1],
                        hybrid_multiplier,
                    )
                )
            merge_seconds += time.perf_counter() - merge_start
        current = merged
        max_nodes = max(max_nodes, len(current))

    return current[0], merge_seconds, max_nodes


def merge_partials_chunked(
    partials: list[tuple[gmpy2.mpz, gmpy2.mpz, gmpy2.mpz]],
    hybrid_multiplier: HybridGpuMultiplier,
) -> tuple[tuple[ChunkInteger, ChunkInteger, ChunkInteger], float, float, int]:
    conversion_seconds = 0.0
    merge_seconds = 0.0
    max_nodes = 0
    merged: list[tuple[ChunkInteger, ChunkInteger, ChunkInteger]] = []
    pending: tuple[ChunkInteger, ChunkInteger, ChunkInteger] | None = None

    for partial in partials:
        conversion_start = time.perf_counter()
        chunk_partial = (
            mpz_to_chunk_integer(partial[0], hybrid_multiplier.chunk_format),
            mpz_to_chunk_integer(partial[1], hybrid_multiplier.chunk_format),
            mpz_to_chunk_integer(partial[2], hybrid_multiplier.chunk_format),
        )
        conversion_seconds += time.perf_counter() - conversion_start
        if pending is None:
            pending = chunk_partial
            continue

        merge_start = time.perf_counter()
        merged.append(
            merge_chudnovsky_partials_chunked(
                pending,
                chunk_partial,
                hybrid_multiplier,
            )
        )
        merge_seconds += time.perf_counter() - merge_start
        pending = None
        max_nodes = max(max_nodes, len(merged) + (1 if pending is not None else 0))

    if pending is not None:
        merged.append(pending)
    current = merged
    max_nodes = max(max_nodes, len(current))

    while len(current) > 1:
        next_round: list[tuple[ChunkInteger, ChunkInteger, ChunkInteger]] = []
        for index in range(0, len(current), 2):
            merge_start = time.perf_counter()
            if index + 1 == len(current):
                next_round.append(current[index])
            else:
                next_round.append(
                    merge_chudnovsky_partials_chunked(
                        current[index],
                        current[index + 1],
                        hybrid_multiplier,
                    )
                )
            merge_seconds += time.perf_counter() - merge_start
        current = next_round
        max_nodes = max(max_nodes, len(current))

    return current[0], conversion_seconds, merge_seconds, max_nodes


def merge_partials_hybrid_frontier_stream(
    partials: Iterable[tuple[gmpy2.mpz, gmpy2.mpz, gmpy2.mpz]],
    hybrid_multiplier: HybridGpuMultiplier,
) -> tuple[tuple[gmpy2.mpz, gmpy2.mpz, gmpy2.mpz], float, int]:
    frontier: list[tuple[gmpy2.mpz, gmpy2.mpz, gmpy2.mpz] | None] = []
    merge_seconds = 0.0
    max_nodes = 0
    seen = 0

    for partial in partials:
        seen += 1
        push_seconds, live_nodes = frontier_push(
            frontier,
            partial,
            lambda left, right: merge_chudnovsky_partials_hybrid(left, right, hybrid_multiplier),
        )
        merge_seconds += push_seconds
        max_nodes = max(max_nodes, live_nodes)

    if seen == 0:
        raise ValueError("merge_partials_hybrid_frontier_stream requires at least one partial")

    merged, reduce_seconds = frontier_reduce(
        frontier,
        lambda left, right: merge_chudnovsky_partials_hybrid(left, right, hybrid_multiplier),
    )
    merge_seconds += reduce_seconds
    return merged, merge_seconds, max_nodes


def merge_partials_chunked_frontier_stream(
    partials: Iterable[tuple[gmpy2.mpz, gmpy2.mpz, gmpy2.mpz]],
    hybrid_multiplier: HybridGpuMultiplier,
) -> tuple[tuple[ChunkInteger, ChunkInteger, ChunkInteger], float, float, int]:
    frontier: list[tuple[ChunkInteger, ChunkInteger, ChunkInteger] | None] = []
    conversion_seconds = 0.0
    merge_seconds = 0.0
    max_nodes = 0
    seen = 0

    for partial in partials:
        seen += 1
        conversion_start = time.perf_counter()
        chunk_partial = (
            mpz_to_chunk_integer(partial[0], hybrid_multiplier.chunk_format),
            mpz_to_chunk_integer(partial[1], hybrid_multiplier.chunk_format),
            mpz_to_chunk_integer(partial[2], hybrid_multiplier.chunk_format),
        )
        conversion_seconds += time.perf_counter() - conversion_start

        push_seconds, live_nodes = frontier_push(
            frontier,
            chunk_partial,
            lambda left, right: merge_chudnovsky_partials_chunked(left, right, hybrid_multiplier),
        )
        merge_seconds += push_seconds
        max_nodes = max(max_nodes, live_nodes)

    if seen == 0:
        raise ValueError("merge_partials_chunked_frontier_stream requires at least one partial")

    merged, reduce_seconds = frontier_reduce(
        frontier,
        lambda left, right: merge_chudnovsky_partials_chunked(left, right, hybrid_multiplier),
    )
    merge_seconds += reduce_seconds
    return merged, conversion_seconds, merge_seconds, max_nodes


def resolve_parallel_config(digits: int) -> tuple[int, int]:
    logical_cpus = mp.cpu_count() or 1
    workers = min(32, logical_cpus)
    term_count = estimate_chudnovsky_term_count(digits)
    target_chunk_terms = math.ceil(term_count / PROJECT2_PARTIAL_TARGET_TASKS)
    chunk_terms = nearest_power_of_two(max(PROJECT2_MIN_CHUNK_TERMS, target_chunk_terms))
    return workers, chunk_terms


def compute_pi_digits_integer_gpu_hybrid(
    digits: int,
    workers: int,
    chunk_terms: int,
    bs_leaf_terms: int,
    gpu_threshold_digits: int,
    gpu_stages: str,
    gpu_chunk_format: str,
    guard_digits: int = PROJECT2_GUARD_DIGITS,
    sqrt_mode: str = "mpz-isqrt",
    division_mode: str = "mpz-div",
    gpu_memory_budget_gb: float | None = None,
    stream_partials: bool = False,
) -> tuple[gmpy2.mpz, int, HybridMulStats, PipelinePhaseStats]:
    phase_stats = PipelinePhaseStats()
    total_digits = digits + guard_digits
    terms = estimate_chudnovsky_term_count(digits, guard_digits=guard_digits)
    hybrid_multiplier = HybridGpuMultiplier(
        threshold_digits=gpu_threshold_digits,
        enable_gpu=True,
        use_gpu_for_merge=(gpu_stages == "merge-and-final"),
        use_gpu_for_final=True,
        chunk_format=gpu_chunk_format,
        gpu_memory_budget_gb=gpu_memory_budget_gb,
    )
    partials = None
    q = None
    t = None
    q_chunks = None
    t_chunks = None

    if stream_partials:
        phase_stats.streaming_partials_used = True
        stream_start = time.perf_counter()
        partial_iter = iter_chudnovsky_partials_parallel(
            terms,
            workers,
            chunk_terms,
            leaf_terms=bs_leaf_terms,
        )
        if gpu_stages == "merge-and-final" and gpu_chunk_format == "binary16":
            (
                (_, q_chunks, t_chunks),
                phase_stats.chunk_conversion_seconds,
                phase_stats.merge_tree_seconds,
                phase_stats.merge_frontier_max_nodes,
            ) = merge_partials_chunked_frontier_stream(
                partial_iter,
                hybrid_multiplier,
            )
            phase_stats.partial_stream_seconds = time.perf_counter() - stream_start
            phase_stats.partial_generation_seconds = max(
                0.0,
                phase_stats.partial_stream_seconds
                - phase_stats.chunk_conversion_seconds
                - phase_stats.merge_tree_seconds,
            )
        else:
            (
                (_, q, t),
                phase_stats.merge_tree_seconds,
                phase_stats.merge_frontier_max_nodes,
            ) = merge_partials_hybrid_frontier_stream(
                partial_iter,
                hybrid_multiplier,
            )
            phase_stats.partial_stream_seconds = time.perf_counter() - stream_start
            phase_stats.partial_generation_seconds = max(
                0.0,
                phase_stats.partial_stream_seconds - phase_stats.merge_tree_seconds,
            )
    else:
        partial_start = time.perf_counter()
        partials = compute_chudnovsky_partials_parallel(
            terms,
            workers,
            chunk_terms,
            leaf_terms=bs_leaf_terms,
        )
        phase_stats.partial_generation_seconds = time.perf_counter() - partial_start
    sqrt_compute_start = time.perf_counter()
    sqrt_term = None
    sqrt_chunks = None
    if sqrt_mode == "mpfr-exact":
        scale = gmpy2.mpz(10) ** total_digits
        sqrt_term = compute_sqrt_term_mpfr_exact(total_digits)
    elif sqrt_mode == "mpz-isqrt":
        scale = gmpy2.mpz(10) ** total_digits
        sqrt_term = gmpy2.isqrt(gmpy2.mpz(10005) * scale * scale)
    elif sqrt_mode == "chunk-gpu-rsqrt-prototype":
        if not (gpu_stages == "merge-and-final" and gpu_chunk_format == "binary16"):
            raise ValueError("chunk-gpu-rsqrt-prototype currently requires merge-and-final + binary16")
        sqrt_chunks, phase_stats.sqrt_scale_seconds = compute_sqrt_term_chunks_gpu_prototype(total_digits, hybrid_multiplier)
    else:
        raise ValueError(f"unsupported sqrt_mode: {sqrt_mode}")
    phase_stats.sqrt_compute_seconds = time.perf_counter() - sqrt_compute_start

    if gpu_stages == "merge-and-final" and gpu_chunk_format == "binary16":
        if not stream_partials:
            (
                (_, q_chunks, t_chunks),
                phase_stats.chunk_conversion_seconds,
                phase_stats.merge_tree_seconds,
                phase_stats.merge_frontier_max_nodes,
            ) = merge_partials_chunked(
                partials,
                hybrid_multiplier,
            )
            del partials

        if sqrt_chunks is None:
            sqrt_convert_start = time.perf_counter()
            sqrt_chunks = mpz_to_chunk_integer(sqrt_term, gpu_chunk_format)
            phase_stats.sqrt_conversion_seconds = time.perf_counter() - sqrt_convert_start
        else:
            phase_stats.sqrt_conversion_seconds = 0.0

        final_mul_start = time.perf_counter()
        numerator_chunks = hybrid_multiplier.mul_chunks(sqrt_chunks, q_chunks, stage="final")
        phase_stats.merge_tree_seconds += time.perf_counter() - final_mul_start

        resolved_division_mode = division_mode
        if division_mode == "auto":
            if hybrid_multiplier.stats.budget_cpu_calls > 0 or hybrid_multiplier.stats.oom_cpu_calls > 0:
                resolved_division_mode = "mpz-div"
            else:
                resolved_division_mode = "newton-chunk-gpu-seed-prototype"
        phase_stats.resolved_division_mode = resolved_division_mode

        if resolved_division_mode in {"newton-chunk-prototype", "newton-chunk-gpu-seed-prototype"}:
            division_start = time.perf_counter()
            scaled_numerator_chunks = mul_chunk_integer_small(
                numerator_chunks,
                426880,
                hybrid_multiplier.chunk_base,
            )
            quotient_chunks, newton_stats = hybrid_multiplier.divide_chunks_newton_prototype(
                scaled_numerator_chunks,
                t_chunks,
                seed_mode=("gpu-doubling" if resolved_division_mode == "newton-chunk-gpu-seed-prototype" else "mpz-exact"),
            )
            phase_stats.final_division_seconds = time.perf_counter() - division_start
            phase_stats.newton_reciprocal_seconds = newton_stats.reciprocal_seconds
            phase_stats.newton_qd_multiply_seconds = newton_stats.qd_multiply_seconds
            phase_stats.newton_correction_seconds = newton_stats.correction_seconds
            phase_stats.newton_reciprocal_iterations = newton_stats.reciprocal_iterations
            phase_stats.newton_correction_digits = newton_stats.correction_digits
            phase_stats.newton_cpu_correction_used = newton_stats.cpu_correction_used
            phase_stats.newton_chunk_fastpath_used = newton_stats.chunk_fastpath_used

            final_to_mpz_start = time.perf_counter()
            pi_scaled = chunk_integer_to_mpz(quotient_chunks, gpu_chunk_format)
            phase_stats.final_chunk_to_mpz_seconds = time.perf_counter() - final_to_mpz_start
            numerator = None
            t = None
        else:
            final_to_mpz_start = time.perf_counter()
            numerator = chunk_integer_to_mpz(numerator_chunks, gpu_chunk_format)
            t = chunk_integer_to_mpz(t_chunks, gpu_chunk_format)
            phase_stats.final_chunk_to_mpz_seconds = time.perf_counter() - final_to_mpz_start
            pi_scaled = None
    else:
        phase_stats.resolved_division_mode = division_mode
        if not stream_partials:
            (_, q, t), phase_stats.merge_tree_seconds, phase_stats.merge_frontier_max_nodes = merge_partials_hybrid(
                partials,
                hybrid_multiplier,
            )
            del partials
        numerator = hybrid_multiplier.mul(sqrt_term, q, stage="final")
        pi_scaled = None

    if pi_scaled is None:
        division_start = time.perf_counter()
        scaled_numerator = gmpy2.mpz(426880) * numerator
        resolved_division_mode = phase_stats.resolved_division_mode or division_mode
        if resolved_division_mode == "auto":
            resolved_division_mode = "mpz-div"
            phase_stats.resolved_division_mode = resolved_division_mode
        if resolved_division_mode == "mpfr-exact":
            pi_scaled, phase_stats.division_fallback_used = exact_division_via_mpfr(
                scaled_numerator,
                t,
                quotient_digits_hint=total_digits + 2,
            )
        elif resolved_division_mode == "mpz-div":
            pi_scaled = scaled_numerator // t
            phase_stats.division_fallback_used = False
        else:
            raise ValueError(f"unsupported division_mode: {resolved_division_mode}")
        phase_stats.final_division_seconds = time.perf_counter() - division_start
    else:
        phase_stats.division_fallback_used = False
    if guard_digits:
        pi_scaled //= gmpy2.mpz(10) ** guard_digits
    return pi_scaled, terms, hybrid_multiplier.stats, phase_stats


def benchmark_digits(
    digits_list: list[int],
    gpu_threshold_digits: int,
    gpu_stages: str,
    gpu_chunk_format: str,
    csv_path: Path,
    output_path: Path | None,
    workers_override: int | None = None,
    chunk_terms_override: int | None = None,
    bs_leaf_terms: int = PROJECT2_BS_LEAF_TERMS,
    sqrt_mode: str = "mpz-isqrt",
    division_mode: str = "mpz-div",
    chunk_arith_backend: str = PROJECT2_CHUNK_ARITH_BACKEND,
    gpu_memory_budget_gb: float | None = None,
    stream_partials: bool = False,
) -> tuple[list[dict[str, object]], str, int]:
    rows: list[dict[str, object]] = []
    highest_digits = max(digits_list)
    highest_output = ""

    set_chunk_arith_backend(chunk_arith_backend)
    if gpu_stages == "merge-and-final" and gpu_chunk_format == "binary16" and chunk_arith_backend != "python":
        ensure_cuda_chunk_ops_ready(required=(chunk_arith_backend == "cuda-ext"))

    for digits in digits_list:
        if workers_override is None or chunk_terms_override is None:
            workers, chunk_terms = resolve_parallel_config(digits)
            if workers_override is not None:
                workers = workers_override
            if chunk_terms_override is not None:
                chunk_terms = chunk_terms_override
        else:
            workers = workers_override
            chunk_terms = chunk_terms_override
        start = time.perf_counter()
        pi_digits, terms, hybrid_stats, phase_stats = compute_pi_digits_integer_gpu_hybrid(
            digits=digits,
            workers=workers,
            chunk_terms=chunk_terms,
            bs_leaf_terms=bs_leaf_terms,
            gpu_threshold_digits=gpu_threshold_digits,
            gpu_stages=gpu_stages,
            gpu_chunk_format=gpu_chunk_format,
            sqrt_mode=sqrt_mode,
            division_mode=division_mode,
            gpu_memory_budget_gb=gpu_memory_budget_gb,
            stream_partials=stream_partials,
        )
        elapsed = time.perf_counter() - start
        prefix_ok = pi_prefix_matches_reference(pi_digits, digits)
        if digits == highest_digits and output_path is not None:
            highest_output = format_pi_digits_integer(pi_digits, digits)

        rows.append(
            {
                "digits": digits,
                "terms": terms,
                "seconds": elapsed,
                "digits_per_second": digits / elapsed,
                "workers_used": workers,
                "chunk_terms": chunk_terms,
                "bs_leaf_terms": bs_leaf_terms,
                "gpu_threshold_digits": gpu_threshold_digits,
                "gpu_memory_budget_gb": gpu_memory_budget_gb,
                "gpu_stages": gpu_stages,
                "gpu_chunk_format": gpu_chunk_format,
                "stream_partials": stream_partials,
                "sqrt_mode": sqrt_mode,
                "division_mode": division_mode,
                "resolved_division_mode": phase_stats.resolved_division_mode or division_mode,
                "chunk_arith_backend": chunk_arith_backend,
                "gpu_calls": hybrid_stats.gpu_calls,
                "cpu_calls": hybrid_stats.cpu_calls,
                "merge_gpu_calls": hybrid_stats.merge_gpu_calls,
                "final_gpu_calls": hybrid_stats.final_gpu_calls,
                "gpu_prepare_seconds": hybrid_stats.gpu_prepare_seconds,
                "gpu_backend_total_seconds": hybrid_stats.gpu_backend_total_seconds,
                "gpu_kernel_seconds": hybrid_stats.gpu_kernel_seconds,
                "gpu_finalize_seconds": hybrid_stats.gpu_finalize_seconds,
                "gpu_peak_gb": hybrid_stats.gpu_peak_gb,
                "gpu_max_operand_digits": hybrid_stats.gpu_max_operand_digits,
                "budget_cpu_calls": hybrid_stats.budget_cpu_calls,
                "oom_cpu_calls": hybrid_stats.oom_cpu_calls,
                "budget_max_operand_digits": hybrid_stats.budget_max_operand_digits,
                "budget_max_estimated_peak_gb": hybrid_stats.budget_max_estimated_peak_gb,
                "partial_generation_seconds": phase_stats.partial_generation_seconds,
                "partial_stream_seconds": phase_stats.partial_stream_seconds,
                "streaming_partials_used": phase_stats.streaming_partials_used,
                "sqrt_compute_seconds": phase_stats.sqrt_compute_seconds,
                "chunk_conversion_seconds": phase_stats.chunk_conversion_seconds,
                "merge_tree_seconds": phase_stats.merge_tree_seconds,
                "merge_frontier_max_nodes": phase_stats.merge_frontier_max_nodes,
                "sqrt_conversion_seconds": phase_stats.sqrt_conversion_seconds,
                "final_chunk_to_mpz_seconds": phase_stats.final_chunk_to_mpz_seconds,
                "final_division_seconds": phase_stats.final_division_seconds,
                "division_fallback_used": phase_stats.division_fallback_used,
                "newton_reciprocal_seconds": phase_stats.newton_reciprocal_seconds,
                "newton_qd_multiply_seconds": phase_stats.newton_qd_multiply_seconds,
                "newton_correction_seconds": phase_stats.newton_correction_seconds,
                "newton_reciprocal_iterations": phase_stats.newton_reciprocal_iterations,
                "newton_correction_digits": phase_stats.newton_correction_digits,
                "newton_cpu_correction_used": phase_stats.newton_cpu_correction_used,
                "newton_chunk_fastpath_used": phase_stats.newton_chunk_fastpath_used,
                "prefix_matches_reference": prefix_ok,
            }
        )

        print(
            "benchmark",
            f"digits={digits}",
            f"terms={terms}",
            f"seconds={elapsed:.6f}",
            f"digits_per_second={digits / elapsed:.6f}",
            f"workers={workers}",
            f"chunk_terms={chunk_terms}",
            f"bs_leaf_terms={bs_leaf_terms}",
            f"gpu_stages={gpu_stages}",
            f"gpu_chunk_format={gpu_chunk_format}",
            f"stream_partials={stream_partials}",
            f"sqrt_mode={sqrt_mode}",
            f"division_mode={division_mode}",
            f"resolved_division_mode={phase_stats.resolved_division_mode or division_mode}",
            f"chunk_arith_backend={chunk_arith_backend}",
            f"gpu_memory_budget_gb={gpu_memory_budget_gb}",
            f"gpu_calls={hybrid_stats.gpu_calls}",
            f"merge_gpu_calls={hybrid_stats.merge_gpu_calls}",
            f"final_gpu_calls={hybrid_stats.final_gpu_calls}",
            f"gpu_backend_total_seconds={hybrid_stats.gpu_backend_total_seconds:.6f}",
            f"gpu_prepare_seconds={hybrid_stats.gpu_prepare_seconds:.6f}",
            f"gpu_finalize_seconds={hybrid_stats.gpu_finalize_seconds:.6f}",
            f"gpu_kernel_seconds={hybrid_stats.gpu_kernel_seconds:.6f}",
            f"gpu_peak_gb={hybrid_stats.gpu_peak_gb:.6f}",
            f"budget_cpu_calls={hybrid_stats.budget_cpu_calls}",
            f"oom_cpu_calls={hybrid_stats.oom_cpu_calls}",
            f"budget_max_estimated_peak_gb={hybrid_stats.budget_max_estimated_peak_gb:.6f}",
            f"partial_generation_seconds={phase_stats.partial_generation_seconds:.6f}",
            f"partial_stream_seconds={phase_stats.partial_stream_seconds:.6f}",
            f"streaming_partials_used={phase_stats.streaming_partials_used}",
            f"sqrt_compute_seconds={phase_stats.sqrt_compute_seconds:.6f}",
            f"chunk_conversion_seconds={phase_stats.chunk_conversion_seconds:.6f}",
            f"merge_tree_seconds={phase_stats.merge_tree_seconds:.6f}",
            f"merge_frontier_max_nodes={phase_stats.merge_frontier_max_nodes}",
            f"final_chunk_to_mpz_seconds={phase_stats.final_chunk_to_mpz_seconds:.6f}",
            f"final_division_seconds={phase_stats.final_division_seconds:.6f}",
            f"division_fallback_used={phase_stats.division_fallback_used}",
            f"newton_reciprocal_seconds={phase_stats.newton_reciprocal_seconds:.6f}",
            f"newton_qd_multiply_seconds={phase_stats.newton_qd_multiply_seconds:.6f}",
            f"newton_correction_seconds={phase_stats.newton_correction_seconds:.6f}",
            f"newton_reciprocal_iterations={phase_stats.newton_reciprocal_iterations}",
            f"newton_correction_digits={phase_stats.newton_correction_digits}",
            f"newton_cpu_correction_used={phase_stats.newton_cpu_correction_used}",
            f"newton_chunk_fastpath_used={phase_stats.newton_chunk_fastpath_used}",
            f"prefix_ok={prefix_ok}",
        )

    write_csv(csv_path, rows)

    if output_path is not None:
        output_path.write_text(highest_output + "\n", encoding="utf-8")

    return rows, highest_output, highest_digits


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid Pi backend: CPU binary splitting plus GPU exact multiplication.")
    parser.add_argument("--profile", choices=sorted(PROJECT2_HYBRID_PROFILES), default="fast-auto")
    parser.add_argument("--digits-list", default="100000,500000,1000000")
    parser.add_argument("--gpu-threshold-digits", type=int, default=None)
    parser.add_argument("--gpu-memory-budget-gb", type=float, default=None)
    parser.add_argument("--gpu-stages", choices=["final-only", "merge-and-final"], default=None)
    parser.add_argument("--gpu-chunk-format", choices=["decimal4", "binary16"], default=None)
    parser.add_argument(
        "--sqrt-mode",
        choices=["mpz-isqrt", "mpfr-exact", "chunk-gpu-rsqrt-prototype"],
        default=None,
    )
    parser.add_argument(
        "--division-mode",
        choices=["auto", "mpz-div", "mpfr-exact", "newton-chunk-prototype", "newton-chunk-gpu-seed-prototype"],
        default=None,
    )
    parser.add_argument("--chunk-arith-backend", choices=["python", "cuda-ext", "auto"], default=None)
    parser.add_argument("--stream-partials", dest="stream_partials", action="store_true")
    parser.add_argument("--no-stream-partials", dest="stream_partials", action="store_false")
    parser.set_defaults(stream_partials=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunk-terms", type=int, default=None)
    parser.add_argument("--bs-leaf-terms", type=int, default=PROJECT2_BS_LEAF_TERMS)
    parser.add_argument("--csv", default="result/project2_gpu_hybrid_pi_benchmark.csv")
    parser.add_argument("--output", default="result/project2_pi_gpu_hybrid_1000000_digits.txt")
    return parser


def apply_cli_profile(args: argparse.Namespace) -> argparse.Namespace:
    profile = PROJECT2_HYBRID_PROFILES[args.profile]
    if args.gpu_threshold_digits is None:
        args.gpu_threshold_digits = int(profile["gpu_threshold_digits"])
    if args.gpu_memory_budget_gb is None:
        args.gpu_memory_budget_gb = profile.get("gpu_memory_budget_gb")
    if args.gpu_stages is None:
        args.gpu_stages = str(profile["gpu_stages"])
    if args.gpu_chunk_format is None:
        args.gpu_chunk_format = str(profile["gpu_chunk_format"])
    if args.sqrt_mode is None:
        args.sqrt_mode = str(profile["sqrt_mode"])
    if args.division_mode is None:
        args.division_mode = str(profile["division_mode"])
    if args.chunk_arith_backend is None:
        args.chunk_arith_backend = str(profile["chunk_arith_backend"])
    if args.stream_partials is None:
        args.stream_partials = bool(profile.get("stream_partials", False))
    return args


def main() -> None:
    args = apply_cli_profile(build_argument_parser().parse_args())
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    digits_list = parse_digits_list(args.digits_list)
    benchmark_digits(
        digits_list=digits_list,
        gpu_threshold_digits=args.gpu_threshold_digits,
        gpu_stages=args.gpu_stages,
        gpu_chunk_format=args.gpu_chunk_format,
        csv_path=Path(args.csv),
        output_path=Path(args.output) if args.output else None,
        workers_override=args.workers,
        chunk_terms_override=args.chunk_terms,
        bs_leaf_terms=args.bs_leaf_terms,
        sqrt_mode=args.sqrt_mode,
        division_mode=args.division_mode,
        chunk_arith_backend=args.chunk_arith_backend,
        gpu_memory_budget_gb=args.gpu_memory_budget_gb,
        stream_partials=args.stream_partials,
    )


if __name__ == "__main__":
    main()
