from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


DECIMAL_BASE = 10_000
CHUNK_DIGITS = 4
SPLIT_BASE = 100
SPLIT_DIGITS = 2
FFT_SAFETY_LIMIT = 1 << 50
BATCHED_LANE_CHUNK_LIMIT = 30_000_000


def next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def parse_digits_list(text: str) -> list[int]:
    values = [int(token) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("digits list must not be empty")
    return values


def digits_to_chunks(digits: int) -> int:
    return (digits + CHUNK_DIGITS - 1) // CHUNK_DIGITS


def estimate_fft_working_set_bytes(
    chunk_count_a: int,
    chunk_count_b: int,
    *,
    use_batched: bool | None = None,
) -> int:
    if chunk_count_a <= 0 or chunk_count_b <= 0:
        raise ValueError("chunk counts must be positive")

    valid_length = chunk_count_a + chunk_count_b - 1
    fft_length = next_power_of_two(valid_length)
    if use_batched is None:
        use_batched = max(chunk_count_a, chunk_count_b) <= BATCHED_LANE_CHUNK_LIMIT

    int64_bytes = 8
    float64_bytes = 8
    complex128_bytes = 16

    input_and_split_bytes = int64_bytes * (4 * chunk_count_a + 4 * chunk_count_b)
    resident_coeff_bytes = int64_bytes * (6 * valid_length)

    if use_batched:
        lane_count = 3
        temp_bytes = (
            float64_bytes * (2 * lane_count * fft_length + lane_count * valid_length)
            + complex128_bytes * (2 * lane_count * (fft_length // 2 + 1))
            + int64_bytes * (lane_count * valid_length)
        )
    else:
        temp_bytes = (
            float64_bytes * (2 * fft_length + valid_length)
            + complex128_bytes * (2 * (fft_length // 2 + 1))
            + int64_bytes * valid_length
        )

    return input_and_split_bytes + resident_coeff_bytes + temp_bytes


def estimate_fft_working_set_gb(
    chunk_count_a: int,
    chunk_count_b: int,
    *,
    use_batched: bool | None = None,
) -> float:
    return estimate_fft_working_set_bytes(
        chunk_count_a,
        chunk_count_b,
        use_batched=use_batched,
    ) / (1024**3)


def normalize_decimal_text(text: str) -> tuple[int, str]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("decimal input must not be empty")

    sign = 1
    if stripped[0] in "+-":
        sign = -1 if stripped[0] == "-" else 1
        stripped = stripped[1:]

    if not stripped or any(char not in "0123456789" for char in stripped):
        raise ValueError("decimal input must contain only digits with an optional leading sign")

    stripped = stripped.lstrip("0") or "0"
    return (1 if stripped == "0" else sign), stripped


def decimal_string_to_chunks(text: str) -> tuple[int, torch.Tensor]:
    sign, digits = normalize_decimal_text(text)
    chunks = [
        int(digits[max(0, end - CHUNK_DIGITS) : end])
        for end in range(len(digits), 0, -CHUNK_DIGITS)
    ]
    return sign, torch.tensor(chunks, dtype=torch.int64, device="cpu")


def chunks_to_decimal_string(chunks: torch.Tensor, sign: int = 1) -> str:
    values = chunks.to(dtype=torch.int64, device="cpu").tolist()
    text = str(values[-1]) + "".join(f"{value:0{CHUNK_DIGITS}d}" for value in reversed(values[:-1]))
    if sign < 0 and text != "0":
        return "-" + text
    return text


def write_decimal_chunks_file(chunks: torch.Tensor, path: Path, sign: int = 1, batch_size: int = 100_000) -> None:
    values = chunks.to(dtype=torch.int64, device="cpu").numpy()
    with path.open("w", encoding="utf-8") as handle:
        if sign < 0 and not (values.size == 1 and int(values[0]) == 0):
            handle.write("-")
        handle.write(str(int(values[-1])))
        for end in range(values.size - 1, 0, -batch_size):
            start = max(0, end - batch_size)
            block = values[start:end][::-1]
            handle.write("".join(f"{int(value):0{CHUNK_DIGITS}d}" for value in block))
        handle.write("\n")


def normalize_int64_contiguous(chunks: torch.Tensor) -> torch.Tensor:
    normalized = chunks
    if normalized.dtype != torch.int64:
        normalized = normalized.to(dtype=torch.int64)
    if not normalized.is_contiguous():
        normalized = normalized.contiguous()
    return normalized


def select_effective_chunks(chunks: torch.Tensor, declared_length: int | None, name: str) -> torch.Tensor:
    normalized = normalize_int64_contiguous(chunks)
    if declared_length is None:
        return normalized
    if declared_length <= 0 or declared_length > int(normalized.numel()):
        raise ValueError(f"{name}_length must be in [1, {int(normalized.numel())}]")
    if declared_length == int(normalized.numel()):
        return normalized
    prefix = normalized.narrow(0, 0, declared_length)
    if not prefix.is_contiguous():
        prefix = prefix.contiguous()
    return prefix


def is_zero_trimmed_chunks(chunks: torch.Tensor) -> bool:
    return chunks.numel() == 1 and int(chunks[0].item()) == 0


def random_chunks_for_digits(digits: int, seed: int) -> torch.Tensor:
    chunk_count = digits_to_chunks(digits)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    chunks = torch.randint(0, DECIMAL_BASE, (chunk_count,), dtype=torch.int64, generator=generator)
    if chunks[-1].item() == 0:
        chunks[-1] = 1
    return chunks


@dataclass
class MultiplyStats:
    chunk_count_a: int
    chunk_count_b: int
    valid_length: int
    fft_length: int
    decimal_base: int
    split_base: int
    split_seconds: float
    upload_seconds: float
    kernel_seconds: float
    download_seconds: float
    combine_seconds: float
    normalize_seconds: float
    total_seconds: float
    carry_passes: int
    peak_gpu_gb: float
    device: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class GpuFftMultiplier:
    def __init__(
        self,
        device: str = "cuda",
        decimal_base: int = DECIMAL_BASE,
        split_base: int = SPLIT_BASE,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; cannot initialize GPU FFT backend.")
        if decimal_base != split_base * split_base:
            raise ValueError("decimal_base must equal split_base^2 for the two-lane exact backend.")

        self.device = torch.device(device)
        self.decimal_base = decimal_base
        self.split_base = split_base
        self.device_name = torch.cuda.get_device_name(self.device)

    def estimate_lane_bound(self, chunk_count_a: int, chunk_count_b: int) -> int:
        max_lane_value = 2 * self.split_base - 2
        return min(chunk_count_a, chunk_count_b) * max_lane_value * max_lane_value

    def ensure_supported(self, chunk_count_a: int, chunk_count_b: int) -> None:
        lane_bound = self.estimate_lane_bound(chunk_count_a, chunk_count_b)
        if lane_bound >= FFT_SAFETY_LIMIT:
            raise ValueError(
                "operand is too large for the current exact two-lane float64 FFT backend: "
                f"estimated lane bound {lane_bound} exceeds {FFT_SAFETY_LIMIT}"
            )

    def _split_chunks(self, chunks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lower = torch.remainder(chunks, self.split_base)
        upper = torch.div(chunks, self.split_base, rounding_mode="floor")
        summed = lower + upper
        return lower, upper, summed

    def _convolve_lane(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        fft_length: int,
        valid_length: int,
    ) -> tuple[torch.Tensor, float, float]:
        stage_start = time.perf_counter()
        lhs_gpu = torch.zeros(fft_length, dtype=torch.float64, device=self.device)
        rhs_gpu = torch.zeros(fft_length, dtype=torch.float64, device=self.device)
        lhs_gpu[: lhs.numel()] = lhs.to(dtype=torch.float64)
        rhs_gpu[: rhs.numel()] = rhs.to(dtype=torch.float64)
        torch.cuda.synchronize(self.device)
        stage_seconds = time.perf_counter() - stage_start

        kernel_start = time.perf_counter()
        lhs_fft = torch.fft.rfft(lhs_gpu)
        rhs_fft = torch.fft.rfft(rhs_gpu)
        conv = torch.fft.irfft(lhs_fft * rhs_fft, n=fft_length)[:valid_length]
        coeffs = torch.round(conv).to(torch.int64)
        torch.cuda.synchronize(self.device)
        kernel_seconds = time.perf_counter() - kernel_start

        del lhs_gpu, rhs_gpu, lhs_fft, rhs_fft, conv
        return coeffs, stage_seconds, kernel_seconds

    def _convolve_lanes_batched(
        self,
        lhs_lanes: tuple[torch.Tensor, ...],
        rhs_lanes: tuple[torch.Tensor, ...],
        fft_length: int,
        valid_length: int,
    ) -> tuple[torch.Tensor, float, float]:
        if len(lhs_lanes) != len(rhs_lanes):
            raise ValueError("lhs_lanes and rhs_lanes must have the same length")

        stage_start = time.perf_counter()
        lane_count = len(lhs_lanes)
        lhs_gpu = torch.zeros((lane_count, fft_length), dtype=torch.float64, device=self.device)
        rhs_gpu = torch.zeros((lane_count, fft_length), dtype=torch.float64, device=self.device)
        for lane_index, lhs in enumerate(lhs_lanes):
            lhs_gpu[lane_index, : lhs.numel()] = lhs.to(dtype=torch.float64)
        for lane_index, rhs in enumerate(rhs_lanes):
            rhs_gpu[lane_index, : rhs.numel()] = rhs.to(dtype=torch.float64)
        torch.cuda.synchronize(self.device)
        stage_seconds = time.perf_counter() - stage_start

        kernel_start = time.perf_counter()
        lhs_fft = torch.fft.rfft(lhs_gpu, dim=-1)
        rhs_fft = torch.fft.rfft(rhs_gpu, dim=-1)
        conv = torch.fft.irfft(lhs_fft * rhs_fft, n=fft_length, dim=-1)[:, :valid_length]
        coeffs = torch.round(conv).to(torch.int64)
        torch.cuda.synchronize(self.device)
        kernel_seconds = time.perf_counter() - kernel_start

        del lhs_gpu, rhs_gpu, lhs_fft, rhs_fft, conv
        return coeffs, stage_seconds, kernel_seconds

    def _normalize_coefficients_gpu(self, coeffs: torch.Tensor) -> tuple[torch.Tensor, int]:
        normalized = coeffs.to(dtype=torch.int64, device=self.device, copy=True)
        carry_passes = 0

        while True:
            carry = torch.div(normalized, self.decimal_base, rounding_mode="floor")
            if int(torch.count_nonzero(carry).item()) == 0:
                break

            normalized = normalized - carry * self.decimal_base
            carry_body = carry[:-1]

            if int(torch.count_nonzero(carry[-1:]).item()) != 0:
                normalized = torch.cat((normalized, carry[-1:].clone()))

            if carry_body.numel() > 0:
                normalized[1 : 1 + carry_body.numel()] += carry_body

            carry_passes += 1

        non_zero = torch.nonzero(normalized, as_tuple=False)
        if non_zero.numel() == 0:
            return torch.tensor([0], dtype=torch.int64, device=self.device), carry_passes
        normalized = normalized[: int(non_zero[-1].item()) + 1]
        return normalized, carry_passes

    def multiply_chunks(
        self,
        a_chunks: torch.Tensor,
        b_chunks: torch.Tensor,
        output_device: str = "cpu",
        assume_nonzero_trimmed: bool = False,
        a_length: int | None = None,
        b_length: int | None = None,
    ) -> tuple[torch.Tensor, MultiplyStats]:
        if a_chunks.ndim != 1 or b_chunks.ndim != 1:
            raise ValueError("chunk inputs must be 1-D tensors")
        if a_chunks.numel() == 0 or b_chunks.numel() == 0:
            raise ValueError("chunk inputs must not be empty")
        if output_device not in {"cpu", "cuda"}:
            raise ValueError("output_device must be either 'cpu' or 'cuda'")

        a_chunks = select_effective_chunks(a_chunks, a_length, "a")
        b_chunks = select_effective_chunks(b_chunks, b_length, "b")
        same_device_inputs = (
            a_chunks.device.type == "cuda"
            and b_chunks.device.type == "cuda"
            and a_chunks.device == self.device
            and b_chunks.device == self.device
        )

        if assume_nonzero_trimmed:
            a_is_zero = is_zero_trimmed_chunks(a_chunks)
            b_is_zero = is_zero_trimmed_chunks(b_chunks)
        else:
            a_is_zero = int(torch.count_nonzero(a_chunks).item()) == 0
            b_is_zero = int(torch.count_nonzero(b_chunks).item()) == 0
        if a_is_zero or b_is_zero:
            zero_device = self.device if output_device == "cuda" else torch.device("cpu")
            stats = MultiplyStats(
                chunk_count_a=int(a_chunks.numel()),
                chunk_count_b=int(b_chunks.numel()),
                valid_length=1,
                fft_length=1,
                decimal_base=self.decimal_base,
                split_base=self.split_base,
                split_seconds=0.0,
                upload_seconds=0.0,
                kernel_seconds=0.0,
                download_seconds=0.0,
                combine_seconds=0.0,
                normalize_seconds=0.0,
                total_seconds=0.0,
                carry_passes=0,
                peak_gpu_gb=0.0,
                device=self.device_name,
            )
            return torch.tensor([0], dtype=torch.int64, device=zero_device), stats

        self.ensure_supported(int(a_chunks.numel()), int(b_chunks.numel()))
        torch.cuda.reset_peak_memory_stats(self.device)

        total_start = time.perf_counter()
        valid_length = int(a_chunks.numel() + b_chunks.numel() - 1)
        fft_length = next_power_of_two(valid_length)

        if same_device_inputs:
            a_gpu = a_chunks.to(device=self.device, dtype=torch.int64, copy=False)
            b_gpu = b_chunks.to(device=self.device, dtype=torch.int64, copy=False)
            upload_seconds = 0.0
        else:
            upload_start = time.perf_counter()
            a_gpu = a_chunks.to(device=self.device, dtype=torch.int64)
            b_gpu = b_chunks.to(device=self.device, dtype=torch.int64)
            torch.cuda.synchronize(self.device)
            upload_seconds = time.perf_counter() - upload_start

        split_start = time.perf_counter()
        a_lower, a_upper, a_sum = self._split_chunks(a_gpu)
        b_lower, b_upper, b_sum = self._split_chunks(b_gpu)
        torch.cuda.synchronize(self.device)
        split_seconds = time.perf_counter() - split_start

        kernel_seconds = 0.0
        download_seconds = 0.0

        use_batched = max(int(a_chunks.numel()), int(b_chunks.numel())) <= BATCHED_LANE_CHUNK_LIMIT
        if use_batched:
            try:
                coeffs_batched, stage, ker = self._convolve_lanes_batched(
                    (a_lower, a_upper, a_sum),
                    (b_lower, b_upper, b_sum),
                    fft_length,
                    valid_length,
                )
                upload_seconds += stage
                kernel_seconds += ker
                coeff_ll = coeffs_batched[0]
                coeff_hh = coeffs_batched[1]
                coeff_sum = coeffs_batched[2]
                del coeffs_batched
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                use_batched = False

        if not use_batched:
            coeff_ll, stage, ker = self._convolve_lane(a_lower, b_lower, fft_length, valid_length)
            upload_seconds += stage
            kernel_seconds += ker

            coeff_hh, stage, ker = self._convolve_lane(a_upper, b_upper, fft_length, valid_length)
            upload_seconds += stage
            kernel_seconds += ker

            coeff_sum, stage, ker = self._convolve_lane(a_sum, b_sum, fft_length, valid_length)
            upload_seconds += stage
            kernel_seconds += ker

        combine_start = time.perf_counter()
        coeff_cross = coeff_sum - coeff_ll - coeff_hh
        combined = coeff_ll + self.split_base * coeff_cross + (self.split_base * self.split_base) * coeff_hh
        torch.cuda.synchronize(self.device)
        combine_seconds = time.perf_counter() - combine_start

        normalize_start = time.perf_counter()
        normalized_gpu, carry_passes = self._normalize_coefficients_gpu(combined)
        torch.cuda.synchronize(self.device)
        normalize_seconds = time.perf_counter() - normalize_start

        if output_device == "cuda":
            normalized = normalized_gpu
            download_seconds = 0.0
        else:
            download_start = time.perf_counter()
            normalized = normalized_gpu.to(dtype=torch.int64, device="cpu")
            torch.cuda.synchronize(self.device)
            download_seconds = time.perf_counter() - download_start

        total_seconds = time.perf_counter() - total_start
        peak_gpu_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)

        del a_gpu, b_gpu
        del a_lower, a_upper, a_sum, b_lower, b_upper, b_sum
        del coeff_ll, coeff_hh, coeff_sum, coeff_cross, combined
        if output_device != "cuda":
            del normalized_gpu

        stats = MultiplyStats(
            chunk_count_a=int(a_chunks.numel()),
            chunk_count_b=int(b_chunks.numel()),
            valid_length=valid_length,
            fft_length=fft_length,
            decimal_base=self.decimal_base,
            split_base=self.split_base,
            split_seconds=split_seconds,
            upload_seconds=upload_seconds,
            kernel_seconds=kernel_seconds,
            download_seconds=download_seconds,
            combine_seconds=combine_seconds,
            normalize_seconds=normalize_seconds,
            total_seconds=total_seconds,
            carry_passes=carry_passes,
            peak_gpu_gb=peak_gpu_gb,
            device=self.device_name,
        )
        return normalized, stats

    def multiply_decimal_strings(
        self,
        lhs_text: str,
        rhs_text: str,
        output_path: Path | None = None,
    ) -> tuple[str | None, MultiplyStats]:
        sign_lhs, lhs_chunks = decimal_string_to_chunks(lhs_text)
        sign_rhs, rhs_chunks = decimal_string_to_chunks(rhs_text)
        sign = sign_lhs * sign_rhs
        result_chunks, stats = self.multiply_chunks(lhs_chunks, rhs_chunks)
        if output_path is not None:
            write_decimal_chunks_file(result_chunks, output_path, sign)
            return None, stats
        return chunks_to_decimal_string(result_chunks, sign), stats


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact big-integer multiplication backend using CUDA FFT with two-lane splitting.")
    parser.add_argument("--lhs", help="Left decimal operand.")
    parser.add_argument("--rhs", help="Right decimal operand.")
    parser.add_argument("--lhs-file", help="Path to a file containing the left decimal operand.")
    parser.add_argument("--rhs-file", help="Path to a file containing the right decimal operand.")
    parser.add_argument("--output", help="Optional output file for the exact product.")
    parser.add_argument("--preview-digits", type=int, default=80, help="When not writing to a file, print at most this many leading digits.")
    return parser


def load_operand(text: str | None, file_path: str | None) -> str:
    if text is not None and file_path is not None:
        raise ValueError("provide either inline operand text or a file path, not both")
    if text is not None:
        return text
    if file_path is not None:
        return Path(file_path).read_text(encoding="utf-8").strip()
    raise ValueError("missing operand")


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    lhs = load_operand(args.lhs, args.lhs_file)
    rhs = load_operand(args.rhs, args.rhs_file)
    multiplier = GpuFftMultiplier()
    output_path = Path(args.output) if args.output else None
    result_text, stats = multiplier.multiply_decimal_strings(lhs, rhs, output_path=output_path)

    if result_text is not None:
        preview = result_text[: args.preview_digits]
        if len(result_text) > args.preview_digits:
            preview += "..."
        print(preview)

    for key, value in stats.as_dict().items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
