from __future__ import annotations

import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import torch
import triton
import triton.language as tl

import gpu_ultra_fractal as gpu_ref


ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
RESULT_DIR = ROOT / "result"
GPU_TRITON_DIR = RESULT_DIR / "gpu_triton"
AXIS_MIN = -1.8
AXIS_MAX = 1.8
SQRT3_2 = math.sqrt(3.0) / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Triton Newton kernel against the current PyTorch path.")
    parser.add_argument("--sizes", type=str, default="1024,1536")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--output-prefix", type=str, default="triton_newton_bench")
    return parser.parse_args()


def ensure_result_dir() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    GPU_TRITON_DIR.mkdir(parents=True, exist_ok=True)


def make_run_dir(base_dir: Path, output_prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = base_dir / f"{output_prefix}__{stamp}"
    attempt = 1
    while candidate.exists():
        attempt += 1
        candidate = base_dir / f"{output_prefix}__{stamp}_{attempt:02d}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


@triton.jit
def newton_direct_kernel(
    zr0_ptr,
    zi0_ptr,
    root_ptr,
    iter_ptr,
    n_points,
    tol2,
    late_tol2,
    BLOCK_SIZE: tl.constexpr,
    MAX_ITER: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_points

    zr = tl.load(zr0_ptr + offs, mask=mask, other=0.0)
    zi = tl.load(zi0_ptr + offs, mask=mask, other=0.0)
    active = mask
    root = tl.full([BLOCK_SIZE], -1, tl.int32)
    iters = tl.zeros([BLOCK_SIZE], tl.int32)

    for step in tl.static_range(0, MAX_ITER):
        z2r = zr * zr - zi * zi
        z2i = 2.0 * zr * zi
        den = z2r * z2r + z2i * z2i
        safe = active & (den > 1.0e-14)

        invr = tl.where(safe, (1.0 / 3.0) * z2r / den, 0.0)
        invi = tl.where(safe, -(1.0 / 3.0) * z2i / den, 0.0)
        zr_next = (2.0 / 3.0) * zr + invr
        zi_next = (2.0 / 3.0) * zi + invi
        zr = tl.where(active, zr_next, zr)
        zi = tl.where(active, zi_next, zi)

        d0 = (zr - 1.0) * (zr - 1.0) + zi * zi
        tmp = zr + 0.5
        d1i = zi - 0.8660254037844386
        d2i = zi + 0.8660254037844386
        d1 = tmp * tmp + d1i * d1i
        d2 = tmp * tmp + d2i * d2i

        root_idx = tl.zeros([BLOCK_SIZE], tl.int32)
        min_d = d0
        pick1 = d1 < min_d
        min_d = tl.where(pick1, d1, min_d)
        root_idx = tl.where(pick1, 1, root_idx)
        pick2 = d2 < min_d
        min_d = tl.where(pick2, d2, min_d)
        root_idx = tl.where(pick2, 2, root_idx)

        just = active & (min_d < tol2)
        root = tl.where(just, root_idx, root)
        iters = tl.where(just, step + 1, iters)
        active = active & ~just

    d0 = (zr - 1.0) * (zr - 1.0) + zi * zi
    tmp = zr + 0.5
    d1i = zi - 0.8660254037844386
    d2i = zi + 0.8660254037844386
    d1 = tmp * tmp + d1i * d1i
    d2 = tmp * tmp + d2i * d2i

    root_idx = tl.zeros([BLOCK_SIZE], tl.int32)
    min_d = d0
    pick1 = d1 < min_d
    min_d = tl.where(pick1, d1, min_d)
    root_idx = tl.where(pick1, 1, root_idx)
    pick2 = d2 < min_d
    min_d = tl.where(pick2, d2, min_d)
    root_idx = tl.where(pick2, 2, root_idx)

    late = active & (min_d < late_tol2)
    root = tl.where(late, root_idx, root)
    iters = tl.where(late, MAX_ITER, iters)

    tl.store(root_ptr + offs, root, mask=mask)
    tl.store(iter_ptr + offs, iters, mask=mask)


def build_plane(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(AXIS_MIN, AXIS_MAX, n, dtype=torch.float32, device=device)
    y = torch.linspace(AXIS_MIN, AXIS_MAX, n, dtype=torch.float32, device=device)
    return x, y


def flatten_plane(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    zr0 = x[None, :].expand(y.shape[0], -1).contiguous().reshape(-1)
    zi0 = y[:, None].expand(-1, x.shape[0]).contiguous().reshape(-1)
    return zr0, zi0


def summarize_output(root_map: torch.Tensor, iters: torch.Tensor) -> dict[str, float]:
    total = root_map.numel()
    converged = root_map >= 0
    converged_total = int(converged.sum().item())
    iter_total = int(iters[converged].to(torch.int64).sum().item()) if converged_total else 0
    return {
        "convergence_fraction": converged_total / total,
        "mean_iterations": (iter_total / converged_total) if converged_total else float("nan"),
        "root0_fraction": float((root_map == 0).sum().item() / total),
        "root1_fraction": float((root_map == 1).sum().item() / total),
        "root2_fraction": float((root_map == 2).sum().item() / total),
    }


def run_reference_newton(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    roots = torch.tensor(
        [
            1.0 + 0.0j,
            complex(-0.5, SQRT3_2),
            complex(-0.5, -SQRT3_2),
        ],
        dtype=torch.complex64,
        device=x.device,
    )
    root_map, iters, *_ = gpu_ref.run_tile_dense(
        "newton",
        x,
        y,
        roots,
        5.0e-7,
        55,
        complex(0.2, 0.2),
        1,
        True,
    )
    return root_map, iters


def run_triton_newton(x: torch.Tensor, y: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    zr0, zi0 = flatten_plane(x, y)
    n_points = zr0.numel()
    root = torch.empty(n_points, dtype=torch.int32, device=x.device)
    iters = torch.empty(n_points, dtype=torch.int32, device=x.device)
    grid = lambda meta: (triton.cdiv(n_points, meta["BLOCK_SIZE"]),)
    newton_direct_kernel[grid](
        zr0,
        zi0,
        root,
        iters,
        n_points,
        5.0e-7 * 5.0e-7,
        (10.0 * 5.0e-7) ** 2,
        BLOCK_SIZE=block_size,
        MAX_ITER=55,
        num_warps=4,
    )
    return root.view(y.shape[0], x.shape[0]).to(torch.int16), iters.view(y.shape[0], x.shape[0]).to(torch.int16)


def time_call(fn, reps: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    ensure_result_dir()
    run_dir = make_run_dir(GPU_TRITON_DIR, args.output_prefix)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton benchmark.")

    device = torch.device("cuda")
    sizes = [int(item.strip()) for item in args.sizes.split(",") if item.strip()]
    rows: list[dict[str, float | int | str]] = []
    log_lines: list[str] = []

    for size in sizes:
        x, y = build_plane(size, device)
        ref_root, ref_iters = run_reference_newton(x, y)
        tri_root, tri_iters = run_triton_newton(x, y, args.block_size)

        ref_stats = summarize_output(ref_root, ref_iters)
        tri_stats = summarize_output(tri_root, tri_iters)

        root_diff = int((ref_root != tri_root).sum().item())
        iter_diff = int((ref_iters != tri_iters).sum().item())
        max_iter_diff = int(torch.max(torch.abs(ref_iters.to(torch.int32) - tri_iters.to(torch.int32))).item())

        compiled_note = "complex eager baseline vs Triton fused direct formula"
        log_lines.append(f"size={size}")
        log_lines.append(f"note={compiled_note}")
        log_lines.append(f"root_diff_pixels={root_diff}")
        log_lines.append(f"iter_diff_pixels={iter_diff}")
        log_lines.append(f"max_iter_diff={max_iter_diff}")
        for key in ("convergence_fraction", "mean_iterations", "root0_fraction", "root1_fraction", "root2_fraction"):
            log_lines.append(f"{key}_delta={ref_stats[key] - tri_stats[key]}")

        # Warm up once more before timing to avoid one-time setup in the measurements.
        run_triton_newton(x, y, args.block_size)
        torch.cuda.synchronize()

        ref_time = time_call(lambda: run_reference_newton(x, y), args.reps)
        tri_time = time_call(lambda: run_triton_newton(x, y, args.block_size), args.reps)

        log_lines.append(f"reference_seconds={ref_time}")
        log_lines.append(f"triton_seconds={tri_time}")
        log_lines.append("")

        rows.append(
            {
                "size": size,
                "block_size": args.block_size,
                "reps": args.reps,
                "root_diff_pixels": root_diff,
                "iter_diff_pixels": iter_diff,
                "max_iter_diff": max_iter_diff,
                "convergence_fraction_delta": ref_stats["convergence_fraction"] - tri_stats["convergence_fraction"],
                "mean_iterations_delta": ref_stats["mean_iterations"] - tri_stats["mean_iterations"],
                "root0_fraction_delta": ref_stats["root0_fraction"] - tri_stats["root0_fraction"],
                "root1_fraction_delta": ref_stats["root1_fraction"] - tri_stats["root1_fraction"],
                "root2_fraction_delta": ref_stats["root2_fraction"] - tri_stats["root2_fraction"],
                "reference_seconds": ref_time,
                "triton_seconds": tri_time,
                "speedup": ref_time / tri_time if tri_time else float("nan"),
            }
        )

    csv_path = run_dir / f"{args.output_prefix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "size",
                "block_size",
                "reps",
                "root_diff_pixels",
                "iter_diff_pixels",
                "max_iter_diff",
                "convergence_fraction_delta",
                "mean_iterations_delta",
                "root0_fraction_delta",
                "root1_fraction_delta",
                "root2_fraction_delta",
                "reference_seconds",
                "triton_seconds",
                "speedup",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    log_path = run_dir / f"{args.output_prefix}.log"
    log_path.write_text(f"run_dir={run_dir}\n\n" + "\n".join(log_lines), encoding="utf-8")
    print(log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
