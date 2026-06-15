from __future__ import annotations

import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

import doublefloat_newton_prototype as df_proto
import gpu_ultra_fractal as gpu_ref


ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
RESULT_DIR = ROOT / "result"
ANALYSIS_DIR = RESULT_DIR / "analysis"
AXIS_MIN = -1.8
AXIS_MAX = 1.8
SQRT3_2 = math.sqrt(3.0) / 2.0
SPLITTER = 4097.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triton double-float Newton prototype benchmark.")
    parser.add_argument("--coarse-grid", type=int, default=512)
    parser.add_argument("--test-grid", type=int, default=1024)
    parser.add_argument("--tol", type=float, default=5.0e-7)
    parser.add_argument("--max-iter", type=int, default=55)
    parser.add_argument("--zoom-step-ulp-ratio", type=float, default=0.5)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--post-polish-steps", type=int, default=2)
    parser.add_argument("--iter-replay-threshold", type=int, default=10)
    parser.add_argument("--iter-replay-dilation", type=int, default=1)
    parser.add_argument("--output-prefix", type=str, default="triton_doublefloat_prototype")
    return parser.parse_args()


def ensure_dirs() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


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
def ds_renorm(h, l):
    s = h + l
    e = l - (s - h)
    return s, e


@triton.jit
def ds_add_basic(ah, al, bh, bl):
    s = ah + bh
    v = s - ah
    t = ((bh - v) + (ah - (s - v))) + al + bl
    return ds_renorm(s, t)


@triton.jit
def ds_sub_basic(ah, al, bh, bl):
    return ds_add_basic(ah, al, -bh, -bl)


@triton.jit
def split_f32(a):
    c = 4097.0 * a
    ahi = c - (c - a)
    alo = a - ahi
    return ahi, alo


@triton.jit
def two_prod(a, b):
    p = a * b
    ahi, alo = split_f32(a)
    bhi, blo = split_f32(b)
    err = ((ahi * bhi - p) + ahi * blo + alo * bhi) + alo * blo
    return p, err


@triton.jit
def ds_mul_basic(ah, al, bh, bl):
    p, e = two_prod(ah, bh)
    e = e + ah * bl + al * bh
    return ds_renorm(p, e)


@triton.jit
def ds_square_basic(ah, al):
    return ds_mul_basic(ah, al, ah, al)


@triton.jit
def ds_mul_scalar_basic(ah, al, scalar):
    zero = tl.zeros_like(ah)
    scalar_vec = zero + scalar
    return ds_mul_basic(ah, al, scalar_vec, zero)


@triton.jit
def ds_reciprocal_real_basic(dh, dl):
    qh = 1.0 / dh
    ql = tl.zeros_like(qh)
    oneh = tl.zeros_like(qh) + 1.0
    onel = tl.zeros_like(qh)
    for _ in tl.static_range(0, 2):
        ph, pl = ds_mul_basic(dh, dl, qh, ql)
        eh, el = ds_sub_basic(oneh, onel, ph, pl)
        ch, cl = ds_mul_basic(qh, ql, eh, el)
        qh, ql = ds_add_basic(qh, ql, ch, cl)
    return qh, ql


@triton.jit
def newton_doublefloat_basic_kernel(
    zr_hi0_ptr,
    zr_lo0_ptr,
    zi_hi0_ptr,
    zi_lo0_ptr,
    zr_hi_out_ptr,
    zr_lo_out_ptr,
    zi_hi_out_ptr,
    zi_lo_out_ptr,
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

    zr_hi = tl.load(zr_hi0_ptr + offs, mask=mask, other=0.0)
    zr_lo = tl.load(zr_lo0_ptr + offs, mask=mask, other=0.0)
    zi_hi = tl.load(zi_hi0_ptr + offs, mask=mask, other=0.0)
    zi_lo = tl.load(zi_lo0_ptr + offs, mask=mask, other=0.0)

    active = mask
    root = tl.full([BLOCK_SIZE], -1, tl.int32)
    iters = tl.zeros([BLOCK_SIZE], tl.int32)

    for step in tl.range(0, MAX_ITER):
        zr2_hi, zr2_lo = ds_square_basic(zr_hi, zr_lo)
        zi2_hi, zi2_lo = ds_square_basic(zi_hi, zi_lo)
        z2r_hi, z2r_lo = ds_sub_basic(zr2_hi, zr2_lo, zi2_hi, zi2_lo)

        zrzi_hi, zrzi_lo = ds_mul_basic(zr_hi, zr_lo, zi_hi, zi_lo)
        z2i_hi, z2i_lo = ds_mul_scalar_basic(zrzi_hi, zrzi_lo, 2.0)

        a2_hi, a2_lo = ds_square_basic(z2r_hi, z2r_lo)
        b2_hi, b2_lo = ds_square_basic(z2i_hi, z2i_lo)
        den_hi, den_lo = ds_add_basic(a2_hi, a2_lo, b2_hi, b2_lo)
        den_approx = den_hi + den_lo
        safe = active & (den_approx > 1.0e-14)

        safe_den_hi = tl.where(safe, den_hi, 1.0)
        safe_den_lo = tl.where(safe, den_lo, 0.0)
        recip_hi, recip_lo = ds_reciprocal_real_basic(safe_den_hi, safe_den_lo)

        invr_hi, invr_lo = ds_mul_basic(z2r_hi, z2r_lo, recip_hi, recip_lo)
        invr_hi, invr_lo = ds_mul_scalar_basic(invr_hi, invr_lo, 1.0 / 3.0)
        invi_hi, invi_lo = ds_mul_basic(z2i_hi, z2i_lo, recip_hi, recip_lo)
        invi_hi, invi_lo = ds_mul_scalar_basic(invi_hi, invi_lo, -1.0 / 3.0)

        invr_hi = tl.where(safe, invr_hi, 0.0)
        invr_lo = tl.where(safe, invr_lo, 0.0)
        invi_hi = tl.where(safe, invi_hi, 0.0)
        invi_lo = tl.where(safe, invi_lo, 0.0)

        lhs_r_hi, lhs_r_lo = ds_mul_scalar_basic(zr_hi, zr_lo, 2.0 / 3.0)
        lhs_i_hi, lhs_i_lo = ds_mul_scalar_basic(zi_hi, zi_lo, 2.0 / 3.0)

        next_r_hi, next_r_lo = ds_add_basic(lhs_r_hi, lhs_r_lo, invr_hi, invr_lo)
        next_i_hi, next_i_lo = ds_add_basic(lhs_i_hi, lhs_i_lo, invi_hi, invi_lo)

        zr_hi = tl.where(active, next_r_hi, zr_hi)
        zr_lo = tl.where(active, next_r_lo, zr_lo)
        zi_hi = tl.where(active, next_i_hi, zi_hi)
        zi_lo = tl.where(active, next_i_lo, zi_lo)

        zr = zr_hi + zr_lo
        zi = zi_hi + zi_lo
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

    zr = zr_hi + zr_lo
    zi = zi_hi + zi_lo
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

    tl.store(zr_hi_out_ptr + offs, zr_hi, mask=mask)
    tl.store(zr_lo_out_ptr + offs, zr_lo, mask=mask)
    tl.store(zi_hi_out_ptr + offs, zi_hi, mask=mask)
    tl.store(zi_lo_out_ptr + offs, zi_lo, mask=mask)
    tl.store(root_ptr + offs, root, mask=mask)
    tl.store(iter_ptr + offs, iters, mask=mask)


def split_doublefloat_flat(x64: np.ndarray, y64: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_hi = x64.astype(np.float32)
    x_lo = (x64 - x_hi.astype(np.float64)).astype(np.float32)
    y_hi = y64.astype(np.float32)
    y_lo = (y64 - y_hi.astype(np.float64)).astype(np.float32)

    zr_hi = torch.from_numpy(np.broadcast_to(x_hi[None, :], (y64.shape[0], x64.shape[0])).copy().reshape(-1)).to(device=device)
    zr_lo = torch.from_numpy(np.broadcast_to(x_lo[None, :], (y64.shape[0], x64.shape[0])).copy().reshape(-1)).to(device=device)
    zi_hi = torch.from_numpy(np.broadcast_to(y_hi[:, None], (y64.shape[0], x64.shape[0])).copy().reshape(-1)).to(device=device)
    zi_lo = torch.from_numpy(np.broadcast_to(y_lo[:, None], (y64.shape[0], x64.shape[0])).copy().reshape(-1)).to(device=device)
    return zr_hi, zr_lo, zi_hi, zi_lo


@torch.no_grad()
def run_triton_doublefloat_basic(
    x64: np.ndarray,
    y64: np.ndarray,
    tol: float,
    max_iter: int,
    block_size: int,
    num_warps: int,
    post_polish_steps: int,
    iter_replay_threshold: int,
    iter_replay_dilation: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    zr_hi, zr_lo, zi_hi, zi_lo = split_doublefloat_flat(x64, y64, device)
    n_points = zr_hi.numel()
    zr_hi_out = torch.empty_like(zr_hi)
    zr_lo_out = torch.empty_like(zr_lo)
    zi_hi_out = torch.empty_like(zi_hi)
    zi_lo_out = torch.empty_like(zi_lo)
    root = torch.empty(n_points, dtype=torch.int32, device=device)
    iters = torch.empty(n_points, dtype=torch.int32, device=device)
    grid = lambda meta: (triton.cdiv(n_points, meta["BLOCK_SIZE"]),)
    newton_doublefloat_basic_kernel[grid](
        zr_hi,
        zr_lo,
        zi_hi,
        zi_lo,
        zr_hi_out,
        zr_lo_out,
        zi_hi_out,
        zi_lo_out,
        root,
        iters,
        n_points,
        tol * tol,
        (10.0 * tol) ** 2,
        BLOCK_SIZE=block_size,
        MAX_ITER=max_iter,
        num_warps=num_warps,
    )
    root_map = root.view(y64.shape[0], x64.shape[0]).to(torch.int16)
    iter_map = iters.view(y64.shape[0], x64.shape[0]).to(torch.int16)
    if post_polish_steps <= 0:
        return root_map, iter_map

    zr64 = zr_hi_out.view(y64.shape[0], x64.shape[0]).to(torch.float64) + zr_lo_out.view(y64.shape[0], x64.shape[0]).to(torch.float64)
    zi64 = zi_hi_out.view(y64.shape[0], x64.shape[0]).to(torch.float64) + zi_lo_out.view(y64.shape[0], x64.shape[0]).to(torch.float64)
    z = torch.complex(zr64, zi64)
    safe = torch.abs(z * z) > 1.0e-28
    for _ in range(post_polish_steps):
        z2 = z * z
        safe = torch.abs(z2) > 1.0e-28
        inv = torch.zeros_like(z)
        if bool(torch.any(safe).item()):
            inv[safe] = (1.0 / 3.0) / z2[safe]
        z = (2.0 / 3.0) * z + inv

    roots = torch.tensor(
        [1.0 + 0.0j, complex(-0.5, SQRT3_2), complex(-0.5, -SQRT3_2)],
        dtype=torch.complex128,
        device=device,
    )
    distances = torch.stack([torch.abs(z - root_val) for root_val in roots], dim=0)
    mins, indices = torch.min(distances, dim=0)
    polished_root = torch.full_like(root_map, -1)
    polished_root = torch.where(mins < (10.0 * tol), indices.to(torch.int16), polished_root)
    keep_kernel = root_map >= 0
    polished_root = torch.where(keep_kernel, indices.to(torch.int16), polished_root)
    if iter_replay_threshold <= 0:
        return polished_root, iter_map

    suspect = build_boundary_mask(polished_root, iter_replay_dilation)
    suspect |= iter_map >= iter_replay_threshold
    suspect |= polished_root != root_map
    row_idx, col_idx, root_sel, iter_sel = replay_reference_newton_selected(x64, y64, suspect, tol, max_iter)
    if row_idx.numel() > 0:
        polished_root[row_idx, col_idx] = root_sel
        iter_map[row_idx, col_idx] = iter_sel
    return polished_root, iter_map


@torch.no_grad()
def run_triton_fp32(
    x64: np.ndarray,
    y64: np.ndarray,
    tol: float,
    max_iter: int,
    block_size: int,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    x = torch.tensor(x64, dtype=torch.float32, device=device)
    y = torch.tensor(y64, dtype=torch.float32, device=device)
    root_map, iter_map, *_ = gpu_ref.run_tile_triton_newton(
        x=x,
        y=y,
        tol=tol,
        max_iter=max_iter,
        factor=1,
        block_size=block_size,
        num_warps=num_warps,
    )
    return root_map.to(torch.int16), iter_map.to(torch.int16)


def benchmark(name: str, fn, warmup: int = 1, reps: int = 1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    root_map = None
    iter_map = None
    for _ in range(reps):
        root_map, iter_map = fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / reps
    print(f"[{name}] elapsed={elapsed:.4f}s", flush=True)
    return root_map, iter_map, elapsed


def unique_pair_count(values64: np.ndarray) -> int:
    hi = values64.astype(np.float32)
    lo = (values64 - hi.astype(np.float64)).astype(np.float32)
    pairs = np.stack((hi, lo), axis=1)
    return int(np.unique(pairs, axis=0).shape[0])


def build_boundary_mask(root_map: torch.Tensor, dilation: int) -> torch.Tensor:
    boundary = torch.zeros_like(root_map, dtype=torch.bool)
    boundary[1:, :] |= root_map[1:, :] != root_map[:-1, :]
    boundary[:-1, :] |= root_map[:-1, :] != root_map[1:, :]
    boundary[:, 1:] |= root_map[:, 1:] != root_map[:, :-1]
    boundary[:, :-1] |= root_map[:, :-1] != root_map[:, 1:]
    boundary |= root_map < 0
    if dilation <= 0:
        return boundary
    dilated = boundary.clone()
    frontier = boundary
    for _ in range(dilation):
        grown = frontier.clone()
        grown[1:, :] |= frontier[:-1, :]
        grown[:-1, :] |= frontier[1:, :]
        grown[:, 1:] |= frontier[:, :-1]
        grown[:, :-1] |= frontier[:, 1:]
        dilated |= grown
        frontier = grown
    return dilated


@torch.no_grad()
def replay_reference_newton_selected(
    x64: np.ndarray,
    y64: np.ndarray,
    mask: torch.Tensor,
    tol: float,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = mask.device
    coords = torch.nonzero(mask, as_tuple=False)
    if coords.numel() == 0:
        empty_idx = torch.empty((0,), dtype=torch.int64, device=device)
        empty_root = torch.empty((0,), dtype=torch.int16, device=device)
        empty_iter = torch.empty((0,), dtype=torch.int16, device=device)
        return empty_idx, empty_idx, empty_root, empty_iter

    row_idx = coords[:, 0]
    col_idx = coords[:, 1]
    x = torch.tensor(x64, dtype=torch.float64, device=device)[col_idx]
    y = torch.tensor(y64, dtype=torch.float64, device=device)[row_idx]
    z = torch.complex(x, y)
    roots = torch.tensor(
        [1.0 + 0.0j, complex(-0.5, SQRT3_2), complex(-0.5, -SQRT3_2)],
        dtype=torch.complex128,
        device=device,
    )
    converged = torch.zeros(z.shape, dtype=torch.bool, device=device)
    root_sel = torch.full(z.shape, -1, dtype=torch.int16, device=device)
    iter_sel = torch.zeros(z.shape, dtype=torch.int16, device=device)

    for step in range(1, max_iter + 1):
        z2 = z * z
        safe = torch.abs(z2) > 1.0e-28
        inv = torch.zeros_like(z)
        if bool(torch.any(safe).item()):
            inv[safe] = (1.0 / 3.0) / z2[safe]
        z_next = (2.0 / 3.0) * z + inv
        z = torch.where(converged, z, z_next)

        distances = torch.stack([torch.abs(z - root_val) for root_val in roots], dim=0)
        mins, indices = torch.min(distances, dim=0)
        just = (~converged) & (mins < tol)
        root_sel[just] = indices[just].to(torch.int16)
        iter_sel[just] = step
        converged |= just
        if bool(torch.all(converged).item()):
            break

    distances = torch.stack([torch.abs(z - root_val) for root_val in roots], dim=0)
    mins, indices = torch.min(distances, dim=0)
    late = (~converged) & (mins < 10.0 * tol)
    root_sel[late] = indices[late].to(torch.int16)
    iter_sel[late] = max_iter
    return row_idx, col_idx, root_sel, iter_sel


def summarize_output(root_map: torch.Tensor, iter_map: torch.Tensor) -> dict[str, float]:
    total = root_map.numel()
    converged = root_map >= 0
    converged_total = int(converged.sum().item())
    iter_total = int(iter_map[converged].to(torch.int64).sum().item()) if converged_total else 0
    return {
        "convergence_fraction": converged_total / total,
        "mean_iterations": (iter_total / converged_total) if converged_total else float("nan"),
        "root0_fraction": float((root_map == 0).sum().item() / total),
        "root1_fraction": float((root_map == 1).sum().item() / total),
        "root2_fraction": float((root_map == 2).sum().item() / total),
    }


def summarize_case(
    case_name: str,
    x64: np.ndarray,
    y64: np.ndarray,
    tol: float,
    max_iter: int,
    block_size: int,
    num_warps: int,
    reps: int,
    post_polish_steps: int,
    iter_replay_threshold: int,
    iter_replay_dilation: int,
) -> list[dict[str, float | int | str]]:
    fp32_root, fp32_iter, fp32_time = benchmark(
        "fp32_triton",
        lambda: run_triton_fp32(x64, y64, tol, max_iter, block_size, num_warps),
        reps=reps,
    )
    fp64_root, fp64_iter, fp64_time = benchmark(
        "fp64_pytorch",
        lambda: df_proto.run_reference_case(x64, y64, "complex128", tol, max_iter),
        reps=reps,
    )
    ds_py_root, ds_py_iter, ds_py_time = benchmark(
        "doublefloat_pytorch",
        lambda: df_proto.run_newton_doublefloat(x64, y64, tol, max_iter, "basic"),
        reps=reps,
    )
    ds_tri_root, ds_tri_iter, ds_tri_time = benchmark(
        "doublefloat_triton",
        lambda: run_triton_doublefloat_basic(
            x64,
            y64,
            tol,
            max_iter,
            block_size,
            num_warps,
            post_polish_steps,
            iter_replay_threshold,
            iter_replay_dilation,
        ),
        reps=reps,
    )

    fp32_unique_x = int(np.unique(x64.astype(np.float32)).size)
    fp32_unique_y = int(np.unique(y64.astype(np.float32)).size)
    fp64_unique_x = int(np.unique(x64).size)
    fp64_unique_y = int(np.unique(y64).size)
    ds_unique_x = unique_pair_count(x64)
    ds_unique_y = unique_pair_count(y64)

    backends = {
        "fp32_triton": (fp32_root, fp32_iter, fp32_time, fp32_unique_x, fp32_unique_y),
        "fp64_pytorch": (fp64_root, fp64_iter, fp64_time, fp64_unique_x, fp64_unique_y),
        "doublefloat_pytorch": (ds_py_root, ds_py_iter, ds_py_time, ds_unique_x, ds_unique_y),
        "doublefloat_triton": (ds_tri_root, ds_tri_iter, ds_tri_time, ds_unique_x, ds_unique_y),
    }

    ref_root = fp64_root
    ref_iter = fp64_iter
    results = []
    for backend, (root_map, iter_map, elapsed, unique_x, unique_y) in backends.items():
        root_diff = int((root_map != ref_root).sum().item())
        iter_diff = int((iter_map != ref_iter).sum().item())
        max_iter_diff = int(torch.max(torch.abs(iter_map.to(torch.int32) - ref_iter.to(torch.int32))).item())
        stats = summarize_output(root_map, iter_map)
        results.append(
            {
                "case": case_name,
                "backend": backend,
                "grid": root_map.shape[0],
                "span": float(x64[-1] - x64[0]),
                "elapsed_seconds": elapsed,
                "convergence_fraction": stats["convergence_fraction"],
                "mean_iterations": stats["mean_iterations"],
                "root0_fraction": stats["root0_fraction"],
                "root1_fraction": stats["root1_fraction"],
                "root2_fraction": stats["root2_fraction"],
                "root_diff_vs_fp64": root_diff,
                "iter_diff_vs_fp64": iter_diff,
                "max_iter_diff_vs_fp64": max_iter_diff,
                "unique_x_count": unique_x,
                "unique_y_count": unique_y,
            }
        )
    return results


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton double-float prototype.")

    ensure_dirs()
    run_dir = make_run_dir(ANALYSIS_DIR, args.output_prefix)

    boundary_x, boundary_y = df_proto.find_boundary_center(args.coarse_grid, args.tol, args.max_iter)
    ulp_ref = float(np.spacing(np.float32(max(abs(boundary_x), abs(boundary_y), 1.0))))
    zoom_span = args.zoom_step_ulp_ratio * ulp_ref * (args.test_grid - 1)

    full_axis = df_proto.build_axis(0.0, AXIS_MAX - AXIS_MIN, args.test_grid)
    zoom_x = df_proto.build_axis(boundary_x, zoom_span, args.test_grid)
    zoom_y = df_proto.build_axis(boundary_y, zoom_span, args.test_grid)

    rows = []
    rows.extend(
        summarize_case(
            "full_plane",
            full_axis,
            full_axis,
            args.tol,
            args.max_iter,
            args.block_size,
            args.num_warps,
            args.reps,
            args.post_polish_steps,
            args.iter_replay_threshold,
            args.iter_replay_dilation,
        )
    )
    rows.extend(
        summarize_case(
            "boundary_zoom",
            zoom_x,
            zoom_y,
            args.tol,
            args.max_iter,
            args.block_size,
            args.num_warps,
            args.reps,
            args.post_polish_steps,
            args.iter_replay_threshold,
            args.iter_replay_dilation,
        )
    )

    csv_path = run_dir / "prototype_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_name = {(row["case"], row["backend"]): row for row in rows}
    lines = [
        "Triton double-float Newton prototype",
        "===================================",
        "",
        f"run_dir={run_dir}",
        f"boundary_center=({boundary_x:.12f}, {boundary_y:.12f})",
        f"boundary_ulp_reference={ulp_ref:.10e}",
        f"zoom_step_ulp_ratio={args.zoom_step_ulp_ratio}",
        f"zoom_span={zoom_span:.10e}",
        f"block_size={args.block_size}",
        f"num_warps={args.num_warps}",
        f"reps={args.reps}",
        f"post_polish_steps={args.post_polish_steps}",
        f"iter_replay_threshold={args.iter_replay_threshold}",
        f"iter_replay_dilation={args.iter_replay_dilation}",
        "",
    ]

    for row in rows:
        lines.append(
            f"{row['case']} / {row['backend']}: elapsed={row['elapsed_seconds']:.4f}s, "
            f"root_diff_vs_fp64={row['root_diff_vs_fp64']}, iter_diff_vs_fp64={row['iter_diff_vs_fp64']}, "
            f"max_iter_diff_vs_fp64={row['max_iter_diff_vs_fp64']}, "
            f"unique_x={row['unique_x_count']}/{row['grid']}, unique_y={row['unique_y_count']}/{row['grid']}"
        )
        lines.append(
            f"  convergence={row['convergence_fraction']:.6f}, mean_iterations={row['mean_iterations']:.4f}, "
            f"basin=({row['root0_fraction']:.6f}, {row['root1_fraction']:.6f}, {row['root2_fraction']:.6f})"
        )

    for case_name in ("full_plane", "boundary_zoom"):
        ds_py = by_name[(case_name, "doublefloat_pytorch")]
        ds_tri = by_name[(case_name, "doublefloat_triton")]
        speedup = float(ds_py["elapsed_seconds"]) / float(ds_tri["elapsed_seconds"])
        lines.append("")
        lines.append(
            f"{case_name} speedup doublefloat_triton_over_pytorch={speedup:.3f}x"
        )

    log_path = run_dir / "prototype_summary.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
