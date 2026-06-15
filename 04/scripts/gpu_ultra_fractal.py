from __future__ import annotations

import argparse
import csv
import gc
import math
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import triton_doublefloat_backend as df_triton_backend

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
RESULT_DIR = ROOT / "result"
GPU_PYTORCH_DIR = RESULT_DIR / "gpu_pytorch"
GPU_TRITON_DIR = RESULT_DIR / "gpu_triton"
AXIS_MIN = -1.8
AXIS_MAX = 1.8


if TRITON_AVAILABLE:

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU tiled fractal renderer for HW/04 Problem 1.")
    parser.add_argument("--compute-grid", type=int, default=80000)
    parser.add_argument("--render-grid", type=int, default=5000)
    parser.add_argument("--render-tile", type=int, default=250)
    parser.add_argument("--compute-tile", type=int, default=20000)
    parser.add_argument("--tol", type=float, default=5.0e-7)
    parser.add_argument("--newton-max-iter", type=int, default=55)
    parser.add_argument("--secant-max-iter", type=int, default=65)
    parser.add_argument("--delta-real", type=float, default=0.2)
    parser.add_argument("--delta-imag", type=float, default=0.2)
    parser.add_argument("--methods", type=str, default="newton,secant")
    parser.add_argument("--output-prefix", type=str, default="problem1_gpu_ultra")
    parser.add_argument("--dtype", choices=("complex64", "complex128"), default="complex64")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--min-render-tile", type=int, default=125)
    parser.add_argument("--disable-newton-symmetry", action="store_true")
    parser.add_argument("--newton-backend", choices=("pytorch", "triton", "doublefloat-triton"), default="pytorch")
    parser.add_argument("--newton-direct-formula", action="store_true")
    parser.add_argument("--triton-block-size", type=int, default=256)
    parser.add_argument("--triton-num-warps", type=int, default=4)
    parser.add_argument("--doublefloat-post-polish-steps", type=int, default=2)
    parser.add_argument("--doublefloat-iter-replay-threshold", type=int, default=8)
    parser.add_argument("--doublefloat-iter-replay-dilation", type=int, default=1)
    parser.add_argument("--stats-only", action="store_true")
    active_group = parser.add_mutually_exclusive_group()
    active_group.add_argument("--enable-active-compression", dest="active_compression", action="store_true")
    active_group.add_argument("--disable-active-compression", dest="active_compression", action="store_false")
    parser.set_defaults(active_compression=False)
    return parser.parse_args()


def ensure_result_dir() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    GPU_PYTORCH_DIR.mkdir(parents=True, exist_ok=True)
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


def complex_dtype(name: str) -> torch.dtype:
    return torch.complex64 if name == "complex64" else torch.complex128


def real_dtype(name: str) -> torch.dtype:
    return torch.float32 if name == "complex64" else torch.float64


def palettes() -> tuple[np.ndarray, np.ndarray]:
    vivid = np.array(
        [
            [0.96, 0.31, 0.27],
            [0.16, 0.67, 0.89],
            [0.96, 0.74, 0.22],
        ],
        dtype=np.float32,
    )
    pastel = np.array(
        [
            [0.95, 0.67, 0.66],
            [0.61, 0.81, 0.92],
            [0.96, 0.89, 0.61],
        ],
        dtype=np.float32,
    )
    return vivid, pastel


def basin_image(root_map: np.ndarray, iters: np.ndarray, max_iter: int, palette: np.ndarray) -> np.ndarray:
    image = np.zeros(root_map.shape + (3,), dtype=np.uint8)
    image[:, :] = np.array([20, 20, 20], dtype=np.uint8)
    converged = root_map >= 0
    if np.any(converged):
        base = (palette[root_map[converged]] * 255.0).astype(np.float32)
        shade = 0.30 + 0.70 * (1.0 - (iters[converged] - 1.0) / max_iter)
        image[converged] = np.clip(np.rint(255.0 - (255.0 - base) * shade[:, None]), 0, 255).astype(np.uint8)
    return image


def classify_roots(z: torch.Tensor, roots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.stack([torch.abs(z - root) for root in roots], dim=0)
    mins, indices = torch.min(distances, dim=0)
    return indices, mins


def downsample_tile(root_map: torch.Tensor, iters: torch.Tensor, factor: int) -> tuple[torch.Tensor, torch.Tensor]:
    out_h = root_map.shape[0] // factor
    out_w = root_map.shape[1] // factor

    reshaped_root = root_map.view(out_h, factor, out_w, factor)
    counts = torch.stack([(reshaped_root == idx).sum(dim=(1, 3)) for idx in range(3)], dim=0)
    converged_pixels = counts.sum(dim=0)
    dominant = counts.argmax(dim=0).to(torch.int8)
    small_root = torch.full((out_h, out_w), -1, dtype=torch.int8, device=root_map.device)
    small_root = torch.where(converged_pixels > 0, dominant, small_root)

    converged_mask = (root_map >= 0).view(out_h, factor, out_w, factor)
    iter_sum = (iters.float() * (root_map >= 0).float()).view(out_h, factor, out_w, factor).sum(dim=(1, 3))
    count = converged_mask.sum(dim=(1, 3))
    small_iters = torch.zeros((out_h, out_w), dtype=torch.float32, device=root_map.device)
    small_iters = torch.where(count > 0, iter_sum / count.clamp_min(1), small_iters)
    return small_root, small_iters


def release_cuda_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def swap_conjugate_root_labels(root_map: np.ndarray) -> None:
    mask1 = root_map == 1
    mask2 = root_map == 2
    root_map[mask1] = 2
    root_map[mask2] = 1


def mirror_newton_result(root_map_full: np.ndarray, iters_full: np.ndarray) -> None:
    midpoint = root_map_full.shape[0] // 2
    if midpoint == 0:
        return

    if root_map_full.shape[0] % 2 == 0:
        source = root_map_full[midpoint:][::-1].copy()
        source_iters = iters_full[midpoint:][::-1]
    else:
        source = root_map_full[midpoint + 1 :][::-1].copy()
        source_iters = iters_full[midpoint + 1 :][::-1]

    swap_conjugate_root_labels(source)
    root_map_full[:midpoint] = source
    iters_full[:midpoint] = source_iters


@torch.no_grad()
def run_tile_dense(
    method: str,
    x: torch.Tensor,
    y: torch.Tensor,
    roots: torch.Tensor,
    tol: float,
    max_iter: int,
    delta: complex,
    factor: int,
    newton_direct_formula: bool,
    return_downsample: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int, int, np.ndarray]:
    real = x[None, :].expand(y.shape[0], -1)
    imag = y[:, None].expand(-1, x.shape[0])
    base = torch.complex(real, imag)

    if method == "newton":
        z = base.clone()
    else:
        z_prev = base.clone()
        z = z_prev + torch.tensor(delta, dtype=base.dtype, device=base.device)

    root_map = torch.full(z.shape, -1, dtype=torch.int8, device=z.device)
    iters = torch.zeros(z.shape, dtype=torch.int16, device=z.device)
    converged = torch.zeros(z.shape, dtype=torch.bool, device=z.device)

    for step in range(1, max_iter + 1):
        if method == "newton":
            if newton_direct_formula:
                z2 = z * z
                safe = torch.abs(z2) > 1.0e-14
                inv = torch.zeros_like(z)
                if bool(torch.any(safe).item()):
                    inv[safe] = (1.0 / 3.0) / z2[safe]
                z_next = (2.0 / 3.0) * z + inv
                z = torch.where(converged, z, z_next)
            else:
                derivative = 3.0 * z * z
                safe = torch.abs(derivative) > 1.0e-14
                update = torch.zeros_like(z)
                safe_vals = z[safe]
                update[safe] = (safe_vals**3 - 1.0) / derivative[safe]
                z = torch.where(converged, z, z - update)
            root_idx, distance = classify_roots(z, roots)
            just = (~converged) & (distance < tol)
        else:
            f_prev = z_prev**3 - 1.0
            f_curr = z**3 - 1.0
            denom = f_curr - f_prev
            safe = torch.abs(denom) > 1.0e-14
            update = torch.zeros_like(z)
            update[safe] = f_curr[safe] * (z[safe] - z_prev[safe]) / denom[safe]
            z_next = z - update
            z_prev = torch.where(converged, z_prev, z)
            z = torch.where(converged, z, z_next)
            residual = torch.abs(z**3 - 1.0)
            root_idx, distance = classify_roots(z, roots)
            just = (~converged) & ((residual < tol) | (distance < tol))

        root_map[just] = root_idx[just].to(torch.int8)
        iters[just] = step
        converged |= just

        if bool(torch.all(converged).item()):
            break

    if method == "newton":
        root_idx, distance = classify_roots(z, roots)
        late = (~converged) & (distance < 10.0 * tol)
        root_map[late] = root_idx[late].to(torch.int8)
        iters[late] = max_iter
        converged |= late

    root_counts = np.array([(root_map == idx).sum().item() for idx in range(3)], dtype=np.int64)
    converged_total = int((root_map >= 0).sum().item())
    iter_total = int(iters[root_map >= 0].to(torch.int64).sum().item()) if converged_total else 0
    if not return_downsample:
        return None, None, converged_total, iter_total, root_counts
    small_root, small_iters = downsample_tile(root_map, iters, factor)
    return small_root, small_iters, converged_total, iter_total, root_counts


@torch.no_grad()
def run_tile_triton_newton(
    x: torch.Tensor,
    y: torch.Tensor,
    tol: float,
    max_iter: int,
    factor: int,
    block_size: int,
    num_warps: int,
    return_downsample: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int, int, np.ndarray]:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available in the current Python environment.")

    zr0 = x[None, :].expand(y.shape[0], -1).contiguous().reshape(-1)
    zi0 = y[:, None].expand(-1, x.shape[0]).contiguous().reshape(-1)
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
        tol * tol,
        (10.0 * tol) ** 2,
        BLOCK_SIZE=block_size,
        MAX_ITER=max_iter,
        num_warps=num_warps,
    )
    root_map = root.view(y.shape[0], x.shape[0]).to(torch.int8)
    iter_map = iters.view(y.shape[0], x.shape[0]).to(torch.int16)
    root_counts = np.array([(root_map == idx).sum().item() for idx in range(3)], dtype=np.int64)
    converged_total = int((root_map >= 0).sum().item())
    iter_total = int(iter_map[root_map >= 0].to(torch.int64).sum().item()) if converged_total else 0
    if not return_downsample:
        return None, None, converged_total, iter_total, root_counts
    small_root, small_iters = downsample_tile(root_map, iter_map, factor)
    return small_root, small_iters, converged_total, iter_total, root_counts


@torch.no_grad()
def run_tile_active(
    method: str,
    x: torch.Tensor,
    y: torch.Tensor,
    roots: torch.Tensor,
    tol: float,
    max_iter: int,
    delta: complex,
    factor: int,
    newton_direct_formula: bool,
    return_downsample: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int, int, np.ndarray]:
    height = y.shape[0]
    width = x.shape[0]
    real = x[None, :].expand(height, -1)
    imag = y[:, None].expand(-1, width)
    base = torch.complex(real, imag).reshape(-1)

    total_points = base.numel()
    active_idx = torch.arange(total_points, device=base.device, dtype=torch.int64)
    root_map = torch.full((total_points,), -1, dtype=torch.int8, device=base.device)
    iters = torch.zeros((total_points,), dtype=torch.int16, device=base.device)

    if method == "newton":
        z_active = base.clone()
    else:
        delta_tensor = torch.tensor(delta, dtype=base.dtype, device=base.device)
        z_prev_active = base.clone()
        z_active = z_prev_active + delta_tensor

    for step in range(1, max_iter + 1):
        if active_idx.numel() == 0:
            break

        if method == "newton":
            if newton_direct_formula:
                z2 = z_active * z_active
                safe = torch.abs(z2) > 1.0e-14
                inv = torch.zeros_like(z_active)
                if bool(torch.any(safe).item()):
                    inv[safe] = (1.0 / 3.0) / z2[safe]
                z_active = (2.0 / 3.0) * z_active + inv
            else:
                derivative = 3.0 * z_active * z_active
                safe = torch.abs(derivative) > 1.0e-14
                update = torch.zeros_like(z_active)
                if bool(torch.any(safe).item()):
                    safe_vals = z_active[safe]
                    update[safe] = (safe_vals**3 - 1.0) / derivative[safe]
                z_active = z_active - update
            root_idx, distance = classify_roots(z_active, roots)
            just = distance < tol
        else:
            f_prev = z_prev_active**3 - 1.0
            f_curr = z_active**3 - 1.0
            denom = f_curr - f_prev
            safe = torch.abs(denom) > 1.0e-14
            update = torch.zeros_like(z_active)
            if bool(torch.any(safe).item()):
                update[safe] = f_curr[safe] * (z_active[safe] - z_prev_active[safe]) / denom[safe]
            z_next = z_active - update
            residual = torch.abs(z_next**3 - 1.0)
            root_idx, distance = classify_roots(z_next, roots)
            just = (residual < tol) | (distance < tol)

        if bool(torch.any(just).item()):
            settled_idx = active_idx[just]
            root_map[settled_idx] = root_idx[just].to(torch.int8)
            iters[settled_idx] = step

        keep = ~just
        if method == "newton":
            z_active = z_active[keep]
        else:
            z_prev_active = z_active[keep]
            z_active = z_next[keep]
        active_idx = active_idx[keep]

    if method == "newton" and active_idx.numel() > 0:
        root_idx, distance = classify_roots(z_active, roots)
        late = distance < 10.0 * tol
        if bool(torch.any(late).item()):
            settled_idx = active_idx[late]
            root_map[settled_idx] = root_idx[late].to(torch.int8)
            iters[settled_idx] = max_iter

    root_map_2d = root_map.view(height, width)
    iters_2d = iters.view(height, width)
    root_counts = np.array([(root_map_2d == idx).sum().item() for idx in range(3)], dtype=np.int64)
    converged_total = int((root_map_2d >= 0).sum().item())
    iter_total = int(iters_2d[root_map_2d >= 0].to(torch.int64).sum().item()) if converged_total else 0
    if not return_downsample:
        return None, None, converged_total, iter_total, root_counts
    small_root, small_iters = downsample_tile(root_map_2d, iters_2d, factor)
    return small_root, small_iters, converged_total, iter_total, root_counts


def compute_method_once(
    method: str,
    compute_grid: int,
    render_grid: int,
    render_tile: int,
    use_newton_symmetry: bool,
    use_active_compression: bool,
    newton_backend: str,
    newton_direct_formula: bool,
    triton_block_size: int,
    triton_num_warps: int,
    doublefloat_post_polish_steps: int,
    doublefloat_iter_replay_threshold: int,
    doublefloat_iter_replay_dilation: int,
    tol: float,
    max_iter: int,
    delta: complex,
    dtype_name: str,
    device_name: str,
) -> tuple[dict[str, np.ndarray | float], dict[str, float]]:
    factor = compute_grid // render_grid
    device = torch.device(device_name)
    c_dtype = complex_dtype(dtype_name)
    axis_dtype = torch.float64 if (method == "newton" and newton_backend == "doublefloat-triton") else real_dtype(dtype_name)
    roots = torch.tensor(
        [
            1.0 + 0.0j,
            np.exp(2j * np.pi / 3.0),
            np.exp(4j * np.pi / 3.0),
        ],
        dtype=c_dtype,
        device=device,
    )
    axis = torch.linspace(AXIS_MIN, AXIS_MAX, compute_grid, dtype=axis_dtype, device=device)
    root_map_full = np.full((render_grid, render_grid), -1, dtype=np.int8)
    iters_full = np.zeros((render_grid, render_grid), dtype=np.float32)
    symmetry_used = method == "newton" and use_newton_symmetry
    midpoint = render_grid // 2

    row_ranges: list[tuple[int, int, str]] = []
    if symmetry_used:
        if render_grid % 2 == 1:
            row_ranges.append((midpoint, midpoint + 1, "center"))
            row_start = midpoint + 1
        else:
            row_start = midpoint
        while row_start < render_grid:
            row_end = min(render_grid, row_start + render_tile)
            row_ranges.append((row_start, row_end, "upper"))
            row_start = row_end
    else:
        row_start = 0
        while row_start < render_grid:
            row_end = min(render_grid, row_start + render_tile)
            row_ranges.append((row_start, row_end, "full"))
            row_start = row_end

    full_converged_total = 0
    full_iter_total = 0
    full_root_counts = np.zeros(3, dtype=np.int64)
    upper_converged_total = 0
    upper_iter_total = 0
    upper_root_counts = np.zeros(3, dtype=np.int64)
    center_converged_total = 0
    center_iter_total = 0
    center_root_counts = np.zeros(3, dtype=np.int64)

    row_tile_count = len(row_ranges)
    col_tile_count = math.ceil(render_grid / render_tile)
    active_compression_used = bool(method != "newton" and use_active_compression)
    newton_backend_used = newton_backend if method == "newton" else "pytorch"
    method_start = time.perf_counter()
    release_cuda_memory(device)
    torch.cuda.reset_peak_memory_stats(device)

    for row_block, (render_row_start, render_row_end, row_kind) in enumerate(row_ranges, start=1):
        compute_row_start = render_row_start * factor
        compute_row_end = render_row_end * factor
        y = axis[compute_row_start:compute_row_end]
        row_converged_total = 0
        row_iter_total = 0
        row_root_counts = np.zeros(3, dtype=np.int64)

        for col_block in range(col_tile_count):
            render_col_start = col_block * render_tile
            render_col_end = min(render_grid, render_col_start + render_tile)
            compute_col_start = render_col_start * factor
            compute_col_end = render_col_end * factor
            x = axis[compute_col_start:compute_col_end]

            if method == "newton" and newton_backend == "triton":
                small_root, small_iters, conv_count, tile_iter_total, counts = run_tile_triton_newton(
                    x=x,
                    y=y,
                    tol=tol,
                    max_iter=max_iter,
                    factor=factor,
                    block_size=triton_block_size,
                    num_warps=triton_num_warps,
                )
            elif method == "newton" and newton_backend == "doublefloat-triton":
                small_root, small_iters, conv_count, tile_iter_total, counts = df_triton_backend.run_tile_triton_doublefloat_newton(
                    x=x,
                    y=y,
                    tol=tol,
                    max_iter=max_iter,
                    factor=factor,
                    block_size=triton_block_size,
                    num_warps=triton_num_warps,
                    post_polish_steps=doublefloat_post_polish_steps,
                    iter_replay_threshold=doublefloat_iter_replay_threshold,
                    iter_replay_dilation=doublefloat_iter_replay_dilation,
                )
            else:
                tile_runner = run_tile_active if active_compression_used else run_tile_dense
                small_root, small_iters, conv_count, tile_iter_total, counts = tile_runner(
                    method=method,
                    x=x,
                    y=y,
                    roots=roots,
                    tol=tol,
                    max_iter=max_iter,
                    delta=delta,
                    factor=factor,
                    newton_direct_formula=newton_direct_formula,
                )

            root_map_full[render_row_start:render_row_end, render_col_start:render_col_end] = (
                small_root.cpu().numpy()
            )
            iters_full[render_row_start:render_row_end, render_col_start:render_col_end] = (
                small_iters.cpu().numpy()
            )
            row_converged_total += conv_count
            row_iter_total += tile_iter_total
            row_root_counts += counts

        if row_kind == "upper":
            upper_converged_total += row_converged_total
            upper_iter_total += row_iter_total
            upper_root_counts += row_root_counts
        elif row_kind == "center":
            center_converged_total += row_converged_total
            center_iter_total += row_iter_total
            center_root_counts += row_root_counts
        else:
            full_converged_total += row_converged_total
            full_iter_total += row_iter_total
            full_root_counts += row_root_counts

        print(
            f"[{method}] completed row tile {row_block}/{row_tile_count}",
            flush=True,
        )

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - method_start
    peak_gpu_mem_gib = torch.cuda.max_memory_allocated(device) / (1024.0**3)

    if symmetry_used:
        mirror_newton_result(root_map_full, iters_full)
        full_converged_total = center_converged_total + 2 * upper_converged_total
        full_iter_total = center_iter_total + 2 * upper_iter_total
        full_root_counts = np.array(
            [
                center_root_counts[0] + 2 * upper_root_counts[0],
                center_root_counts[1] + upper_root_counts[1] + upper_root_counts[2],
                center_root_counts[2] + upper_root_counts[1] + upper_root_counts[2],
            ],
            dtype=np.int64,
        )

    total_points = compute_grid * compute_grid

    data = {
        "method": method,
        "root_map": root_map_full,
        "iters": iters_full,
        "tol": tol,
        "max_iter": max_iter,
        "axis_min": AXIS_MIN,
        "axis_max": AXIS_MAX,
        "grid_size": compute_grid,
        "render_grid": render_grid,
        "convergence_fraction": full_converged_total / total_points,
        "mean_iterations": (full_iter_total / full_converged_total) if full_converged_total else float("nan"),
        "root0_fraction": float(full_root_counts[0] / total_points),
        "root1_fraction": float(full_root_counts[1] / total_points),
        "root2_fraction": float(full_root_counts[2] / total_points),
    }
    metrics = {
        "method": method,
        "elapsed_seconds": elapsed,
        "peak_gpu_mem_gib": peak_gpu_mem_gib,
        "render_tile": render_tile,
        "symmetry_used": symmetry_used,
        "active_compression_used": active_compression_used,
        "newton_direct_formula_used": bool(method == "newton" and (newton_backend in ("triton", "doublefloat-triton") or newton_direct_formula)),
        "newton_backend_used": newton_backend_used,
    }
    return data, metrics


def compute_method(
    method: str,
    compute_grid: int,
    render_grid: int,
    render_tile: int,
    min_render_tile: int,
    use_newton_symmetry: bool,
    use_active_compression: bool,
    newton_backend: str,
    newton_direct_formula: bool,
    triton_block_size: int,
    triton_num_warps: int,
    doublefloat_post_polish_steps: int,
    doublefloat_iter_replay_threshold: int,
    doublefloat_iter_replay_dilation: int,
    tol: float,
    max_iter: int,
    delta: complex,
    dtype_name: str,
    device_name: str,
) -> tuple[dict[str, np.ndarray | float], dict[str, float]]:
    device = torch.device(device_name)
    attempt_tile = render_tile

    while True:
        try:
            return compute_method_once(
                method=method,
                compute_grid=compute_grid,
                render_grid=render_grid,
                render_tile=attempt_tile,
                use_newton_symmetry=use_newton_symmetry,
                use_active_compression=use_active_compression,
                newton_backend=newton_backend,
                newton_direct_formula=newton_direct_formula,
                triton_block_size=triton_block_size,
                triton_num_warps=triton_num_warps,
                doublefloat_post_polish_steps=doublefloat_post_polish_steps,
                doublefloat_iter_replay_threshold=doublefloat_iter_replay_threshold,
                doublefloat_iter_replay_dilation=doublefloat_iter_replay_dilation,
                tol=tol,
                max_iter=max_iter,
                delta=delta,
                dtype_name=dtype_name,
                device_name=device_name,
            )
        except torch.OutOfMemoryError:
            release_cuda_memory(device)
            if attempt_tile <= min_render_tile:
                raise

            next_tile = max(min_render_tile, attempt_tile // 2)
            if next_tile == attempt_tile:
                next_tile = max(min_render_tile, attempt_tile - 1)
            if next_tile == attempt_tile:
                raise

            print(
                f"[{method}] CUDA OOM at render_tile={attempt_tile}; retrying with render_tile={next_tile}",
                flush=True,
            )
            attempt_tile = next_tile


def compute_method_stats_only(
    method: str,
    compute_grid: int,
    compute_tile: int,
    use_newton_symmetry: bool,
    use_active_compression: bool,
    newton_backend: str,
    newton_direct_formula: bool,
    triton_block_size: int,
    triton_num_warps: int,
    doublefloat_post_polish_steps: int,
    doublefloat_iter_replay_threshold: int,
    doublefloat_iter_replay_dilation: int,
    tol: float,
    max_iter: int,
    delta: complex,
    dtype_name: str,
    device_name: str,
) -> tuple[dict[str, float | str | bool], dict[str, float | str | bool]]:
    device = torch.device(device_name)
    c_dtype = complex_dtype(dtype_name)
    axis_dtype = torch.float64 if (method == "newton" and newton_backend == "doublefloat-triton") else real_dtype(dtype_name)
    roots = torch.tensor(
        [
            1.0 + 0.0j,
            np.exp(2j * np.pi / 3.0),
            np.exp(4j * np.pi / 3.0),
        ],
        dtype=c_dtype,
        device=device,
    )
    grid_spacing = (AXIS_MAX - AXIS_MIN) / (compute_grid - 1)
    float32_ulp = float(np.spacing(np.float32(AXIS_MAX)))
    ulp_ratio = grid_spacing / float32_ulp
    symmetry_used = method == "newton" and use_newton_symmetry
    midpoint = compute_grid // 2

    row_ranges: list[tuple[int, int, str]] = []
    if symmetry_used:
        if compute_grid % 2 == 1:
            row_ranges.append((midpoint, midpoint + 1, "center"))
            row_start = midpoint + 1
        else:
            row_start = midpoint
        while row_start < compute_grid:
            row_end = min(compute_grid, row_start + compute_tile)
            row_ranges.append((row_start, row_end, "upper"))
            row_start = row_end
    else:
        row_start = 0
        while row_start < compute_grid:
            row_end = min(compute_grid, row_start + compute_tile)
            row_ranges.append((row_start, row_end, "full"))
            row_start = row_end

    full_converged_total = 0
    full_iter_total = 0
    full_root_counts = np.zeros(3, dtype=np.int64)
    upper_converged_total = 0
    upper_iter_total = 0
    upper_root_counts = np.zeros(3, dtype=np.int64)
    center_converged_total = 0
    center_iter_total = 0
    center_root_counts = np.zeros(3, dtype=np.int64)

    active_compression_used = bool(method != "newton" and use_active_compression)
    newton_backend_used = newton_backend if method == "newton" else "pytorch"
    row_tile_count = len(row_ranges)
    col_tile_count = math.ceil(compute_grid / compute_tile)
    step = grid_spacing

    method_start = time.perf_counter()
    release_cuda_memory(device)
    torch.cuda.reset_peak_memory_stats(device)

    for row_block, (row_start, row_end, row_kind) in enumerate(row_ranges, start=1):
        row_idx = torch.arange(row_start, row_end, dtype=axis_dtype, device=device)
        y = AXIS_MIN + row_idx * step
        row_converged_total = 0
        row_iter_total = 0
        row_root_counts = np.zeros(3, dtype=np.int64)

        for col_block in range(col_tile_count):
            col_start = col_block * compute_tile
            col_end = min(compute_grid, col_start + compute_tile)
            col_idx = torch.arange(col_start, col_end, dtype=axis_dtype, device=device)
            x = AXIS_MIN + col_idx * step

            if method == "newton" and newton_backend == "triton":
                _, _, conv_count, tile_iter_total, counts = run_tile_triton_newton(
                    x=x,
                    y=y,
                    tol=tol,
                    max_iter=max_iter,
                    factor=1,
                    block_size=triton_block_size,
                    num_warps=triton_num_warps,
                    return_downsample=False,
                )
            elif method == "newton" and newton_backend == "doublefloat-triton":
                _, _, conv_count, tile_iter_total, counts = df_triton_backend.run_tile_triton_doublefloat_newton(
                    x=x,
                    y=y,
                    tol=tol,
                    max_iter=max_iter,
                    factor=1,
                    block_size=triton_block_size,
                    num_warps=triton_num_warps,
                    post_polish_steps=doublefloat_post_polish_steps,
                    iter_replay_threshold=doublefloat_iter_replay_threshold,
                    iter_replay_dilation=doublefloat_iter_replay_dilation,
                    return_downsample=False,
                )
            else:
                tile_runner = run_tile_active if active_compression_used else run_tile_dense
                _, _, conv_count, tile_iter_total, counts = tile_runner(
                    method=method,
                    x=x,
                    y=y,
                    roots=roots,
                    tol=tol,
                    max_iter=max_iter,
                    delta=delta,
                    factor=1,
                    newton_direct_formula=newton_direct_formula,
                    return_downsample=False,
                )

            row_converged_total += conv_count
            row_iter_total += tile_iter_total
            row_root_counts += counts

        if row_kind == "upper":
            upper_converged_total += row_converged_total
            upper_iter_total += row_iter_total
            upper_root_counts += row_root_counts
        elif row_kind == "center":
            center_converged_total += row_converged_total
            center_iter_total += row_iter_total
            center_root_counts += row_root_counts
        else:
            full_converged_total += row_converged_total
            full_iter_total += row_iter_total
            full_root_counts += row_root_counts

        print(f"[{method}] completed compute row tile {row_block}/{row_tile_count}", flush=True)

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - method_start
    peak_gpu_mem_gib = torch.cuda.max_memory_allocated(device) / (1024.0**3)

    if symmetry_used:
        full_converged_total = center_converged_total + 2 * upper_converged_total
        full_iter_total = center_iter_total + 2 * upper_iter_total
        full_root_counts = np.array(
            [
                center_root_counts[0] + 2 * upper_root_counts[0],
                center_root_counts[1] + upper_root_counts[1] + upper_root_counts[2],
                center_root_counts[2] + upper_root_counts[1] + upper_root_counts[2],
            ],
            dtype=np.int64,
        )

    total_points = compute_grid * compute_grid
    data = {
        "method": method,
        "grid_size": compute_grid,
        "compute_tile": compute_tile,
        "stats_only": True,
        "grid_spacing": grid_spacing,
        "float32_ulp_at_axis_max": float32_ulp,
        "grid_to_float32_ulp_ratio": ulp_ratio,
        "convergence_fraction": full_converged_total / total_points,
        "mean_iterations": (full_iter_total / full_converged_total) if full_converged_total else float("nan"),
        "root0_fraction": float(full_root_counts[0] / total_points),
        "root1_fraction": float(full_root_counts[1] / total_points),
        "root2_fraction": float(full_root_counts[2] / total_points),
    }
    metrics = {
        "method": method,
        "elapsed_seconds": elapsed,
        "peak_gpu_mem_gib": peak_gpu_mem_gib,
        "compute_tile": compute_tile,
        "symmetry_used": symmetry_used,
        "active_compression_used": active_compression_used,
        "newton_direct_formula_used": bool(method == "newton" and (newton_backend in ("triton", "doublefloat-triton") or newton_direct_formula)),
        "newton_backend_used": newton_backend_used,
        "stats_only": True,
        "grid_spacing": grid_spacing,
        "float32_ulp_at_axis_max": float32_ulp,
        "grid_to_float32_ulp_ratio": ulp_ratio,
    }
    return data, metrics


def write_csv(rows: list[dict[str, float | str]], path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(results: list[dict[str, np.ndarray | float]], out_path: Path) -> None:
    vivid, pastel = palettes()
    fig, axes = plt.subplots(2, len(results), figsize=(7 * len(results), 12), constrained_layout=True)
    if len(results) == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, result in enumerate(results):
        for row, (palette_name, palette) in enumerate((("vivid", vivid), ("pastel", pastel))):
            ax = axes[row, col]
            image = basin_image(result["root_map"], result["iters"], int(result["max_iter"]), palette)
            ax.imshow(
                image,
                origin="lower",
                extent=[AXIS_MIN, AXIS_MAX, AXIS_MIN, AXIS_MAX],
            )
            ax.set_title(f"{result['method'].title()} / {palette_name}")
            ax.set_xlabel("Re(z)")
            ax.set_ylabel("Im(z)")

    fig.suptitle("GPU tiled fractal rendering for HW/04 Problem 1", fontsize=15)
    fig.savefig(out_path, dpi=360)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_result_dir()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current Python environment.")
    if args.device != "cuda":
        raise RuntimeError("This script currently supports CUDA only.")
    if not args.stats_only and args.compute_grid % args.render_grid != 0:
        raise ValueError("compute-grid must be an integer multiple of render-grid.")
    if args.newton_backend in ("triton", "doublefloat-triton") and not TRITON_AVAILABLE:
        raise RuntimeError("The Triton backend was requested, but Triton is not installed.")
    if args.triton_block_size <= 0:
        raise ValueError("triton-block-size must be positive.")
    if args.triton_num_warps <= 0:
        raise ValueError("triton-num-warps must be positive.")
    if args.compute_tile <= 0:
        raise ValueError("compute-tile must be positive.")
    if args.doublefloat_post_polish_steps < 0:
        raise ValueError("doublefloat-post-polish-steps cannot be negative.")
    if args.doublefloat_iter_replay_threshold < 0:
        raise ValueError("doublefloat-iter-replay-threshold cannot be negative.")
    if args.doublefloat_iter_replay_dilation < 0:
        raise ValueError("doublefloat-iter-replay-dilation cannot be negative.")

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    factor = (args.compute_grid // args.render_grid) if not args.stats_only else None
    if args.render_tile <= 0:
        raise ValueError("render-tile must be positive.")
    if args.min_render_tile <= 0:
        raise ValueError("min-render-tile must be positive.")
    if args.min_render_tile > args.render_tile:
        raise ValueError("min-render-tile cannot exceed render-tile.")

    delta = complex(args.delta_real, args.delta_imag)
    run_base_dir = GPU_TRITON_DIR if args.newton_backend in ("triton", "doublefloat-triton") else GPU_PYTORCH_DIR
    run_dir = make_run_dir(run_base_dir, args.output_prefix)
    summary_rows: list[dict[str, float | str]] = []
    resource_rows: list[dict[str, float | str]] = []
    result_blocks: list[dict[str, np.ndarray | float]] = []

    for method in methods:
        max_iter = args.newton_max_iter if method == "newton" else args.secant_max_iter
        if args.stats_only:
            data, metrics = compute_method_stats_only(
                method=method,
                compute_grid=args.compute_grid,
                compute_tile=args.compute_tile,
                use_newton_symmetry=not args.disable_newton_symmetry,
                use_active_compression=args.active_compression,
                newton_backend=args.newton_backend,
                newton_direct_formula=args.newton_direct_formula,
                triton_block_size=args.triton_block_size,
                triton_num_warps=args.triton_num_warps,
                doublefloat_post_polish_steps=args.doublefloat_post_polish_steps,
                doublefloat_iter_replay_threshold=args.doublefloat_iter_replay_threshold,
                doublefloat_iter_replay_dilation=args.doublefloat_iter_replay_dilation,
                tol=args.tol,
                max_iter=max_iter,
                delta=delta,
                dtype_name=args.dtype,
                device_name=args.device,
            )
        else:
            data, metrics = compute_method(
                method=method,
                compute_grid=args.compute_grid,
                render_grid=args.render_grid,
                render_tile=args.render_tile,
                min_render_tile=args.min_render_tile,
                use_newton_symmetry=not args.disable_newton_symmetry,
                use_active_compression=args.active_compression,
                newton_backend=args.newton_backend,
                newton_direct_formula=args.newton_direct_formula,
                triton_block_size=args.triton_block_size,
                triton_num_warps=args.triton_num_warps,
                doublefloat_post_polish_steps=args.doublefloat_post_polish_steps,
                doublefloat_iter_replay_threshold=args.doublefloat_iter_replay_threshold,
                doublefloat_iter_replay_dilation=args.doublefloat_iter_replay_dilation,
                tol=args.tol,
                max_iter=max_iter,
                delta=delta,
                dtype_name=args.dtype,
                device_name=args.device,
            )
            result_blocks.append(data)
        summary_rows.append(
            {
                "method": method,
                "tol": args.tol,
                "max_iter": max_iter,
                "grid_size": args.compute_grid,
                "render_grid": args.render_grid if not args.stats_only else "",
                "render_tile": metrics.get("render_tile", ""),
                "compute_tile": metrics.get("compute_tile", ""),
                "stats_only": metrics.get("stats_only", False),
                "symmetry_used": metrics["symmetry_used"],
                "active_compression_used": metrics["active_compression_used"],
                "newton_backend_used": metrics["newton_backend_used"],
                "newton_direct_formula_used": metrics["newton_direct_formula_used"],
                "grid_spacing": metrics.get("grid_spacing", ""),
                "float32_ulp_at_axis_max": metrics.get("float32_ulp_at_axis_max", ""),
                "grid_to_float32_ulp_ratio": metrics.get("grid_to_float32_ulp_ratio", ""),
                "convergence_fraction": data["convergence_fraction"],
                "mean_iterations": data["mean_iterations"],
                "root0_fraction": data["root0_fraction"],
                "root1_fraction": data["root1_fraction"],
                "root2_fraction": data["root2_fraction"],
            }
        )
        resource_rows.append(
            {
                "method": method,
                "device": torch.cuda.get_device_name(0),
                "dtype": args.dtype,
                "compute_grid": args.compute_grid,
                "render_grid": args.render_grid if not args.stats_only else "",
                "render_tile": metrics.get("render_tile", ""),
                "compute_tile": metrics.get("compute_tile", ""),
                "stats_only": metrics.get("stats_only", False),
                "symmetry_used": metrics["symmetry_used"],
                "active_compression_used": metrics["active_compression_used"],
                "newton_backend_used": metrics["newton_backend_used"],
                "newton_direct_formula_used": metrics["newton_direct_formula_used"],
                "grid_spacing": metrics.get("grid_spacing", ""),
                "float32_ulp_at_axis_max": metrics.get("float32_ulp_at_axis_max", ""),
                "grid_to_float32_ulp_ratio": metrics.get("grid_to_float32_ulp_ratio", ""),
                "elapsed_seconds": metrics["elapsed_seconds"],
                "peak_gpu_mem_gib": metrics["peak_gpu_mem_gib"],
            }
        )
        release_cuda_memory(torch.device(args.device))

    plot_path = None
    if not args.stats_only:
        plot_path = run_dir / f"{args.output_prefix}.png"
        plot_results(result_blocks, plot_path)

    write_csv(
        summary_rows,
        run_dir / f"{args.output_prefix}_summary.csv",
        [
            "method",
            "tol",
            "max_iter",
            "grid_size",
            "render_grid",
            "render_tile",
            "compute_tile",
            "stats_only",
            "symmetry_used",
            "active_compression_used",
            "newton_backend_used",
            "newton_direct_formula_used",
            "grid_spacing",
            "float32_ulp_at_axis_max",
            "grid_to_float32_ulp_ratio",
            "convergence_fraction",
            "mean_iterations",
            "root0_fraction",
            "root1_fraction",
            "root2_fraction",
        ],
    )
    write_csv(
        resource_rows,
        run_dir / f"{args.output_prefix}_resources.csv",
        [
            "method",
            "device",
            "dtype",
            "compute_grid",
            "render_grid",
            "render_tile",
            "compute_tile",
            "stats_only",
            "symmetry_used",
            "active_compression_used",
            "newton_backend_used",
            "newton_direct_formula_used",
            "grid_spacing",
            "float32_ulp_at_axis_max",
            "grid_to_float32_ulp_ratio",
            "elapsed_seconds",
            "peak_gpu_mem_gib",
        ],
    )

    log_path = run_dir / f"{args.output_prefix}.log"
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("GPU ultra fractal run\n")
        fh.write("====================\n\n")
        fh.write(f"run_dir={run_dir}\n")
        fh.write(
            f"device={torch.cuda.get_device_name(0)}, dtype={args.dtype}, "
            f"compute_grid={args.compute_grid}, render_grid={args.render_grid}, "
            f"requested_render_tile={args.render_tile}, min_render_tile={args.min_render_tile}, "
            f"compute_tile={args.compute_tile}, factor={factor}, tol={args.tol}, "
            f"stats_only={args.stats_only}, active_compression={args.active_compression}, "
            f"requested_newton_backend={args.newton_backend}, "
            f"requested_newton_direct_formula={args.newton_direct_formula}, "
            f"triton_block_size={args.triton_block_size}, triton_num_warps={args.triton_num_warps}\n\n"
        )
        for summary, resource in zip(summary_rows, resource_rows, strict=True):
            grid_spacing = resource["grid_spacing"]
            ulp_ratio = resource["grid_to_float32_ulp_ratio"]
            grid_spacing_text = f"{grid_spacing:.10e}" if isinstance(grid_spacing, (int, float)) else ""
            ulp_ratio_text = f"{ulp_ratio:.6f}" if isinstance(ulp_ratio, (int, float)) else ""
            fh.write(
                f"{summary['method']}: convergence_fraction={summary['convergence_fraction']:.6f}, "
                f"mean_iterations={summary['mean_iterations']:.4f}, "
                f"basin fractions=({summary['root0_fraction']:.6f}, {summary['root1_fraction']:.6f}, {summary['root2_fraction']:.6f}), "
                f"grid_spacing={grid_spacing_text}, "
                f"grid_to_float32_ulp_ratio={ulp_ratio_text}, "
                f"symmetry={resource['symmetry_used']}, "
                f"active_compression={resource['active_compression_used']}, "
                f"newton_backend={resource['newton_backend_used']}, "
                f"newton_direct_formula={resource['newton_direct_formula_used']}, "
                f"render_tile={resource['render_tile']}, compute_tile={resource['compute_tile']}, "
                f"elapsed={resource['elapsed_seconds']:.2f}s, peak_gpu_mem={resource['peak_gpu_mem_gib']:.2f} GiB\n"
            )
        if plot_path is not None:
            fh.write(f"\nfigure={plot_path.name}\n")

    print(log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
