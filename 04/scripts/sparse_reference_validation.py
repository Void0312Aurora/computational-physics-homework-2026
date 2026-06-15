from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import doublefloat_newton_prototype as df_proto
import gpu_ultra_fractal as gpu_ref
import triton_doublefloat_backend as df_triton_backend


ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
RESULT_DIR = ROOT / "result"
ANALYSIS_DIR = RESULT_DIR / "analysis"
AXIS_SPAN = gpu_ref.AXIS_MAX - gpu_ref.AXIS_MIN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse reference validation for high-grid Newton patches.")
    parser.add_argument("--coarse-grid", type=int, default=512)
    parser.add_argument("--patch-grid", type=int, default=512)
    parser.add_argument("--global-grids", type=str, default="8192,262144,1048576,4194304,16777216,33554432")
    parser.add_argument("--center-mode", choices=("fixed", "tracked"), default="tracked")
    parser.add_argument("--tol", type=float, default=5.0e-7)
    parser.add_argument("--max-iter", type=int, default=55)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--post-polish-steps", type=int, default=2)
    parser.add_argument("--iter-replay-threshold", type=int, default=8)
    parser.add_argument("--iter-replay-dilation", type=int, default=1)
    parser.add_argument("--output-prefix", type=str, default="sparse_reference_validation")
    return parser.parse_args()


def parse_global_grids(spec: str) -> list[int]:
    values = [int(item.strip()) for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError("global-grids must contain at least one integer.")
    for value in values:
        if value <= 1:
            raise ValueError("global-grids values must be greater than 1.")
    return values


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


def build_patch_axis(center: float, step: float, n: int) -> np.ndarray:
    offsets = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    return center + offsets * step


def unique_pair_count(values64: np.ndarray) -> int:
    hi = values64.astype(np.float32)
    lo = (values64 - hi.astype(np.float64)).astype(np.float32)
    pairs = np.stack((hi, lo), axis=1)
    return int(np.unique(pairs, axis=0).shape[0])


def select_boundary_center_from_patch(x64: np.ndarray, y64: np.ndarray, root_map: torch.Tensor) -> tuple[float, float]:
    grid = root_map.detach().cpu().numpy()
    best_score = -1
    best_pos = (grid.shape[0] // 2, grid.shape[1] // 2)
    for i in range(1, grid.shape[0] - 1):
        block = grid[i - 1 : i + 2]
        for j in range(1, grid.shape[1] - 1):
            patch = block[:, j - 1 : j + 2]
            score = np.unique(patch).size
            if score > best_score:
                best_score = score
                best_pos = (i, j)
    row, col = best_pos
    return float(x64[col]), float(y64[row])


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


def benchmark(fn) -> tuple[torch.Tensor, torch.Tensor, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    root_map, iter_map = fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return root_map, iter_map, elapsed


def run_fp32_triton(
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


def run_doublefloat_triton(
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
    x = torch.tensor(x64, dtype=torch.float64, device=device)
    y = torch.tensor(y64, dtype=torch.float64, device=device)
    root_map, iter_map, *_ = df_triton_backend.run_tile_triton_doublefloat_newton(
        x=x,
        y=y,
        tol=tol,
        max_iter=max_iter,
        factor=1,
        block_size=block_size,
        num_warps=num_warps,
        post_polish_steps=post_polish_steps,
        iter_replay_threshold=iter_replay_threshold,
        iter_replay_dilation=iter_replay_dilation,
    )
    return root_map.to(torch.int16), iter_map.to(torch.int16)


def warm_up(
    center_x: float,
    center_y: float,
    tol: float,
    max_iter: int,
    block_size: int,
    num_warps: int,
    post_polish_steps: int,
    iter_replay_threshold: int,
    iter_replay_dilation: int,
) -> None:
    warm_grid = 64
    step = AXIS_SPAN / (8192 - 1)
    x64 = build_patch_axis(center_x, step, warm_grid)
    y64 = build_patch_axis(center_y, step, warm_grid)
    run_fp32_triton(x64, y64, tol, max_iter, block_size, num_warps)
    run_doublefloat_triton(
        x64,
        y64,
        tol,
        max_iter,
        block_size,
        num_warps,
        post_polish_steps,
        0,
        iter_replay_dilation,
    )
    run_doublefloat_triton(
        x64,
        y64,
        tol,
        max_iter,
        block_size,
        num_warps,
        post_polish_steps,
        iter_replay_threshold,
        iter_replay_dilation,
    )
    df_proto.run_reference_case(x64, y64, "complex128", tol, max_iter)
    gpu_ref.release_cuda_memory(torch.device("cuda"))


def compare_patch(
    global_grid: int,
    patch_grid: int,
    center_x: float,
    center_y: float,
    tol: float,
    max_iter: int,
    block_size: int,
    num_warps: int,
    post_polish_steps: int,
    iter_replay_threshold: int,
    iter_replay_dilation: int,
) -> tuple[list[dict[str, float | int | str]], tuple[float, float]]:
    step = AXIS_SPAN / (global_grid - 1)
    span = step * (patch_grid - 1)
    x64 = build_patch_axis(center_x, step, patch_grid)
    y64 = build_patch_axis(center_y, step, patch_grid)

    fp32_unique_x = int(np.unique(x64.astype(np.float32)).size)
    fp32_unique_y = int(np.unique(y64.astype(np.float32)).size)
    df_unique_x = unique_pair_count(x64)
    df_unique_y = unique_pair_count(y64)
    fp64_unique_x = int(np.unique(x64).size)
    fp64_unique_y = int(np.unique(y64).size)
    ulp_ref = float(np.spacing(np.float32(max(abs(center_x), abs(center_y), 1.0))))
    ulp_ratio = step / ulp_ref

    fp32_root, fp32_iter, fp32_time = benchmark(
        lambda: run_fp32_triton(x64, y64, tol, max_iter, block_size, num_warps)
    )
    df_base_root, df_base_iter, df_base_time = benchmark(
        lambda: run_doublefloat_triton(
            x64,
            y64,
            tol,
            max_iter,
            block_size,
            num_warps,
            post_polish_steps,
            0,
            iter_replay_dilation,
        )
    )
    df_t8_root, df_t8_iter, df_t8_time = benchmark(
        lambda: run_doublefloat_triton(
            x64,
            y64,
            tol,
            max_iter,
            block_size,
            num_warps,
            post_polish_steps,
            iter_replay_threshold,
            iter_replay_dilation,
        )
    )
    fp64_root, fp64_iter, fp64_time = benchmark(
        lambda: df_proto.run_reference_case(x64, y64, "complex128", tol, max_iter)
    )

    rows: list[dict[str, float | int | str]] = []
    backends = {
        "fp32_triton": (fp32_root, fp32_iter, fp32_time, fp32_unique_x, fp32_unique_y),
        "doublefloat_triton_base": (df_base_root, df_base_iter, df_base_time, df_unique_x, df_unique_y),
        "doublefloat_triton_t8": (df_t8_root, df_t8_iter, df_t8_time, df_unique_x, df_unique_y),
        "fp64_pytorch": (fp64_root, fp64_iter, fp64_time, fp64_unique_x, fp64_unique_y),
    }

    for backend, (root_map, iter_map, elapsed, unique_x, unique_y) in backends.items():
        root_diff = int((root_map != fp64_root).sum().item())
        iter_diff = int((iter_map != fp64_iter).sum().item())
        max_iter_diff = int(torch.max(torch.abs(iter_map.to(torch.int32) - fp64_iter.to(torch.int32))).item())
        stats = summarize_output(root_map, iter_map)
        rows.append(
            {
                "global_grid": global_grid,
                "patch_grid": patch_grid,
                "backend": backend,
                "center_x": center_x,
                "center_y": center_y,
                "dx": step,
                "patch_span": span,
                "float32_ulp_ref": ulp_ref,
                "dx_to_float32_ulp_ratio": ulp_ratio,
                "elapsed_seconds": elapsed,
                "convergence_fraction": stats["convergence_fraction"],
                "mean_iterations": stats["mean_iterations"],
                "root0_fraction": stats["root0_fraction"],
                "root1_fraction": stats["root1_fraction"],
                "root2_fraction": stats["root2_fraction"],
                "root_diff_vs_fp64": root_diff,
                "iter_diff_vs_fp64": iter_diff,
                "max_iter_diff_vs_fp64": max_iter_diff,
                "root_diff_ratio_vs_fp64": root_diff / root_map.numel(),
                "iter_diff_ratio_vs_fp64": iter_diff / iter_map.numel(),
                "unique_x_count": unique_x,
                "unique_y_count": unique_y,
                "unique_x_ratio": unique_x / patch_grid,
                "unique_y_ratio": unique_y / patch_grid,
            }
        )

    gpu_ref.release_cuda_memory(torch.device("cuda"))
    next_center = select_boundary_center_from_patch(x64, y64, fp64_root)
    return rows, next_center


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for sparse reference validation.")

    ensure_dirs()
    run_dir = make_run_dir(ANALYSIS_DIR, args.output_prefix)
    global_grids = parse_global_grids(args.global_grids)

    center_x, center_y = df_proto.find_boundary_center(args.coarse_grid, args.tol, args.max_iter)
    warm_up(
        center_x=center_x,
        center_y=center_y,
        tol=args.tol,
        max_iter=args.max_iter,
        block_size=args.block_size,
        num_warps=args.num_warps,
        post_polish_steps=args.post_polish_steps,
        iter_replay_threshold=args.iter_replay_threshold,
        iter_replay_dilation=args.iter_replay_dilation,
    )

    rows: list[dict[str, float | int | str]] = []
    current_center = (center_x, center_y)
    for global_grid in global_grids:
        patch_rows, next_center = compare_patch(
                global_grid=global_grid,
                patch_grid=args.patch_grid,
                center_x=current_center[0],
                center_y=current_center[1],
                tol=args.tol,
                max_iter=args.max_iter,
                block_size=args.block_size,
                num_warps=args.num_warps,
                post_polish_steps=args.post_polish_steps,
                iter_replay_threshold=args.iter_replay_threshold,
                iter_replay_dilation=args.iter_replay_dilation,
            )
        rows.extend(patch_rows)
        if args.center_mode == "tracked":
            current_center = next_center

    csv_path = run_dir / "sparse_reference_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "Sparse reference validation",
        "===========================",
        "",
        f"run_dir={run_dir}",
        f"boundary_center=({center_x:.12f}, {center_y:.12f})",
        f"center_mode={args.center_mode}",
        f"global_grids={','.join(str(item) for item in global_grids)}",
        f"patch_grid={args.patch_grid}",
        f"tol={args.tol}",
        f"max_iter={args.max_iter}",
        f"block_size={args.block_size}",
        f"num_warps={args.num_warps}",
        f"post_polish_steps={args.post_polish_steps}",
        f"iter_replay_threshold={args.iter_replay_threshold}",
        f"iter_replay_dilation={args.iter_replay_dilation}",
        "",
    ]

    for global_grid in global_grids:
        lines.append(f"global_grid={global_grid}")
        subset = [row for row in rows if row["global_grid"] == global_grid]
        center_row = subset[0]
        lines.append(
            f"  center=({center_row['center_x']:.12f}, {center_row['center_y']:.12f})"
        )
        for backend in ("fp32_triton", "doublefloat_triton_base", "doublefloat_triton_t8", "fp64_pytorch"):
            row = next(item for item in subset if item["backend"] == backend)
            lines.append(
                f"  {backend}: elapsed={row['elapsed_seconds']:.4f}s, "
                f"dx_to_float32_ulp_ratio={row['dx_to_float32_ulp_ratio']:.4f}, "
                f"root_diff={row['root_diff_vs_fp64']}, iter_diff={row['iter_diff_vs_fp64']}, "
                f"max_iter_diff={row['max_iter_diff_vs_fp64']}, "
                f"unique_x={row['unique_x_count']}/{args.patch_grid}, unique_y={row['unique_y_count']}/{args.patch_grid}"
            )
        lines.append("")

    log_path = run_dir / "sparse_reference_validation.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
