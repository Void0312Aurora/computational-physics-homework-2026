from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import gpu_ultra_fractal as gpu_ref


ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
RESULT_DIR = ROOT / "result"
ANALYSIS_DIR = RESULT_DIR / "analysis"
AXIS_MIN = -1.8
AXIS_MAX = 1.8
SQRT3_2 = np.sqrt(3.0) / 2.0
SPLITTER = 4097.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prototype double-float Newton benchmark on GPU.")
    parser.add_argument("--coarse-grid", type=int, default=512)
    parser.add_argument("--test-grid", type=int, default=512)
    parser.add_argument("--tol", type=float, default=5.0e-7)
    parser.add_argument("--max-iter", type=int, default=55)
    parser.add_argument("--zoom-step-ulp-ratio", type=float, default=0.5)
    parser.add_argument("--arith-mode", choices=("basic", "refined"), default="basic")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-prefix", type=str, default="doublefloat_newton_prototype")
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


def build_axis(center: float, span: float, n: int) -> np.ndarray:
    start = center - span / 2.0
    stop = center + span / 2.0
    return np.linspace(start, stop, n, dtype=np.float64)


def split_doublefloat(values64: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    hi = values64.astype(np.float32)
    lo = (values64 - hi.astype(np.float64)).astype(np.float32)
    return torch.from_numpy(hi).cuda(), torch.from_numpy(lo).cuda()


def ds_renorm(hi: torch.Tensor, lo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = hi + lo
    e = lo - (s - hi)
    return s, e


def two_sum(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = a + b
    bb = s - a
    err = (a - (s - bb)) + (b - bb)
    return s, err


def quick_two_sum(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = a + b
    err = b - (s - a)
    return s, err


def ds_add_basic(ah: torch.Tensor, al: torch.Tensor, bh: torch.Tensor, bl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = ah + bh
    v = s - ah
    t = ((bh - v) + (ah - (s - v))) + al + bl
    return ds_renorm(s, t)


def ds_add_refined(ah: torch.Tensor, al: torch.Tensor, bh: torch.Tensor, bl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s, e = two_sum(ah, bh)
    t, f = two_sum(al, bl)
    e = e + t
    sh, sl = quick_two_sum(s, e)
    sl = sl + f
    return ds_renorm(sh, sl)


def ds_sub_basic(ah: torch.Tensor, al: torch.Tensor, bh: torch.Tensor, bl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return ds_add_basic(ah, al, -bh, -bl)


def ds_sub_refined(ah: torch.Tensor, al: torch.Tensor, bh: torch.Tensor, bl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return ds_add_refined(ah, al, -bh, -bl)


def two_prod(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p = a * b
    ca = SPLITTER * a
    a_hi = ca - (ca - a)
    a_lo = a - a_hi
    cb = SPLITTER * b
    b_hi = cb - (cb - b)
    b_lo = b - b_hi
    err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return p, err


def ds_mul_basic(ah: torch.Tensor, al: torch.Tensor, bh: torch.Tensor, bl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p, e = two_prod(ah, bh)
    e = e + ah * bl + al * bh
    return ds_renorm(p, e)


def ds_mul_refined(ah: torch.Tensor, al: torch.Tensor, bh: torch.Tensor, bl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p, e = two_prod(ah, bh)
    e = e + ah * bl + al * bh + al * bl
    ph, pl = quick_two_sum(p, e)
    return ds_renorm(ph, pl)


def ds_mul_scalar_with(mul_fn, ah: torch.Tensor, al: torch.Tensor, scalar: float) -> tuple[torch.Tensor, torch.Tensor]:
    bs = torch.full_like(ah, scalar)
    bz = torch.zeros_like(ah)
    return mul_fn(ah, al, bs, bz)


def ds_square_basic(ah: torch.Tensor, al: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return ds_mul_basic(ah, al, ah, al)


def ds_square_refined(ah: torch.Tensor, al: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p, e = two_prod(ah, ah)
    e = e + 2.0 * ah * al + al * al
    ph, pl = quick_two_sum(p, e)
    return ds_renorm(ph, pl)


def ds_reciprocal_real_basic(dh: torch.Tensor, dl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    qh = 1.0 / dh
    ql = torch.zeros_like(qh)
    oneh = torch.ones_like(qh)
    onel = torch.zeros_like(qh)
    for _ in range(2):
        ph, pl = ds_mul_basic(dh, dl, qh, ql)
        eh, el = ds_sub_basic(oneh, onel, ph, pl)
        ch, cl = ds_mul_basic(qh, ql, eh, el)
        qh, ql = ds_add_basic(qh, ql, ch, cl)
    return qh, ql


def ds_reciprocal_real_refined(dh: torch.Tensor, dl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    qh = 1.0 / (dh + dl)
    ql = torch.zeros_like(qh)
    oneh = torch.ones_like(qh)
    onel = torch.zeros_like(qh)
    for _ in range(3):
        ph, pl = ds_mul_refined(dh, dl, qh, ql)
        eh, el = ds_sub_refined(oneh, onel, ph, pl)
        ch, cl = ds_mul_refined(qh, ql, eh, el)
        qh, ql = ds_add_refined(qh, ql, ch, cl)
    return qh, ql


def classify_from_estimate(zr_hi: torch.Tensor, zr_lo: torch.Tensor, zi_hi: torch.Tensor, zi_lo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    zr = zr_hi.to(torch.float64) + zr_lo.to(torch.float64)
    zi = zi_hi.to(torch.float64) + zi_lo.to(torch.float64)
    d0 = (zr - 1.0) ** 2 + zi**2
    d1 = (zr + 0.5) ** 2 + (zi - SQRT3_2) ** 2
    d2 = (zr + 0.5) ** 2 + (zi + SQRT3_2) ** 2
    distances = torch.stack((d0, d1, d2), dim=0)
    mins, indices = torch.min(distances, dim=0)
    return indices.to(torch.int16), torch.sqrt(mins)


@torch.no_grad()
def run_newton_doublefloat(
    x64: np.ndarray,
    y64: np.ndarray,
    tol: float,
    max_iter: int,
    arith_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if arith_mode == "refined":
        add_fn = ds_add_refined
        sub_fn = ds_sub_refined
        mul_fn = ds_mul_refined
        square_fn = ds_square_refined
        reciprocal_fn = ds_reciprocal_real_refined
    else:
        add_fn = ds_add_basic
        sub_fn = ds_sub_basic
        mul_fn = ds_mul_basic
        square_fn = ds_square_basic
        reciprocal_fn = ds_reciprocal_real_basic

    x_hi_1d, x_lo_1d = split_doublefloat(x64)
    y_hi_1d, y_lo_1d = split_doublefloat(y64)

    zr_hi = x_hi_1d[None, :].expand(y_hi_1d.shape[0], -1).clone()
    zr_lo = x_lo_1d[None, :].expand(y_lo_1d.shape[0], -1).clone()
    zi_hi = y_hi_1d[:, None].expand(-1, x_hi_1d.shape[0]).clone()
    zi_lo = y_lo_1d[:, None].expand(-1, x_lo_1d.shape[0]).clone()

    converged = torch.zeros_like(zr_hi, dtype=torch.bool)
    root_map = torch.full_like(zr_hi, -1, dtype=torch.int16)
    iters = torch.zeros_like(zr_hi, dtype=torch.int16)

    for step in range(1, max_iter + 1):
        z2r_hi, z2r_lo = sub_fn(*square_fn(zr_hi, zr_lo), *square_fn(zi_hi, zi_lo))
        zrzi_hi, zrzi_lo = mul_fn(zr_hi, zr_lo, zi_hi, zi_lo)
        z2i_hi, z2i_lo = ds_mul_scalar_with(mul_fn, zrzi_hi, zrzi_lo, 2.0)

        den_hi, den_lo = add_fn(*square_fn(z2r_hi, z2r_lo), *square_fn(z2i_hi, z2i_lo))
        recip_hi, recip_lo = reciprocal_fn(den_hi, den_lo)

        invr_hi, invr_lo = mul_fn(z2r_hi, z2r_lo, recip_hi, recip_lo)
        invr_hi, invr_lo = ds_mul_scalar_with(mul_fn, invr_hi, invr_lo, 1.0 / 3.0)

        invi_hi, invi_lo = mul_fn(z2i_hi, z2i_lo, recip_hi, recip_lo)
        invi_hi, invi_lo = ds_mul_scalar_with(mul_fn, invi_hi, invi_lo, -1.0 / 3.0)

        lhs_r_hi, lhs_r_lo = ds_mul_scalar_with(mul_fn, zr_hi, zr_lo, 2.0 / 3.0)
        lhs_i_hi, lhs_i_lo = ds_mul_scalar_with(mul_fn, zi_hi, zi_lo, 2.0 / 3.0)

        next_r_hi, next_r_lo = add_fn(lhs_r_hi, lhs_r_lo, invr_hi, invr_lo)
        next_i_hi, next_i_lo = add_fn(lhs_i_hi, lhs_i_lo, invi_hi, invi_lo)

        zr_hi = torch.where(converged, zr_hi, next_r_hi)
        zr_lo = torch.where(converged, zr_lo, next_r_lo)
        zi_hi = torch.where(converged, zi_hi, next_i_hi)
        zi_lo = torch.where(converged, zi_lo, next_i_lo)

        root_idx, distance = classify_from_estimate(zr_hi, zr_lo, zi_hi, zi_lo)
        just = (~converged) & (distance < tol)
        root_map[just] = root_idx[just]
        iters[just] = step
        converged |= just
        if bool(torch.all(converged).item()):
            break

    root_idx, distance = classify_from_estimate(zr_hi, zr_lo, zi_hi, zi_lo)
    late = (~converged) & (distance < 10.0 * tol)
    root_map[late] = root_idx[late]
    iters[late] = max_iter
    return root_map, iters


@torch.no_grad()
def run_reference_case(x64: np.ndarray, y64: np.ndarray, dtype_name: str, tol: float, max_iter: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    r_dtype = gpu_ref.real_dtype(dtype_name)
    c_dtype = gpu_ref.complex_dtype(dtype_name)
    roots = torch.tensor(
        [1.0 + 0.0j, complex(-0.5, SQRT3_2), complex(-0.5, -SQRT3_2)],
        dtype=c_dtype,
        device=device,
    )
    x = torch.tensor(x64, dtype=r_dtype, device=device)
    y = torch.tensor(y64, dtype=r_dtype, device=device)
    root_map, iters, *_ = gpu_ref.run_tile_dense(
        "newton",
        x,
        y,
        roots,
        tol,
        max_iter,
        complex(0.2, 0.2),
        1,
        True,
        True,
    )
    return root_map, iters


def benchmark(name: str, fn) -> tuple[torch.Tensor, torch.Tensor, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    root_map, iters = fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"[{name}] elapsed={elapsed:.4f}s", flush=True)
    return root_map, iters, elapsed


def find_boundary_center(coarse_grid: int, tol: float, max_iter: int) -> tuple[float, float]:
    axis = np.linspace(AXIS_MIN, AXIS_MAX, coarse_grid, dtype=np.float64)
    root_map, _ = run_reference_case(axis, axis, "complex128", tol, max_iter)
    grid = root_map.cpu().numpy()
    best_score = -1
    best_pos = (coarse_grid // 2, coarse_grid // 2)
    for i in range(1, coarse_grid - 1):
        block = grid[i - 1 : i + 2]
        for j in range(1, coarse_grid - 1):
            patch = block[:, j - 1 : j + 2]
            score = np.unique(patch).size
            if score > best_score:
                best_score = score
                best_pos = (i, j)
    row, col = best_pos
    return float(axis[col]), float(axis[row])


def unique_pair_count(hi: np.ndarray, lo: np.ndarray) -> int:
    pairs = np.stack((hi, lo), axis=1)
    return int(np.unique(pairs, axis=0).shape[0])


def summarize_case(
    case_name: str,
    x64: np.ndarray,
    y64: np.ndarray,
    tol: float,
    max_iter: int,
    arith_mode: str,
) -> list[dict[str, float | int | str]]:
    fp32_root, fp32_iter, fp32_time = benchmark("fp32", lambda: run_reference_case(x64, y64, "complex64", tol, max_iter))
    fp64_root, fp64_iter, fp64_time = benchmark("fp64", lambda: run_reference_case(x64, y64, "complex128", tol, max_iter))
    ds_label = f"doublefloat_{arith_mode}"
    ds_root, ds_iter, ds_time = benchmark(ds_label, lambda: run_newton_doublefloat(x64, y64, tol, max_iter, arith_mode))

    fp32_unique_x = int(np.unique(x64.astype(np.float32)).size)
    fp64_unique_x = int(np.unique(x64).size)
    fp32_unique_y = int(np.unique(y64.astype(np.float32)).size)
    fp64_unique_y = int(np.unique(y64).size)
    ds_x_hi = x64.astype(np.float32)
    ds_x_lo = (x64 - ds_x_hi.astype(np.float64)).astype(np.float32)
    ds_y_hi = y64.astype(np.float32)
    ds_y_lo = (y64 - ds_y_hi.astype(np.float64)).astype(np.float32)
    ds_unique_x = unique_pair_count(ds_x_hi, ds_x_lo)
    ds_unique_y = unique_pair_count(ds_y_hi, ds_y_lo)

    results = []
    backends = {
        "fp32": (fp32_root, fp32_iter, fp32_time, fp32_unique_x, fp32_unique_y),
        "fp64": (fp64_root, fp64_iter, fp64_time, fp64_unique_x, fp64_unique_y),
        ds_label: (ds_root, ds_iter, ds_time, ds_unique_x, ds_unique_y),
    }
    ref_root = fp64_root
    ref_iter = fp64_iter
    for backend, (root_map, iter_map, elapsed, unique_x, unique_y) in backends.items():
        root_diff = int((root_map != ref_root).sum().item())
        iter_diff = int((iter_map != ref_iter).sum().item())
        max_iter_diff = int(torch.max(torch.abs(iter_map.to(torch.int32) - ref_iter.to(torch.int32))).item())
        total = root_map.numel()
        converged = root_map >= 0
        converged_total = int(converged.sum().item())
        iter_total = int(iter_map[converged].to(torch.int64).sum().item()) if converged_total else 0
        results.append(
            {
                "case": case_name,
                "backend": backend,
                "grid": root_map.shape[0],
                "span": float(x64[-1] - x64[0]),
                "elapsed_seconds": elapsed,
                "convergence_fraction": converged_total / total,
                "mean_iterations": (iter_total / converged_total) if converged_total else float("nan"),
                "root0_fraction": float((root_map == 0).sum().item() / total),
                "root1_fraction": float((root_map == 1).sum().item() / total),
                "root2_fraction": float((root_map == 2).sum().item() / total),
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
        raise RuntimeError("CUDA is required for the double-float prototype.")
    if args.device != "cuda":
        raise RuntimeError("This prototype currently supports CUDA only.")

    ensure_dirs()
    run_dir = make_run_dir(ANALYSIS_DIR, args.output_prefix)

    boundary_x, boundary_y = find_boundary_center(args.coarse_grid, args.tol, args.max_iter)
    ulp_ref = float(np.spacing(np.float32(max(abs(boundary_x), abs(boundary_y), 1.0))))
    zoom_span = args.zoom_step_ulp_ratio * ulp_ref * (args.test_grid - 1)

    full_axis = build_axis(0.0, AXIS_MAX - AXIS_MIN, args.test_grid)
    zoom_x = build_axis(boundary_x, zoom_span, args.test_grid)
    zoom_y = build_axis(boundary_y, zoom_span, args.test_grid)

    rows = []
    rows.extend(summarize_case("full_plane", full_axis, full_axis, args.tol, args.max_iter, args.arith_mode))
    rows.extend(summarize_case("boundary_zoom", zoom_x, zoom_y, args.tol, args.max_iter, args.arith_mode))

    csv_path = run_dir / "prototype_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "Double-float Newton prototype",
        "============================",
        "",
        f"run_dir={run_dir}",
        f"boundary_center=({boundary_x:.12f}, {boundary_y:.12f})",
        f"boundary_ulp_reference={ulp_ref:.10e}",
        f"zoom_step_ulp_ratio={args.zoom_step_ulp_ratio}",
        f"arith_mode={args.arith_mode}",
        f"zoom_span={zoom_span:.10e}",
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
    log_path = run_dir / "prototype_summary.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
