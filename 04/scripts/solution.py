from __future__ import annotations

import csv
import math
import multiprocessing as mp_pool
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import psutil


ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
RESULT_DIR = ROOT / "result"
mp.mp.dps = 80

FRACTAL_RENDER_GRID = 5000
FRACTAL_SUPERSAMPLE = 4
FRACTAL_GRID_SIZE = FRACTAL_RENDER_GRID * FRACTAL_SUPERSAMPLE
FRACTAL_WORKERS = min(80, os.cpu_count() or 8)
FRACTAL_TOL = 5.0e-7
FRACTAL_NEWTON_MAX_ITER = 55
FRACTAL_SECANT_MAX_ITER = 65
FRACTAL_SECANT_DELTA = 0.2 + 0.2j
FRACTAL_FIGSIZE = (15, 14)
FRACTAL_DPI = 360


def ensure_result_dir() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def classify_roots(z: np.ndarray, roots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = np.stack([np.abs(z - root) for root in roots], axis=0)
    indices = np.argmin(distances, axis=0)
    mins = np.min(distances, axis=0)
    return indices, mins


class ResourceMonitor:
    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.proc = psutil.Process(os.getpid())
        self.cpu_samples: list[float] = []
        self.peak_rss = 0
        self.peak_cpu = 0.0

    def start(self) -> None:
        psutil.cpu_percent(interval=None)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            rss = 0
            processes = [self.proc]
            try:
                processes.extend(self.proc.children(recursive=True))
            except psutil.Error:
                pass
            for process in processes:
                try:
                    rss += process.memory_info().rss
                except psutil.Error:
                    continue
            self.peak_rss = max(self.peak_rss, rss)
            cpu_sample = psutil.cpu_percent(interval=self.interval)
            self.cpu_samples.append(cpu_sample)
            self.peak_cpu = max(self.peak_cpu, cpu_sample)

    def stop(self) -> dict[str, float]:
        self.stop_event.set()
        self.thread.join()
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0
        return {
            "avg_cpu_percent": avg_cpu,
            "peak_cpu_percent": self.peak_cpu,
            "peak_rss_gib": self.peak_rss / (1024.0**3),
        }


def _problem1_row_ranges(grid_size: int, render_grid: int, workers: int) -> list[tuple[int, int]]:
    factor = grid_size // render_grid
    render_edges = np.linspace(0, render_grid, workers + 1, dtype=int)
    ranges: list[tuple[int, int]] = []
    for render_start, render_end in zip(render_edges[:-1], render_edges[1:], strict=True):
        if render_end > render_start:
            ranges.append((int(render_start * factor), int(render_end * factor)))
    return ranges


def _downsample_fractal_block(root_map: np.ndarray, iters: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    out_h = root_map.shape[0] // factor
    out_w = root_map.shape[1] // factor
    reshaped_root = root_map.reshape(out_h, factor, out_w, factor)
    counts = np.stack([(reshaped_root == idx).sum(axis=(1, 3)) for idx in range(3)], axis=0)
    converged_pixels = counts.sum(axis=0)

    small_root = np.full((out_h, out_w), -1, dtype=np.int8)
    has_converged = converged_pixels > 0
    if np.any(has_converged):
        dominant = np.argmax(counts, axis=0).astype(np.int8)
        small_root[has_converged] = dominant[has_converged]

    mask = (root_map >= 0).reshape(out_h, factor, out_w, factor)
    iter_sum = (
        (iters.astype(np.float32) * (root_map >= 0)).reshape(out_h, factor, out_w, factor).sum(axis=(1, 3))
    )
    count_float = mask.sum(axis=(1, 3))
    small_iters = np.zeros((out_h, out_w), dtype=np.float32)
    nonzero = count_float > 0
    small_iters[nonzero] = iter_sum[nonzero] / count_float[nonzero]
    return small_root, small_iters


def _fractal_worker(
    method: str,
    row_start: int,
    row_end: int,
    grid_size: int,
    render_grid: int,
    tol: float,
    max_iter: int,
    delta: complex,
) -> tuple[int, int, np.ndarray, np.ndarray, int, int, np.ndarray]:
    factor = grid_size // render_grid
    roots = np.array(
        [
            1.0 + 0.0j,
            np.exp(2j * np.pi / 3.0),
            np.exp(4j * np.pi / 3.0),
        ],
        dtype=np.complex128,
    )
    axis = np.linspace(-1.8, 1.8, grid_size, dtype=np.float64)
    base = axis[None, :] + 1j * axis[row_start:row_end, None]

    if method == "newton":
        z = base.copy()
    else:
        z_prev = base.copy()
        z = z_prev + delta

    root_map = np.full(z.shape, -1, dtype=np.int8)
    iters = np.zeros(z.shape, dtype=np.int16)
    converged = np.zeros(z.shape, dtype=bool)

    for step in range(1, max_iter + 1):
        remaining = ~converged
        if not np.any(remaining):
            break

        if method == "newton":
            z_remaining = z[remaining]
            derivative = 3.0 * z_remaining * z_remaining
            safe = np.abs(derivative) > 1.0e-14
            z_next = z_remaining.copy()
            z_safe = z_remaining[safe]
            z_next[safe] = z_safe - (z_safe**3 - 1.0) / derivative[safe]
            z[remaining] = z_next

            current = ~converged
            root_idx, distance = classify_roots(z[current], roots)
            just = distance < tol
        else:
            z_prev_remaining = z_prev[remaining]
            z_remaining = z[remaining]
            f_prev = z_prev_remaining**3 - 1.0
            f_curr = z_remaining**3 - 1.0
            denom = f_curr - f_prev
            safe = np.abs(denom) > 1.0e-14
            z_next = z_remaining.copy()
            z_next[safe] = (
                z_remaining[safe]
                - f_curr[safe] * (z_remaining[safe] - z_prev_remaining[safe]) / denom[safe]
            )
            z_prev[remaining] = z_remaining
            z[remaining] = z_next

            current = ~converged
            residual = np.abs(z[current] ** 3 - 1.0)
            root_idx, distance = classify_roots(z[current], roots)
            just = (residual < tol) | (distance < tol)

        if np.any(just):
            current_indices = np.flatnonzero(current)
            hit = current_indices[just]
            root_map.flat[hit] = root_idx[just]
            iters.flat[hit] = step
            converged.flat[hit] = True

    if method == "newton":
        root_idx, distance = classify_roots(z[~converged], roots)
        remaining = np.flatnonzero(~converged)
        late = distance < 10.0 * tol
        if np.any(late):
            hit = remaining[late]
            root_map.flat[hit] = root_idx[late]
            iters.flat[hit] = max_iter
            converged.flat[hit] = True

    small_root, small_iters = _downsample_fractal_block(root_map, iters, factor)
    root_counts = np.array([(root_map == idx).sum() for idx in range(3)], dtype=np.int64)
    total_iters = int(iters[converged].astype(np.int64).sum())
    return (
        row_start // factor,
        row_end // factor,
        small_root,
        small_iters,
        int(converged.sum()),
        total_iters,
        root_counts,
    )


def compute_parallel_fractal(
    method: str,
    grid_size: int,
    render_grid: int,
    tol: float,
    max_iter: int,
    delta: complex = FRACTAL_SECANT_DELTA,
    workers: int = FRACTAL_WORKERS,
) -> dict[str, np.ndarray | float]:
    root_map = np.full((render_grid, render_grid), -1, dtype=np.int8)
    iters = np.zeros((render_grid, render_grid), dtype=np.float32)
    total_points = grid_size * grid_size
    converged_total = 0
    iter_total = 0
    root_counts = np.zeros(3, dtype=np.int64)
    row_ranges = _problem1_row_ranges(grid_size, render_grid, workers)

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_pool.get_context("fork")) as executor:
        futures = [
            executor.submit(
                _fractal_worker,
                method,
                row_start,
                row_end,
                grid_size,
                render_grid,
                tol,
                max_iter,
                delta,
            )
            for row_start, row_end in row_ranges
        ]
        for future in as_completed(futures):
            start_render, end_render, small_root, small_iters, converged_count, total_iters, counts = future.result()
            root_map[start_render:end_render] = small_root
            iters[start_render:end_render] = small_iters
            converged_total += converged_count
            iter_total += total_iters
            root_counts += counts

    convergence_fraction = converged_total / total_points
    mean_iterations = iter_total / converged_total if converged_total else float("nan")
    return {
        "method": method,
        "root_map": root_map,
        "iters": iters,
        "tol": tol,
        "max_iter": max_iter,
        "axis_min": -1.8,
        "axis_max": 1.8,
        "grid_size": grid_size,
        "render_grid": render_grid,
        "workers": workers,
        "convergence_fraction": convergence_fraction,
        "mean_iterations": mean_iterations,
        "root0_fraction": float(root_counts[0] / total_points),
        "root1_fraction": float(root_counts[1] / total_points),
        "root2_fraction": float(root_counts[2] / total_points),
    }


def basin_image(root_map: np.ndarray, iters: np.ndarray, max_iter: int, palette: np.ndarray) -> np.ndarray:
    image = np.zeros(root_map.shape + (3,), dtype=np.uint8)
    image[:, :] = np.array([20, 20, 20], dtype=np.uint8)
    converged = root_map >= 0
    if np.any(converged):
        base = (palette[root_map[converged]] * 255.0).astype(np.float32)
        shade = 0.30 + 0.70 * (1.0 - (iters[converged] - 1.0) / max_iter)
        image[converged] = np.clip(np.rint(255.0 - (255.0 - base) * shade[:, None]), 0, 255).astype(np.uint8)
    return image


def plot_problem1(newton: dict[str, np.ndarray | float], secant: dict[str, np.ndarray | float]) -> None:
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
    fig, axes = plt.subplots(2, 2, figsize=FRACTAL_FIGSIZE, constrained_layout=True)
    panels = [
        ("Newton Fractal / vivid", newton, vivid),
        ("Newton Fractal / pastel", newton, pastel),
        ("Secant Fractal / vivid", secant, vivid),
        ("Secant Fractal / pastel", secant, pastel),
    ]
    for ax, (title, data, palette) in zip(axes.ravel(), panels, strict=True):
        image = basin_image(data["root_map"], data["iters"], int(data["max_iter"]), palette)
        ax.imshow(
            image,
            origin="lower",
            extent=[
                float(data["axis_min"]),
                float(data["axis_max"]),
                float(data["axis_min"]),
                float(data["axis_max"]),
            ],
        )
        ax.set_title(title)
        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")
    fig.suptitle("Problem 1: z^3 - 1 root basins by Newton and secant iterations", fontsize=15)
    fig.savefig(RESULT_DIR / "problem1_fractals.png", dpi=FRACTAL_DPI)
    plt.close(fig)


def problem1_summary(newton: dict[str, np.ndarray | float], secant: dict[str, np.ndarray | float]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for method, data in (("newton", newton), ("secant", secant)):
        rows.append(
            {
                "method": method,
                "tol": float(data["tol"]),
                "max_iter": int(data["max_iter"]),
                "grid_size": int(data["grid_size"]),
                "render_grid": int(data["render_grid"]),
                "workers": int(data["workers"]),
                "convergence_fraction": float(data["convergence_fraction"]),
                "mean_iterations": float(data["mean_iterations"]),
                "root0_fraction": float(data["root0_fraction"]),
                "root1_fraction": float(data["root1_fraction"]),
                "root2_fraction": float(data["root2_fraction"]),
            }
        )
    with (RESULT_DIR / "problem1_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "method",
                "tol",
                "max_iter",
                "grid_size",
                "render_grid",
                "workers",
                "convergence_fraction",
                "mean_iterations",
                "root0_fraction",
                "root1_fraction",
                "root2_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def problem1() -> tuple[list[dict[str, float | str]], dict[str, float]]:
    monitor = ResourceMonitor()
    timer_start = time.perf_counter()
    monitor.start()
    newton = compute_parallel_fractal(
        "newton",
        grid_size=FRACTAL_GRID_SIZE,
        render_grid=FRACTAL_RENDER_GRID,
        tol=FRACTAL_TOL,
        max_iter=FRACTAL_NEWTON_MAX_ITER,
        workers=FRACTAL_WORKERS,
    )
    secant = compute_parallel_fractal(
        "secant",
        grid_size=FRACTAL_GRID_SIZE,
        render_grid=FRACTAL_RENDER_GRID,
        tol=FRACTAL_TOL,
        max_iter=FRACTAL_SECANT_MAX_ITER,
        delta=FRACTAL_SECANT_DELTA,
        workers=FRACTAL_WORKERS,
    )
    resource_summary = monitor.stop()
    resource_summary["elapsed_seconds"] = time.perf_counter() - timer_start
    resource_summary["grid_size"] = float(FRACTAL_GRID_SIZE)
    resource_summary["render_grid"] = float(FRACTAL_RENDER_GRID)
    resource_summary["workers"] = float(FRACTAL_WORKERS)

    with (RESULT_DIR / "problem1_resources.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "grid_size",
                "render_grid",
                "workers",
                "elapsed_seconds",
                "avg_cpu_percent",
                "peak_cpu_percent",
                "peak_rss_gib",
            ],
        )
        writer.writeheader()
        writer.writerow(resource_summary)

    plot_problem1(newton, secant)
    return problem1_summary(newton, secant), resource_summary


def hybrid_newton_bisection(
    f: Callable[[float], float],
    df: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1.0e-12,
    max_iter: int = 100,
) -> tuple[float, int]:
    fa = f(a)
    fb = f(b)
    if fa == 0.0:
        return a, 0
    if fb == 0.0:
        return b, 0
    if fa * fb > 0.0:
        raise ValueError("Bracket does not contain a sign change")

    x = 0.5 * (a + b)
    for iteration in range(1, max_iter + 1):
        fx = f(x)
        if abs(fx) < tol or 0.5 * (b - a) < tol:
            return x, iteration
        dfx = df(x)
        candidate = None
        if math.isfinite(dfx) and abs(dfx) > 1.0e-14:
            newton = x - fx / dfx
            if a < newton < b and math.isfinite(newton):
                candidate = newton
        if candidate is None:
            candidate = 0.5 * (a + b)

        fc = f(candidate)
        if fa * fc <= 0.0:
            b, fb = candidate, fc
        else:
            a, fa = candidate, fc
        x = candidate
    return x, max_iter


def scan_sign_changes(
    f: Callable[[float], float], start: float, end: float, samples: int
) -> list[tuple[float, float]]:
    xs = np.linspace(start, end, samples + 1)
    vals = [f(float(x)) for x in xs]
    brackets: list[tuple[float, float]] = []
    for left, right, f_left, f_right in zip(xs[:-1], xs[1:], vals[:-1], vals[1:], strict=True):
        if not (math.isfinite(f_left) and math.isfinite(f_right)):
            continue
        if f_left == 0.0:
            brackets.append((float(left), float(left)))
            continue
        if f_left * f_right < 0.0:
            brackets.append((float(left), float(right)))
    return brackets


def deduplicate_roots(roots: list[float], tol: float = 1.0e-9) -> list[float]:
    unique: list[float] = []
    for root in sorted(roots):
        if not unique or abs(root - unique[-1]) > tol:
            unique.append(root)
    return unique


def problem2() -> list[dict[str, float]]:
    def f(x: float) -> float:
        return 4.0 * math.cos(x) - math.exp(x)

    def df(x: float) -> float:
        return -4.0 * math.sin(x) - math.exp(x)

    brackets = scan_sign_changes(f, -10.0, 2.0, 6000)
    rows: list[dict[str, float]] = []
    roots: list[float] = []
    for a, b in brackets:
        if a == b:
            root = a
            iterations = 0
        else:
            root, iterations = hybrid_newton_bisection(f, df, a, b)
        residual = f(root)
        try:
            ref = float(mp.findroot(lambda t: 4 * mp.cos(t) - mp.e**t, (a, b if b != a else a + 0.05)))
        except Exception:
            ref = root
        roots.append(root)
        rows.append(
            {
                "interval_left": a,
                "interval_right": b,
                "root": root,
                "iterations": iterations,
                "residual": residual,
                "reference": ref,
                "abs_error": abs(root - ref),
            }
        )

    unique = deduplicate_roots([row["root"] for row in rows], tol=1.0e-8)
    filtered: list[dict[str, float]] = []
    for root in unique:
        candidates = [row for row in rows if abs(row["root"] - root) < 1.0e-8]
        filtered.append(min(candidates, key=lambda item: abs(item["residual"])))

    with (RESULT_DIR / "problem2_roots.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "interval_left",
                "interval_right",
                "root",
                "iterations",
                "residual",
                "reference",
                "abs_error",
            ],
        )
        writer.writeheader()
        writer.writerows(filtered)
    return filtered


def brent_root(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1.0e-12,
    max_iter: int = 100,
) -> tuple[float, int]:
    fa = f(a)
    fb = f(b)
    if fa * fb > 0.0:
        raise ValueError("Root must be bracketed")
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = c
    mflag = True

    for iteration in range(1, max_iter + 1):
        if fa != fc and fb != fc:
            s = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:
            s = b - fb * (b - a) / (fb - fa)

        lower = (3.0 * a + b) / 4.0
        upper = b
        if lower > upper:
            lower, upper = upper, lower

        cond1 = not (lower < s < upper)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2.0
        cond3 = (not mflag) and abs(s - b) >= abs(c - d) / 2.0
        cond4 = mflag and abs(b - c) < tol
        cond5 = (not mflag) and abs(c - d) < tol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = 0.5 * (a + b)
            mflag = True
        else:
            mflag = False

        fs = f(s)
        d = c
        c = b
        fc = fb

        if fa * fs < 0.0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        if abs(fb) < tol or abs(b - a) < tol:
            return b, iteration

    return b, max_iter


def safe_problem3_value(kind: str, h: float, x: float) -> float | None:
    if not (0.0 < x < h):
        return None
    inside = h * h - x * x
    if inside <= 0.0:
        return None
    try:
        root = math.sqrt(inside)
        if kind == "f":
            value = x * math.tan(x) - root
        else:
            tan_x = math.tan(x)
            if abs(tan_x) < 1.0e-12:
                return None
            value = x / tan_x + root
    except ValueError:
        return None
    if not math.isfinite(value) or abs(value) > 1.0e6:
        return None
    return value


def scan_problem3_brackets(kind: str, h: float, samples: int = 6000) -> list[tuple[float, float]]:
    eps = 1.0e-6
    singularities: list[float] = []
    if kind == "f":
        k = 0
        while (0.5 + k) * math.pi < h:
            singularities.append((0.5 + k) * math.pi)
            k += 1
    else:
        k = 1
        while k * math.pi < h:
            singularities.append(k * math.pi)
            k += 1
    boundaries = [eps]
    boundaries.extend(point for point in singularities if eps < point < h - eps)
    boundaries.append(h - eps)
    brackets: list[tuple[float, float]] = []
    for left_bound, right_bound in zip(boundaries[:-1], boundaries[1:], strict=True):
        left = left_bound + eps
        right = right_bound - eps
        if left >= right:
            continue
        xs = np.linspace(left, right, samples)
        prev_x = None
        prev_value = None
        for x in xs:
            value = safe_problem3_value(kind, h, float(x))
            if value is None:
                prev_x = None
                prev_value = None
                continue
            if prev_x is not None and prev_value is not None and prev_value * value < 0.0:
                brackets.append((prev_x, float(x)))
            prev_x = float(x)
            prev_value = value
    return brackets


def problem3() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    h_values = [0.2, 0.5, 1.0, 2.0]
    for h in h_values:
        for kind in ("f", "g"):
            brackets = scan_problem3_brackets(kind, h)
            if not brackets:
                rows.append(
                    {
                        "function": kind,
                        "h": h,
                        "root": "",
                        "iterations": "",
                        "residual": "",
                        "reference": "",
                        "abs_error": "",
                        "status": "no root detected in (0, h)",
                    }
                )
                continue
            for a, b in brackets:
                func = lambda x, kind=kind, h=h: safe_problem3_value(kind, h, x)  # noqa: E731
                root, iterations = brent_root(lambda x: float(func(x)), a, b)
                residual = float(func(root))
                if kind == "f":
                    ref_mp = mp.findroot(
                        lambda t: t * mp.tan(t) - mp.sqrt(h * h - t * t),
                        (a, b),
                    )
                else:
                    ref_mp = mp.findroot(
                        lambda t: t / mp.tan(t) + mp.sqrt(h * h - t * t),
                        (a, b),
                    )
                ref = float(ref_mp)
                rows.append(
                    {
                        "function": kind,
                        "h": h,
                        "root": root,
                        "iterations": iterations,
                        "residual": residual,
                        "reference": ref,
                        "abs_error": abs(root - ref),
                        "status": "ok",
                    }
                )

    with (RESULT_DIR / "problem3_roots.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "function",
                "h",
                "root",
                "iterations",
                "residual",
                "reference",
                "abs_error",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def cubic_roots_shengjin(a: float, b: float, c: float, d: float) -> list[float]:
    if a == 0.0:
        raise ValueError("Leading coefficient must be non-zero")
    aa = b / a
    bb = c / a
    cc = d / a
    p = bb - aa * aa / 3.0
    q = 2.0 * aa**3 / 27.0 - aa * bb / 3.0 + cc
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3

    if disc <= 0.0:
        radius = 2.0 * math.sqrt(-p / 3.0)
        cos_arg = -(q / 2.0) / math.sqrt(-(p / 3.0) ** 3)
        cos_arg = max(-1.0, min(1.0, cos_arg))
        theta = math.acos(cos_arg)
        roots = [
            radius * math.cos((theta + 2.0 * math.pi * k) / 3.0) - aa / 3.0
            for k in range(3)
        ]
        return sorted(roots)

    def cbrt(x: float) -> float:
        return math.copysign(abs(x) ** (1.0 / 3.0), x)

    u = cbrt(-q / 2.0 + math.sqrt(disc))
    v = cbrt(-q / 2.0 - math.sqrt(disc))
    return [u + v - aa / 3.0]


def problem4() -> list[dict[str, float]]:
    coeffs = (1.0, -70.5, 1533.54, -10082.44)
    roots = cubic_roots_shengjin(*coeffs)

    def poly(x: float) -> float:
        return ((x**3) - 70.5 * x * x + 1533.54 * x - 10082.44)

    rows = []
    for root in roots:
        rows.append({"root": root, "residual": poly(root)})

    with (RESULT_DIR / "problem4_shengjin_roots.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["root", "residual"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


G = 6.674e-11
M = 5.974e24
m = 7.348e22
R = 3.844e8
OMEGA = 2.662e-6


def l1_function(r: float) -> float:
    return G * M / (r * r) - G * m / ((R - r) * (R - r)) - OMEGA * OMEGA * r


def l1_derivative(r: float) -> float:
    return -2.0 * G * M / (r**3) - 2.0 * G * m / ((R - r) ** 3) - OMEGA * OMEGA


def newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1.0e-12,
    max_iter: int = 50,
) -> tuple[float, int]:
    x = x0
    for iteration in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        x_new = x - fx / dfx
        if abs(x_new - x) < tol * max(1.0, abs(x_new)):
            return x_new, iteration
        x = x_new
    return x, max_iter


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1.0e-12,
    max_iter: int = 80,
) -> tuple[float, int]:
    prev = x0
    curr = x1
    f_prev = f(prev)
    f_curr = f(curr)
    for iteration in range(1, max_iter + 1):
        denom = f_curr - f_prev
        if abs(denom) < 1.0e-20:
            return curr, iteration
        nxt = curr - f_curr * (curr - prev) / denom
        if abs(nxt - curr) < tol * max(1.0, abs(nxt)):
            return nxt, iteration
        prev, curr = curr, nxt
        f_prev, f_curr = f_curr, f(curr)
    return curr, max_iter


def problem5() -> list[dict[str, float | str]]:
    ref = float(
        mp.findroot(
            lambda r: G * M / (r * r) - G * m / ((R - r) ** 2) - OMEGA**2 * r,
            3.2e8,
        )
    )
    newton_root, newton_iters = newton(l1_function, l1_derivative, 3.2e8)
    secant_root, secant_iters = secant(l1_function, 3.0e8, 3.3e8)
    rows: list[dict[str, float | str]] = []
    for method, root, iterations in (
        ("newton", newton_root, newton_iters),
        ("secant", secant_root, secant_iters),
    ):
        rows.append(
            {
                "method": method,
                "root_m": root,
                "iterations": iterations,
                "residual": l1_function(root),
                "reference": ref,
                "abs_error": abs(root - ref),
            }
        )

    with (RESULT_DIR / "problem5_l1.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["method", "root_m", "iterations", "residual", "reference", "abs_error"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_log(
    problem1_rows: list[dict[str, float | str]],
    problem1_resources: dict[str, float],
    problem2_rows: list[dict[str, float]],
    problem3_rows: list[dict[str, float | str]],
    problem4_rows: list[dict[str, float]],
    problem5_rows: list[dict[str, float | str]],
) -> None:
    log_path = RESULT_DIR / "temp-01.log"
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("HW/04 numerical results\n")
        fh.write("=======================\n\n")

        fh.write("Problem 1: Newton/Secant fractal summary for z^3 - 1 = 0\n")
        fh.write(
            f"Tolerance = {FRACTAL_TOL:.1e}, study window = [-1.8, 1.8] x [-1.8, 1.8], "
            f"compute grid = {FRACTAL_GRID_SIZE} x {FRACTAL_GRID_SIZE}, "
            f"render grid = {FRACTAL_RENDER_GRID} x {FRACTAL_RENDER_GRID}, "
            f"workers = {FRACTAL_WORKERS}, figure dpi = {FRACTAL_DPI}\n"
        )
        for row in problem1_rows:
            fh.write(
                "  {method}: convergence_fraction={convergence_fraction:.6f}, "
                "mean_iterations={mean_iterations:.4f}, basin fractions="
                "({root0_fraction:.6f}, {root1_fraction:.6f}, {root2_fraction:.6f})\n".format(**row)
            )
        fh.write(
            f"  resource summary: elapsed={problem1_resources['elapsed_seconds']:.2f} s, "
            f"avg_cpu_percent={problem1_resources['avg_cpu_percent']:.2f}, "
            f"peak_cpu_percent={problem1_resources['peak_cpu_percent']:.2f}, "
            f"peak_rss_gib={problem1_resources['peak_rss_gib']:.2f}\n"
        )
        fh.write("  Figure: result/problem1_fractals.png\n\n")

        fh.write("Problem 2: safeguarded Newton-Bisection roots of 4 cos(x) - exp(x)\n")
        fh.write("Study interval fixed to [-10, 2] because the function has infinitely many negative roots.\n")
        for row in problem2_rows:
            fh.write(
                f"  bracket=[{row['interval_left']:.6f}, {row['interval_right']:.6f}] "
                f"root={row['root']:.15f}, iterations={int(row['iterations'])}, "
                f"residual={row['residual']:.3e}, abs_error={row['abs_error']:.3e}\n"
            )
        fh.write("\n")

        fh.write("Problem 3: Brent-style Muller-Brent results\n")
        for row in problem3_rows:
            if row["status"] != "ok":
                fh.write(
                    f"  {row['function']}(x), h={row['h']}: {row['status']}\n"
                )
            else:
                fh.write(
                    f"  {row['function']}(x), h={row['h']}: root={float(row['root']):.15f}, "
                    f"iterations={int(row['iterations'])}, residual={float(row['residual']):.3e}, "
                    f"abs_error={float(row['abs_error']):.3e}\n"
                )
        fh.write("\n")

        fh.write("Problem 4: Shengjin / closed-form cubic roots\n")
        for row in problem4_rows:
            fh.write(
                f"  root={row['root']:.15f}, residual={row['residual']:.3e}\n"
            )
        fh.write("\n")

        fh.write("Problem 5: Earth-Moon L1\n")
        for row in problem5_rows:
            fh.write(
                f"  {row['method']}: root={float(row['root_m']):.12f} m, "
                f"iterations={int(row['iterations'])}, residual={float(row['residual']):.3e}, "
                f"abs_error={float(row['abs_error']):.3e}\n"
            )


def main() -> None:
    ensure_result_dir()

    problem1_rows, problem1_resources = problem1()
    problem2_rows = problem2()
    problem3_rows = problem3()
    problem4_rows = problem4()
    problem5_rows = problem5()
    write_log(problem1_rows, problem1_resources, problem2_rows, problem3_rows, problem4_rows, problem5_rows)


if __name__ == "__main__":
    main()
