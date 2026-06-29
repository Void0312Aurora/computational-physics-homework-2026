from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.solution import (  # noqa: E402
    FRACTAL_NEWTON_MAX_ITER,
    FRACTAL_SECANT_DELTA,
    FRACTAL_SECANT_MAX_ITER,
    FRACTAL_TOL,
    classify_roots,
)


RESULT_DIR = ROOT / "result" / "analysis"
ROOTS = np.array(
    [
        1.0 + 0.0j,
        np.exp(2j * np.pi / 3.0),
        np.exp(4j * np.pi / 3.0),
    ],
    dtype=np.complex128,
)


def iterate_grid(
    method: str,
    grid: int,
    tol: float,
    max_iter: int,
    delta: complex = FRACTAL_SECANT_DELTA,
) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.8, 1.8, grid, dtype=np.float64)
    base = axis[None, :] + 1j * axis[:, None]
    root_map = np.full((grid, grid), -1, dtype=np.int8)
    iters = np.zeros((grid, grid), dtype=np.int16)
    converged = np.zeros((grid, grid), dtype=bool)

    if method == "newton":
        z = base.copy()
    elif method == "secant":
        z_prev = base.copy()
        z = z_prev + delta
    else:
        raise ValueError(f"unsupported method: {method}")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
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
                root_idx, distance = classify_roots(z[current], ROOTS)
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
                root_idx, distance = classify_roots(z[current], ROOTS)
                just = (residual < tol) | (distance < tol)

            if np.any(just):
                current_indices = np.flatnonzero(current)
                hit = current_indices[just]
                root_map.flat[hit] = root_idx[just]
                iters.flat[hit] = step
                converged.flat[hit] = True

        if method == "newton":
            remaining = ~converged
            root_idx, distance = classify_roots(z[remaining], ROOTS)
            late = distance < 10.0 * tol
            if np.any(late):
                hit = np.flatnonzero(remaining)[late]
                root_map.flat[hit] = root_idx[late]
                iters.flat[hit] = max_iter
                converged.flat[hit] = True

    return root_map, iters


def neighbor_metrics(root_map: np.ndarray) -> dict[str, float]:
    left = root_map[:, :-1]
    right = root_map[:, 1:]
    lower = root_map[:-1, :]
    upper = root_map[1:, :]

    horizontal_both = (left >= 0) & (right >= 0)
    vertical_both = (lower >= 0) & (upper >= 0)
    both_converged_pairs = int(horizontal_both.sum() + vertical_both.sum())
    root_flips = int(((left != right) & horizontal_both).sum() + ((lower != upper) & vertical_both).sum())

    total_pairs = int(left.size + lower.size)
    failed_pairs = int(((left < 0) | (right < 0)).sum() + ((lower < 0) | (upper < 0)).sum())

    return {
        "neighbor_root_flip_fraction": root_flips / both_converged_pairs
        if both_converged_pairs
        else float("nan"),
        "failed_neighbor_pair_fraction": failed_pairs / total_pairs if total_pairs else float("nan"),
    }


def summarize_case(
    case: str,
    method: str,
    delta: complex | None,
    grid: int,
    tol: float,
    max_iter: int,
    root_map: np.ndarray,
    iters: np.ndarray,
    elapsed_seconds: float,
) -> dict[str, float | int | str]:
    converged = root_map >= 0
    total = root_map.size
    converged_count = int(converged.sum())
    conv_iters = iters[converged].astype(np.float64)
    quantiles = (
        np.percentile(conv_iters, [50.0, 90.0, 95.0, 99.0])
        if converged_count
        else np.array([float("nan")] * 4)
    )
    metrics = neighbor_metrics(root_map)
    return {
        "case": case,
        "method": method,
        "delta_real": float(delta.real) if delta is not None else float("nan"),
        "delta_imag": float(delta.imag) if delta is not None else float("nan"),
        "tol": tol,
        "max_iter": max_iter,
        "grid": grid,
        "points": total,
        "converged": converged_count,
        "convergence_fraction": converged_count / total,
        "failure_fraction": 1.0 - converged_count / total,
        "mean_iterations": float(conv_iters.mean()) if converged_count else float("nan"),
        "p50_iterations": float(quantiles[0]),
        "p90_iterations": float(quantiles[1]),
        "p95_iterations": float(quantiles[2]),
        "p99_iterations": float(quantiles[3]),
        "max_iterations": int(conv_iters.max()) if converged_count else 0,
        "root0_fraction": float((root_map == 0).sum() / total),
        "root1_fraction": float((root_map == 1).sum() / total),
        "root2_fraction": float((root_map == 2).sum() / total),
        "neighbor_root_flip_fraction": metrics["neighbor_root_flip_fraction"],
        "failed_neighbor_pair_fraction": metrics["failed_neighbor_pair_fraction"],
        "elapsed_seconds": elapsed_seconds,
    }


def sensitivity_row(
    case: str,
    delta: complex,
    baseline_case: str,
    baseline_delta: complex,
    baseline_root: np.ndarray,
    baseline_summary: dict[str, float | int | str],
    root_map: np.ndarray,
    summary: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    both = (baseline_root >= 0) & (root_map >= 0)
    both_count = int(both.sum())
    disagreement = int((baseline_root[both] != root_map[both]).sum()) if both_count else 0
    total = root_map.size
    return {
        "case": case,
        "delta_real": float(delta.real),
        "delta_imag": float(delta.imag),
        "baseline_case": baseline_case,
        "baseline_delta_real": float(baseline_delta.real),
        "baseline_delta_imag": float(baseline_delta.imag),
        "grid": int(summary["grid"]),
        "both_converged_fraction": both_count / total,
        "root_disagreement_fraction": disagreement / both_count if both_count else float("nan"),
        "convergence_fraction_delta": float(summary["convergence_fraction"])
        - float(baseline_summary["convergence_fraction"]),
        "mean_iterations_delta": float(summary["mean_iterations"]) - float(baseline_summary["mean_iterations"]),
        "p95_iterations_delta": float(summary["p95_iterations"]) - float(baseline_summary["p95_iterations"]),
        "p99_iterations_delta": float(summary["p99_iterations"]) - float(baseline_summary["p99_iterations"]),
    }


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify Problem 1 secant convergence and second-initial-point sensitivity."
    )
    parser.add_argument("--grid", type=int, default=1024, help="Square grid size for lightweight analysis.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grid < 16:
        raise SystemExit("--grid must be at least 16")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cases: list[tuple[str, str, complex | None, int]] = [
        ("newton", "newton", None, FRACTAL_NEWTON_MAX_ITER),
        ("secant_delta_0p2_0p2i", "secant", FRACTAL_SECANT_DELTA, FRACTAL_SECANT_MAX_ITER),
        ("secant_delta_0p1_0p1i", "secant", 0.1 + 0.1j, FRACTAL_SECANT_MAX_ITER),
        ("secant_delta_0p3_0p3i", "secant", 0.3 + 0.3j, FRACTAL_SECANT_MAX_ITER),
        ("secant_delta_0p2_0p0i", "secant", 0.2 + 0.0j, FRACTAL_SECANT_MAX_ITER),
        ("secant_delta_0p0_0p2i", "secant", 0.0 + 0.2j, FRACTAL_SECANT_MAX_ITER),
    ]

    roots_by_case: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, float | int | str]] = {}
    rows: list[dict[str, float | int | str]] = []
    for case, method, delta, max_iter in cases:
        start = time.perf_counter()
        root_map, iters = iterate_grid(
            method=method,
            grid=args.grid,
            tol=FRACTAL_TOL,
            max_iter=max_iter,
            delta=delta if delta is not None else FRACTAL_SECANT_DELTA,
        )
        elapsed = time.perf_counter() - start
        summary = summarize_case(case, method, delta, args.grid, FRACTAL_TOL, max_iter, root_map, iters, elapsed)
        roots_by_case[case] = root_map
        summaries[case] = summary
        rows.append(summary)
        print(
            f"{case}: convergence={summary['convergence_fraction']:.8f}, "
            f"mean_iter={summary['mean_iterations']:.4f}, elapsed={elapsed:.2f}s"
        )

    baseline_case = "secant_delta_0p2_0p2i"
    sensitivity_rows = [
        sensitivity_row(
            case=case,
            delta=delta,
            baseline_case=baseline_case,
            baseline_delta=FRACTAL_SECANT_DELTA,
            baseline_root=roots_by_case[baseline_case],
            baseline_summary=summaries[baseline_case],
            root_map=roots_by_case[case],
            summary=summaries[case],
        )
        for case, method, delta, _max_iter in cases
        if method == "secant" and delta is not None
    ]

    write_csv(RESULT_DIR / "problem1_secant_stability_summary.csv", rows)
    write_csv(RESULT_DIR / "problem1_secant_delta_sensitivity.csv", sensitivity_rows)


if __name__ == "__main__":
    main()
