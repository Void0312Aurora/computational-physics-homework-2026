from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from scripts.solution import (
    FRACTAL_NEWTON_MAX_ITER,
    FRACTAL_TOL,
    compute_parallel_fractal,
)  # noqa: E402

ANALYSIS_DIR = ROOT / "result" / "analysis"
BINARY = ROOT / "problem1_cpu_ultra"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_single_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one row in {csv_path}")
    return rows[0]


def finite_fraction_fields(row: dict[str, str]) -> bool:
    keys = [
        "convergence_fraction",
        "mean_iterations",
        "root0_fraction",
        "root1_fraction",
        "root2_fraction",
    ]
    for key in keys:
        try:
            value = float(row[key])
        except (KeyError, ValueError):
            return False
        if not np.isfinite(value):
            return False
    return True


def run_cpu_ultra(
    args: argparse.Namespace, root_map_path: Path, iter_map_path: Path
) -> tuple[dict[str, str], str, Path]:
    run_prefix = f"{args.output_prefix}_cpu_ultra"
    root_map_arg = root_map_path.relative_to(ROOT).as_posix()
    iter_map_arg = iter_map_path.relative_to(ROOT).as_posix()
    cmd = [
        str(BINARY),
        "--compute-grid",
        str(args.compute_grid),
        "--render-grid",
        str(args.render_grid),
        "--tile-rows",
        str(args.tile_rows),
        "--threads",
        str(args.threads),
        "--max-iter",
        str(args.max_iter),
        "--tol",
        str(args.tol),
        "--output-prefix",
        run_prefix,
        "--write-root-map",
        root_map_arg,
        "--write-iter-map",
        iter_map_arg,
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    source_csv = ANALYSIS_DIR / f"{run_prefix}.csv"
    source_command = shlex.join(["./problem1_cpu_ultra", *cmd[1:]])
    return load_single_row(source_csv), source_command, source_csv


def summarize(
    args: argparse.Namespace,
    cpu_row: dict[str, str],
    source_command: str,
    source_csv: Path,
) -> dict[str, Any]:
    reference = compute_parallel_fractal(
        "newton",
        grid_size=args.compute_grid,
        render_grid=args.render_grid,
        tol=args.tol,
        max_iter=args.max_iter,
        workers=args.reference_workers,
    )
    root_map_path = ANALYSIS_DIR / f"{args.output_prefix}_cpu_ultra_roots.bin"
    iter_map_path = ANALYSIS_DIR / f"{args.output_prefix}_cpu_ultra_iters.bin"
    cpu_roots = (
        np.fromfile(root_map_path, dtype=np.uint8)
        .reshape(args.render_grid, args.render_grid)
        .astype(np.int16)
    )
    cpu_roots[cpu_roots == 255] = -1
    cpu_iters = np.fromfile(iter_map_path, dtype=np.float32).reshape(
        args.render_grid, args.render_grid
    )
    ref_roots = np.asarray(reference["root_map"], dtype=np.int16)
    ref_iters = np.asarray(reference["iters"], dtype=np.float32)

    root_diff_mask = cpu_roots != ref_roots
    both_converged = (cpu_roots >= 0) & (ref_roots >= 0)
    iter_abs_diff = np.abs(cpu_iters[both_converged] - ref_iters[both_converged])
    root_fraction_diffs = [
        abs(
            float(cpu_row[f"root{idx}_fraction"])
            - float(reference[f"root{idx}_fraction"])
        )
        for idx in range(3)
    ]
    root_fraction_sum = sum(float(cpu_row[f"root{idx}_fraction"]) for idx in range(3))
    convergence_abs_diff = abs(
        float(cpu_row["convergence_fraction"])
        - float(reference["convergence_fraction"])
    )
    mean_iterations_abs_diff = abs(
        float(cpu_row["mean_iterations"]) - float(reference["mean_iterations"])
    )
    root_diff_fraction = float(np.mean(root_diff_mask))
    iter_mean_abs_diff = float(np.mean(iter_abs_diff)) if iter_abs_diff.size else 0.0
    iter_max_abs_diff = float(np.max(iter_abs_diff)) if iter_abs_diff.size else 0.0

    field_sanity_ok = (
        finite_fraction_fields(cpu_row)
        and 0.0 <= float(cpu_row["convergence_fraction"]) <= 1.0
        and 0.0 <= root_fraction_sum <= 1.0 + args.fraction_tolerance
    )
    accepted = (
        field_sanity_ok
        and root_diff_fraction <= args.root_diff_threshold
        and max(root_fraction_diffs) <= args.fraction_tolerance
        and convergence_abs_diff <= args.fraction_tolerance
        and mean_iterations_abs_diff <= args.iteration_mean_threshold
        and iter_mean_abs_diff <= args.iteration_map_mean_threshold
    )

    return {
        "accepted": accepted,
        "generated_at": utc_now(),
        "compute_grid": args.compute_grid,
        "render_grid": args.render_grid,
        "tile_rows": args.tile_rows,
        "threads": args.threads,
        "reference_workers": args.reference_workers,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "warmup_runs": args.warmup_runs,
        "timed_repeats": args.timed_repeats,
        "timing_scope": args.timing_scope,
        "root_diff_fraction": root_diff_fraction,
        "root_diff_pixels": int(np.count_nonzero(root_diff_mask)),
        "render_pixels": int(args.render_grid * args.render_grid),
        "iter_mean_abs_diff": iter_mean_abs_diff,
        "iter_max_abs_diff": iter_max_abs_diff,
        "max_root_fraction_abs_diff": max(root_fraction_diffs),
        "convergence_abs_diff": convergence_abs_diff,
        "mean_iterations_abs_diff": mean_iterations_abs_diff,
        "cpu_convergence_fraction": float(cpu_row["convergence_fraction"]),
        "reference_convergence_fraction": float(reference["convergence_fraction"]),
        "cpu_mean_iterations": float(cpu_row["mean_iterations"]),
        "reference_mean_iterations": float(reference["mean_iterations"]),
        "cpu_root0_fraction": float(cpu_row["root0_fraction"]),
        "cpu_root1_fraction": float(cpu_row["root1_fraction"]),
        "cpu_root2_fraction": float(cpu_row["root2_fraction"]),
        "reference_root0_fraction": float(reference["root0_fraction"]),
        "reference_root1_fraction": float(reference["root1_fraction"]),
        "reference_root2_fraction": float(reference["root2_fraction"]),
        "root_fraction_sum": root_fraction_sum,
        "field_sanity_ok": field_sanity_ok,
        "root_diff_threshold": args.root_diff_threshold,
        "fraction_tolerance": args.fraction_tolerance,
        "iteration_mean_threshold": args.iteration_mean_threshold,
        "iteration_map_mean_threshold": args.iteration_map_mean_threshold,
        "source_command": source_command,
        "source_csv": source_csv.relative_to(ROOT).as_posix(),
        "root_map_path": root_map_path.relative_to(ROOT).as_posix(),
        "iter_map_path": iter_map_path.relative_to(ROOT).as_posix(),
    }


def write_csv(row: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "accepted",
        "compute_grid",
        "render_grid",
        "tile_rows",
        "threads",
        "reference_workers",
        "max_iter",
        "tol",
        "warmup_runs",
        "timed_repeats",
        "timing_scope",
        "root_diff_fraction",
        "root_diff_pixels",
        "render_pixels",
        "iter_mean_abs_diff",
        "iter_max_abs_diff",
        "max_root_fraction_abs_diff",
        "convergence_abs_diff",
        "mean_iterations_abs_diff",
        "cpu_convergence_fraction",
        "reference_convergence_fraction",
        "cpu_mean_iterations",
        "reference_mean_iterations",
        "field_sanity_ok",
        "source_command",
        "source_csv",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Small-grid correctness gate for CPU ultra vs Python reference."
    )
    parser.add_argument("--compute-grid", type=int, default=128)
    parser.add_argument("--render-grid", type=int, default=32)
    parser.add_argument("--tile-rows", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--reference-workers", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=FRACTAL_NEWTON_MAX_ITER)
    parser.add_argument("--tol", type=float, default=FRACTAL_TOL)
    parser.add_argument("--output-prefix", default="problem1_correctness_gate")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--timed-repeats", type=int, default=1)
    parser.add_argument("--timing-scope", default="small_grid_correctness_full_run")
    parser.add_argument("--root-diff-threshold", type=float, default=0.02)
    parser.add_argument("--fraction-tolerance", type=float, default=0.02)
    parser.add_argument("--iteration-mean-threshold", type=float, default=0.25)
    parser.add_argument("--iteration-map-mean-threshold", type=float, default=0.25)
    args = parser.parse_args()

    if args.compute_grid <= 0 or args.render_grid <= 0:
        raise ValueError("--compute-grid and --render-grid must be positive")
    if args.compute_grid % args.render_grid != 0:
        raise ValueError("--compute-grid must be an integer multiple of --render-grid")
    if args.render_grid % 2 != 0:
        raise ValueError("--render-grid must be even for CPU ultra symmetry")
    if args.warmup_runs != 0 or args.timed_repeats != 1:
        raise ValueError(
            "correctness gate records timing metadata but runs exactly one timed small-grid check"
        )
    if not BINARY.exists():
        raise FileNotFoundError(f"missing CPU ultra binary: {BINARY}")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    root_map_path = ANALYSIS_DIR / f"{args.output_prefix}_cpu_ultra_roots.bin"
    iter_map_path = ANALYSIS_DIR / f"{args.output_prefix}_cpu_ultra_iters.bin"
    cpu_row, source_command, source_csv = run_cpu_ultra(
        args, root_map_path, iter_map_path
    )
    row = summarize(args, cpu_row, source_command, source_csv)

    csv_path = ANALYSIS_DIR / f"{args.output_prefix}.csv"
    json_path = ANALYSIS_DIR / f"{args.output_prefix}.json"
    write_csv(row, csv_path)
    json_path.write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"accepted={row['accepted']}")
    print(f"csv={csv_path.relative_to(ROOT).as_posix()}")
    print(f"json={json_path.relative_to(ROOT).as_posix()}")
    if not row["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
