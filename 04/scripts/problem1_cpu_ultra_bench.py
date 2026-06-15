from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
RESULT_DIR = ROOT / "result" / "analysis"
BINARY = ROOT / "problem1_cpu_ultra"


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def load_single_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one result row in {csv_path}")
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a benchmark sweep for problem1_cpu_ultra.")
    parser.add_argument("--compute-grids", default="20000,40000,80000")
    parser.add_argument("--render-grid", type=int, default=5000)
    parser.add_argument("--tile-rows", type=int, default=64)
    parser.add_argument("--threads", type=int, default=88)
    parser.add_argument("--max-iter", type=int, default=55)
    parser.add_argument("--tol", type=float, default=5.0e-7)
    parser.add_argument("--output-prefix", default="problem1_cpu_ultra_bench")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for compute_grid in parse_int_list(args.compute_grids):
        run_prefix = f"{args.output_prefix}_{compute_grid}"
        cmd = [
            str(BINARY),
            "--compute-grid",
            str(compute_grid),
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
        ]
        subprocess.run(cmd, check=True, cwd=ROOT)
        row = load_single_row(RESULT_DIR / f"{run_prefix}.csv")
        row["run_prefix"] = run_prefix
        rows.append(row)

    summary_path = RESULT_DIR / f"{args.output_prefix}_summary.csv"
    fieldnames = ["run_prefix"] + list(rows[0].keys())
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
