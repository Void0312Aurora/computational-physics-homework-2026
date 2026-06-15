from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
RESULT_DIR = ROOT / "result"
ANALYSIS_DIR = RESULT_DIR / "analysis"
BINARY = ROOT / "problem1_cpu_ultra"

AXIS_MIN = -1.8
AXIS_MAX = 1.8
FRACTAL_DPI = 360
FRACTAL_FIGSIZE = (15, 7)


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


def load_single_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one row in {csv_path}")
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cpu-ultra Newton render and create a final PNG.")
    parser.add_argument("--compute-grid", type=int, default=80000)
    parser.add_argument("--render-grid", type=int, default=5000)
    parser.add_argument("--tile-rows", type=int, default=256)
    parser.add_argument("--threads", type=int, default=88)
    parser.add_argument("--max-iter", type=int, default=55)
    parser.add_argument("--tol", type=float, default=5.0e-7)
    parser.add_argument("--output-prefix", default="problem1_cpu_ultra_render")
    parser.add_argument("--final-image", default="problem1_cpu_ultra_fractals.png")
    args = parser.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    root_path = ANALYSIS_DIR / f"{args.output_prefix}_root_map.bin"
    iter_path = ANALYSIS_DIR / f"{args.output_prefix}_iter_map.bin"
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
        args.output_prefix,
        "--write-root-map",
        str(root_path),
        "--write-iter-map",
        str(iter_path),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)

    root_map = np.fromfile(root_path, dtype=np.uint8).reshape(args.render_grid, args.render_grid)
    root_map = root_map.astype(np.int16)
    root_map[root_map == 255] = -1
    iters = np.fromfile(iter_path, dtype=np.float32).reshape(args.render_grid, args.render_grid)
    vivid, pastel = palettes()

    fig, axes = plt.subplots(1, 2, figsize=FRACTAL_FIGSIZE, constrained_layout=True)
    for ax, title, palette in zip(
        axes,
        ("CPU ultra Newton / vivid", "CPU ultra Newton / pastel"),
        (vivid, pastel),
        strict=True,
    ):
        image = basin_image(root_map, iters, args.max_iter, palette)
        ax.imshow(
            image,
            origin="lower",
            extent=[AXIS_MIN, AXIS_MAX, AXIS_MIN, AXIS_MAX],
        )
        ax.set_title(title)
        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")
    fig.suptitle("Problem 1: cpu-ultra Newton fractal", fontsize=15)
    final_image = RESULT_DIR / args.final_image
    fig.savefig(final_image, dpi=FRACTAL_DPI)
    plt.close(fig)

    summary_row = load_single_row(ANALYSIS_DIR / f"{args.output_prefix}.csv")
    summary_csv = RESULT_DIR / "problem1_cpu_ultra_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)


if __name__ == "__main__":
    main()
