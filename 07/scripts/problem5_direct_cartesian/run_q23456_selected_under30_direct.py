from __future__ import annotations

import argparse
import csv
from io import StringIO
import os
from pathlib import Path
import subprocess

from problem5_core import build_binary, reference_volume, write_csv


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results"
BINARY = ROOT / "direct_tensor_midpoint_orthant_mixed_q26_tile_batch_avx2"

DEFAULT_THREADS = 88
DEFAULT_TAIL_DIMENSION = 8
DEFAULT_PREFIX_CHUNK_POINTS = 128
DEFAULT_BATCH_PREFIXES = 16
STORAGE_RANK8 = 1

FIELDS = [
    "dimension",
    "q_pattern",
    "q2_axis_count",
    "q3_axis_count",
    "q4_axis_count",
    "q5_axis_count",
    "q6_axis_count",
    "tail_dimension",
    "prefix_chunk_points",
    "batch_prefixes",
    "storage_mode",
    "total_points",
    "direct_inside_points",
    "direct_estimate",
    "reference_volume",
    "direct_relative_error",
    "within_10_percent",
    "direct_total_runtime_s",
    "direct_build_runtime_s",
    "direct_count_runtime_s",
    "direct_count_points_per_s",
    "threads",
    "predicted_inside_points",
    "predicted_estimate",
    "predicted_relative_error",
    "inside_points_match_prediction",
    "estimate_abs_diff_prediction",
    "mode",
]


def q_pattern(counts: tuple[int, int, int, int, int]) -> str:
    return f"2^{counts[0]} 3^{counts[1]} 4^{counts[2]} 5^{counts[3]} 6^{counts[4]}"


def read_selected_rows(dimensions: set[int]) -> list[dict[str, str]]:
    path = RESULT_DIR / "q23456_selected_under30_results.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if int(row["dimension"]) in dimensions]


def read_existing(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_single_csv(stdout: str) -> dict[str, str]:
    rows = list(csv.DictReader(StringIO(stdout)))
    if len(rows) != 1:
        raise RuntimeError(f"Unexpected CSV output: {stdout!r}")
    return rows[0]


def run_direct(
    counts: tuple[int, int, int, int, int],
    *,
    tail_dimension: int,
    threads: int,
    prefix_chunk_points: int,
    batch_prefixes: int,
) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("OMP_PROC_BIND", "spread")
    env.setdefault("OMP_PLACES", "cores")
    completed = subprocess.run(
        [
            str(BINARY),
            str(counts[0]),
            str(counts[1]),
            str(counts[2]),
            str(counts[3]),
            str(counts[4]),
            str(tail_dimension),
            str(threads),
            str(prefix_chunk_points),
            str(batch_prefixes),
            str(STORAGE_RANK8),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    return read_single_csv(completed.stdout)


def normalize(selected: dict[str, str], raw: dict[str, str]) -> dict[str, object]:
    dimension = int(raw["dimension"])
    counts = tuple(int(raw[f"q{q}_axis_count"]) for q in [2, 3, 4, 5, 6])
    direct_estimate = float(raw["estimate"])
    reference = reference_volume(dimension)
    relative_error = abs(direct_estimate - reference) / reference
    predicted_estimate = float(selected["coefficient_estimate"])
    direct_inside = int(raw["inside_points"])
    predicted_inside = int(selected["coefficient_inside_points"])
    return {
        "dimension": dimension,
        "q_pattern": q_pattern(counts),
        "q2_axis_count": counts[0],
        "q3_axis_count": counts[1],
        "q4_axis_count": counts[2],
        "q5_axis_count": counts[3],
        "q6_axis_count": counts[4],
        "tail_dimension": raw["tail_dimension"],
        "prefix_chunk_points": raw["prefix_chunk_points"],
        "batch_prefixes": raw["batch_prefixes"],
        "storage_mode": raw["storage_mode"],
        "total_points": raw["total_points"],
        "direct_inside_points": direct_inside,
        "direct_estimate": raw["estimate"],
        "reference_volume": f"{reference:.16e}",
        "direct_relative_error": f"{relative_error:.6e}",
        "within_10_percent": str(relative_error <= 0.10).lower(),
        "direct_total_runtime_s": raw["total_runtime_s"],
        "direct_build_runtime_s": raw["build_runtime_s"],
        "direct_count_runtime_s": raw["count_runtime_s"],
        "direct_count_points_per_s": raw["count_points_per_s"],
        "threads": raw["threads"],
        "predicted_inside_points": predicted_inside,
        "predicted_estimate": selected["coefficient_estimate"],
        "predicted_relative_error": selected["predicted_relative_error"],
        "inside_points_match_prediction": str(direct_inside == predicted_inside).lower(),
        "estimate_abs_diff_prediction": f"{abs(direct_estimate - predicted_estimate):.18e}",
        "mode": raw["mode"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", default="27,28,29,30")
    parser.add_argument("--tail-dimension", type=int, default=DEFAULT_TAIL_DIMENSION)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--prefix-chunk-points", type=int, default=DEFAULT_PREFIX_CHUNK_POINTS)
    parser.add_argument("--batch-prefixes", type=int, default=DEFAULT_BATCH_PREFIXES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dimensions = {int(item) for item in args.dimensions.split(",") if item.strip()}
    output_csv = RESULT_DIR / "q23456_selected_under30_direct_results.csv"
    existing = [] if args.force else read_existing(output_csv)
    done = {int(row["dimension"]) for row in existing}

    rows: list[dict[str, object]] = list(existing)
    build_binary()
    for selected in read_selected_rows(dimensions):
        dimension = int(selected["dimension"])
        if dimension in done:
            continue
        counts = tuple(int(selected[f"q{q}_axis_count"]) for q in [2, 3, 4, 5, 6])
        raw = run_direct(
            counts,
            tail_dimension=args.tail_dimension,
            threads=args.threads,
            prefix_chunk_points=args.prefix_chunk_points,
            batch_prefixes=args.batch_prefixes,
        )
        rows.append(normalize(selected, raw))
        rows.sort(key=lambda row: int(row["dimension"]))
        write_csv(output_csv, rows, FIELDS)
        done.add(dimension)

    log_lines = [
        "q23456 selected under-30 direct runs",
        "====================================",
        f"dimensions={','.join(str(item) for item in sorted(dimensions))}",
        f"completed_dimensions={','.join(str(item) for item in sorted(done))}",
        "",
    ]
    for row in sorted(rows, key=lambda item: int(item["dimension"])):
        if int(row["dimension"]) in dimensions:
            log_lines.append(
                f"n={row['dimension']}: pattern={row['q_pattern']}, "
                f"inside={row['direct_inside_points']}, "
                f"estimate={row['direct_estimate']}, "
                f"relerr={row['direct_relative_error']}, "
                f"count_runtime_s={row['direct_count_runtime_s']}, "
                f"match_prediction={row['inside_points_match_prediction']}"
            )
    log_lines.append("")
    log_lines.append(f"Direct CSV written to {output_csv}")
    (RESULT_DIR / "q23456_selected_under30_direct_results_run.log").write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
