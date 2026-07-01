from __future__ import annotations

import csv
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results"
Q_VALUES = (2, 3, 4, 5, 6)
COMMON_LCM_Q = 60
THRESHOLD_UNITS = (2 * COMMON_LCM_Q) ** 2


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_binary() -> None:
    subprocess.run(["make", "direct_tensor_midpoint_orthant_mixed_q26_tile_batch_avx2"], cwd=ROOT, check=True)


def reference_volume(dimension: int) -> float:
    return math.pi ** (0.5 * dimension) / math.gamma(0.5 * dimension + 1.0)


def q_pattern(counts: tuple[int, int, int, int, int]) -> str:
    return " ".join(f"{q}^{count}" for q, count in zip(Q_VALUES, counts))


def q345_main_counts(dimension: int) -> tuple[int, int, int, int, int]:
    k = round(dimension / 3.0)
    return (0, k, dimension - 2 * k, k, 0)


def point_count(counts: tuple[int, int, int, int, int]) -> int:
    total = 1
    for q, count in zip(Q_VALUES, counts):
        total *= q**count
    return total


def roughness(counts: tuple[int, int, int, int, int]) -> float:
    return sum(count / float(q * q) for q, count in zip(Q_VALUES, counts))


def axis_unit_squares(q: int) -> list[int]:
    scale = COMMON_LCM_Q // q
    return [((2 * digit + 1) * scale) ** 2 for digit in range(q)]


def coefficient_inside_points(counts: tuple[int, int, int, int, int]) -> int:
    distribution = [0] * (THRESHOLD_UNITS + 1)
    distribution[0] = 1
    for q, axis_count in zip(Q_VALUES, counts):
        values = axis_unit_squares(q)
        for _ in range(axis_count):
            next_distribution = [0] * (THRESHOLD_UNITS + 1)
            for partial_sum, count in enumerate(distribution):
                if count == 0:
                    continue
                for value in values:
                    new_sum = partial_sum + value
                    if new_sum <= THRESHOLD_UNITS:
                        next_distribution[new_sum] += count
            distribution = next_distribution
    return sum(distribution)


def coefficient_estimate(counts: tuple[int, int, int, int, int], inside_points: int) -> float:
    dimension = sum(counts)
    return inside_points * (2.0**dimension) / float(point_count(counts))
