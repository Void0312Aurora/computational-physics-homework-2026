from __future__ import annotations

import csv
import itertools
import math
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def exact_ball_volume(n: int) -> float:
    return math.pi ** (0.5 * n) / math.gamma(0.5 * n + 1.0)


def cc_abscissa(order: int, i: int) -> float:
    if order == 1:
        return 0.0
    value = math.cos((order - i) * math.pi / (order - 1))
    if 2 * i - 1 == order:
        value = 0.0
    return value


def cc_weights(order: int) -> list[float]:
    if order == 1:
        return [2.0]

    weights = [0.0] * order
    for i in range(1, order + 1):
        theta = (i - 1) * math.pi / (order - 1)
        value = 1.0
        for j in range(1, (order - 1) // 2 + 1):
            factor = 1.0 if 2 * j == (order - 1) else 2.0
            value -= factor * math.cos(2.0 * j * theta) / (4 * j * j - 1)
        if i == 1 or i == order:
            value /= (order - 1)
        else:
            value = 2.0 * value / (order - 1)
        weights[i - 1] = value
    return weights


def weak_compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in weak_compositions(total - first, parts - 1):
            yield (first,) + rest


def level_to_order_closed(level_1d: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(1 if level == 0 else 2**level + 1 for level in level_1d)


def sparse_grid_cc_size(dim_num: int, level_max: int) -> int:
    if level_max < 0:
        return 0
    if level_max == 0:
        return 1

    new_1d = [0] * (level_max + 1)
    new_1d[0] = 1
    new_1d[1] = 2
    factor = 1
    for level in range(2, level_max + 1):
        factor *= 2
        new_1d[level] = factor

    total_points = 0
    for level in range(level_max + 1):
        for level_1d in weak_compositions(level, dim_num):
            product = 1
            for value in level_1d:
                product *= new_1d[value]
            total_points += product
    return total_points


def sparse_grid_cc_unit(dim_num: int, level_max: int) -> dict[tuple[int, ...], float]:
    order_max = 1 if level_max == 0 else 2**level_max + 1
    grid_weights: dict[tuple[int, ...], float] = {}
    level_min = max(0, level_max + 1 - dim_num)

    for level in range(level_min, level_max + 1):
        coeff = ((-1) ** (level_max - level)) * math.comb(dim_num - 1, level_max - level)
        for level_1d in weak_compositions(level, dim_num):
            order_1d = level_to_order_closed(level_1d)
            one_d_weights = [tuple(0.5 * weight for weight in cc_weights(order)) for order in order_1d]
            one_d_indices = [range(order) for order in order_1d]

            for local_index in itertools.product(*one_d_indices):
                scaled_index = []
                local_weight = float(coeff)
                for dim, idx in enumerate(local_index):
                    local_weight *= one_d_weights[dim][idx]
                    level_dim = level_1d[dim]
                    if level_dim == 0:
                        scaled_index.append((order_max - 1) // 2)
                    else:
                        scaled_index.append(idx * (2 ** (level_max - level_dim)))
                key = tuple(scaled_index)
                grid_weights[key] = grid_weights.get(key, 0.0) + local_weight

    return {key: value for key, value in grid_weights.items() if abs(value) > 1e-14}


def index_to_unit_coordinate(level_max: int, index: int) -> float:
    order_max = 1 if level_max == 0 else 2**level_max + 1
    return 0.5 * (1.0 + cc_abscissa(order_max, index + 1))


def smootherstep(z: np.ndarray) -> np.ndarray:
    x = 0.5 * (z + 1.0)
    return ((6.0 * x - 15.0) * x + 10.0) * x * x * x


def quintic_indicator(r2: np.ndarray, delta: float) -> np.ndarray:
    z = (r2 - 1.0) / delta
    out = np.empty_like(z)
    left = z <= -1.0
    right = z >= 1.0
    mid = (~left) & (~right)
    out[left] = 1.0
    out[right] = 0.0
    if np.any(mid):
        out[mid] = 1.0 - smootherstep(z[mid])
    return out


def tanh_indicator(r2: np.ndarray, delta: float) -> np.ndarray:
    return 0.5 * (1.0 - np.tanh((r2 - 1.0) / delta))


def sparse_grid_r2_weights(dim_num: int, level_max: int) -> tuple[np.ndarray, np.ndarray]:
    grid_weights = sparse_grid_cc_unit(dim_num, level_max)
    r2 = np.empty(len(grid_weights), dtype=float)
    weights = np.empty(len(grid_weights), dtype=float)
    for k, (index_tuple, weight) in enumerate(grid_weights.items()):
        point = [index_to_unit_coordinate(level_max, index) for index in index_tuple]
        r2[k] = sum(x * x for x in point)
        weights[k] = weight
    return r2, weights


def integrate_smoothed_ball(dim_num: int, level_max: int, family: str, delta: float) -> tuple[float, float, int]:
    r2, weights = sparse_grid_r2_weights(dim_num, level_max)
    if family == "quintic":
        values = quintic_indicator(r2, delta)
    elif family == "tanh":
        values = tanh_indicator(r2, delta)
    else:
        raise ValueError(f"Unknown smoothing family: {family}")
    orthant_estimate = float(np.dot(weights, values))
    weight_sum = float(np.sum(weights))
    return orthant_estimate * (2.0**dim_num), weight_sum, len(weights)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    cases = [
        {"dimension": 5, "level_max": 4},
        {"dimension": 5, "level_max": 5},
        {"dimension": 8, "level_max": 6},
        {"dimension": 10, "level_max": 6},
        {"dimension": 12, "level_max": 5},
        {"dimension": 15, "level_max": 5},
        {"dimension": 20, "level_max": 4},
    ]
    deltas = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    families = ["quintic", "tanh"]

    rows: list[dict[str, object]] = []
    log_lines = [
        "Problem 5 sparse-grid smoothed-indicator experiment",
        "===================================================",
        "Method: Smolyak sparse grid with nested Clenshaw-Curtis rules on [0,1]^n,",
        "then multiply by 2^n. The hard indicator is replaced by a smooth transition",
        "across the sphere boundary, tested with quintic and tanh profiles.",
        "",
    ]

    for case in cases:
        dim_num = case["dimension"]
        level_max = case["level_max"]
        predicted_points = sparse_grid_cc_size(dim_num, level_max)
        exact = exact_ball_volume(dim_num)
        log_lines.append(
            f"dimension n={dim_num}, level={level_max}, predicted_points={predicted_points}, exact={exact:.15e}"
        )
        best_row: dict[str, object] | None = None

        for family in families:
            for delta in deltas:
                start = time.time()
                estimate, weight_sum, unique_points = integrate_smoothed_ball(dim_num, level_max, family, delta)
                elapsed = time.time() - start
                rel_error = abs(estimate - exact) / exact
                weight_error = abs(weight_sum - 1.0)
                row = {
                    "dimension": dim_num,
                    "level_max": level_max,
                    "family": family,
                    "delta": delta,
                    "predicted_points": predicted_points,
                    "unique_points": unique_points,
                    "estimate": f"{estimate:.16e}",
                    "exact": f"{exact:.16e}",
                    "relative_error": f"{rel_error:.6e}",
                    "weight_sum_error": f"{weight_error:.6e}",
                    "elapsed_s": f"{elapsed:.3f}",
                }
                rows.append(row)
                log_lines.append(
                    "  "
                    f"{family:7s}, delta={delta:7.4f}, unique_points={unique_points}, "
                    f"estimate={estimate:.12e}, rel={rel_error:.3e}, "
                    f"weight_sum_error={weight_error:.3e}, time={elapsed:.2f}s"
                )
                if best_row is None or float(row["relative_error"]) < float(best_row["relative_error"]):
                    best_row = row

        if best_row is not None:
            log_lines.append(
                "  "
                f"best: family={best_row['family']}, delta={float(best_row['delta']):.4f}, "
                f"estimate={best_row['estimate']}, rel={best_row['relative_error']}"
            )
        log_lines.append("")

    write_csv(
        RESULT_DIR / "problem5_sparse_grid_smoothed_experiment.csv",
        rows,
        [
            "dimension",
            "level_max",
            "family",
            "delta",
            "predicted_points",
            "unique_points",
            "estimate",
            "exact",
            "relative_error",
            "weight_sum_error",
            "elapsed_s",
        ],
    )
    (RESULT_DIR / "problem5_sparse_grid_smoothed_experiment.log").write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
