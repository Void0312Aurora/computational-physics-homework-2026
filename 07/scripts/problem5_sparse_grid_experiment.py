from __future__ import annotations

import csv
import itertools
import math
import time
from pathlib import Path


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


def integrate_unit_ball_sparse_grid(dim_num: int, level_max: int) -> tuple[float, float, int]:
    grid_weights = sparse_grid_cc_unit(dim_num, level_max)
    orthant_estimate = 0.0
    weight_sum = 0.0
    for index_tuple, weight in grid_weights.items():
        point = [index_to_unit_coordinate(level_max, index) for index in index_tuple]
        weight_sum += weight
        if sum(x * x for x in point) <= 1.0 + 1e-15:
            orthant_estimate += weight
    return orthant_estimate * (2.0**dim_num), weight_sum, len(grid_weights)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    dims = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    max_points = 300_000
    rows: list[dict[str, object]] = []
    log_lines = [
        "Problem 5 sparse-grid experiment",
        "================================",
        "Method: Smolyak sparse grid with nested Clenshaw-Curtis rules on [0,1]^n, then multiply by 2^n.",
        "This is an experimental alternative to cubature, kept separate from the main homework script.",
        "",
    ]

    for dim_num in dims:
        exact = exact_ball_volume(dim_num)
        log_lines.append(f"dimension n={dim_num}, exact={exact:.15e}")
        for level_max in range(0, 8):
            predicted_points = sparse_grid_cc_size(dim_num, level_max)
            if predicted_points > max_points:
                log_lines.append(
                    f"  level={level_max}: skipped because predicted sparse-grid size {predicted_points} exceeds {max_points}"
                )
                break

            start = time.time()
            estimate, weight_sum, unique_points = integrate_unit_ball_sparse_grid(dim_num, level_max)
            elapsed = time.time() - start
            rel_error = abs(estimate - exact) / exact
            weight_error = abs(weight_sum - 1.0)

            rows.append(
                {
                    "dimension": dim_num,
                    "level_max": level_max,
                    "predicted_points": predicted_points,
                    "unique_points": unique_points,
                    "estimate": f"{estimate:.16e}",
                    "exact": f"{exact:.16e}",
                    "relative_error": f"{rel_error:.6e}",
                    "weight_sum_error": f"{weight_error:.6e}",
                    "elapsed_s": f"{elapsed:.3f}",
                }
            )
            log_lines.append(
                "  "
                f"level={level_max}, predicted_points={predicted_points}, unique_points={unique_points}, "
                f"estimate={estimate:.12e}, rel={rel_error:.3e}, weight_sum_error={weight_error:.3e}, "
                f"time={elapsed:.2f}s"
            )
        log_lines.append("")

    write_csv(
        RESULT_DIR / "problem5_sparse_grid_experiment.csv",
        rows,
        [
            "dimension",
            "level_max",
            "predicted_points",
            "unique_points",
            "estimate",
            "exact",
            "relative_error",
            "weight_sum_error",
            "elapsed_s",
        ],
    )
    (RESULT_DIR / "problem5_sparse_grid_experiment.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
