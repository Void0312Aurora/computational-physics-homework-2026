from __future__ import annotations

import csv
import math
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"
getcontext().prec = 80
PI_OVER_4_DECIMAL = Decimal(
    "0.785398163397448309615660845819875721049292349843776455243736148076954101571552"
)


ArrayFunc = Callable[[np.ndarray], np.ndarray]



def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def composite_simpson(f: ArrayFunc, a: float, b: float, n: int) -> float:
    if n % 2 != 0:
        raise ValueError("Composite Simpson's rule needs an even number of slices.")
    x = np.linspace(a, b, n + 1)
    y = np.asarray(f(x), dtype=float)
    h = (b - a) / n
    return float(h / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])))


def problem4(log: list[str]) -> None:
    n = 1024
    cases = [
        {
            "case": "a",
            "integral": "int_-1^1 sqrt(1-x^2) dx",
            "transformation": "x = sin(theta), theta in [-pi/2, pi/2]",
            "value": composite_simpson(lambda theta: np.cos(theta) ** 2, -math.pi / 2.0, math.pi / 2.0, n),
            "exact": math.pi / 2.0,
        },
        {
            "case": "b",
            "integral": "int_0^pi sin^2(theta) dtheta",
            "transformation": "direct Simpson integration",
            "value": composite_simpson(lambda theta: np.sin(theta) ** 2, 0.0, math.pi, n),
            "exact": math.pi / 2.0,
        },
        {
            "case": "c",
            "integral": "int_0^inf dx / ((1+x) sqrt(x))",
            "transformation": "x = tan^2(theta), theta in [0, pi/2]",
            "value": composite_simpson(lambda theta: np.full_like(theta, 2.0), 0.0, math.pi / 2.0, n),
            "exact": math.pi,
        },
    ]
    rows = []
    for case in cases:
        rows.append(
            {
                "case": case["case"],
                "integral": case["integral"],
                "transformation": case["transformation"],
                "slices": n,
                "estimate": f"{float(case['value']):.16f}",
                "exact": f"{float(case['exact']):.16f}",
                "actual_error": f"{abs(float(case['value']) - float(case['exact'])):.6e}",
            }
        )
    write_csv(
        RESULT_DIR / "problem4_integrals.csv",
        rows,
        ["case", "integral", "transformation", "slices", "estimate", "exact", "actual_error"],
    )

    log.append("Problem 4: transformed integrals")
    for row in rows:
        log.append(
            f"  ({row['case']}) estimate={row['estimate']}, exact={row['exact']}, "
            f"error={row['actual_error']}"
        )
    log.append("")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    problem4(log)
    print("\n".join(log))


if __name__ == "__main__":
    main()
