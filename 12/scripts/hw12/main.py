from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np

from .config import CONFIG, PROBLEM_SEEDS, RESULT, SEED
from .io_utils import ensure_dirs
from .problem1 import problem1
from .problem2 import problem2
from .problem3 import problem3
from .problem4 import problem4
from .problem5 import problem5
from .rng_diagnostics import rng_diagnostics


def main() -> None:
    ensure_dirs()
    log: list[str] = [
        "HW/12 random-walk workflow run",
        f"Seed: {SEED}",
        f"Problem seeds: {PROBLEM_SEEDS}",
        "Pseudo-random default: NumPy default_rng / PCG64 family.",
        "Custom PRNG comparison: xoshiro256** and Park-Miller LCG.",
    ]
    with (RESULT / "hw12_config.json").open("w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)
    summary = {
        "seed": SEED,
        "config": CONFIG,
        "rng_note": {
            "default": "NumPy np.random.default_rng, PCG64-family bit generator",
            "custom_compared": [
                "xoshiro256** with SplitMix64 seeding, unbiased bounded integers, and jump substreams",
                "Park-Miller LCG from HW11 style, with rejection-sampled bounded integers",
            ],
            "quasi_random": "SciPy scrambled Sobol for whole-path direction variables",
        },
        "problem1": problem1(np.random.default_rng(PROBLEM_SEEDS["problem1"]), log),
        "problem2": problem2(np.random.default_rng(PROBLEM_SEEDS["problem2"]), log),
        "problem3": problem3(log),
        "problem4": problem4(np.random.default_rng(PROBLEM_SEEDS["problem4"]), log),
        "problem5": problem5(np.random.default_rng(PROBLEM_SEEDS["problem5"]), log),
        "rng_diagnostics": rng_diagnostics(PROBLEM_SEEDS["rng_diagnostics"], log),
    }
    summary_path = RESULT / "hw12_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.append(f"Wrote {summary_path.relative_to(RESULT.parent)}")
    log.append("All computations completed successfully.")
    (RESULT / "temp-01.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
