from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "result"
SEED = 20260603
PROBLEM_SEEDS = {
    "problem1": SEED + 101,
    "problem2": SEED + 202,
    "problem4": SEED + 404,
    "problem5": SEED + 505,
    "rng_diagnostics": SEED + 606,
}

CONFIG = {
    "seed": SEED,
    "problem_seeds": PROBLEM_SEEDS,
    "problem1": {
        "n_steps": 1000,
        "fixed_paths": [50, 500, 5000],
        "variable_paths": 500,
        "quasi_paths": [128, 512, 2048, 8192],
    },
    "problem2": {
        "fjc_N": [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
        "frc_N": [10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
        "frc_theta_deg": 68.0,
        "fjc_long_chain_samples": 4000,
    },
    "problem3": {
        "N_values": [16, 32, 64, 128, 256, 512, 1024],
        "biased_p": 0.7,
    },
    "polymer_fit": {
        "primary_min_N": 64,
        "sensitivity_min_N": [32, 64, 128, 256],
        "bootstrap_replicates": 800,
    },
    "rng_diagnostics": {
        "generators": ["numpy_pcg64", "xoshiro256ss", "park_miller"],
        "uniform_n": 200000,
        "hist_bins": 20,
        "integer_n": 200000,
        "integer_low": 0,
        "integer_high": 6,
        "hw11_integral_samples": [1000, 10000, 100000],
        "hw11_integral_repeats": 50,
    },
}
