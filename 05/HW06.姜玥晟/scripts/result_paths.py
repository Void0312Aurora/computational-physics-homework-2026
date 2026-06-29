from __future__ import annotations

import csv
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR

for search_path in (SCRIPT_DIR, PROJECT_ROOT):
    search_text = str(search_path)
    if search_text not in sys.path:
        sys.path.insert(0, search_text)

default_result_name = "results" if PROJECT_ROOT.name.startswith("HW") else "result"
RESULT_DIR = Path(os.environ.get("HW05_RESULT_DIR", PROJECT_ROOT / default_result_name)).resolve()
USE_GROUPED_RESULTS = PROJECT_ROOT.name.startswith("HW")
PROBLEM1_RESULT_DIR = RESULT_DIR / "problem1" if USE_GROUPED_RESULTS else RESULT_DIR
PROBLEM2_RESULT_DIR = RESULT_DIR / "problem2" if USE_GROUPED_RESULTS else RESULT_DIR
PROBLEM3_RESULT_DIR = RESULT_DIR / "problem3" if USE_GROUPED_RESULTS else RESULT_DIR
PROJECT2_RESULT_DIR = RESULT_DIR / "project2" if USE_GROUPED_RESULTS else RESULT_DIR
PROJECT2_OUTPUT_DIR = PROJECT2_RESULT_DIR / "output" if USE_GROUPED_RESULTS else RESULT_DIR
PROJECT2_MANIFEST_DIR = PROJECT2_RESULT_DIR / "manifest" if USE_GROUPED_RESULTS else RESULT_DIR
PROJECT2_BENCHMARK_DIR = PROJECT2_RESULT_DIR / "benchmarks" if USE_GROUPED_RESULTS else RESULT_DIR
PROJECT2_CPU_BENCHMARK_DIR = PROJECT2_BENCHMARK_DIR / "cpu" if USE_GROUPED_RESULTS else RESULT_DIR
PROJECT2_REFERENCE_DIR = PROJECT2_RESULT_DIR / "references" if USE_GROUPED_RESULTS else RESULT_DIR
PROJECT2_YCRUNCHER_DIR = PROJECT2_REFERENCE_DIR / "ycruncher" if USE_GROUPED_RESULTS else RESULT_DIR
RUN_RESULT_DIR = RESULT_DIR / "run" if USE_GROUPED_RESULTS else RESULT_DIR


def ensure_result_dir(*paths: Path) -> None:
    targets = paths or [
        PROBLEM1_RESULT_DIR,
        PROBLEM2_RESULT_DIR,
        PROBLEM3_RESULT_DIR,
        PROJECT2_OUTPUT_DIR,
        PROJECT2_MANIFEST_DIR,
        PROJECT2_CPU_BENCHMARK_DIR,
        PROJECT2_YCRUNCHER_DIR,
        RUN_RESULT_DIR,
    ]
    for path in targets:
        path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
