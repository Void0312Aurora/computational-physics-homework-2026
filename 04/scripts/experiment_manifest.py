from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
RESULT_DIR = ROOT / "result"
ANALYSIS_DIR = RESULT_DIR / "analysis"
MANIFEST_PATH = ANALYSIS_DIR / "experiment_manifest.json"
METADATA_PATH = ANALYSIS_DIR / "metadata.json"
REPORT_MANIFEST_PATH = ANALYSIS_DIR / "report_manifest.json"

ARTIFACT_PATTERNS = [
    "result/problem1_cpu_ultra_summary.csv",
    "result/problem1_cpu_ultra_fractals.png",
    "result/analysis/problem1_cpu_profile*.csv",
    "result/analysis/problem1_cpu_profile.png",
    "result/analysis/problem1_correctness_gate*.csv",
    "result/analysis/problem1_correctness_gate*.json",
    "result/analysis/problem1_cpu_ultra_*.csv",
    "result/analysis/problem1_scaling_frontier.csv",
    "result/analysis/problem1_scaling_frontier.png",
    "result/analysis/problem1_precision_frontier.csv",
    "result/analysis/problem1_precision_frontier.png",
    "result/analysis/problem1_frontier_curves__*/problem1_scaling_frontier.csv",
    "result/analysis/problem1_frontier_curves__*/problem1_scaling_frontier.png",
    "result/analysis/problem1_frontier_curves__*/problem1_precision_frontier.csv",
    "result/analysis/problem1_frontier_curves__*/problem1_precision_frontier.png",
    "result/analysis/problem1_gpu_triton_extreme_stats*/extreme_stats_summary.csv",
    "result/analysis/sparse_reference_validation_*/*.csv",
    "result/gpu_pytorch/*/*_summary.csv",
    "result/gpu_pytorch/*/*_resources.csv",
    "result/gpu_triton/*/*_summary.csv",
    "result/gpu_triton/*/*_resources.csv",
    "result/gpu_triton/triton_newton_bench*/*.csv",
]

REPORT_ARTIFACT_PATHS = [
    "docs/answer/answer.md",
    "docs/answer/answer.pdf",
    "docs/answer/answer.docx",
    "docs/answer/latex/answer.tex",
    "docs/answer/latex/answer.pdf",
    "result/problem1_fractals.png",
    "result/problem1_summary.csv",
    "result/problem1_resources.csv",
    "result/analysis/problem1_cpu_profile.csv",
    "result/analysis/problem1_cpu_profile_raw.csv",
    "result/analysis/problem1_cpu_profile.png",
    "result/problem1_cpu_ultra_fractals.png",
    "result/problem1_cpu_ultra_summary.csv",
    "result/analysis/problem1_cpu_ultra_bench_summary.csv",
    "result/analysis/problem1_cpu_ultra_bench_samples.csv",
    "result/analysis/problem1_cpu_ultra_bench_tile256_summary.csv",
    "result/analysis/problem1_cpu_ultra_bench_tile256_samples.csv",
    "result/analysis/tune_t88_r256.csv",
    "result/analysis/problem1_scaling_frontier.csv",
    "result/analysis/problem1_scaling_frontier.png",
    "result/analysis/problem1_precision_frontier.csv",
    "result/analysis/problem1_precision_frontier.png",
    "result/gpu_pytorch/problem1_gpu_100k_newton/problem1_gpu_100k_newton_summary.csv",
    "result/gpu_pytorch/problem1_gpu_100k_newton/problem1_gpu_100k_newton_resources.csv",
    "result/gpu_triton/problem1_gpu_newton_triton_100k_rt625/problem1_gpu_newton_triton_100k_rt625_summary.csv",
    "result/gpu_triton/problem1_gpu_newton_triton_100k_rt625/problem1_gpu_newton_triton_100k_rt625_resources.csv",
    "result/gpu_triton/problem1_gpu_newton_triton_stats_1m__20260408-195452/problem1_gpu_newton_triton_stats_1m_summary.csv",
    "result/gpu_triton/problem1_gpu_newton_triton_stats_1m__20260408-195452/problem1_gpu_newton_triton_stats_1m_resources.csv",
    "result/gpu_triton/problem1_gpu_newton_triton_stats_2m__20260408-195815/problem1_gpu_newton_triton_stats_2m_summary.csv",
    "result/gpu_triton/problem1_gpu_newton_triton_stats_2m__20260408-195815/problem1_gpu_newton_triton_stats_2m_resources.csv",
    "result/problem2_roots.csv",
    "result/problem3_roots.csv",
    "result/problem4_shengjin_roots.csv",
    "result/problem5_l1.csv",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            return list(reader.fieldnames or []), rows
    except UnicodeDecodeError:
        return [], []


def unique_values(rows: list[dict[str, str]], key: str) -> list[str]:
    values = sorted({row[key] for row in rows if row.get(key) not in (None, "")})
    return values


def numeric_values(rows: list[dict[str, str]], keys: list[str]) -> list[float]:
    values: list[float] = []
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value in (None, ""):
                continue
            try:
                values.append(float(value))
            except ValueError:
                pass
    return values


def is_large_scale(path: Path, rows: list[dict[str, str]]) -> bool:
    text = rel(path).lower()
    if re.search(
        r"(^|[_/-])(40k|80k|100k|250k|500k|1m|2m|80000|100000|250000|500000|1000000|2000000)([_./-]|$)",
        text,
    ):
        return True
    values = numeric_values(rows, ["compute_grid", "grid_size", "global_grid"])
    return bool(values and max(values) >= 40_000)


def source_command_for(path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    recorded = unique_values(rows, "source_command")
    if recorded:
        return {"source_command": recorded, "source_command_inferred": False}

    path_text = rel(path)
    name = path.name
    parent = path.parent.name

    if path_text.startswith("docs/answer/"):
        if path.suffix == ".tex":
            return {
                "source_command": "manual LaTeX snapshot from docs/answer/answer.md; verify with make check-source",
                "source_command_inferred": True,
            }
        if path_text.startswith("docs/answer/latex/"):
            return {
                "source_command": "make -C docs/answer/latex",
                "source_command_inferred": True,
            }
        return {"source_command": "make docs", "source_command_inferred": True}
    if path_text in {
        "result/problem1_fractals.png",
        "result/problem1_summary.csv",
        "result/problem1_resources.csv",
        "result/problem2_roots.csv",
        "result/problem3_roots.csv",
        "result/problem4_shengjin_roots.csv",
        "result/problem5_l1.csv",
    }:
        return {"source_command": "make run", "source_command_inferred": True}
    if "problem1_cpu_ultra_bench" in name:
        return {
            "source_command": 'make cpu-ultra-bench CPU_ULTRA_BENCH_ARGS="--compute-grids 20000,40000,80000 --render-grid 5000 --tile-rows 64 --threads 88 --repeats 3 --warmup-runs 1 --timing-scope process_full_run --output-prefix problem1_cpu_ultra_bench"',
            "source_command_inferred": True,
        }
    if "problem1_cpu_ultra_render" in name or path_text.endswith(
        "problem1_cpu_ultra_summary.csv"
    ):
        return {
            "source_command": 'make cpu-ultra-render CPU_ULTRA_RENDER_ARGS="--compute-grid 80000 --render-grid 5000 --tile-rows 256 --threads 88 --output-prefix problem1_cpu_ultra_render_80k"',
            "source_command_inferred": True,
        }
    if "problem1_cpu_profile" in name:
        return {
            "source_command": 'make cpu-profile CPU_PROFILE_ARGS="--worker-grid 4096 --worker-render-grid 1024 --worker-counts 1,8,16,32,64 --grid-sizes 1024,2048,4096,8192 --grid-workers 64 --repeats 3 --warmup-runs 1 --timing-scope full_compute_plus_resource_monitor --output-prefix problem1_cpu_profile"',
            "source_command_inferred": True,
        }
    if name.startswith("tune_t"):
        return {
            "source_command": "CPU ultra tile-row tuning sweep; exact parameters are encoded in the CSV row",
            "source_command_inferred": True,
        }
    if "problem1_correctness_gate" in name:
        return {
            "source_command": 'make correctness-gate CORRECTNESS_ARGS="--compute-grid 128 --render-grid 32 --tile-rows 8 --threads 2 --reference-workers 2 --output-prefix problem1_correctness_gate"',
            "source_command_inferred": True,
        }
    if "problem1_scaling_frontier" in name or "problem1_precision_frontier" in name:
        return {
            "source_command": "make frontier-curves",
            "source_command_inferred": True,
        }
    if "sparse_reference_validation" in path_text:
        return {
            "source_command": 'make sparse-validate SPARSE_VALIDATE_ARGS="..."',
            "source_command_inferred": True,
        }
    if "triton_newton_bench" in path_text:
        return {
            "source_command": f'make triton-bench TRITON_ARGS="--output-prefix {parent}"',
            "source_command_inferred": True,
        }
    if path_text.startswith("result/gpu_pytorch/") or path_text.startswith(
        "result/gpu_triton/"
    ):
        prefix = parent.split("__", 1)[0]
        return {
            "source_command": f'make gpu-ultra GPU_ARGS="... --output-prefix {prefix}"',
            "source_command_inferred": True,
        }
    if "doublefloat_newton_prototype" in path_text:
        return {
            "source_command": 'make doublefloat-proto DOUBLEFLOAT_ARGS="..."',
            "source_command_inferred": True,
        }
    if "triton_doublefloat_prototype" in path_text:
        return {
            "source_command": 'make triton-doublefloat-proto TRITON_DOUBLEFLOAT_ARGS="..."',
            "source_command_inferred": True,
        }
    return {"source_command": None, "source_command_inferred": False}


def artifact_kind(path: Path) -> str:
    if path.suffix == ".csv":
        return "csv"
    if path.suffix == ".json":
        return "json"
    if path.suffix == ".png":
        return "figure"
    return path.suffix.lstrip(".") or "file"


def artifact_record(path: Path) -> dict[str, Any]:
    columns: list[str] = []
    rows: list[dict[str, str]] = []
    if path.suffix == ".csv":
        columns, rows = read_csv_rows(path)

    stat = path.stat()
    record: dict[str, Any] = {
        "path": rel(path),
        "kind": artifact_kind(path),
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "sha256": sha256_file(path),
        "is_large_scale": is_large_scale(path, rows),
    }
    record.update(source_command_for(path, rows))

    source_csv = unique_values(rows, "source_csv")
    if source_csv:
        record["source_csv"] = source_csv
    elif path.suffix == ".png":
        sibling_csv = path.with_suffix(".csv")
        if sibling_csv.exists():
            record["source_csv"] = rel(sibling_csv)

    if path.suffix == ".csv":
        record["columns"] = columns
        record["row_count"] = len(rows)
        for key in (
            "compute_grid",
            "grid_size",
            "global_grid",
            "render_grid",
            "tile_rows",
            "threads",
            "render_tile",
            "compute_tile",
            "warmup_runs",
            "timed_repeats",
            "timing_scope",
        ):
            values = unique_values(rows, key)
            if values:
                record[f"{key}_values"] = values
    return record


def collect_artifacts() -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for pattern in ARTIFACT_PATTERNS:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    return [artifact_record(path) for path in sorted(paths, key=rel)]


def collect_report_artifacts() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_path in REPORT_ARTIFACT_PATHS:
        path = ROOT / raw_path
        if path.is_file():
            records.append(artifact_record(path))
    return records


def first_line(path: Path, prefix: str) -> str | None:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.startswith(prefix):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def mem_total_kib() -> int | None:
    value = first_line(Path("/proc/meminfo"), "MemTotal")
    if not value:
        return None
    parts = value.split()
    try:
        return int(parts[0])
    except (IndexError, ValueError):
        return None


def command_first_line(args: list[str], timeout: float = 3.0) -> str | None:
    if shutil.which(args[0]) is None:
        return None
    try:
        proc = subprocess.run(
            args, check=False, capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (proc.stdout or proc.stderr).strip()
    if not text:
        return None
    return text.splitlines()[0].strip()


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def nvidia_smi() -> dict[str, Any] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout).strip()}
    devices = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            devices.append(
                {"name": parts[0], "driver_version": parts[1], "memory_total": parts[2]}
            )
    return {"devices": devices}


def write_metadata() -> dict[str, Any]:
    packages = {
        name: package_version(name)
        for name in ("numpy", "matplotlib", "mpmath", "psutil", "torch", "triton")
    }
    env_keys = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "PYTHONPATH",
    ]
    metadata: dict[str, Any] = {
        "generated_at": utc_now(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu": {
            "model": first_line(Path("/proc/cpuinfo"), "model name"),
            "logical_count": os.cpu_count(),
            "affinity_count": (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
        },
        "memory": {
            "mem_total_kib": mem_total_kib(),
        },
        "packages": packages,
        "environment": {key: os.environ.get(key) for key in env_keys},
        "tools": {
            "gcc": command_first_line(["gcc", "--version"]),
            "g++": command_first_line(["g++", "--version"]),
            "make": command_first_line(["make", "--version"]),
        },
        "gpu": nvidia_smi(),
        "build_notes": {
            "cpu_ultra_compile_command": "gcc -O3 -march=native -ffast-math -fopenmp -o problem1_cpu_ultra scripts/problem1_cpu_ultra.c -lm",
            "large_scale_rerun_performed": False,
        },
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    metadata = write_metadata()
    artifacts = collect_artifacts()
    report_artifacts = collect_report_artifacts()
    report_manifest = {
        "generated_at": utc_now(),
        "root": str(ROOT),
        "report_source": "docs/answer/answer.md",
        "artifact_count": len(report_artifacts),
        "artifacts": report_artifacts,
        "purpose": "direct report dependency list for HW/04 answer content and conclusion evidence",
    }
    REPORT_MANIFEST_PATH.write_text(
        json.dumps(report_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "generated_at": utc_now(),
        "root": str(ROOT),
        "metadata_path": rel(METADATA_PATH),
        "report_manifest_path": rel(REPORT_MANIFEST_PATH),
        "artifact_count": len(artifacts),
        "large_scale_artifact_count": sum(
            1 for artifact in artifacts if artifact["is_large_scale"]
        ),
        "large_scale_rerun_performed": False,
        "artifacts": artifacts,
        "metadata_digest": hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"metadata={rel(METADATA_PATH)}")
    print(f"manifest={rel(MANIFEST_PATH)}")
    print(f"report_manifest={rel(REPORT_MANIFEST_PATH)}")
    print(f"artifacts={len(artifacts)}")
    print(f"report_artifacts={len(report_artifacts)}")
    print(f"large_scale_artifacts={manifest['large_scale_artifact_count']}")


if __name__ == "__main__":
    main()
