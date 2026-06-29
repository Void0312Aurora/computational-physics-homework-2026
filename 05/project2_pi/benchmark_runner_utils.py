from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from . import homework_bridge

T = TypeVar("T")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CpuRouteBenchmark:
    route: str
    group: str
    target_digits: int
    seconds: float
    digits_per_second: float
    prefix_ok: bool
    status: str
    notes: str


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def median_iqr(values: list[float]) -> tuple[float, float]:
    return percentile(values, 0.5), percentile(values, 0.75) - percentile(values, 0.25)


def format_samples(values: list[float], precision: int) -> str:
    return ";".join(f"{value:.{precision}f}" for value in values)


def iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_order_strategy(*, shuffle: bool, grouped: bool) -> str:
    if shuffle:
        return "shuffle"
    if grouped:
        return "grouped"
    return "interleave"


def order_items(items: list[T], *, strategy: str, seed: int) -> list[T]:
    ordered = list(items)
    if strategy == "shuffle":
        rng = random.Random(seed)
        rng.shuffle(ordered)
    return ordered


def write_metadata_json(path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def benchmark_cpp_gmp(
    digits: int,
    threads: int | None = None,
    chunk_terms: int | None = None,
    leaf_terms: int | None = None,
    task_terms: int | None = None,
    parallel_mode: str | None = None,
) -> CpuRouteBenchmark:
    binary = PROJECT_ROOT / "project2_pi" / "bin" / "project2_gmp_backend"
    auto_config = homework_bridge.resolve_project2_cpp_config(digits)
    if threads is None:
        threads = auto_config.threads
    if chunk_terms is None:
        chunk_terms = auto_config.chunk_terms
    if leaf_terms is None:
        leaf_terms = auto_config.leaf_terms
    if task_terms is None:
        task_terms = auto_config.task_terms
    if parallel_mode is None:
        parallel_mode = auto_config.parallel_mode
    parallel_mode = homework_bridge.normalize_project2_cpp_parallel_mode(parallel_mode)
    route = {
        "chunked": "cpp_gmp_openmp",
        "tasks": "cpp_gmp_openmp_tasks",
        "frontier": "cpp_gmp_openmp_frontier",
    }[parallel_mode]
    try:
        proc = _run_command(
            [
                str(binary),
                "--digits",
                str(digits),
                "--threads",
                str(threads),
                "--chunk-terms",
                str(chunk_terms),
                "--leaf-terms",
                str(leaf_terms),
                "--task-terms",
                str(task_terms),
                "--parallel-mode",
                parallel_mode,
            ],
            cwd=PROJECT_ROOT,
        )
    except OSError as exc:
        return CpuRouteBenchmark(
            route, "full_scale", digits, 0.0, 0.0, False, "failed", str(exc)
        )
    if proc.returncode != 0:
        return CpuRouteBenchmark(
            route=route,
            group="full_scale",
            target_digits=digits,
            seconds=0.0,
            digits_per_second=0.0,
            prefix_ok=False,
            status="failed",
            notes=proc.stderr.strip() or "cpp backend failed",
        )
    values = homework_bridge.parse_key_values(proc.stdout)
    prefix_ok = values.get("prefix_ok", "False") == "True"
    return CpuRouteBenchmark(
        route=route,
        group="full_scale",
        target_digits=digits,
        seconds=float(values["seconds"]),
        digits_per_second=float(values["digits_per_second"]),
        prefix_ok=prefix_ok,
        status="ok" if prefix_ok else "prefix_mismatch",
        notes=(
            f"threads={threads},chunk_terms={chunk_terms},leaf_terms={leaf_terms},"
            f"task_terms={task_terms},parallel_mode={parallel_mode}"
        ),
    )


def benchmark_cpp_levelpool(
    digits: int,
    threads: int | None = None,
    chunk_terms: int | None = None,
    leaf_terms: int | None = None,
) -> CpuRouteBenchmark:
    binary = homework_bridge.resolve_project2_cpp_levelpool_binary()
    if not binary.is_file():
        raise RuntimeError(
            "project2_gmp_levelpool_backend is unavailable; run `make cpp_levelpool_backend` first"
        )

    auto_config = homework_bridge.resolve_project2_cpp_config(digits)
    if threads is None:
        threads = auto_config.threads
    if chunk_terms is None:
        chunk_terms = auto_config.chunk_terms
    if leaf_terms is None:
        leaf_terms = auto_config.leaf_terms

    try:
        proc = _run_command(
            [
                str(binary),
                "--digits",
                str(digits),
                "--threads",
                str(threads),
                "--chunk-terms",
                str(chunk_terms),
                "--leaf-terms",
                str(leaf_terms),
            ],
            cwd=PROJECT_ROOT,
        )
    except OSError as exc:
        return CpuRouteBenchmark(
            "cpp_gmp_levelpool",
            "full_scale",
            digits,
            0.0,
            0.0,
            False,
            "failed",
            str(exc),
        )
    if proc.returncode != 0:
        return CpuRouteBenchmark(
            route="cpp_gmp_levelpool",
            group="full_scale",
            target_digits=digits,
            seconds=0.0,
            digits_per_second=0.0,
            prefix_ok=False,
            status="failed",
            notes=proc.stderr.strip() or "levelpool backend failed",
        )
    values = homework_bridge.parse_key_values(proc.stdout)
    prefix_ok = values.get("prefix_ok", "False") == "True"
    return CpuRouteBenchmark(
        route="cpp_gmp_levelpool",
        group="full_scale",
        target_digits=digits,
        seconds=float(values["seconds"]),
        digits_per_second=float(values["digits_per_second"]),
        prefix_ok=prefix_ok,
        status="ok" if prefix_ok else "prefix_mismatch",
        notes=f"threads={threads},chunk_terms={chunk_terms},leaf_terms={leaf_terms},representation=levelpool",
    )


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _summarize_ranges(values: list[int]) -> str:
    if not values:
        return ""
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _cpu_affinity() -> str:
    if not hasattr(os, "sched_getaffinity"):
        return "unavailable"
    try:
        return _summarize_ranges(sorted(os.sched_getaffinity(0)))
    except OSError:
        return "unavailable"


def _cpu_governor() -> str:
    paths = sorted(
        Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")
    )
    governors = sorted({value for path in paths if (value := _read_text(path))})
    return ",".join(governors) if governors else "unavailable"


def _cpu_frequency_mhz() -> str:
    value = _read_text(Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"))
    if value is None:
        return "unavailable"
    try:
        return f"{int(value) / 1000.0:.1f}"
    except ValueError:
        return value


def _nvidia_smi_summary() -> str:
    if shutil.which("nvidia-smi") is None:
        return "unavailable"
    query = "name,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total"
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    if proc.returncode != 0:
        return "unavailable"
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return " | ".join(lines) if lines else "unavailable"


def collect_system_context() -> dict[str, str]:
    try:
        loadavg = ",".join(f"{value:.2f}" for value in os.getloadavg())
    except OSError:
        loadavg = "unavailable"
    return {
        "loadavg": loadavg,
        "omp_proc_bind": os.environ.get("OMP_PROC_BIND", ""),
        "omp_places": os.environ.get("OMP_PLACES", ""),
        "process_cpu_affinity": _cpu_affinity(),
        "cpu_governor": _cpu_governor(),
        "cpu_frequency_mhz": _cpu_frequency_mhz(),
        "nvidia_smi": _nvidia_smi_summary(),
    }
