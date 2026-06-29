# HW/04

## Overview

This directory contains a Python implementation for all five problems from `docs/problem`.

- Problem 1: Newton and secant fractals for `z^3 - 1 = 0`
- Problem 2: safeguarded Newton-Bisection for `4 cos(x) - e^x`
- Problem 3: Brent-style Muller-Brent root finding for the trigonometric-sqrt equations
- Problem 4: Shengjin-style closed-form solution for the cubic polynomial
- Problem 5: Newton and secant solution of the Earth-Moon `L1` equation

For Problem 1, the current default build uses an aggressive parallel fractal configuration:

- compute grid: `20000 x 20000`
- render grid: `5000 x 5000`
- tolerance: `5e-7`
- iteration cap: `55` for Newton, `65` for secant
- worker count: `80`
- output figure: `15 x 14` inches at `360 dpi`

An additional GPU implementation is available in `gpu_ultra_fractal.py`.

- It uses `PyTorch CUDA` plus tiled rendering.
- It keeps GPU memory bounded by tile size instead of whole-image size.
- It is intended for higher compute grids such as `40000 x 40000`, `80000 x 80000`, or above.
- For `Newton`, it can exploit conjugate symmetry and usually delivers an almost `2x` speedup.
- It now stops each tile early once every point in that tile has converged, which cuts a large amount of wasted iteration work.
- It also provides an experimental `--newton-direct-formula` mode that rewrites the Newton step as `z_{n+1} = (2/3)z + 1/(3z^2)` and yielded about `17%` extra speedup on the tested `40k` and `100k` runs, with only tiny boundary-level statistical drift.
- It also includes an experimental active-point compression mode, but this is currently disabled by default because indexed sparse updates are slower than dense tensor math on this GPU.
- `secant` keeps the full plane, but now supports automatic tile fallback on CUDA OOM.
- `Newton` now also has an integrated experimental `--newton-backend triton` path.
- On larger runs this Triton backend is dramatically faster than the PyTorch path while preserving pixel-identical output in the tested `40k` and `100k` renders.
- On very small runs with tiny tiles it can still lose to PyTorch because the kernel-launch overhead dominates.

There is also a separate Triton benchmark in `triton_newton_bench.py`.

- It benchmarks a fused Newton direct-formula kernel against the current PyTorch dense path.
- In the isolated-tile tests it delivered very large speedups over the dense PyTorch reference.
- The corresponding kernel has now been integrated into the main Newton renderer as an opt-in backend.

There is also a prototype double-float experiment in `doublefloat_newton_prototype.py`.

- It represents each coordinate as a `hi + lo` pair of `float32` values.
- It is currently an analysis prototype for Newton only, not a production rendering backend.
- Its main purpose is to compare `fp32`, `double-float`, and `fp64` when coordinate spacing becomes the limiting factor.
- In the current smoke and `512 x 512` prototype runs it preserved the zoom-case coordinate uniqueness of `fp64`, while costing roughly `6x` the `fp32` runtime and still staying well below the measured Triton `fp64` slowdown.
- It now supports `--arith-mode basic|refined`; in the current `1024 x 1024` comparison the refined arithmetic did not improve the measured classification metrics, so `basic` remains the more practical default.

There is also a Triton version of the double-float prototype in `triton_doublefloat_prototype.py`.

- It benchmarks `fp32 Triton`, `fp64 PyTorch`, `double-float PyTorch`, and `double-float Triton` on the same full-plane and boundary-zoom cases.
- The current kernel uses a non-unrolled Triton loop; fully static unrolling made compilation impractically slow for the heavier double-float arithmetic.
- In the current `1024` and `2048` tests it preserved the key zoom-case coordinate-uniqueness win of the double-float representation while massively accelerating the double-float path.
- It now also supports a light `--post-polish-steps` float64 polishing pass; this helped the `1024` full-plane run recover the same root-difference count as the PyTorch double-float prototype.
- It also supports selective `fp64` iteration replay on suspicious pixels. With `--post-polish-steps 2 --iter-replay-threshold 8 --iter-replay-dilation 1`, the current prototype drove `root_diff_vs_fp64` to zero at both `1024` and `2048`, while still remaining substantially faster than the PyTorch double-float path.
- On the full plane its iteration-map differences remain somewhat larger, so it is already a strong throughput prototype but not yet the final numerical reference.

The double-float Triton route is now also integrated into the main renderer through `triton_doublefloat_backend.py`.

- Use it with `--newton-backend doublefloat-triton`.
- It inherits the same main-chain tile scheduler, run-directory layout, and Newton conjugate-symmetry acceleration as the existing `fp32` Triton backend.
- Unlike the plain `fp32` path, it always builds Newton coordinates in `float64` before splitting them into `hi + lo` pairs, so it does not silently fall back to `float32` grid spacing.
- Its current recommended accuracy-oriented settings match the prototype: `--doublefloat-post-polish-steps 2 --doublefloat-iter-replay-threshold 8 --doublefloat-iter-replay-dilation 1`.

There is also a sparse high-grid validation script in `sparse_reference_validation.py`.

- It validates only local patches instead of rendering the full plane.
- Each patch uses the same step size that a much larger nominal global grid would have used.
- It compares `fp32 Triton`, `double-float Triton` without replay, `double-float Triton` with the current `t8` replay, and an `fp64` reference on the exact same patch.
- `--center-mode tracked` recenters each deeper patch on the most boundary-sensitive point of the previous `fp64` patch, which is much better for checking whether relative precision degrades as the nominal grid grows.

There is also a frontier plotting helper in `problem1_frontier_curves.py`.

- It turns the existing extreme-grid sweep and sparse-validation CSV outputs into reusable scaling and precision-frontier plots.
- The scaling figure focuses on elapsed time, effective throughput, peak memory, and `dx / ULP32` as the grid grows.
- The precision figure focuses on `fp32`, `double-float`, and `double-float + t8 replay` behavior near the `float32` coordinate-resolution limit.

There is also a supplementary CPU profiling helper in `problem1_cpu_profile.py`.

- It keeps the original CPU baseline intact and adds a lighter reproducible worker sweep plus grid sweep for Problem 1.
- It is meant to document the CPU-side parallel scaling and memory growth of the homework's core fractal workflow.
- It writes a compact CSV and a summary figure under `result/analysis/`, so the report can state clearly that the assignment was already completed with a CPU implementation before any GPU acceleration was discussed.

There is also a specialized high-throughput CPU Newton implementation in `problem1_cpu_ultra.c`.

- It is not a replacement for the original homework script; it is a focused performance path for very large Problem 1 Newton grids.
- It uses `C + OpenMP`, direct-form Newton updates, conjugate-symmetry reuse, and tile-by-tile streaming downsampling.
- Its memory footprint is bounded mainly by the `render_grid` output buffers plus one small tile accumulator, so it stays near constant RSS as `compute_grid` increases.
- On this workstation it was used as an additional performance experiment for `80000 x 80000` Newton runs; the measured runtime and memory figures should be read as implementation results, not as part of the original homework requirements.

## Requirements

- `python3`
- `numpy`
- `matplotlib`
- `mpmath`
- `/home/void0312/Cpy/.venv/bin/python` with `pypandoc`
- `xelatex` if PDF export is needed

## Usage

Run the numerical workflow:

```bash
make run
```

Regenerate results and export documentation:

```bash
make docs
```

Run the GPU ultra renderer with the script defaults:

```bash
make gpu-ultra
```

Run the GPU renderer with explicit overrides:

```bash
make gpu-ultra GPU_ARGS="--compute-grid 40000 --render-grid 5000 --render-tile 250 --methods newton --output-prefix problem1_gpu_bench40k_newton"
```

Disable the Newton symmetry optimization for baseline comparison:

```bash
make gpu-ultra GPU_ARGS="--compute-grid 12000 --render-grid 3000 --render-tile 250 --methods newton --disable-newton-symmetry --output-prefix problem1_gpu_newton_sym_off_smoke"
```

Enable the experimental active-compression path for comparison:

```bash
make gpu-ultra GPU_ARGS="--compute-grid 12000 --render-grid 3000 --render-tile 250 --methods newton,secant --enable-active-compression --output-prefix problem1_gpu_active_on_smoke"
```

Enable the experimental direct Newton update formula:

```bash
make gpu-ultra GPU_ARGS="--compute-grid 40000 --render-grid 5000 --render-tile 250 --methods newton --newton-direct-formula --output-prefix problem1_gpu_newton_direct_40k"
```

Use the integrated Triton backend for Newton:

```bash
make gpu-ultra GPU_ARGS="--compute-grid 100000 --render-grid 5000 --render-tile 1000 --methods newton --newton-backend triton --output-prefix problem1_gpu_newton_triton_100k_rt1000"
```

Use a lower-VRAM Triton configuration with nearly the same `100k` runtime:

```bash
make gpu-ultra GPU_ARGS="--compute-grid 100000 --render-grid 5000 --render-tile 625 --methods newton --newton-backend triton --output-prefix problem1_gpu_newton_triton_100k_rt625"
```

Use the integrated double-float Triton backend in the main renderer:

```bash
make gpu-ultra GPU_ARGS="--compute-grid 1024 --render-grid 256 --render-tile 128 --methods newton --newton-backend doublefloat-triton --doublefloat-post-polish-steps 2 --doublefloat-iter-replay-threshold 8 --doublefloat-iter-replay-dilation 1 --output-prefix main_chain_doublefloat_triton_smoke"
```

Use the same backend in stats-only mode for larger compute grids:

```bash
make gpu-ultra GPU_ARGS="--compute-grid 8192 --compute-tile 2048 --stats-only --methods newton --newton-backend doublefloat-triton --doublefloat-post-polish-steps 2 --doublefloat-iter-replay-threshold 8 --doublefloat-iter-replay-dilation 1 --output-prefix main_chain_doublefloat_triton_stats_8k"
```

这些 GPU 命令现在都会把结果写到新的 run 目录里，例如：

- `result/gpu_pytorch/<output_prefix>__YYYYMMDD-HHMMSS/`
- `result/gpu_triton/<output_prefix>__YYYYMMDD-HHMMSS/`

Use the stats-only extreme mode for much larger compute grids without rendering:

```bash
make gpu-ultra GPU_ARGS="--compute-grid 1000000 --compute-tile 12500 --stats-only --methods newton --newton-backend triton --output-prefix problem1_gpu_newton_triton_stats_1m"
```

Run the Triton Newton benchmark:

```bash
make triton-bench TRITON_ARGS="--sizes 2048,3072 --reps 3 --block-size 256 --output-prefix triton_newton_bench_v2"
```

Run the prototype double-float comparison:

```bash
make doublefloat-proto DOUBLEFLOAT_ARGS="--coarse-grid 512 --test-grid 512 --output-prefix doublefloat_newton_prototype_512"
```

Run the refined arithmetic variant explicitly:

```bash
make doublefloat-proto DOUBLEFLOAT_ARGS="--coarse-grid 512 --test-grid 1024 --arith-mode refined --output-prefix doublefloat_newton_prototype_1024_refined"
```

Run the Triton double-float prototype:

```bash
make triton-doublefloat-proto TRITON_DOUBLEFLOAT_ARGS="--coarse-grid 512 --test-grid 1024 --reps 10 --post-polish-steps 2 --iter-replay-threshold 8 --iter-replay-dilation 1 --output-prefix triton_doublefloat_prototype_1024_r10"
```

Run the sparse tracked high-grid validation:

```bash
make sparse-validate SPARSE_VALIDATE_ARGS="--patch-grid 512 --global-grids 8192,262144,1048576,4194304,16777216,33554432 --center-mode tracked --output-prefix sparse_reference_validation_tracked"
```

Generate the Problem 1 scaling and precision-frontier plots from the latest analysis CSVs:

```bash
make frontier-curves
```

Generate the supplementary CPU worker/grid profile:

```bash
make cpu-profile CPU_PROFILE_ARGS="--worker-grid 4096 --worker-render-grid 1024 --worker-counts 1,8,16,32,64 --grid-sizes 1024,2048,4096,8192 --grid-workers 64 --repeats 3 --warmup-runs 1 --timing-scope full_compute_plus_resource_monitor --output-prefix problem1_cpu_profile"
```

`problem1_cpu_profile.py` writes timed samples to `result/analysis/<prefix>_raw.csv`, aggregate mean/std/median/IQR/best statistics to `result/analysis/<prefix>.csv`, and plots elapsed/throughput error bars from the aggregate table.

By default the worker sweep appends `os.cpu_count()` to the supplied `--worker-counts`.
Use `--no-include-cpu-count` for tiny smoke tests that should run only the
explicit worker counts.

Run only the five per-problem import smoke checks without starting the large Problem 1 fractal render:

```bash
for n in 1 2 3 4 5; do make problem$n PROBLEM_ARGS="--import-smoke"; done
```

Run the specialized high-throughput CPU Newton path:

```bash
make cpu-ultra CPU_ULTRA_ARGS="--compute-grid 80000 --render-grid 5000 --tile-rows 256 --threads 88 --output-prefix problem1_cpu_ultra_80k"
```

Sweep several large-grid CPU ultra runs and summarize them:

```bash
make cpu-ultra-bench CPU_ULTRA_BENCH_ARGS="--compute-grids 20000,40000,80000 --render-grid 5000 --tile-rows 64 --threads 88 --repeats 3 --warmup-runs 1 --timing-scope process_full_run --output-prefix problem1_cpu_ultra_bench"
make cpu-ultra-bench CPU_ULTRA_BENCH_ARGS="--compute-grids 80000 --render-grid 5000 --tile-rows 256 --threads 88 --repeats 3 --warmup-runs 1 --timing-scope process_full_run_tile256 --output-prefix problem1_cpu_ultra_bench_tile256"
```

For lightweight smoke validation, keep the same runner on small grids:

```bash
make cpu-ultra-bench CPU_ULTRA_BENCH_ARGS="--compute-grids 64,128 --render-grid 32 --tile-rows 8 --threads 2 --repeats 2 --warmup-runs 0 --output-prefix problem1_cpu_ultra_bench_smoke"
```

The bench runner writes per-run samples to `result/analysis/<prefix>_samples.csv` and aggregate mean/std/median/IQR/best statistics to `result/analysis/<prefix>_summary.csv` while preserving each timed sample's `source_command` and `source_csv`. The `problem1_cpu_ultra_bench*` files are the repeat-aware benchmark sweeps; the `tune_t*_r*.csv` files are the older single-run tile-row sweep; `result/problem1_cpu_ultra_summary.csv` describes the final render path.

Run the small-grid correctness gate for CPU ultra against the Python reference:

```bash
make correctness-gate CORRECTNESS_ARGS="--compute-grid 128 --render-grid 32 --tile-rows 8 --threads 2 --reference-workers 2 --output-prefix problem1_correctness_gate"
```

The correctness gate writes `result/analysis/problem1_correctness_gate.csv` and `.json`, including root-map difference rate, iteration-map difference, root-fraction/convergence deltas, field sanity status, and the same timing-scope metadata fields used by the benchmark runners.

Refresh the lightweight experiment manifest, report dependency manifest, and environment metadata from existing artifacts:

```bash
make experiment-manifest
```

This writes the full artifact inventory to `result/analysis/experiment_manifest.json`
and the report-only dependency list to `result/analysis/report_manifest.json`.

Or point the plotting script at specific analysis files:

```bash
make frontier-curves FRONTIER_ARGS="--extreme-summary result/analysis/problem1_gpu_triton_extreme_stats__20260408-201128/extreme_stats_summary.csv --sparse-validation result/analysis/sparse_reference_validation_tracked__20260408-223101/sparse_reference_validation.csv --output-prefix problem1_frontier_curves_manual"
```

## Outputs

- `result/`
  - 根目录保留主流程结果和文档直接引用的文件，例如 `temp-01.log`、`problem1_fractals.png`、`problem1_summary.csv`、`problem2_roots.csv`、`problem3_roots.csv`、`problem4_shengjin_roots.csv`、`problem5_l1.csv`
  - `result/gpu_pytorch/` 存放 PyTorch CUDA 路线的实验结果，每次新测试自动创建一个独立 run 目录
  - `result/gpu_triton/` 存放 Triton 集成后端和独立 Triton benchmark 的结果，每次新测试自动创建一个独立 run 目录
  - `result/analysis/` 存放 CPU 补充剖面、`cpu-ultra` 大网格 benchmark、跨后端对照、调参结论，以及 `problem1_frontier_curves*` 这类汇总曲线目录
  - `result/analysis/experiment_manifest.json` 和 `result/analysis/metadata.json` 记录扩展实验产物的 source path、source command、checksum、mtime、是否大规模，以及当前 Python/platform/CPU/OMP/GPU 可见环境
  - 详细说明见 `result/README.md`
- `docs/answer/answer.docx`
- `docs/answer/answer.pdf`
