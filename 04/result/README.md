# Result Layout

`result/` 现在按用途分成两层：

- 根目录保留当前作业主流程直接生成和文档直接引用的结果。
- GPU 相关的实验结果按后端和分析用途分到子目录，避免继续堆在根目录。
- 对于新的 GPU 测试，脚本会在对应类别目录下自动创建一个新的 run 文件夹，而不是把文件直接堆到类别目录里。

## Root Files

这些文件是 `solution.py` 的主输出，也是 `docs/answer/answer.md` 当前直接引用的结果：

- `temp-01.log`
- `problem1_fractals.png`
- `problem1_summary.csv`
- `problem1_resources.csv`
- `problem2_roots.csv`
- `problem3_roots.csv`
- `problem4_shengjin_roots.csv`
- `problem5_l1.csv`

## GPU Experiment Folders

- `gpu_pytorch/`
  - 存放 `gpu_ultra_fractal.py` 在 PyTorch 后端下的实验结果
  - 包括 smoke、40k/80k/100k、对称优化、early-exit、direct-formula、active-compression、secant 准备测试等
  - 新运行的目录名形如 `<output_prefix>__YYYYMMDD-HHMMSS`

- `gpu_triton/`
  - 存放 Triton 集成后端和独立 Triton benchmark 的结果
  - 包括 Triton smoke、40k/100k 集成测试、不同 render tile 配置、block-size 调参、`triton_newton_bench_*`
  - 也包括 `stats-only extreme` 这种不生成图像、只累计全局统计的大网格压力测试
  - 现在还包括 `doublefloat-triton` 主链路接入后的 smoke、`stats-only` 和高精度对照 run
  - 新运行的目录名同样使用 `<output_prefix>__YYYYMMDD-HHMMSS`

- `analysis/`
  - 存放 CPU 补充剖面、跨后端对照和结论性摘要
  - 当前包括 Triton 集成对照表、block tuning 结论、大网格 extreme sweep 摘要
  - 现在也包括 `problem1_cpu_profile.*` 这类 CPU worker/grid 剖面文件，用于说明作业主流程本身的 CPU 可复现性与资源尺度；`problem1_cpu_profile_raw.csv` 是 timed samples，`problem1_cpu_profile.csv` 是 mean/std/median/IQR/best 聚合表
  - `make review-smoke` 生成 `problem1_cpu_profile_review_smoke*`、`problem1_cpu_ultra_bench_review_smoke*` 和 `problem1_correctness_gate_review_smoke*`；前两类 benchmark 用小网格记录 `warmup_runs=1`、`timed_repeats=3`，correctness gate 则是单次 CPU ultra vs Python reference acceptance check，不触发大尺寸 CPU/GPU/Triton 重跑
  - 现在还包括 `problem1_cpu_ultra_*` 与 `tune_t*_r*.csv` 这类大网格 CPU ultra benchmark 和调参结果，用于记录 `C + OpenMP` 分块 Newton 路线在 `80000 x 80000` 级别上的实际吞吐与常量级 RSS
  - `problem1_cpu_ultra_bench_samples.csv` 是 CPU ultra per-run samples；`problem1_cpu_ultra_bench_summary.csv` 是 `tile_rows=64` compute-grid bench sweep 聚合表；`problem1_cpu_ultra_bench_tile256_summary.csv` 是 `80000 -> 5000, tile_rows=256` 的 repeat-aware 聚合表；`tune_t*_r*.csv` 是较早的单次 tile-row sweep；`problem1_cpu_ultra_summary.csv` 是 final render 摘要，这几类口径不要混读
  - `problem1_correctness_gate.csv` 和 `.json` 是小网格 CPU ultra vs Python reference acceptance gate，记录 root-map/iteration-map/root-fraction/convergence 差异与 `warmup_runs/timed_repeats/timing_scope`
  - 也包括双浮点 `iter replay` 总结、replay mask tradeoff，以及 sparse high-grid reference validation
  - `experiment_manifest.json` 由 `make experiment-manifest` 从已有 CSV/PNG/JSON 轻量生成，记录 source path、source command、checksum、mtime、是否大规模、CSV 口径字段和 timing metadata；`report_manifest.json` 是报告正文实际引用产物的窄依赖清单；`metadata.json` 记录当前 Python/platform/CPU/OMP/GPU 可见环境
  - 这类目录通常按分析主题命名，而不是按时间戳命名

## Note

较早的历史结果已经按“每个测试一个目录”回填整理，但当时没有统一的时间戳命名，因此它们一般直接使用测试前缀作为目录名。
