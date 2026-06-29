# Plan

## Scope

- Target folder: `HW/04`
- Problem source: `HW/04/docs/problem/image.png`, `image_2.png`, `image_3.png`, `image_4.png`
- This revision focuses on strengthening `docs/answer/*`:
  - keep the original CPU homework workflow explicit, rather than letting later GPU progress dominate the narrative
  - add a supplementary CPU profile for Problem 1 so the report contains reproducible CPU-side scaling and memory evidence
  - add a dedicated CPU ultra Newton path for very large Problem 1 grids, prioritizing constant-memory tiling and raw throughput
  - add missing derivations for Problems 2 to 5
  - expand Problem 1 discussion beyond the minimum submission outputs
  - fold the existing large-grid GPU, Triton, double-float, and sparse-validation progress into the final report
  - add reusable Problem 1 frontier plots so later scaling or precision runs can be summarized without manual spreadsheet work

## Problem Summary

- Chinese translation:
  - Problem 1: 对 `f(z)=z^3-1=0` 使用 Newton-Raphson 生成分形图，调节收敛容差使图像美观，并给出两种配色方案；可选地用 secant 方法重做并讨论是否还能看到分形。
  - Problem 2: 使用完整的 Newton-Raphson + Bisection 混合法求 `f(x)=4 cos x - e^x` 的根。
  - Problem 3: 使用完整的 Müller-Brent/Brent 型算法求
    `f(x)=x tan x - sqrt(h^2-x^2)` 与 `g(x)=x cot x + sqrt(h^2-x^2)` 的根，可自行选择 `h`，例如 `0.2, 0.5, 1.0` 等。
  - Problem 4: 使用范盛金公式求三次方程 `x^3 - 70.5x^2 + 1533.54x - 10082.44 = 0` 的根。
  - Problem 5: 使用 Newton 或 secant 方法求地月系统 `L1` 点方程的数值解，并达到至少四位有效数字。
- Required outputs:
  - 数值方法实现
  - Problem 1 的分形图
  - 各题结果表与日志
  - `docs/answer` 最终答案文档
- Numerical or implementation constraints:
  - Problem 1 需要复平面迭代和收敛分类
  - Problem 2 方程在负半轴有无限多个根，需要说明计算区间
  - Problem 3 需要在函数定义域和三角函数奇点之间做安全寻根
  - Problem 4 需要给出闭式求根并验证残差
  - Problem 5 需要高精度参考值核对

## Approach

- Language/tool choice:
  - 使用 `Python` 统一实现数值方法、图像与结果表
  - 使用 `numpy`、`matplotlib`、`mpmath`
  - 使用 `pypandoc` 导出 `docs/answer`
- Core algorithm or script plan:
  - Problem 1: 生成 Newton fractal，并追加 secant fractal；采用并行高采样计算后降采样输出，兼顾高分辨率与可交付图像尺寸
  - Problem 1 CPU 补充实验：增加单独的 worker sweep 与 grid sweep，说明 CPU 主流程在不同并行度和不同网格下的耗时、吞吐率与内存增长；benchmark runner 输出 timed raw samples 与 mean/std/median/IQR/best 聚合表，并记录 `warmup_runs`、`timed_repeats`、`timing_scope`
  - Problem 1 CPU ultra 实现：使用 `C + OpenMP`、Newton 直写公式、上半平面对称复用和流式降采样，目标是在 `80000 x 80000` 级别上显著压低时间与内存；bench wrapper 同样保留 per-run samples、aggregate summary、source command/source CSV 和冷热启动口径字段
  - Problem 1 附加 GPU 路线：使用 `PyTorch CUDA` 做 tiled streaming 计算，在固定显存预算下支持更高 compute grid
  - Problem 1 附加高精度 GPU 路线：把 `double-float Triton` 后端接入主链路，复用 Newton 共轭对称、分块调度和 run 目录管理，并保留 `post-polish + t8 replay` 的精度修正
  - Problem 1 精度验证：增加 sparse reference validation，只在边界敏感局部 patch 上对比 `fp32 / double-float / fp64`，并支持 tracked center 模式以检查名义全局网格继续增大时的相对精度变化；另外增加小网格 `problem1_correctness_gate.py`，用 CPU ultra 输出的 root/iteration map 与 Python reference 对照，形成可自动失败的 acceptance gate
  - Problem 1 后处理分析：增加 frontier plotting 脚本，从已有 extreme sweep 和 sparse validation CSV 自动生成规模扩展曲线与精度前沿曲线
  - 复现实验索引：增加轻量 `experiment_manifest.py`，从已有扩展 CSV/PNG 生成 `result/analysis/experiment_manifest.json`，并配套生成 `metadata.json` 记录 Python、平台、CPU、线程环境和可见 GPU 信息；同时生成 `report_manifest.json`，只列报告正文直接引用的 CSV/PNG/PDF 依赖；该步骤只做文件索引和环境采样，不触发 80k/2m 或 GPU 大规模重跑
  - 分题入口复现：`problem1.py` 到 `problem5.py` 显式导入 `scripts.solution`，避免误命中根目录 `solution.py` 包装器；提供 `--import-smoke` 用于验证 Makefile 分题入口导入链路
  - 文档增强：在 `docs/answer/answer.md` 中补充算法公式、推导链、结果表和大网格实验摘要，并说明哪些内容属于主提交流水线、哪些内容属于 `HW/04` 内的扩展研究进展
  - Problem 2: 在代表性区间 `[-10, 2]` 扫描变号区间，并用 safeguarded Newton-bisection 求所有根
  - Problem 3: 对若干 `h` 值扫描括区，使用 Brent 型算法求根，并记录某些 `h` 下 `g(x)` 无根的情况
  - Problem 4: 实现三次方程三实根分支的闭式公式，并用残差验证
  - Problem 5: 用 Newton 与 secant 求 `L1` 点，并与高精度参考解比较
- Output files to create:
  - `solution.py`
  - `gpu_ultra_fractal.py`
  - `problem1_cpu_profile.py`
  - `problem1_cpu_ultra.c`
  - `problem1_cpu_ultra_bench.py`
  - `problem1_correctness_gate.py`
  - `experiment_manifest.py`
  - `triton_doublefloat_backend.py`
  - `sparse_reference_validation.py`
  - `Makefile`
  - `README.md`
  - `result/temp-01.log`
  - `result/problem1_fractals.png`
  - `result/problem2_roots.csv`
  - `result/problem3_roots.csv`
  - `result/problem4_shengjin_roots.csv`
  - `result/problem5_l1.csv`
  - `docs/answer/answer.md`
  - `docs/answer/render_docs.py`

## Testing

- Commands to run:
  - `make docs`
  - `for n in 1 2 3 4 5; do make problem$n PROBLEM_ARGS="--import-smoke"; done`
  - `make cpu-profile CPU_PROFILE_ARGS="--worker-grid 64 --worker-render-grid 32 --worker-counts 1,2 --no-include-cpu-count --grid-sizes 64,128 --grid-workers 2 --repeats 2 --output-prefix problem1_cpu_profile_smoke"`
  - `make cpu-ultra-bench CPU_ULTRA_BENCH_ARGS="--compute-grids 64,128 --render-grid 32 --tile-rows 8 --threads 2 --repeats 2 --output-prefix problem1_cpu_ultra_bench_smoke"`
  - `make correctness-gate CORRECTNESS_ARGS="--compute-grid 128 --render-grid 32 --tile-rows 8 --threads 2 --reference-workers 2 --output-prefix problem1_correctness_gate"`
  - `make experiment-manifest`
- Expected checks:
  - Problem 1 图像正常生成，并能区分 basins
  - `doublefloat-triton` 主链路在小尺寸渲染和 stats-only 模式下均可运行，并保持主链路对称优化
  - CPU supplementary profile 能生成 worker/grid 两组剖面，并输出 raw samples、aggregate CSV 与 `problem1_cpu_profile.png`
  - CPU ultra 路线能输出 `80000 x 80000` 的真实 benchmark 结果，并验证是否满足 `180 s` 内完成的目标；小规模 smoke 只验证 repeats/summary 框架，不替代大规模重跑
  - correctness gate 能在 `128/32` 小网格下通过 CPU ultra vs Python reference 的 root-map、iteration-map、root fraction、convergence 和 field sanity 检查
  - sparse validation 能生成局部 patch 对照结果，并记录不同 nominal global grid 下的 `root_diff_vs_fp64`、`iter_diff_vs_fp64` 与坐标唯一性
  - frontier plotting 脚本能自动发现最新的 extreme sweep 与 tracked sparse validation 数据，并输出图像与摘要 CSV
  - experiment manifest 能生成 `result/analysis/experiment_manifest.json`、`report_manifest.json` 与 `metadata.json`，记录扩展图表/CSV 的 source path、source command、checksum、mtime 和是否大规模，不重跑重实验
  - 分题入口 smoke 能确认 `make problem1` 到 `make problem5` 的 wrapper 导入链路不会再命中根目录 `solution.py`
  - Problem 2 混合法在选定区间内找到全部根，并与高精度核对
  - Problem 3 Brent 型算法对选定 `h` 值稳定收敛
  - Problem 4 得到三个实根并使残差接近零
  - Problem 5 Newton 与 secant 给出一致的 `L1` 距离
  - `answer.docx` 与 `answer.pdf` 成功生成
  - 渲染后的 `answer.pdf` 要能体现新增推导和扩展实验，不出现图片丢失、公式错位或明显表格拥挤

## Risks

- Known numerical pitfalls:
  - Problem 1 secant 版本对初始双点选择敏感
  - CPU ultra 路线为追求吞吐，仅覆盖 Problem 1 的 Newton 情形，不追求与通用主脚本同等的功能范围
  - Problem 2 若不限定区间，会遇到无限多个负根
  - Problem 3 需避开 `tan`、`cot` 奇点与 `sqrt(h^2-x^2)` 定义域边界
  - Problem 4 闭式公式的三角分支需要谨慎实现
  - Problem 5 牛顿法需要合适初值
- Toolchain or environment concerns:
  - 依赖 `numpy`、`matplotlib`、`mpmath`
  - 文档导出依赖 `xelatex` 与 `pypandoc`
  - 新增内容较多后，需要额外检查 PDF 的分页与表格宽度
