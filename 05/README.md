# HW/05 Project Notes

`HW/05` 现在已经按“冻结基线”和“下一代架构设计”重新整理。

## Frozen Baseline

- 主线实现：`project2_pi.gmp_backend.cpp` + `project2_pi.homework_bridge`
- 默认 homework 后端：优化后的 `cpp_gmp_openmp_optimized`
  - `solution.py` 现在默认优先走纯 CPU C++ 主线；如需强制切回 GPU，可显式设置 `PROJECT2_HOMEWORK_BACKEND=gpu`
  - CPU C++ 主线二进制现在支持 `parallel_mode=chunked|tasks|frontier`；默认仍保留 `chunked`，`tasks` 与 `frontier` 都只作为显式实验入口
  - 另有独立的 `project2_gmp_levelpool_backend`，用于承载更接近表示层的 `levelpool` 预分配/复用重构原型
- 冻结状态：
  - 当前 CPU 路线已经冻结为基线，不再继续做 `parallel_mode` 变体、`levelpool` 补丁或参数扫描
  - 冻结说明见 `project2_pi/FROZEN_BASELINE.md`
  - 下一代设计见 `project2_pi/NEXTGEN_CPU_ARCHITECTURE.md`
- GPU 吞吐探索 profile：`fast-auto`
  - 现已带自适应除法分流：正常窗口保留 GPU Newton，检测到高位数回退风险时自动切回 `mpz-div`
- 高位数保守 profile：`high-digits-auto`
- 有界工作集 profile：`bounded-stream-auto`
  - 通过 `stream_partials + frontier merge + 10 GB` FFT 预算，把 GPU 工作集压到更稳定的窗口
- 推荐入口：`make run`
- 大位数证据入口：`make project2_manifest`
  - 只读取已有 `result/project2_pi_100000000_digits.txt`，生成 sidecar manifest、SHA256、首尾/抽样窗口和历史 y-cruncher 引用
  - 不属于默认 `make run`，不会启动 `100M/140M/2.5B` 级重算
- 统一测速入口：`make benchmark_pi`
- 极限探索入口：`make benchmark_extremes`
- CPU runner 稳健统计：
  - `benchmark_cpu_tasktree` 与 `benchmark_cpu_representation` 现在默认按 interleaved 顺序运行；如需随机化可加 `--shuffle --seed <n>`，如需复查旧口径可显式用 `--grouped`
  - 默认重复数提升为 `--repeats 5`；长任务若降到 `2~3` 次，需要在结果说明中标注为长任务降级口径
  - CSV 保留 `seconds_samples` / `digits_per_second_samples`，并新增 `median_seconds`、`iqr_seconds`、`median_digits_per_second`、`iqr_digits_per_second`
  - 同名 sidecar metadata（默认 `<csv>.metadata.json`）记录 `run_index`、执行顺序、`loadavg`、OpenMP 亲和性变量、进程 CPU affinity、CPU governor/frequency 与可用时的 `nvidia-smi` 摘要
  - `make benchmark_cpu_smoke` 只跑 `100k` 位、4 线程、5 次重复，生成 `project2_cpu_cpp_parallel_modes_smoke.*`、`project2_cpu_cpp_representation_smoke.*` 及 metadata，用作审计 smoke，不触发 `100M/140M/2.5B`

本地最新测速结果见 `result/project2_route_benchmark_current.{csv,md}`。在 `10,000,000` digits 的同口径测试里：

- `gpu_hybrid_merge_fast_auto`：约 `5.02e6 digits/s`
- `gpu_hybrid_merge_fast_python`：约 `5.10e6 digits/s`
- `cpp_gmp_levelpool`：约 `2.62e6 digits/s`
- `cpp_gmp_openmp`：约 `2.61e6 digits/s`
- `cpp_gmp_openmp_frontier`：约 `2.46e6 digits/s`
- `python_gmp_bridge`：约 `2.10e6 digits/s`
- `gpu_hybrid_legacy_default`：约 `2.00e6 digits/s`

这意味着在 `10M` 这档上，GPU fast 路线仍然是最快的；而在 CPU 路线内部，`levelpool` 已经给出了最后一轮“表示层仍停留在 `mpz_t` 对象树内”的证据。当前项目的判断是不再继续沿这条旧 CPU 主线深挖，而是把它冻结成对照基线。

## Next-Gen CPU

后续纯 CPU 工作不再继续堆在 `gmp_backend.cpp` 上，而是要单独开启下一代架构线：

- 不再以 `Triple(mpz_t p, q, t)` 作为热路径基本单位
- 不再允许运行时计算路径依赖 `mpz_t`、`mpn_*` 或旧 GMP bridge
- 改为 `page/block engine + destructive reduction + RNS/NTT` 大乘法域
- 最终 reciprocal / sqrt / division 也要进入新 engine，而不是桥接回旧后端

当前 next-gen 进度：

- `project2_pi/nextgen_cpu/` 已经落地 `PageArena + BigHandle + TripleSlot`
- 已有 `canonical 32-bit block -> residue inject -> cached multi-modulus NTT -> direct CRT carry -> canonical` 的 CPU 侧乘法 smoke
- 已有 quick benchmark：`result/project2_nextgen_cpu_multiply_benchmark.{csv,log}`
- 已有 crossover benchmark：`result/project2_nextgen_cpu_multiply_benchmark_extended.{csv,log}`
- 已有 destructive product tree benchmark：`result/project2_nextgen_cpu_tree_benchmark.{csv,log}`
- 当前 benchmark 结论已经从“只保证 exact”推进到“中大规模开始反超 schoolbook”：
  - 在 quick benchmark 的 `256` blocks 上，`speedup_rns_vs_schoolbook` 约 `0.355`
  - 在 extended benchmark 的 `1024` blocks 上，`speedup_rns_vs_schoolbook` 约 `1.282`
  - 在 extended benchmark 的 `4096` blocks 上，`speedup_rns_vs_schoolbook` 约 `4.109`
  - 在 tree benchmark 的 `64 leaves x 256 blocks` workload 上，adaptive `RNS` product tree 约 `3.477x`
- 这说明新线已经不再只是 correctness 骨架，而是开始出现明确的规模优势区间
- 同时当前 tree benchmark 也说明：吞吐优势已经能延续到树形归并，但 `peak_live_blocks` 还没有下降，说明上层节点仍未真正停留在 transform-resident 域

设计文档见：

- `project2_pi/NEXTGEN_CPU_ARCHITECTURE.md`

## Extreme Exploration

更大的 CPU / 混合极限探索结果保存在：

- `result/project2_cpu_hybrid_extremes.{csv,md}`
- `result/project2_hybrid_extremes_limit_window.{csv,md}`
- `result/project2_cpu_cpp_extremes_deep.{csv,md}`
- `result/project2_cpu_cpp_reoptimized_summary.{csv,md}`
- `result/project2_cpu_cpp_parallel_modes_summary.{csv,md}`
- `result/project2_cpu_cpp_representation_summary.{csv,md}`
- `result/project2_cpu_cpp_tasktree_summary.{csv,md}`
- `result/project2_hybrid_refactor_summary.{csv,md}`
- `result/project2_hybrid_adaptive_division_summary.{csv,md}`
- `result/project2_hybrid_bounded_stream_summary.{csv,md}`

当前可以先记住几个结论：

- 混合主线 `gpu_hybrid_fast_auto`
  - 历史窗口测试里，`140M` digits 曾在当前 `RTX 3090 24GB` 上触发 CUDA OOM
  - 本轮重构后，`140M` digits 可稳定跑通，约 `6.22e6 digits/s`
  - 继续加入自适应除法后，`140M` digits 保持 `newton-chunk-gpu-seed-prototype`，约 `6.13e6 digits/s`
  - 继续加入自适应除法后，`160M` digits 自动切到 `mpz-div`，约 `3.18e6 digits/s`
- 混合保守线 `gpu_hybrid_high_digits_auto`
  - 通过 `12 GB` 的 GPU 乘法预算把超大乘法主动留给 CPU
  - `100M` digits 保持 `newton-chunk-gpu-seed-prototype`，约 `4.47e6 digits/s`
  - `140M` digits 自动切到 `mpz-div`，约 `2.35e6 digits/s`
- 混合有界工作集线 `gpu_hybrid_bounded_stream_auto`
  - 用流式 partial 生成与 frontier merge 把 `merge_frontier_max_nodes` 压到 `4`
  - `100M` digits 自动切到 `mpz-div`，约 `2.30e6 digits/s`，`gpu_peak_gb` 约 `8.65`
  - `140M` digits 自动切到 `mpz-div`，约 `1.45e6 digits/s`，`gpu_peak_gb` 约 `8.39`
  - `160M` digits 自动切到 `mpz-div`，约 `1.42e6 digits/s`，`gpu_peak_gb` 约 `8.72`
- 纯 CPU 优化主线 `cpp_gmp_openmp_optimized`
  - 通过 `leaf_terms` 小块迭代和 OpenMP 并行 merge，把高位数最佳分块从 `524288~1048576` 前移到 `131072~262144`
  - `100M` digits 约 `2.11e6 digits/s`
  - `150M` digits 成功，约 `1.80e6 digits/s`
  - `200M` digits 成功，约 `1.83e6 digits/s`
- 纯 CPU `task-tree` 调度原型
  - 已把 `parallel_mode/tasks` 和 `task_terms` 接入主配置链、极限 benchmark 入口和独立对比脚本
  - 在最新一轮 `44 threads, chunk_terms=task_terms=131072, leaf_terms=8` 的重复测试里，`tasks` 与 `chunked` 基本处于同一量级：`10M` 几乎持平，`50M` 略慢，`100M` 约快 `2.8%`
  - 这说明更深一层的任务调度改造还不够稳定，不足以单独把当前 `mpz_t` 架构推到 `y-cruncher` 那个数量级
- 纯 CPU `frontier` 直接重构原型
  - 新增 `parallel_mode=frontier`，把并行 partial 生成和按区间顺序的 frontier reduce 串成一条 CPU 流水线，并把 merge 改成原地 `mpz` 更新
  - 在 `44 threads, chunk_terms=task_terms=131072, leaf_terms=8`、`10M/50M/100M` 各 2 次重复测试里，`frontier` 的平均吞吐分别只有约 `2.28e6 / 1.32e6 / 0.73e6 digits/s`
  - 相比同轮 `chunked`，`frontier` 在 `50M` 约慢 `39%`、在 `100M` 约慢 `56%`，说明“减少 live nodes + 流式归并”本身并不足以解决当前 CPU 主线的体系瓶颈
- 纯 CPU 表示层 `levelpool` 原型
  - 新增独立二进制 `project2_gmp_levelpool_backend`，把 chunk 内递归 scratch 与跨 chunk merge level 都改成预分配 `ReservedTriple` 池，主动减少 `mpz_t` 对象构造、清理和整轮 merge 向量重建
  - 在 `44 threads, chunk_terms=131072, leaf_terms=8` 的表示层对比里，`levelpool` 在 `50M` 平均约 `2.31e6 digits/s`，比同轮 `chunked/tasks` 略快；但在 `100M` 平均约 `1.94e6 digits/s`，比 `chunked/tasks` 慢约 `4%~5%`
  - 这说明它和 `frontier` 不同，并没有出现明显吞吐崩塌；但当前实现还只是“有效原型”，不足以直接取代默认 `chunked` 主线

这说明当前机器上的两条路线分工已经很清楚：

- 如果目标是“最快”，继续沿 `hybrid fast-auto` 追吞吐
- 如果目标是“显存更可控但仍想保留明显 GPU 收益”，先试 `hybrid high-digits-auto`
- 如果目标是“把 GPU 工作集压到约 10 GB 并接受速度明显下降”，可试 `hybrid bounded-stream-auto`
- 如果目标是“继续往上顶位数”或“默认稳定地算更高位数”，直接用 `cpp_gmp_openmp_optimized` 更合理

## Current Layout

- `solution.py`
  - Homework 05 主入口；Project 2 默认优先走优化后的 CPU C++ 主线，环境变量可覆盖。
- `project2_pi/`
  - 当前唯一继续优化的 Pi 计算主线，包含 C++ CPU 主程序、GPU FFT 后端、hybrid 探索程序、测速脚本与作业桥接模块。
- `project2_gpu_native_rns/`
  - 已封存的原生 RNS 研究分支；保留结果和实现用于参考，不再作为主线。
- `project2_gpu_throughput_mainline/`
  - 已封存的 throughput-first 研究分支；保留结果和实现用于参考，不再作为主线。
- `cuda/`
  - `project2_pi` 依赖的 CUDA chunk-ops 扩展源码。
- `result/`
  - 当前工作目录下生成的 benchmark、结果和中间产物。
- `docs/`
  - 题目、计划与答案文档。
- `HW06.姜玥晟/` 和 `HW06.姜玥晟.tar`
  - 历史提交快照，视为归档内容。

## Recommended Entry Points

- 跑作业主流程：`make run`
- 生成答案文档：`make docs`
- 生成 Project 2 大输出 manifest：`make project2_manifest` 或 `make reproduce_project2_report_manifest`
- 跑纯 CPU C++ 主线：`make cpp_backend`
- 跑 GPU hybrid 探索：`make gpu_pi_hybrid`
- 跑 legacy hybrid 对照：`make gpu_pi_hybrid_legacy`
- 复跑路线测速矩阵：`make benchmark_pi`
- 跑 CPU / 混合极限探索：`make benchmark_extremes`
- 跑纯 CPU `chunked vs tasks` 对比：`make benchmark_cpu_tasktree`
- 跑纯 CPU `chunked / tasks / frontier` 对比：`make benchmark_cpu_parallel_modes`
- 跑纯 CPU `chunked / tasks / levelpool` 表示层对比：`make benchmark_cpu_representation`
- 跑 5-repeat CPU benchmark smoke：`make benchmark_cpu_smoke`
- 跑下一代 CPU 骨架 smoke：`make nextgen_cpu_smoke`
- 跑下一代 CPU isolated multiply benchmark：`make benchmark_nextgen_cpu`
- 跑下一代 CPU crossover benchmark：`make benchmark_nextgen_cpu_extended`
- 跑下一代 CPU product tree benchmark：`make benchmark_nextgen_cpu_tree`
- 跑 GPU FFT benchmark：`make fft_gpu_benchmark`
- 编译 C++ CPU 基线：`make cpp_backend`
- 编译 C++ `levelpool` 原型：`make cpp_levelpool_backend`
- 编译下一代 CPU 骨架：`make nextgen_cpu_backend`
- 编译 `mpn` benchmark：`make mpn_benchmark`

## Archived Routes

- `project2_pi.gpu_pi_full_cuda`
  - 仍可运行，但已归档为 prototype；详情见 `project2_pi/ARCHIVED_ROUTES.md`
- `project2_gpu_native_rns`
  - 已归档；详情见 `project2_gpu_native_rns/ARCHIVED.md`
- `project2_gpu_throughput_mainline`
  - 已归档；详情见 `project2_gpu_throughput_mainline/ARCHIVED.md`

## Notes

- `docs/answer/*` 和 `HW06.姜玥晟/*` 中保留了一些整理前的历史路径，作为存档对照。
- `PROJECT2_HOMEWORK_BACKEND` 现在默认是 `cpu`；如果需要强制切回 hybrid，可设置 `PROJECT2_HOMEWORK_BACKEND=gpu` 或 `auto`。
- 如需手动覆盖优化后的 CPU 参数，可设置 `PROJECT2_CPP_THREADS`、`PROJECT2_CPP_CHUNK_TERMS`、`PROJECT2_CPP_LEAF_TERMS`、`PROJECT2_CPP_TASK_TERMS`、`PROJECT2_CPP_PARALLEL_MODE`。
- 如果需要恢复更大的 benchmark 列表，可设置 `PROJECT2_BENCHMARK_DIGITS=...`。
- `100M/140M/2.5B` 级结果视为显式实验或历史保留产物；默认 `make run` 不生成这些规模。复核报告引用时先运行 `make project2_manifest`，查看 `result/project2_pi_100000000_digits.txt.manifest.json` 与 `result/ycruncher/validation/` 中的历史校验记录。
