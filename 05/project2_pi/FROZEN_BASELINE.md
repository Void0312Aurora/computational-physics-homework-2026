# Frozen CPU Baseline

本文件用于明确 `HW/05/project2_pi/` 当前这条纯 CPU 路线已经冻结为基线，不再继续做局部调度/参数/对象池层面的追加优化。

冻结日期：

- `2026-04-21`

## 冻结范围

以下内容保留为“可运行、可验对、可对照”的基线，而不是后续主攻优化对象：

- `gmp_backend.cpp`
  - 当前默认纯 CPU 主线
  - `parallel_mode=chunked|tasks|frontier`
- `gmp_backend_levelpool.cpp`
  - 表示层预分配/复用原型
- `benchmark_cpu_tasktree.py`
- `benchmark_cpu_representation.py`
- `benchmark_matrix.py`
- `benchmark_extremes.py`
- `homework_bridge.py`
- 相关基线结果：
  - `result/project2_cpu_cpp_reoptimized_summary.{csv,md}`
  - `result/project2_cpu_cpp_parallel_modes_summary.{csv,md}`
  - `result/project2_cpu_cpp_representation_summary.{csv,md}`

## 冻结含义

冻结并不表示这些实现会被删除，而是表示：

1. 不再继续给这条旧 CPU 主线追加新的 `parallel_mode`、`frontier` 变体、`levelpool` 变体或其他同层改造。
2. 不再继续围绕 `threads / chunk_terms / leaf_terms / task_terms` 做性能导向的参数扫描。
3. 允许的修改只剩两类：
   - 正确性修复
   - 为后续新架构做基线对照时所需的最小构建/接口修复

## 冻结依据

当前旧主线已经给出了足够清晰的结构性结论：

- `chunked` 优化主线在 `100M` digits 上约 `2.11e6 digits/s`
- `frontier` 在 `50M / 100M` 明显退化，说明“换调度形态”不能解决核心瓶颈
- `levelpool` 在 `50M` 略有收益，但在 `100M` 又回落到比 `chunked/tasks` 慢约 `4%~5%`

这说明旧主线的瓶颈已经不是“树怎么排”或“对象怎么少建一点”这种层级，而是：

- `mpz_t` 对象树本身
- 大整数结果在 merge 过程里的流动方式
- GMP 乘法/除法/开方内核与当前对象表示之间的耦合

## 后续方向

后续所有真正的性能工作，都应迁移到新的独立架构线上，而不是继续在这条基线里打补丁。

下一代设计见：

- `project2_pi/NEXTGEN_CPU_ARCHITECTURE.md`
