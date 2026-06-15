# Throughput Mainline Freeze Snapshot

冻结时间：`2026-04-18`

## 冻结结论

这条 `project2_gpu_throughput_mainline` 路线已经完成了它最重要的研究目标：

1. 证明了 `throughput-first` 的 GPU exact `P/Q/T` grouped scheduler 可以跑通。
2. 证明了 staged exact closure 可以把 root-level modulus growth 压回到一个很小的 fixed-moduli window。
3. 证明了 final host tail 不是当前 mainline 的首要瓶颈，真正的大头已经暴露为 device closure 本体和 one-time CUDA startup。

但这条路线没有完成它最初想挑战的性能目标：

1. 它没有接近 `10M/s -> 30M/s` 的完整 `pi` 吞吐量区间。
2. 它没有超过现有更强的自研完整 `pi` 主线。
3. 它更没有接近本机 `y-cruncher` 的成熟端到端水平。

因此，这条路线在 `2026-04-18` 被正式冻结为：

1. 一个研究型、证明可行性的 GPU exact closure / scheduler 原型。
2. 一个可回看的架构探索分支。
3. 一个不再继续承担“本机完整 `pi` 极限速度主线”职责的工作区。

## 冻结范围

以下内容作为当前冻结基线保留：

1. `src/main.cu`
2. `src/runtime.cu`
3. `include/project2_gpu_throughput_mainline/runtime.cuh`
4. `README.md`
5. `docs/architecture.md`
6. `docs/roadmap.md`
7. `result/` 下已经生成的 benchmark / smoke 日志

冻结规则：

1. 不再继续在这条分支上做无边界微调。
2. 不覆盖现有结果文件，只允许追加新 `output-tag`。
3. 若未来要重启这条路线，必须先明确是“研究恢复”还是“性能主线重构”。

## 当前已完成能力

当前冻结版本已经具备以下能力：

1. grouped batched pointwise / pack / FFT bridge / level planner 基础基准都已落地。
2. generic multi-limb grouped planner 已能稳定覆盖 `mask=63`、`mask=4095` 和 exact `32-bit modulo` 路径。
3. exact-moduli grouped planner 已接上真实 Chudnovsky `P/Q/T` leaves。
4. end-to-end `pi` smoke 已能通过 `16 / 32 / 64 / 128` terms 的 correctness 验证。
5. staged closure 已能在 `2` 模数默认路径和 forced `3 / 4` 模数路径下通过验证。
6. final host `sqrt(10005)` + division tail 已修正 guard digits，并用更快的 host long division 替代旧 bit-by-bit 路径。
7. end-to-end smoke 已不再被 embedded `2500` digits reference cap 卡死，并支持 `--report-decimal-digits` 截断输出。

## 冻结基线结果

### A. 代表性 planner / backbone 结果

这些结果说明 throughput-first GPU backbone 已经成形，但它们不是最终 `pi digits/s`：

1. `exact-moduli-pqt-32full`
   - `avg_pipeline_ms ~= 1507.89`
   - `packed_residue_values_per_second ~= 7.12e8`
   - 结果文件：
     `../result/project2_gpu_throughput_grouped_level_planner_exact_moduli_pqt_grouped_level_planner_exact_moduli_pqt_4096x4096x8.log`

2. `multilimb-mask32full`
   - `avg_pipeline_ms ~= 283.31`
   - `packed_residue_values_per_second ~= 9.47e8`
   - 结果文件：
     `../result/project2_gpu_throughput_grouped_level_planner_multilimb_grouped_level_planner_multilimb_4096x4096x8_mask32full_fp64_limb16.log`

### B. 关键 closure 诊断结果

这组结果用于冻结“当前真正瓶颈在哪里”的判断：

1. `2500d` breakdown diagnostic
   - `term_count = 256`
   - `slot_count = 2048`
   - `effective_closure_modulus_count = 2`
   - `cuda_runtime_init_ms = 210.898`
   - `closure_setup_ms = 14.994`
   - `closure_measured_total_ms = 102.921`
   - `closure_wall_ms = 190.276`
   - `final_host_tail_ms = 3.1161`
   - `prefix_match = 1`
   - 结果文件：
     `../result/project2_gpu_throughput_pi_end_to_end_e2e_256_2048_2500_breakdown_v1.log`

这个诊断结果很关键，因为它说明：

1. final host tail 已经缩到很小。
2. 真正剩余的大头不是 modulus headroom，也不是 host tail。
3. 真正的问题是 device closure 本体和 cold-start overhead。

### C. 冻结时的最终代码锚点

这组结果用于冻结“最新代码状态下，这条路线实际停在什么性能量级”：

1. `2500d` final-code smoke
   - `steady_state_pi_result_ms = 73.196`
   - `steady_state_pi_digits_per_second = 34154.9`
   - `cold_process_pi_digits_per_second = 3865.37`
   - 结果文件：
     `../result/project2_gpu_throughput_pi_end_to_end_e2e_256_2048_2500_nod2dreset_v1.log`

2. `5000d` final-code smoke
   - `steady_state_pi_result_ms = 114.98`
   - `steady_state_pi_digits_per_second = 43485.7`
   - `cold_process_pi_digits_per_second = 4851.59`
   - `closure_setup_ms = 62.3159`
   - `closure_measured_total_ms = 270.372`
   - `prefix_match = 1`
   - 结果文件：
     `../result/project2_gpu_throughput_pi_end_to_end_e2e_512_4096_5000_nod2dreset_v1.log`

冻结判断以这两个 final-code smoke 为准：

1. 路线是正确的。
2. correctness 还在。
3. 但性能量级仍然远低于目标。

## 已验证但被否决的方向

冻结时还需要把失败方向明确记下来，避免以后重复试错：

1. GPU balanced leaf-digit generation
   - 曾尝试把 balanced leaf digits 直接搬到 GPU setup。
   - 实测 `closure_setup_ms` 明显恶化，而不是改善。
   - 代表日志：
     `../result/project2_gpu_throughput_pi_end_to_end_e2e_256_2048_2500_leafgpu_v1.log`
     `../result/project2_gpu_throughput_pi_end_to_end_e2e_512_4096_5000_leafgpu_v1.log`

2. Immutable leaf-digit first-level reuse
   - 这是一个正确的结构清理。
   - 但它只带来了小幅收益，没有改变瓶颈排序。
   - 代表日志：
     `../result/project2_gpu_throughput_pi_end_to_end_e2e_256_2048_2500_nod2dreset_v1.log`
     `../result/project2_gpu_throughput_pi_end_to_end_e2e_512_4096_5000_nod2dreset_v1.log`

## 与其他路线的关系

这一步是决定“为什么冻结”的关键。

先说明口径：

1. 这条 throughput mainline 目前只有 smoke-scale end-to-end `pi` 指标。
2. 现有 `merge-and-final v4` 和 `y-cruncher` 是更完整、更成熟的 full `pi` compute 路线。
3. 它们不是严格 apples-to-apples 的同规模 benchmark，但已经足够用于判断“哪条路线值得继续当主线”。

对比结论：

1. 现有更强的自研完整 `pi` 主线：
   - `gpu_pi_hybrid_merge_gpu_resident_v4`
   - `100000000` digits
   - `2.712093e6 digits/s`
   - 来源：
     `../../result/project2_extreme_backend_summary.csv`

2. 本机 `y-cruncher`：
   - `100000000` digits: `4.708098e7 digits/s`
   - `2500000000` digits: `3.392458e7 digits/s`
   - 来源：
     `../../result/project2_extreme_backend_summary.csv`

3. 本冻结路线：
   - `5000d` final-code smoke: `4.34857e4 digits/s`

这意味着：

1. 它比现有更强的自研完整 `pi` 主线低大约 `62x`。
2. 它比本机 `y-cruncher` 低大约 `10^3x`。
3. 继续在这条分支上做小修小补，不再是高性价比的极限速度路线。

## 冻结后的定位

冻结后的推荐定位如下：

1. 保留这条路线作为“GPU exact closure / staged normalization / grouped scheduler”的研究档案。
2. 不再把它当成完整 `pi` 极限性能主线。
3. 若要继续冲本机完整 `pi` 的速度和位数，应优先回到：
   - `merge-and-final v4` 这一条已经更强的自研主线
   - 或直接使用 `y-cruncher` 作为实际极限工具链

## 只有在什么条件下才值得解冻

这条路线只有在下面条件同时成立时，才值得恢复：

1. 目标从“立刻冲完整 `pi` 极限速度”切换为“继续研究纯 GPU exact architecture”。
2. 接受一次真正的架构重启，而不是继续做 incremental micro-optimization。
3. 计划中的新方向至少包括其中两项：
   - GPU-native big integer / RNS representation 重构
   - persistent CUDA context / allocation reuse across runs
   - GPU-side division / reciprocal / sqrt 主线化
   - 更彻底的 closure semantics 迁移，而不是继续保留 host-side exact tail

如果做不到这三点，就不建议解冻。

## 冻结摘要

一句话总结这条路线：

1. 它已经成功证明“这条架构可以做通”。
2. 但它没有证明“这条分支值得继续承担本机完整 `pi` 速度主线”。
3. 因此，当前最合理的处理方式是：冻结、归档、保留结果，不再继续作为主线推进。
