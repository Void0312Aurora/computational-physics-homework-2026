# Result Index

这份索引用来整理 `HW/05/project2_gpu_native_rns/result` 下的实验结果。

整理原则：

1. 按类别将原始结果文件放入对应子目录，不改文件名。
2. 优先标出每一类结果里当前推荐引用的基线文件。
3. 同一家族里的早期 `v1/v2/...` 文件保留为试错轨迹，但默认不再作为主引用结果。

补充说明：

1. `2026-04-18` 起，这个 `result/` 目录已经补齐为项目内自包含布局。
2. 先前落在 `HW/05/result` 的 native-RNS 历史结果，已导入到这里的对应分类子目录。
3. 当前总览是“当前主线结果 + 历史导入结果”并存，前者用于引用主结论，后者用于追溯优化过程。

## 当前目录布局

1. `correction_domain/`
   - correction-domain / residual-correction 相关验证
2. `end_to_end_smoke/`
   - 只看 correctness 的端到端 prefix 验证
3. `pqt_tensor_smoke/`
   - `P/Q/T` tensor 级验证
4. `pqt_tree_tensor_smoke/`
   - `P/Q/T` tensor-tree exact merge 验证
5. `end_to_end_benchmark/`
   - 含 timing 和 `digits/s` 的端到端 benchmark
6. `runtime_smoke/`
   - 早期 runtime smoke、C 阶段优化历史、lifecycle/block 回归与汇总 CSV
7. `d_semantics/`
   - D1-D4 直接语义 smoke：scaled constant / reciprocal seed / sqrt seed / division
8. `route_and_plan/`
   - E1/E2 route selection 与 E3 partial execution plan
9. `pqt_host_semantics/`
   - host-side exact `P/Q/T` 语义 smoke

## 推荐引用顺序

如果只想快速了解这条 `project2_gpu_native_rns` 路线，建议按下面顺序阅读：

1. end-to-end smoke
   - `end_to_end_smoke/project2_pi_end_to_end_smoke_default_20260417_v2.log`
   - `end_to_end_smoke/project2_pi_end_to_end_smoke_100digits_20260417_v2.log`
2. correction-domain recheck
   - `correction_domain/project2_correction_domain_modswitch_20260417_v2.log`
   - `correction_domain/project2_correction_domain_recheck_after_pi_end_to_end_20260417.log`
3. deep exact tree milestones
   - `pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms32_modswitch_20260417_v1.log`
   - `pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms47_modswitch_20260417_v1.log`
   - `pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms64_modswitch_20260417_v1.log`
4. end-to-end benchmark anchors
   - `end_to_end_benchmark/project2_pi_end_to_end_benchmark_400digits_20260417_v3.log`
   - `end_to_end_benchmark/project2_pi_end_to_end_benchmark_400digits_20260417_v3.csv`
   - `end_to_end_benchmark/project2_pi_end_to_end_benchmark_630digits_20260417_v1.log`
   - `end_to_end_benchmark/project2_pi_end_to_end_benchmark_630digits_20260417_v1.csv`
   - `end_to_end_benchmark/project2_pi_end_to_end_benchmark_870digits_20260417_v1.log`
   - `end_to_end_benchmark/project2_pi_end_to_end_benchmark_870digits_20260417_v1.csv`
5. runtime optimization history
   - `runtime_smoke/project2_cuda_stage1_summary.csv`
   - `runtime_smoke/project2_gpu_native_rns_smoke_noverify_64x4096_c45lifecycle.csv`
   - `runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_c45lifecycle.csv`
   - `runtime_smoke/project2_gpu_native_rns_smoke_full_64x4096_c45lifecycle.csv`
6. D/E planning checkpoints
   - `d_semantics/project2_gpu_native_rns_scaled_constant_smoke_d1.log`
   - `d_semantics/project2_gpu_native_rns_reciprocal_seed_smoke_d2.log`
   - `d_semantics/project2_gpu_native_rns_sqrt_seed_smoke_d3.log`
   - `d_semantics/project2_gpu_native_rns_division_smoke_d4.log`
   - `route_and_plan/project2_gpu_native_rns_pi_route_smoke_100m_e12.log`
   - `route_and_plan/project2_gpu_native_rns_pi_execution_plan_100m_e3.log`

## 分类整理

### A. Correction-domain 结果

这组文件对应 D5 / correction-domain 路线的验证与回归检查。

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `correction_domain/project2_correction_domain_modswitch_20260417_v2.log` | 推荐引用 | 当前 correction-domain modswitch 基线；已在 `freeze_checklist.md` 中引用。 |
| `correction_domain/project2_correction_domain_terms32_recheck_20260417_v1.log` | 保留 | `terms32` 之后的回归复查。 |
| `correction_domain/project2_correction_domain_recheck_after_pi_end_to_end_20260417.log` | 保留 | 接通 end-to-end `pi` 之后的 correction-domain 再验证。 |
| `correction_domain/project2_correction_domain_modswitch_20260417.log` | 已被后续版本替代 | 被 `..._v2.log` 覆盖。 |

### B. End-to-end smoke 结果

这组文件对应 E3 最早的完整 `pi` prefix 闭环验证。

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `end_to_end_smoke/project2_pi_end_to_end_smoke_default_20260417_v2.log` | 推荐引用 | `50 digits` smoke 的当前基线。 |
| `end_to_end_smoke/project2_pi_end_to_end_smoke_100digits_20260417_v2.log` | 推荐引用 | `100 digits` smoke 的当前基线。 |
| `end_to_end_smoke/project2_pi_end_to_end_smoke_default_20260417.log` | 已被后续版本替代 | 被 `..._v2.log` 覆盖。 |
| `end_to_end_smoke/project2_pi_end_to_end_smoke_100digits_20260417.log` | 已被后续版本替代 | 被 `..._v2.log` 覆盖。 |

### C. PQT tensor smoke 结果

这组文件对应不经过完整 tree merge 的 `P/Q/T` tensor 级验证。

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `pqt_tensor_smoke/project2_pi_pqt_tensor_terms4_modswitch_20260417.log` | 保留 | 最早的小规模 tensor smoke。 |
| `pqt_tensor_smoke/project2_pi_pqt_tensor_terms8_modswitch_20260417.log` | 保留 | `8 terms` tensor smoke。 |
| `pqt_tensor_smoke/project2_pi_pqt_tensor_terms16_modswitch_20260417_v3.log` | 推荐引用 | 当前 `16 terms` tensor 基线。 |
| `pqt_tensor_smoke/project2_pi_pqt_tensor_terms64_modswitch_20260417_v1.log` | 推荐引用 | `10` 模数深路径下的 `64 terms` tensor smoke。 |
| `pqt_tensor_smoke/project2_pi_pqt_tensor_terms16_modswitch_20260417.log` | 已被后续版本替代 | 被 `..._v3.log` 覆盖。 |
| `pqt_tensor_smoke/project2_pi_pqt_tensor_terms16_modswitch_20260417_v2.log` | 已被后续版本替代 | 被 `..._v3.log` 覆盖。 |

### D. PQT tree tensor smoke 结果

这组文件对应真正的 tensor-tree exact merge 验证，是 native RNS 主线最关键的一组结构性结果。

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms4_modswitch_20260417.log` | 保留 | 最早的 tree smoke 台阶。 |
| `pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms8_modswitch_20260417_v3.log` | 推荐引用 | 当前 `8 terms` tree 基线。 |
| `pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms16_modswitch_20260417_v4.log` | 推荐引用 | 当前 `16 terms` tree 基线。 |
| `pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms32_modswitch_20260417_v1.log` | 推荐引用 | `32 terms` exact tree milestone。 |
| `pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms47_modswitch_20260417_v1.log` | 推荐引用 | 旧 `7` 模数 ceiling 附近的 `47 terms` 里程碑。 |
| `pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms64_modswitch_20260417_v1.log` | 推荐引用 | `10` 模数扩展后的 `64 terms` 深树里程碑。 |
| `pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms8_modswitch_20260417.log` | 已被后续版本替代 | 被 `..._v3.log` 覆盖。 |
| `pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms16_modswitch_20260417.log` | 已被后续版本替代 | 被 `..._v4.log` 覆盖。 |
| `pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms16_modswitch_20260417_v2.log` | 已被后续版本替代 | 被 `..._v4.log` 覆盖。 |
| `pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms16_modswitch_20260417_v3.log` | 已被后续版本替代 | 被 `..._v4.log` 覆盖。 |

### E. End-to-end benchmark 结果

这组文件对应 E4 的 `digits/s` 端到端 benchmark，是当前最值得引用的结果集合。

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `end_to_end_benchmark/project2_pi_end_to_end_benchmark_400digits_20260417_v3.log` | 推荐引用 | `400 digits` 当前 benchmark 基线；已切到更干净的 `tree_validation_mode` 口径。 |
| `end_to_end_benchmark/project2_pi_end_to_end_benchmark_400digits_20260417_v3.csv` | 推荐引用 | 上述 log 的 CSV 版。 |
| `end_to_end_benchmark/project2_pi_end_to_end_benchmark_630digits_20260417_v1.log` | 推荐引用 | `47 terms / 630 digits` ceiling 附近 benchmark。 |
| `end_to_end_benchmark/project2_pi_end_to_end_benchmark_630digits_20260417_v1.csv` | 推荐引用 | 上述 log 的 CSV 版。 |
| `end_to_end_benchmark/project2_pi_end_to_end_benchmark_870digits_20260417_v1.log` | 推荐引用 | `10` 模数深路径下的 `64 terms / 870 digits` benchmark。 |
| `end_to_end_benchmark/project2_pi_end_to_end_benchmark_870digits_20260417_v1.csv` | 推荐引用 | 上述 log 的 CSV 版。 |
| `end_to_end_benchmark/project2_pi_end_to_end_benchmark_400digits_20260417_v2.log` | 已被后续版本替代 | 被 `..._v3.log` 覆盖。 |
| `end_to_end_benchmark/project2_pi_end_to_end_benchmark_400digits_20260417_v2.csv` | 已被后续版本替代 | 被 `..._v3.csv` 覆盖。 |

### F. Runtime smoke 与优化历史

这组文件主要是从 `HW/05/result` 导入回本项目的 C 阶段历史结果。目前文件较多，重点不是逐个阅读，而是把优化链条留在项目内。

- 当前目录：`runtime_smoke/`
- 当前文件数：`120`
- 推荐先看：
  - `runtime_smoke/project2_cuda_stage1_summary.csv`
  - `runtime_smoke/project2_gpu_native_rns_smoke_noverify_64x4096_c45lifecycle.csv`
  - `runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_c45lifecycle.csv`
  - `runtime_smoke/project2_gpu_native_rns_smoke_full_64x4096_c45lifecycle.csv`
  - `runtime_smoke/project2_gpu_native_rns_lifecycle_block_smoke_c45.log`
- 命名家族大致分为：
  - `project2_gpu_native_rns_smoke_{noverify|sampled|full}_*`
  - `project2_gpu_native_rns_smoke_*_c1midtile / c2finalfuse / c1fwdfinalfuse / c3pointtriplet / c45lifecycle`
  - `project2_gpu_native_rns_smoke_*_d1scaledconst / d2recipseed / d3sqrtseed / d4division / d5correction`

### G. D 阶段语义 smoke

这组文件保存 D1-D4 的直接语义验收日志。

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `d_semantics/project2_gpu_native_rns_scaled_constant_smoke_d1.log` | 推荐引用 | D1 scaled constant 语义验收。 |
| `d_semantics/project2_gpu_native_rns_reciprocal_seed_smoke_d2.log` | 推荐引用 | D2 reciprocal seed 语义验收。 |
| `d_semantics/project2_gpu_native_rns_sqrt_seed_smoke_d3.log` | 推荐引用 | D3 sqrt seed 语义验收。 |
| `d_semantics/project2_gpu_native_rns_division_smoke_d4.log` | 推荐引用 | D4 division prototype 语义验收。 |

### H. Route / Plan 结果

这组文件冻结了从 E1/E2 到 E3 early planning 的路线选择。

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `route_and_plan/project2_gpu_native_rns_pi_route_smoke_1m_e12.log` | 保留 | `1M digits` 路线选择 smoke。 |
| `route_and_plan/project2_gpu_native_rns_pi_route_smoke_100m_e12.log` | 推荐引用 | `100M digits` 路线选择 smoke。 |
| `route_and_plan/project2_gpu_native_rns_pi_execution_plan_1m_e3.log` | 保留 | `1M digits` partial execution plan。 |
| `route_and_plan/project2_gpu_native_rns_pi_execution_plan_100m_e3.log` | 推荐引用 | `100M digits` partial execution plan。 |

### I. Host-side PQT 语义结果

这组文件对应把 `P/Q/T` 对象语义先在 host exact arithmetic 上定死的阶段。

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `pqt_host_semantics/project2_gpu_native_rns_pi_pqt_smoke_8terms_e3.log` | 保留 | `8 terms` host PQT semantic smoke。 |
| `pqt_host_semantics/project2_gpu_native_rns_pi_pqt_smoke_16terms_e3.log` | 推荐引用 | `16 terms` host PQT semantic smoke。 |

## 当前建议保留的“主引用锚点”

如果后续文档、总结或提交材料只想引用最少的一组结果，建议保留下面这些文件作为主锚点：

1. `correction_domain/project2_correction_domain_modswitch_20260417_v2.log`
2. `correction_domain/project2_correction_domain_recheck_after_pi_end_to_end_20260417.log`
3. `end_to_end_smoke/project2_pi_end_to_end_smoke_default_20260417_v2.log`
4. `end_to_end_smoke/project2_pi_end_to_end_smoke_100digits_20260417_v2.log`
5. `pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms32_modswitch_20260417_v1.log`
6. `pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms47_modswitch_20260417_v1.log`
7. `pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms64_modswitch_20260417_v1.log`
8. `end_to_end_benchmark/project2_pi_end_to_end_benchmark_400digits_20260417_v3.log`
9. `end_to_end_benchmark/project2_pi_end_to_end_benchmark_400digits_20260417_v3.csv`
10. `end_to_end_benchmark/project2_pi_end_to_end_benchmark_630digits_20260417_v1.log`
11. `end_to_end_benchmark/project2_pi_end_to_end_benchmark_630digits_20260417_v1.csv`
12. `end_to_end_benchmark/project2_pi_end_to_end_benchmark_870digits_20260417_v1.log`
13. `end_to_end_benchmark/project2_pi_end_to_end_benchmark_870digits_20260417_v1.csv`

## 命名阅读说明

命名模式大致如下：

1. `correction_domain/project2_correction_domain_*`
   - correction-domain / residual-correction 相关验证
2. `end_to_end_smoke/project2_pi_end_to_end_smoke_*`
   - 只看 correctness 的端到端 prefix 验证
3. `pqt_tensor_smoke/project2_pi_pqt_tensor_*`
   - `P/Q/T` tensor 级验证
4. `pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_*`
   - `P/Q/T` tensor-tree exact merge 验证
5. `end_to_end_benchmark/project2_pi_end_to_end_benchmark_*`
   - 含 timing 和 `digits/s` 的端到端 benchmark
6. `runtime_smoke/project2_gpu_native_rns_smoke_*`
   - runtime smoke、优化历史与 staged benchmark 结果
7. `d_semantics/project2_gpu_native_rns_{scaled_constant|reciprocal_seed|sqrt_seed|division}_smoke_*`
   - D1-D4 直接语义 smoke
8. `route_and_plan/project2_gpu_native_rns_{pi_route|pi_execution_plan}_*`
   - 路线选择与执行计划 smoke
9. `pqt_host_semantics/project2_gpu_native_rns_pi_pqt_smoke_*`
   - host-side PQT 对象语义 smoke

后缀说明：

1. 无版本后缀：通常是该系列最早一次落地结果
2. `_v2 / _v3 / _v4`：同一家族的后续改进版
3. 规则上优先看版本号更高的文件，除非本索引显式说明只是“并行保留”而非替代

## 目录使用说明

当前 `project2_gpu_native_rns/README.md` 和 `docs/freeze_checklist.md` 里的相对路径都已经同步更新到新的分类目录。
后续新增结果时，建议继续保持“按类别落子目录、文件名保留原始实验语义”的方式，避免再次回到平铺堆放。
