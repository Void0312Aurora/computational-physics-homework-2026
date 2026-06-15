# Project 2 GPU-Native RNS Reboot

这个子项目是对 `HW/05` 里旧有 `limb + trim/compare + host fallback` 路线的主动重启。目标不是继续沿着保守路径打补丁，而是直接建立一条更激进的原生 GPU 架构：

1. 放弃把“大整数语义”建立在 `base-2^16` carry 链上。
2. 采用 `RNS/CRT` 作为 GPU 上的基础精确表示，尽量把 carry、trim、compare 从热路径拿掉。
3. 把数值主体和元数据都当成 device-resident 对象管理，而不是只把数组放到 GPU、语义仍留在 host。

## 目录

- `include/project2_gpu_native_rns/rns_runtime.cuh`
  原生 CUDA RNS runtime 的公开接口。
- `src/rns_runtime.cu`
  device-resident RNS tensor、编码、点加/减/乘以及 host 侧 CRT reconstruction。
- `src/main.cu`
  smoke test 入口。
- `docs/architecture.md`
  激进架构蓝图与下一阶段模块拆解。
- `docs/freeze_checklist.md`
  当前实现冻结点、任务拆解、变更控制与后续唯一主线。
- `result/README.md`
  本地结果索引；当前项目结果已经按类别收拢到子项目自己的 `result/` 树内。

## 结果布局

当前 `project2_gpu_native_rns` 已改成项目内自包含的结果布局：

1. 新跑出来的 generic smoke 默认写到 `result/runtime_smoke/`
2. 端到端 `pi` benchmark 默认写到 `result/end_to_end_benchmark/`
3. 先前散落在 `HW/05/result` 的 native-RNS 历史结果，已经按类别导入到本项目自己的 `result/` 子目录

如果需要快速找结果，优先看 `result/README.md`。

## 当前已经落地的内容

当前版本不是完整的 `pi` 求解器，而是新的 GPU-native 表示层闭环：

1. `DeviceRnsTensor`
   - 布局：`residues[modulus][value][slot]`
   - 元数据：`sign / logical_slots / scale_bits / level`
2. 原生 CUDA kernel
   - `encode_u64`
   - `pointwise_add`
   - `pointwise_sub`
   - `pointwise_mul`
   - `pairwise_convolution`
     - `ntt_shared` for small padded sizes
     - `ntt_global` staged global-memory path for larger padded sizes
3. host 侧 exact reconstruction
   - 使用 Garner/CRT 把 residue tensor 精确重建回标量，便于验证
4. D1 语义接口
   - `make_scaled_constant_tensor`
   - `encode_scaled_constant_blocks`
   - `drop_moduli_prefix` 之后仍可继续接 block add/mul/convolution
5. D2 语义接口
   - `make_reciprocal_seed_tensor`
   - `encode_reciprocal_seed_blocks`
   - `denominator * reciprocal_seed` 与 `scaled-one` 的误差可以继续留在同一条 RNS / `scale_bits` 域里
6. D3 语义接口
   - `make_sqrt_seed_tensor`
   - `encode_sqrt_seed_blocks`
   - `sqrt_seed^2` 与 `scaled-radicand` 的误差可以继续留在同一条 RNS / `scale_bits` 域里
7. D4 语义接口
   - `make_division_quotient_tensor`
   - `encode_division_quotient_blocks`
   - `quotient * denominator` 与 `scaled-numerator` 的余数可以继续留在同一条 RNS / `scale_bits` 域里
8. D5 共享 correction 语义
   - `compute_residual_correction`
   - `--correction-domain-smoke`
   - reciprocal / sqrt / division 的 residual 与 corrected 都可以通过同一套 API 保持在目标 RNS / `scale_bits` 域里

这意味着新分支已经不是“还没想好表示”的空壳，而是已经有了第一层真正独立于旧 limb 路线的运行时核心。

## 为什么选这条路

旧路线的主要问题不是 kernel 数量不够，而是：

1. `trim / compare / effective_length` 这类长度语义天然要求回 host 决策。
2. reciprocal / sqrt / division 仍不断回落到 `mpz`。
3. carry/borrow 语义会让 GPU kernel 反复遭遇串行依赖。

`RNS/CRT` 路线的核心优点是：

1. residue 通道之间天然并行。
2. pointwise add/sub/mul 完全不需要 carry。
3. 后续可以自然接 `NTT/cuFFT` 风格的卷积与乘法层。

## 当前 smoke test 覆盖

当前 smoke test 会：

1. 在 GPU 上编码一批 30-bit 标量到 RNS tensor。
2. 在 GPU 上执行 pointwise add/sub/mul。
3. 在 GPU 上执行 `slot_count > 1` 的 NTT 卷积。
4. 可选地以 `none / sampled / full` 三种模式执行验证。
5. `full` 模式会下载全量结果并做 host 侧 exact CRT reconstruction。
6. `sampled` 模式只抽取少量 residue 样本回 host，并对抽样点做精确校验。

当前版本的 smoke 程序还会自动把结果写到：

- `result/runtime_smoke/project2_gpu_native_rns_smoke.log`
- `result/runtime_smoke/project2_gpu_native_rns_smoke.csv`

也支持直接改规模做微基准：

```bash
./bin/project2_gpu_native_rns_smoke --value-count 1024 --slot-count 8 --iterations 10
```

纯 GPU 跑分模式：

```bash
./bin/project2_gpu_native_rns_smoke --value-count 256 --slot-count 32 --iterations 10 --input-bits 28 --no-verify --output-tag noverify_256x32
```

或者：

```bash
make run-noverify
```

更大规模的 global-NTT 跑分：

```bash
make run-global-noverify
```

抽样验证模式：

```bash
./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode sampled --verify-samples 64 --output-tag sampled_64x4096
```

默认 smoke（`make gpu_native_rns`，开启 exact verification）：

| 指标 | 数值 |
| --- | --- |
| validated scalar slots | `1024` |
| validated convolution coefficients | `1792` |
| convolution algorithm | `ntt_shared` |
| convolution NTT size | `8` |
| cold kernel time | `337.311676 ms` |
| avg encode time | `0.046784 ms` |
| avg pointwise time | `0.022490 ms` |
| avg convolution time | `0.165069 ms` |
| avg kernel time | `0.234342 ms` |
| avg download + CRT reconstruction | `1.486925 ms` |
| avg end-to-end | `1.739813 ms` |
| avg scalar slots per second, end-to-end | `588568.93` |
| avg convolution coefficients per second, kernel-only | `10856079.45` |
| avg pipeline residue values per second, kernel-only | `88485909.21` |
| avg download/kernel ratio | `6.35x` |

额外的 no-verify 跑分样例（`256 x 32`，纯 GPU benchmark）：

- 结果文件：
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_noverify_256x32.log`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_noverify_256x32.csv`
- `convolution_ntt_size = 64`
- `avg_kernel_ms = 0.515018 ms`
- `avg_end_to_end_ms = 0.526167 ms`
- `avg_scalar_slots_per_second_e2e = 15569189.58`
- `avg_pipeline_residue_values_per_second_kernel = 332540092.57`

大规模 global-NTT full correctness 样例（`2 x 3000`）：

- 结果文件：
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_full_2x3000.log`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_full_2x3000.csv`
- `convolution_algorithm = ntt_global`
- `convolution_ntt_size = 8192`
- `verification_mode = full`
- `checked_scalar_slots = 6000`
- `checked_convolution_coefficients = 11998`
- `avg_kernel_ms = 0.253792`
- `avg_download_reconstruct_ms = 8.457942`
- `avg_end_to_end_ms = 19.076323`
- `status = ok`

同一规模的 sampled verification 样例（`2 x 3000`, `verify_samples = 64`）：

- 结果文件：
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_2x3000.log`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_2x3000.csv`
- `verification_mode = sampled`
- `verification_sample_count = 64`
- `checked_scalar_slots = 64`
- `checked_convolution_coefficients = 64`
- `avg_kernel_ms = 0.163808`
- `avg_download_reconstruct_ms = 0.108521`
- `avg_end_to_end_ms = 0.343013`
- `status = ok`

大规模 global-NTT no-verify 跑分样例（`64 x 4096`）：

- 结果文件：
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_noverify_64x4096_c45lifecycle.log`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_noverify_64x4096_c45lifecycle.csv`
- `convolution_algorithm = ntt_global`
- `input_staging_mode = device_reuse`
- `one_time_input_upload_ms = 0.908806`
- `convolution_ntt_size = 8192`
- `verification_mode = none`
- `avg_encode_ms = 0.030106`
- `avg_convolution_ms = 0.503104`
- `avg_kernel_ms = 0.585498`
- `avg_end_to_end_ms = 0.599273`
- `avg_scalar_slots_per_second_e2e = 437436986.11`
- `avg_pipeline_residue_values_per_second_kernel = 9401971836.37`

大规模 global-NTT sampled verification 样例（`64 x 4096`, `verify_samples = 64`）：

- 结果文件：
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_c45lifecycle.log`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_c45lifecycle.csv`
- `input_staging_mode = device_reuse`
- `one_time_input_upload_ms = 0.880518`
- `verification_mode = sampled`
- `checked_scalar_slots = 64`
- `checked_convolution_coefficients = 64`
- `avg_encode_ms = 0.033005`
- `avg_convolution_ms = 0.500941`
- `avg_kernel_ms = 0.585325`
- `avg_download_reconstruct_ms = 0.090364`
- `avg_end_to_end_ms = 0.779366`
- `avg_scalar_slots_per_second_e2e = 336355620.52`
- `avg_download_over_kernel_ratio = 0.154383`

大规模 global-NTT full verification 样例（`64 x 4096`）：

- 结果文件：
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_full_64x4096_c45lifecycle.log`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_full_64x4096_c45lifecycle.csv`
- `input_staging_mode = device_reuse`
- `one_time_input_upload_ms = 0.941674`
- `verification_mode = full`
- `checked_scalar_slots = 262144`
- `checked_convolution_coefficients = 524224`
- `avg_convolution_ms = 0.503155`
- `avg_kernel_ms = 0.689914`
- `avg_download_reconstruct_ms = 354.231544`
- `avg_end_to_end_ms = 990.866828`
- `avg_scalar_slots_per_second_e2e = 264560.27`
- `avg_download_over_kernel_ratio = 513.44x`

reduced-level lifecycle/block smoke（`2 x 8`, level=`2`）：

- 结果文件：
  - `result/runtime_smoke/project2_gpu_native_rns_lifecycle_block_smoke_c45.log`
- `lifecycle_block_smoke_status = ok`
- `active_modulus_count = 2`
- `product_scale_bits = 18`
- `convolution_logical_slots = 15`

D1 scaled-constant smoke（direct native-RNS constant construction）：

- 结果文件：
  - `result/d_semantics/project2_gpu_native_rns_scaled_constant_smoke_d1.log`
- 命令：
  - `./bin/project2_gpu_native_rns_smoke --scaled-constant-smoke`
- `scaled_constant_smoke_status = ok`
- `full_modulus_count = 3`
- `active_modulus_count = 2`
- `dropped_scale_bits = 9`
- `level_scale_bits = 11`
- `product_scale_bits = 20`
- `convolution_logical_slots = 7`

这条 smoke 不是吞吐 benchmark，而是 D 阶段第一刀的语义验收：它验证 `make_scaled_constant_tensor / encode_scaled_constant_blocks` 已经能直接在 native RNS runtime 里构造 scaled constant，并把它接到 reduced-level lifecycle 与 formal polynomial/block 运算路径上。

D2 reciprocal-seed smoke（direct native-RNS reciprocal prototype）：

- 结果文件：
  - `result/d_semantics/project2_gpu_native_rns_reciprocal_seed_smoke_d2.log`
- 命令：
  - `./bin/project2_gpu_native_rns_smoke --reciprocal-seed-smoke`
- `reciprocal_seed_smoke_status = ok`
- `denominator_coefficient = 13`
- `reciprocal_seed_coefficient = 1290555`
- `product_coefficient = 16777215`
- `ideal_scaled_one_coefficient = 16777216`
- `error_coefficient = 1`
- `denominator_scale_bits = 7`
- `reciprocal_scale_bits = 17`
- `target_product_scale_bits = 24`

这条 smoke 对应 D 阶段第二刀的最小 reciprocal 原型：当前先覆盖 positive monomial scaled-constant，直接构造 reciprocal seed，并验证 `denominator * seed + error = scaled-one` 可以在同一个 reduced-level RNS / `scale_bits` 语义域中闭环。它还不是任意高精度的 full reciprocal/Newton 主线，但已经把 Newton 下一步真正需要的 seed、product、ideal、error 这几类对象迁进了 native RNS。

D3 sqrt-seed smoke（direct native-RNS sqrt prototype）：

- 结果文件：
  - `result/d_semantics/project2_gpu_native_rns_sqrt_seed_smoke_d3.log`
- 命令：
  - `./bin/project2_gpu_native_rns_smoke --sqrt-seed-smoke`
- `sqrt_seed_smoke_status = ok`
- `radicand_coefficient = 10005`
- `sqrt_seed_coefficient = 104883811`
- `square_coefficient = 11000613809883721`
- `target_scaled_radicand_coefficient = 11000613835898880`
- `error_coefficient = 26015159`
- `radicand_scale_bits = 0`
- `sqrt_scale_bits = 20`
- `target_square_scale_bits = 40`

这条 smoke 对应 D 阶段第三刀的最小 sqrt 原型：当前先覆盖 positive monomial scaled-constant，并限制在 even `radicand_scale_bits` 与 `uint64` 可重建验证范围内，直接构造 `floor(sqrt(x) * 2^k)` 的 sqrt seed，再验证 `sqrt_seed^2 + error = scaled_radicand` 可以在同一个 reduced-level RNS / `scale_bits` 语义域中闭环。它还不是任意高精度的 full Newton sqrt，但已经把后续 `sqrt(10005)` 这类路径真正需要的 seed、square、target、error 对象迁进了 native RNS。

D4 division smoke（direct native-RNS division prototype）：

- 结果文件：
  - `result/d_semantics/project2_gpu_native_rns_division_smoke_d4.log`
- 命令：
  - `./bin/project2_gpu_native_rns_smoke --division-smoke`
- `division_smoke_status = ok`
- `numerator_coefficient = 355`
- `denominator_coefficient = 113`
- `quotient_coefficient = 1647099`
- `product_coefficient = 186122187`
- `scaled_numerator_coefficient = 186122240`
- `remainder_coefficient = 53`
- `numerator_scale_bits = 5`
- `denominator_scale_bits = 3`
- `quotient_scale_bits = 21`
- `target_product_scale_bits = 24`

这条 smoke 对应 D 阶段第四刀的最小 division 原型：当前先覆盖 positive monomial scaled-constant，并限制在 `uint64` 可重建验证范围内，直接构造 `floor((numerator / denominator) * 2^k)` 的 quotient prototype，再验证 `quotient * denominator + remainder = scaled_numerator` 可以在同一个 reduced-level RNS / `scale_bits` 语义域中闭环。它还不是完整的 Newton division，但已经把 quotient、product、scaled-numerator、remainder 这些后续除法主线需要的对象迁进了 native RNS。

D5 correction-domain smoke（shared native-RNS residual/correction semantics）：

- 结果文件：
  - `result/correction_domain/project2_gpu_native_rns_correction_domain_smoke_d5.log`
- 命令：
  - `./bin/project2_gpu_native_rns_smoke --correction-domain-smoke`
- `correction_domain_smoke_status = ok`
- `shared_correction_api = compute_residual_correction`
- `active_modulus_count = 2`
- `reciprocal_residual_coefficient = 1`
- `sqrt_residual_coefficient = 26015159`
- `division_residual_coefficient = 53`

这条 smoke 对应 D 阶段第五刀：不是再引入新的算子，而是把 reciprocal、sqrt、division 三类路径里原本分散的 correction 逻辑统一成一个共享接口 `compute_residual_correction(target, approximate, residual, corrected)`。当前 smoke 显式验证这三类 case 的 residual 与 corrected 都保持在与 target 相同的 `moduli / logical_slots / scale_bits / level / sign` 域里，因此 D 阶段最小函数语义闭环已经形成。

E1/E2 pi-route smoke（full-pipeline route selection on native RNS）：

- 结果文件：
  - `result/route_and_plan/project2_gpu_native_rns_pi_route_smoke_1m_e12.log`
  - `result/route_and_plan/project2_gpu_native_rns_pi_route_smoke_100m_e12.log`
- 命令：
  - `./bin/project2_gpu_native_rns_smoke --pi-route-smoke --pi-digits 1000000`
  - `./bin/project2_gpu_native_rns_smoke --pi-route-smoke --pi-digits 100000000`
- `chosen_route = chudnovsky_binary_splitting`
- `preferred_multiply_backbone = exact_multiply_tree_on_native_rns_ntt`
- `final_nonmultiply_steps = 2`
- `1M digits -> chudnovsky_terms = 70516, binary_split_depth = 17`
- `100M digits -> chudnovsky_terms = 7051370, binary_split_depth = 23`
- `rejected_route = agm_like_constant_algorithm`

这条 smoke 对应 E 阶段前两刀：它把“完整 `pi` 路线该怎么选”从口头判断收口成了可执行的冻结结论。当前主线明确选择 `exact Chudnovsky + binary splitting`，因为这条路径能把绝大多数高精度成本重新压回当前已经最强的 `exact multiply / NTT` 主骨架，而不是把系统压力放到 repeated full-precision reciprocal / sqrt / division 上。对于当前这个 native RNS reboot，这比转向 AGM-like 常数算法更有现实推进价值。

E3 partial pi-execution-plan smoke（merge-tree execution skeleton on native RNS）：

- 结果文件：
  - `result/route_and_plan/project2_gpu_native_rns_pi_execution_plan_1m_e3.log`
  - `result/route_and_plan/project2_gpu_native_rns_pi_execution_plan_100m_e3.log`
- 命令：
  - `./bin/project2_gpu_native_rns_smoke --pi-execution-plan-smoke --pi-digits 1000000`
  - `./bin/project2_gpu_native_rns_smoke --pi-execution-plan-smoke --pi-digits 100000000`
- `1M digits -> leaf_terms_per_task = 32, leaf_task_count = 2204, peak_ntt_size = 262144`
- `100M digits -> leaf_terms_per_task = 2048, leaf_task_count = 3444, peak_ntt_size = 33554432`
- `modulus_dynamic_range_bits = 91`
- `100M root_safe_limb_bits = 33`

这条 smoke 对应 E3 的第一刀：它还没有真正开始算 `pi`，但已经把 binary-splitting 的 merge-tree 规模和 exact-multiply 调度压力显式化了。最重要的工程结论有两个。第一，`100M` digits 的 root multiply 已经对应到 `~10.38M slots` 和 `2^25` 级 NTT，这说明后续主战场会是大规模 merge-tree multiply 的端到端落地。第二，当前三模数组合对 `32-bit limbs` 仍然可用，但在 `100M` root 上已经只剩 `33` bits 安全余量，继续冲更高位数时就不能再假装 base/模数组合不是问题。

E3 P/Q/T semantics smoke（exact Chudnovsky leaf/merge semantics on host）：

- 结果文件：
  - `result/pqt_host_semantics/project2_gpu_native_rns_pi_pqt_smoke_8terms_e3.log`
  - `result/pqt_host_semantics/project2_gpu_native_rns_pi_pqt_smoke_16terms_e3.log`
- 命令：
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-smoke --pi-terms 8`
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-smoke --pi-terms 16`
- `8 terms -> node_count = 15, root_t_bits = 434, root_ntt_size = 32`
- `16 terms -> node_count = 31, root_t_bits = 944, root_ntt_size = 64`
- `merge_formula = T_left*Q_right + P_left*T_right`

这条 smoke 对应 E3 的第二刀：它不追求吞吐，而是把 `P/Q/T` 的对象语义定死。当前已经可以在 host-side exact arithmetic 下，直接验证 `leaf(0)`、`leaf(k>0)` 和 merge recurrence 的正确性。这听起来朴素，但意义很大，因为下一步一旦把这些对象迁到 native-RNS tensor 和 GPU merge execution 上，我们至少不会再反复摇摆“节点到底该长什么样、merge 公式到底该怎么接”。

E3 P/Q/T tensor merge smoke（first native-RNS tensor merge execution）：

- 结果文件：
  - `result/pqt_tensor_smoke/project2_gpu_native_rns_pi_pqt_tensor_smoke_2terms_e3.log`
- 命令：
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-tensor-smoke --pi-terms 2`
- `merge_interval = 0:2`
- `chosen_limb_bits = 16`
- `uses_signed_residue_encoding = 1`
- `p_output_slot_count = 1`
- `q_output_slot_count = 4`
- `t_output_slot_count = 5`
- `t_output_ntt_size = 8`
- `metadata_ok = 1`
- `p_match = 1, q_match = 1, t_match = 1`

这条 smoke 对应 E3 的第三刀：它第一次不再停留在 host-side exact 语义，而是真的把一个 Chudnovsky 两叶子 merge 节点搬到 native-RNS tensor 上执行。当前做法是把 `P/Q/T` 编码成 `base-2^16` 的 signed limb polynomial，再把负系数直接映射到模数域里，用现有 convolution/add kernel 跑出 `P=P_left*P_right`、`Q=Q_left*Q_right`、`T=T_left*Q_right + P_left*T_right`，最后在 host 端用中心提升把系数精确验回来。它还不是完整的 bottom-up merge tree，但已经证明当前 runtime 可以承接真实的 `P/Q/T` merge 语义，而不是只会做 route planning 或 host-side semantic smoke。

E3 P/Q/T tensor tree smoke（multi-node bottom-up merge execution）：

- 结果文件：
  - `result/pqt_tree_tensor_smoke/project2_gpu_native_rns_pi_pqt_tensor_tree_smoke_4terms_e3.log`
  - `result/pqt_tree_tensor_smoke/project2_gpu_native_rns_pi_pqt_tensor_tree_smoke_8terms_e3.log`
  - `result/pqt_tree_tensor_smoke/project2_gpu_native_rns_pi_pqt_tensor_tree_smoke_16terms_e3.log`
- 命令：
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-tree-tensor-smoke --pi-terms 4`
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-tree-tensor-smoke --pi-terms 8`
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-tree-tensor-smoke --pi-terms 16`
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-tree-tensor-smoke --pi-terms 32`
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-tree-tensor-smoke --pi-terms 47`
- `4 terms -> chosen_limb_bits = 16, root_t_slot_count = 11, peak_ntt_size = 16, p/q/t match = 1/1/1`
- `8 terms -> chosen_limb_bits = 8, root_t_slot_count = 51, peak_ntt_size = 64, p/q/t match = 1/1/1`
- `16 terms -> chosen_limb_bits = 4, modulus_count = 7, root_t_slot_count = 230, peak_ntt_size = 256, p/q/t match = 1/1/1`
- `32 terms -> chosen_limb_bits = 2, modulus_count = 7, root_t_slot_count = 993, peak_ntt_size = 1024, p/q/t match = 1/1/1`
- `47 terms -> chosen_limb_bits = 1, modulus_count = 7, root_t_slot_count = 3031, peak_ntt_size = 4096, p/q/t match = 1/1/1`
- `64 terms -> chosen_limb_bits = 1, modulus_count = 10, root_t_slot_count = 4224, peak_ntt_size = 8192, p/q/t match = 1/1/1`

这条 smoke 对应 E3 的第四刀：它第一次把 leaf tensor construction、device-side zero-pad 和多节点 bottom-up merge execution 接到了同一条 native-RNS 流水线里。最新一轮扩模之后，当前可用模数组已经先扩展到 `7` 个 NTT-friendly primes，再继续扩展到 `10` 个；默认主线仍保持 `3` 模数，`9..47 terms` 的深树 smoke 走 `7` 模数实验路径，而 `48..64 terms` 会显式切到 `10` 模数深树实验路径。这样一来，`16 terms` 在 `4-bit limbs` 下可以保持 exact，`32 terms` 在 `2-bit limbs` 下保持 exact，`47 terms` 在 `1-bit limbs + 7 moduli` 下保持 exact，而这轮继续推进后，`64 terms` 也已经在 `1-bit limbs + 10 moduli` 下保持 exact；对应的 CRT 动态范围从三模数组合约 `90.47 bits` 提升到七模数组合约 `212.35 bits`，再提升到十模数组合约 `302.73 bits`。这轮还顺手修掉了一个真正卡住 10 模数路径的实现 bug：shared-NTT path 里的 `primitive_root_for_modulus_index()` 仍然只覆盖旧的前 `7` 个模数，新增模数会错误落到 `root = 1`，先在 `term_count = 64` 的两叶子 tensor smoke 上把 `T` 路径打坏。把这个映射补齐之后，`64 terms` 的单节点和整树 exactness 都已经恢复。因此当前阻塞已经再次前移：如果还想继续越过 `64 terms`，下一步不再是修补现有 10 模数路径，而是要继续增加动态范围或改变更深树的表示/收缩策略。

E3 end-to-end `pi` host-closure smoke（first true end-to-end `pi` prefix closure）：

- 结果文件：
  - `result/end_to_end_smoke/project2_pi_end_to_end_smoke_default_20260417_v2.log`
  - `result/end_to_end_smoke/project2_pi_end_to_end_smoke_100digits_20260417_v2.log`
- 命令：
  - `./bin/project2_gpu_native_rns_smoke --pi-end-to-end-smoke`
  - `./bin/project2_gpu_native_rns_smoke --pi-end-to-end-smoke --pi-digits 100`
- `50 digits -> term_count = 8, reference_prefix_digits_checked = 50, chosen_limb_bits = 8, modulus_count = 3, peak_ntt_size = 64, prefix_match = 1`
- `100 digits -> term_count = 10, reference_prefix_digits_checked = 100, chosen_limb_bits = 4, modulus_count = 7, peak_ntt_size = 256, prefix_match = 1`

这条 smoke 对应 E3 的第五刀：它第一次不再停留在 `P/Q/T` 子流程，而是把 GPU tensor-tree 产出的 root `Q/T` 继续接到了最终 Chudnovsky 收口上。当前做法仍然诚实地把 closing 留在 host 侧：用仓内 `HostBigInt` 补齐最小 `isqrt/division` 路径，计算 `floor(pi * 10^digits)`，再和内置参考前缀逐位对齐。它还不是最终想要的 native-RNS final `sqrt/division`，但它已经证明当前主线不只是能把 multiply tree 算对，而是真的已经有了一条完整的端到端 `pi` 前缀闭环。

E4 first end-to-end `pi digits/s` benchmark（first timed end-to-end benchmark harness）：

- 结果文件：
  - `result/end_to_end_benchmark/project2_pi_end_to_end_benchmark_400digits_20260417_v3.log`
  - `result/end_to_end_benchmark/project2_pi_end_to_end_benchmark_400digits_20260417_v3.csv`
- 命令：
  - `./bin/project2_gpu_native_rns_smoke --pi-end-to-end-benchmark --pi-digits 400 --iterations 3 --output-tag 400digits_20260417_v3`
- `term_count = 31, required_terms = 31, reference_prefix_digits_checked = 100`
- `chosen_limb_bits = 2, modulus_count = 7, peak_ntt_size = 1024`
- `tree_validation_mode = skip_intermediate_metadata_downloads_keep_root_exact_match`
- `avg_tree_execution_ms = 57.2298`
- `avg_host_closure_ms = 2.70191`
- `avg_end_to_end_ms = 59.9317`
- `avg_digits_per_second_e2e = 6674.26`
- `prefix_match = 1`

这条 benchmark 对应 E4 的第一刀：它第一次把这条主线真正变成了可计时的端到端 `pi` 流水线，而不再只是“能不能算对”的 smoke。最新一轮又把 benchmark 路径里的中间节点 metadata 下载从 hot path 里拔掉，只保留 root exactness，这样 `400 digits / 31 terms` 的 benchmark 更接近真正要看的 tree 执行本体，端到端吞吐也从早先的 `6.43k digits/s` 提到约 `6.67k digits/s`。

E4 deeper exact-window benchmark（near current 7-modulus exact ceiling）：

- 结果文件：
  - `result/end_to_end_benchmark/project2_pi_end_to_end_benchmark_630digits_20260417_v1.log`
  - `result/end_to_end_benchmark/project2_pi_end_to_end_benchmark_630digits_20260417_v1.csv`
- 命令：
  - `./bin/project2_gpu_native_rns_smoke --pi-end-to-end-benchmark --pi-digits 630 --iterations 3 --output-tag 630digits_20260417_v1`
- `term_count = 47, required_terms = 47, reference_prefix_digits_checked = 100`
- `chosen_limb_bits = 1, modulus_count = 7, peak_ntt_size = 4096`
- `tree_validation_mode = skip_intermediate_metadata_downloads_keep_root_exact_match`
- `avg_tree_execution_ms = 110.515`
- `avg_host_closure_ms = 5.93018`
- `avg_end_to_end_ms = 116.445`
- `avg_digits_per_second_e2e = 5410.26`
- `prefix_match = 1`

这条 benchmark 把 E4 继续推到了当前 7 模数 exact ceiling 附近。它给出的信息很直接：即使把 target 拉到 `630 digits / 47 terms`，当前时间大头仍然是 `tensor tree`，而不是 host-side closing；`avg_host_closure_ms` 只有约 `5.93 ms`，而 tree 本体已经来到约 `110.5 ms`。所以如果我们还想继续往上冲，眼下更关键的不是继续抠 closing 常数，而是解决 `47 terms` 之后的模数动态范围和更深 tree 的执行组织问题。

E4 max current exact-window benchmark（10-modulus deep exact path）：

- 结果文件：
  - `result/end_to_end_benchmark/project2_pi_end_to_end_benchmark_870digits_20260417_v1.log`
  - `result/end_to_end_benchmark/project2_pi_end_to_end_benchmark_870digits_20260417_v1.csv`
- 命令：
  - `./bin/project2_gpu_native_rns_smoke --pi-end-to-end-benchmark --pi-digits 870 --iterations 3 --output-tag 870digits_20260417_v1`
- `term_count = 64, required_terms = 64, reference_prefix_digits_checked = 100`
- `chosen_limb_bits = 1, modulus_count = 10, peak_ntt_size = 8192`
- `tree_validation_mode = skip_intermediate_metadata_downloads_keep_root_exact_match`
- `avg_tree_execution_ms = 189.672`
- `avg_host_closure_ms = 14.3921`
- `avg_end_to_end_ms = 204.064`
- `avg_digits_per_second_e2e = 4263.36`
- `prefix_match = 1`

这条 benchmark 把 E4 直接推进到了当前这套 exact 路径的上边界。它的意义不只是“又多了一条结果”，而是说明新补上的 `10` 模数深树路径已经不只是停留在 tensor-tree smoke，而是真的已经能支撑到 `870 digits / 64 terms` 的端到端 `pi` benchmark。与此同时，它也把下一个边界讲得更清楚了：`900 digits` 已经需要 `66 terms`，所以当前这套路径的现实 ceiling 现在不再是 `47 terms`，而是已经提升到了 `64 terms / 870 digits` 附近。

最新这几轮 `ntt_global` 优化主要来自十点：

1. 每个 stage 的 twiddle 不再在 butterfly 内部反复做 `pow_mod`，而是预计算后放进 constant memory。
2. forward transform 的 `lhs/rhs` 两路 stage 合并进同一个 kernel，减少 launch 和重复 twiddle 开销。
3. global-memory NTT 的临时 workspace 改成持久复用，不再为每次卷积重复 `cudaMalloc/cudaFree`。
4. `lhs/rhs` 的 pad + bit-reverse 输入预处理合并成一次双路 kernel，进一步压低 launch 和准备开销。
5. `encode_u64` 前端新增持久 upload workspace，并把 `lhs/rhs` 的 residue 编码与 metadata 初始化合并成双路 kernel，减少前端两次 `cudaMalloc/cudaFree` 和重复 launch。
6. `length <= 8192` 的 forward/inverse stage twiddle schedule 现在会一次构建后常驻 device cache，主 benchmark 上不再每次卷积都重新做 stage 级 twiddle upload。
7. 最新这一轮把 `pointwise multiply` 和逆变换前的整块 `bit_reverse_inplace` 合并成一个 kernel，避免在 global-NTT 路径里再做一次完整的 transform-wide 读写重排。
8. 紧接着又把 smoke benchmark 的输入准备拆成“一次性 host->device 上传”与“循环内 device-side residue encode”两段；测量环节只复用已上传的 device 输入，并把一次性准备时间单独记成 `one_time_input_upload_ms`。
9. 再往前一步，把 global NTT 中 `length <= 1024` 的初始 stages 改成 shared-memory tile 内的多层融合：forward dual 一次做完 tile 内前几层，inverse 单路也同理，再回到剩余大 stage 的 global-memory 路径。
10. 当前这一刀再把 inverse 路径的最后一层 stage 与 `1 / ntt_size` scale-out 融成一个 kernel，避免 final stage 结束后再额外做一次 full-prefix 读写。

在 `64 x 4096` 这个样例上，当前 C 阶段验收版 `c45lifecycle` 的 `sampled(64)` 数据来到：

- 结果文件：
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_c45lifecycle.log`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_c45lifecycle.csv`
- `one_time_input_upload_ms = 0.880518`
- `avg_encode_ms = 0.033005`
- `avg_convolution_ms = 0.500941`
- `avg_kernel_ms = 0.585325`
- `avg_end_to_end_ms = 0.779366`
- `avg_scalar_slots_per_second_e2e = 336355620.52`

相对早先 `fusedstages` 的 `sampled(64)` 基线，这条最终接受的 C 阶段主线把 `avg_convolution_ms` 从 `0.560947 ms` 压到 `0.500941 ms`，`avg_kernel_ms` 从 `0.650778 ms` 压到 `0.585325 ms`，`avg_end_to_end_ms` 从 `0.913517 ms` 压到 `0.779366 ms`；对应的端到端吞吐从约 `286.96M` 提升到约 `336.36M scalar slots/s`。而且这次接受的不只是 hot-path 提速，还同时补齐了 level lifecycle 与 polynomial/block facade。

这几组数据合起来说明：

1. `full` verify 下，真正的大头已经不是 GPU kernel，而是 host 侧下载与 CRT 精确重建；`64 x 4096` 上这部分现在达到 kernel 的约 `513x`。
2. `sampled` verify 现在能保留 smoke 的正确性护栏，同时把验证成本压到更接近 `no-verify`；同一个 `64 x 4096` case 上，端到端吞吐已经提升到约 `336.4M scalar slots/s`，而 `no-verify` 达到约 `437.4M scalar slots/s`。
3. `validated_*` 代表整批 workload 规模，`checked_*` 才代表这次实际核查了多少点；这样可以在不改吞吐统计口径的前提下引入抽样验证。
4. `avg_encode_ms` 现在表示循环内的 device-side residue encode + metadata 初始化；一次性的输入上传成本已经单独记录为 `one_time_input_upload_ms`。
5. 在 shared-memory 融合初始 stages、forward/inverse final-stage 融合、sampled gather batching 之后，再补上 lifecycle/block 语义接口，`avg_convolution_ms` 已经下降到 `0.50 ms` 级，说明当前 aggressive 路线不只是统计口径改变，而是卷积核本体也在继续得到实打实的结构性收益。
6. 在 `slot_count=4` 这种很小的卷积上，NTT 的常数未必优于朴素卷积；但一旦放大到更像 block multiply 的规模，NTT 路径才开始有研究意义。
7. `ntt_global` 已经不再停留在“理论下一步”，而是有了真实通过 `full` 和 `sampled` 两种验证路径的 `8192` padded size 主线。

这些吞吐指标的含义如下：

1. `avg scalar slots per second, end-to-end`
   - 用 `validated_scalar_slots / avg_end_to_end_ms` 得到
   - 衡量从输入编码到下载验证整条 smoke 流程，每秒能闭环多少个标量槽位
2. `avg convolution coefficients per second, kernel-only`
   - 用 `validated_convolution_coefficients / avg_convolution_ms` 得到
   - 只衡量当前 NTT 卷积 kernel 本身每秒产出多少个系数
3. `avg pipeline residue values per second, kernel-only`
   - 统计两次 encode、三次 pointwise 和一次 convolution 一共写出的 residue 值数量，再除以 `avg_kernel_ms`
   - 这是当前最接近“底层 RNS kernel 吞吐率”的指标
4. `checked_scalar_slots / checked_convolution_coefficients`
   - 只描述验证覆盖范围，不参与吞吐分母
   - 主要用来区分 `sampled` 和 `full` 模式的校验力度

## 冻结入口

当前这条 native RNS 主线已经进入“基线先冻结、任务按清单解锁”的阶段。

- 冻结清单：
  - `docs/freeze_checklist.md`
- 建议使用方式：
  - 先在清单里确认当前冻结基线
  - 当前 Milestone E 的前两步也已经完成，下一步直接按清单进入 `E3 / E4`
  - 新实验只追加，不覆盖当前冻结结果

它们和以前的 `pi digits/s` 不是同一种量纲：

1. 这里测的是底层 runtime 微基准，不是完整 `pi` 求解。
2. 这里的单位是 `slot/s`、`coeff/s`、`residue-value/s`，不是十进制位数每秒。
3. 以前的 `pi digits/s` 包含 binary splitting、merge tree、reciprocal、sqrt、division、格式化输出等整条流水线成本，因此会比这里小很多，也更接近真正的端到端能力。

## 下一阶段

1. D1 到 D5 已经把 `scaled constants / reciprocal seed / sqrt seed / division quotient / shared correction domain` 迁入主线，Milestone D 的最小语义闭环已经完成。
2. E1/E2 也已经冻结：完整 `pi` 路线明确选择 `exact Chudnovsky + binary splitting on native RNS exact multiply backbone`，而不是 AGM-like 常数算法。
3. E3 现在已经有五个可执行切片：binary-splitting merge-tree 的执行计划、小规模 exact `P/Q/T` 对象语义 smoke、第一段 native-RNS tensor merge smoke、小规模 multi-node tensor tree smoke，以及第一条 end-to-end `pi` host-closure smoke。
4. E4 也已经不只是一条起步 benchmark：当前 `400 digits / 31 terms` 的新口径结果约为 `6.67k digits/s`，`630 digits / 47 terms` 的 7 模数 ceiling case 约为 `5.41k digits/s`，而新的 `870 digits / 64 terms` 深树 case 约为 `4.26k digits/s`。
5. 当前下一步不再是泛泛地说“继续冲 32+ terms”，因为 exact 路线的现实窗口已经从 `47 terms` 继续推进到 `64 terms / 870 digits`；真正的下一步是决定如何越过这个新 ceiling，包括更多模数、批量 leaf generation，以及更深 tree 的表示/执行组织。
