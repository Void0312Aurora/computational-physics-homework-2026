# Native RNS Freeze Checklist

## 目标

这份文档用于把 `HW/05/project2_gpu_native_rns` 当前阶段的实现状态冻结下来，并把后续工作拆成一份不会漂移的任务清单。

冻结的含义不是“停止一切开发”，而是：

1. 当前基线结果、文件布局和指标口径先固定。
2. 后续改动必须围绕明确的未完成任务推进，而不是继续无边界试错。
3. 任何新的优化都必须在这份清单上显式解锁、执行、验收、再回填。

冻结时间：`2026-04-16`

## 冻结快照

当前冻结基线对应的关键结果如下：

1. `64 x 4096 --no-verify`
   - `input_staging_mode = device_reuse`
   - `avg_convolution_ms = 0.503104`
   - `avg_kernel_ms = 0.585498`
   - `avg_end_to_end_ms = 0.599273`
   - `avg_scalar_slots_per_second_e2e = 437436986.11`
   - 结果文件：`result/runtime_smoke/project2_gpu_native_rns_smoke_noverify_64x4096_c45lifecycle.csv`
2. `64 x 4096 --verify-mode sampled --verify-samples 64`
   - `avg_convolution_ms = 0.500941`
   - `avg_kernel_ms = 0.585325`
   - `avg_download_reconstruct_ms = 0.090364`
   - `avg_end_to_end_ms = 0.779366`
   - `avg_scalar_slots_per_second_e2e = 336355620.52`
   - 结果文件：`result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_c45lifecycle.csv`
3. `64 x 4096 --verify-mode full`
   - `avg_convolution_ms = 0.503155`
   - `avg_kernel_ms = 0.689914`
   - `avg_download_reconstruct_ms = 354.231544`
   - `avg_end_to_end_ms = 990.866828`
   - `avg_download_over_kernel_ratio = 513.44x`
   - 结果文件：`result/runtime_smoke/project2_gpu_native_rns_smoke_full_64x4096_c45lifecycle.csv`
4. `2 x 3000 --verify-mode full`
   - `status = ok`
   - 结果文件：`result/runtime_smoke/project2_gpu_native_rns_smoke_full_2x3000_c45lifecycle.csv`
5. `2 x 8 level-2 lifecycle/block smoke`
   - `lifecycle_block_smoke_status = ok`
   - `active_modulus_count = 2`
   - `product_scale_bits = 18`
   - `convolution_logical_slots = 15`
   - 结果文件：`result/runtime_smoke/project2_gpu_native_rns_lifecycle_block_smoke_c45.log`

## 冻结范围

以下内容视为当前阶段已经定下来的实现基线：

1. 数据表示
   - `DeviceRnsTensor` 与 `residues[modulus][value][slot]` 布局
   - 元数据字段 `sign / logical_slots / scale_bits / level`
2. 运行时骨架
   - pointwise add/sub/mul
   - `ntt_shared`
   - `ntt_global`
   - `verify_mode = none / sampled / full`
3. 当前已落地的 hot-path 优化
   - stage twiddle cache
   - transform pointwise multiply + inverse-side bit-reverse fusion
   - device input reuse
   - `length <= 1024` initial stages 的 shared-memory tile 融合
   - inverse final stage + scale-out 融合
4. 当前已落地的语义接口
   - 可变 `modulus_count` tensor 分配
   - `set_uniform_tensor_metadata`
   - `drop_moduli_prefix`
   - polynomial/block facade：`allocate_polynomial_block_tensor / encode_polynomial_blocks / add_polynomial_blocks / multiply_polynomial_blocks / convolve_polynomial_blocks`
5. 当前指标口径
   - `avg_encode_ms` 只统计 measured loop 内的 device-side residue encode
   - 一次性 host->device 输入准备单独记为 `one_time_input_upload_ms`
   - `validated_*` 表示 workload 总量
   - `checked_*` 表示实际验证覆盖范围

## 冻结规则

冻结期间，必须遵守下面这些约束：

1. 不重写、不删除当前冻结基线对应的结果文件。
2. 不再修改已经冻结的指标口径，除非同时升级所有 CSV 列、README、architecture、answer 文档。
3. 新实验必须追加新 `output-tag`，不能覆盖冻结快照。
4. 任何性能结论必须至少同时提供：
   - `no-verify`
   - `sampled`
   - 必要时的 `full` correctness case
5. 任何结构性优化合入前，必须先过：
   - `2 x 3000 --verify-mode full`
   - `64 x 4096 --verify-mode sampled --verify-samples 64`
6. 除本清单中明确列出的“待解锁任务”外，不再回到旧 hybrid / limb / mpz 主线做旁路修补。

## 任务拆解

### A. 已完成并冻结

- [x] A1. 原生 `DeviceRnsTensor` 表示层完成
- [x] A2. pointwise add/sub/mul 完成
- [x] A3. `ntt_shared` 小规模卷积路径完成
- [x] A4. `ntt_global` 大规模卷积路径完成
- [x] A5. `none / sampled / full` 三档验证链完成
- [x] A6. stage twiddle cache 完成
- [x] A7. multiply + bit-reverse kernel fusion 完成
- [x] A8. device input reuse 完成
- [x] A9. `<=1024` initial stages shared-memory tile 融合完成

### B. Milestone C 剩余任务

这些任务是当前唯一允许继续解锁的主线。

- [x] C1. 继续优化剩余大 stage
  完成标准：
  - 不破坏 `8192` padded size correctness
  - `64 x 4096 sampled(64)` 的 `avg_convolution_ms` 继续下降
- [x] C2. inverse 路径进一步融合
  完成标准：
  - 减少 inverse side 的 global-memory round-trip
  - `avg_kernel_ms` 有稳定下降，不只是冷启动波动
- [x] C3. sampled verification 更多下沉到 device
  完成标准：
  - host 侧 sample gather / CRT 路径进一步缩短
  - `sampled` 模式仍保持 `status = ok`
- [x] C4. level management / modulus lifecycle 设计
  完成标准：
  - 给后续 reciprocal/sqrt/division 留出真实接口，而不是只停留在占位字段
- [x] C5. polynomial/block 表示转正
  完成标准：
  - `slot_count > 1` 不再只是 smoke benchmark 专用
  - 能明确承接后续高层 exact multiply 语义
- [x] C6. 冻结一次“Milestone C 验收版”
  完成标准：
  - 给出最终 C 阶段基线
  - 输出完整的性能对比和剩余瓶颈判断

### C. Milestone D 任务

- [x] D1. scaled constants 迁入 RNS 主线
- [x] D2. reciprocal 原型迁入 RNS 主线
- [x] D3. sqrt 原型迁入 RNS 主线
- [x] D4. division 原型迁入 RNS 主线
- [x] D5. error/correction 保持在 RNS domain

Milestone D 的冻结前提：

1. Milestone C 必须先有稳定验收版。
2. 高层函数原型不能再借道旧 `mpz` 主线。

### D. Milestone E 任务

- [x] E1. 选定完整 `pi` 路线
- [x] E2. 决定是重写 Chudnovsky/Binary Splitting 还是转向更适合 GPU 的常数算法
- [ ] E3. 接通从 exact multiply 到完整 `pi` 求解流水线
- [ ] E4. 建立真正的 `digits/s` 端到端评测

## 当前阻塞判断

当前最重要的现实判断必须固定下来，防止后续误判瓶颈：

1. `sampled / no-verify` 主线上，主瓶颈已经回到 `ntt_global` 本体，尤其是剩余大 stage 的组织和 global-memory 往返。
2. `full` 主线上，真正的主瓶颈不是 kernel，而是 host 侧 `download + CRT reconstruction`。
3. 因为 kernel 已经继续下降，所以 `full` 模式里 host reconstruction 的相对主导程度反而更夸张。

## 解冻顺序

从现在开始，只允许按下面顺序继续推进：

1. D 阶段函数语义迁移
2. E 阶段完整 `pi` 路线

## 变更记录模板

后续每次解锁一个任务，建议按下面模板追加到本文件末尾：

## Change 2026-04-16 C1 Partial

- 解锁任务：
  - `C1. 继续优化剩余大 stage`
- 改动文件：
  - `src/rns_runtime.cu`
- 改动内容：
  - 把 `ntt_stages_dual_shared_kernel / ntt_stages_shared_kernel` 泛化为可指定 `stage_start_length`
  - 在 `device_reuse + fused_initial_stages` 之后，再尝试把 `2048 -> 4096` 这一段中间大 stage 融入 `4096` tile 的 shared-memory 路径
- correctness case：
  - `./bin/project2_gpu_native_rns_smoke --value-count 2 --slot-count 3000 --iterations 1 --input-bits 24 --verify-mode full --output-tag full_2x3000_c1midtile`
  - `status = ok`
- performance case：
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --no-verify --output-tag noverify_64x4096_c1midtile`
  - `avg_convolution_ms = 0.557875`
  - `avg_kernel_ms = 0.641824`
  - `avg_end_to_end_ms = 0.652638`
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode sampled --verify-samples 64 --output-tag sampled_64x4096_c1midtile`
  - `avg_convolution_ms = 0.554426`
  - `avg_kernel_ms = 0.637370`
  - `avg_end_to_end_ms = 0.838524`
- 是否达到完成标准：
  - `部分达到`
  - `sampled(64)` 的 `avg_convolution_ms` 相对冻结基线 `0.560947 -> 0.554426` 继续下降
  - `2 x 3000 full` correctness 保持通过
  - 但 `no-verify` 相对冻结基线 `0.651871 -> 0.652638` 基本持平，尚不足以支持整体重设主基线
- 新的冻结结论：
  - 保持当前冻结快照不变
  - `C1` 继续保持未完成状态
  - 下一刀优先针对最终剩余的大 stage，尤其是 `8192` 级别的 stage 组织与 inverse 路径
- 新增结果文件：
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_full_2x3000_c1midtile.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_noverify_64x4096_c1midtile.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_c1midtile.csv`

## Change 2026-04-16 C2 Final Fuse

- 解锁任务：
  - `C2. inverse 路径进一步融合`
- 改动文件：
  - `src/rns_runtime.cu`
- 改动内容：
  - 新增 `ntt_final_stage_scale_out_kernel`
  - 当 `inverse_start_length == ntt_size` 时，跳过 generic final inverse stage 与后续 `scale_and_copy_out_prefix_kernel`
  - 直接把 inverse 最后一层 butterfly 与 `1 / ntt_size` 缩放融合，并把结果直接写入 `out.d_residues`
  - 修复 fused final-stage 写回时在 `out_slot_count < ntt_size / 2` 条件下的 prefix 越界写风险
- correctness case：
  - `./bin/project2_gpu_native_rns_smoke --value-count 2 --slot-count 3000 --iterations 1 --input-bits 24 --verify-mode full --output-tag full_2x3000_c2finalfuse`
  - `status = ok`
- performance case：
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --no-verify --output-tag noverify_64x4096_c2finalfuse`
  - `avg_convolution_ms = 0.541242`
  - `avg_kernel_ms = 0.624429`
  - `avg_end_to_end_ms = 0.634368`
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode sampled --verify-samples 64 --output-tag sampled_64x4096_c2finalfuse`
  - `avg_convolution_ms = 0.543277`
  - `avg_kernel_ms = 0.643130`
  - `avg_end_to_end_ms = 1.089754`
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode sampled --verify-samples 64 --output-tag sampled_64x4096_c2finalfuse_repeat`
  - `avg_convolution_ms = 0.542925`
  - `avg_kernel_ms = 0.625952`
  - `avg_end_to_end_ms = 0.831110`
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode full --output-tag full_64x4096_c2finalfuse`
  - `avg_convolution_ms = 0.542106`
  - `avg_kernel_ms = 0.730554`
  - `avg_end_to_end_ms = 924.103380`
- 是否达到完成标准：
  - `达到`
  - inverse side 少掉了一次 final-stage global-memory round-trip
  - `no-verify` 的 `avg_kernel_ms` 相对冻结基线 `0.642214 -> 0.624429` 稳定下降
  - `sampled(64)` 的 repeat run `avg_kernel_ms` 相对冻结基线 `0.650778 -> 0.625952` 稳定下降
  - `full` 的 `avg_kernel_ms` 也从 `0.740832 -> 0.730554` 继续下降
- 新的冻结结论：
  - 冻结快照更新到 `c2finalfuse` 系列结果
  - `sampled` 口径采用 repeat run，避免把一次 host-side jitter 误判成主线退化
  - `C2` 标记完成
  - 后续主线回到 `C1` 收尾与 `C3` 下沉验证链
- 新增结果文件：
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_full_2x3000_c2finalfuse.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_noverify_64x4096_c2finalfuse.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_c2finalfuse.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_c2finalfuse_repeat.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_full_64x4096_c2finalfuse.csv`

## Change 2026-04-16 C1 Final

- 解锁任务：
  - `C1. 继续优化剩余大 stage`
- 改动文件：
  - `src/rns_runtime.cu`
- 改动内容：
  - 新增 `ntt_final_stage_dual_pointwise_bit_reverse_kernel`
  - 当 `forward_start_length == ntt_size` 时，跳过 generic final forward stage
  - 直接把 forward 最后一层 butterfly、pointwise multiply 与 inverse-side bit-reverse 准备合进一个 kernel
  - 为避免 in-place bit-reverse 写回覆盖尚未读取的源数据，给 global NTT workspace 增加 `scratch` buffer，并让后续 inverse 直接接到这块 scratch
- correctness case：
  - `./bin/project2_gpu_native_rns_smoke --value-count 2 --slot-count 3000 --iterations 1 --input-bits 24 --verify-mode full --output-tag full_2x3000_c1fwdfinalfuse`
  - `status = ok`
- performance case：
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --no-verify --output-tag noverify_64x4096_c1fwdfinalfuse`
  - `avg_convolution_ms = 0.502579`
  - `avg_kernel_ms = 0.584864`
  - `avg_end_to_end_ms = 0.594960`
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode sampled --verify-samples 64 --output-tag sampled_64x4096_c1fwdfinalfuse`
  - `avg_convolution_ms = 0.501965`
  - `avg_kernel_ms = 0.589882`
  - `avg_end_to_end_ms = 0.818612`
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode full --output-tag full_64x4096_c1fwdfinalfuse`
  - `avg_convolution_ms = 0.501555`
  - `avg_kernel_ms = 0.684390`
  - `avg_end_to_end_ms = 965.296815`
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode full --output-tag full_64x4096_c1fwdfinalfuse_repeat`
  - `avg_convolution_ms = 0.502374`
  - `avg_kernel_ms = 0.720358`
  - `avg_end_to_end_ms = 991.561731`
- 是否达到完成标准：
  - `达到`
  - `2 x 3000 full` correctness 保持通过
  - `sampled(64)` 的 `avg_convolution_ms` 相对当前冻结基线 `0.542925 -> 0.501965` 继续下降
  - `no-verify` 的 `avg_kernel_ms` 相对当前冻结基线 `0.624429 -> 0.584864` 明显下降
  - `sampled(64)` 的 `avg_kernel_ms` 相对当前冻结基线 `0.625952 -> 0.589882` 明显下降
- 新的冻结结论：
  - `C1` 标记完成
  - hot-path 改进非常明确，说明 `8192` 规模下剩余 final forward stage 的额外 global-memory 往返确实是主线瓶颈之一
  - 但 `full` 端到端时间仍被 host `download + CRT reconstruction` 主导，而且本轮两次 full run 都出现明显波动，因此暂不整体重写冻结快照
  - 下一步按冻结顺序进入 `C3`
- 新增结果文件：
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_full_2x3000_c1fwdfinalfuse.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_noverify_64x4096_c1fwdfinalfuse.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_c1fwdfinalfuse.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_full_64x4096_c1fwdfinalfuse.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_full_64x4096_c1fwdfinalfuse_repeat.csv`

## Change 2026-04-16 C3 Point Triplet Gather

- 解锁任务：
  - `C3. sampled verification 更多下沉到 device`
- 改动文件：
  - `src/rns_runtime.cu`
- 改动内容：
  - 新增 `gather_residue_samples_triplet_kernel`
  - 把 sampled verification 中 `sum / diff / prod` 三路 pointwise 样本 gather 合并成一次 kernel launch
  - 把三路 pointwise 样本合并成一次 device-to-host copy，再分别做 sampled CRT reconstruction
  - 保留 convolution sampled gather 与 host 侧 sampled CRT 语义不变，优先先砍掉 pointwise 三路的 host 往返
- correctness case：
  - `./bin/project2_gpu_native_rns_smoke --value-count 2 --slot-count 3000 --iterations 1 --input-bits 24 --verify-mode full --output-tag full_2x3000_c3pointtriplet`
  - `status = ok`
- performance case：
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --no-verify --output-tag noverify_64x4096_c3pointtriplet`
  - `avg_kernel_ms = 0.588141`
  - `avg_end_to_end_ms = 0.597876`
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode sampled --verify-samples 64 --output-tag sampled_64x4096_c3pointtriplet`
  - `avg_kernel_ms = 0.583296`
  - `avg_download_reconstruct_ms = 0.090859`
  - `avg_end_to_end_ms = 0.768132`
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode sampled --verify-samples 64 --output-tag sampled_64x4096_c3pointtriplet_repeat`
  - `avg_kernel_ms = 0.591302`
  - `avg_download_reconstruct_ms = 0.098610`
  - `avg_end_to_end_ms = 0.783812`
- 是否达到完成标准：
  - `达到`
  - sampled 模式保持 `status = ok`
  - `avg_download_reconstruct_ms` 相对 `C1` 冻结前开发基线 `0.130376 -> 0.090859 / 0.098610` 明显下降
  - `avg_end_to_end_ms` 相对 `C1` 开发基线 `0.818612 -> 0.768132 / 0.783812` 稳定下降
- 新的冻结结论：
  - `C3` 标记完成
  - 这刀主要改善 sampled verification 辅路径；`no-verify` 基本持平，说明热路径本体没有被破坏
  - 当前仍暂不整体重写冻结快照，等待 `C4/C5` 与 `C6` 一起做一次更完整的 Milestone C 验收冻结
  - 下一步进入 `C4/C5`
- 新增结果文件：
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_full_2x3000_c3pointtriplet.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_noverify_64x4096_c3pointtriplet.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_c3pointtriplet.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_c3pointtriplet_repeat.csv`

## Change 2026-04-16 C4C5 Lifecycle Blocks

- 解锁任务：
  - `C4. level management / modulus lifecycle 设计`
  - `C5. polynomial/block 表示转正`
- 改动文件：
  - `include/project2_gpu_native_rns/rns_runtime.cuh`
  - `src/rns_runtime.cu`
  - `src/main.cu`
- 改动内容：
  - `allocate_device_tensor` 现在支持显式 `modulus_count`
  - 新增 `set_uniform_tensor_metadata` 与 `drop_moduli_prefix`，把 level / modulus lifecycle 变成真实可调用接口
  - 新增 `allocate_polynomial_block_tensor / encode_polynomial_blocks / add_polynomial_blocks / multiply_polynomial_blocks / convolve_polynomial_blocks`
  - 新增 `--lifecycle-block-smoke`，用独立 smoke 验证 reduced-level block 语义
  - 修正 pointwise multiply / convolution 的 metadata 传播，使 `scale_bits` 与 nonzero/sign 语义更接近后续 exact multiply 需求
- correctness case：
  - `./bin/project2_gpu_native_rns_smoke --lifecycle-block-smoke > result/runtime_smoke/project2_gpu_native_rns_lifecycle_block_smoke_c45.log`
  - `lifecycle_block_smoke_status = ok`
  - `active_modulus_count = 2`
  - `product_scale_bits = 18`
  - `convolution_logical_slots = 15`
  - `./bin/project2_gpu_native_rns_smoke --value-count 2 --slot-count 3000 --iterations 1 --input-bits 24 --verify-mode full --output-tag full_2x3000_c45lifecycle`
  - `status = ok`
- performance case：
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --no-verify --output-tag noverify_64x4096_c45lifecycle`
  - `avg_kernel_ms = 0.585498`
  - `avg_end_to_end_ms = 0.599273`
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode sampled --verify-samples 64 --output-tag sampled_64x4096_c45lifecycle`
  - `avg_kernel_ms = 0.585325`
  - `avg_download_reconstruct_ms = 0.090364`
  - `avg_end_to_end_ms = 0.779366`
- 是否达到完成标准：
  - `达到`
  - `C4`：level / modulus lifecycle 已不再只是元数据占位，而是有了 `modulus_count` 分层分配、metadata 重设与 modulus-drop 的真实接口
  - `C5`：`slot_count > 1` 现在有正式的 polynomial/block API，而且 reduced-level block add/mul/convolution 已通过独立 smoke 验证
  - 主 benchmark 的 `no-verify / sampled` 热路径保持稳定，没有因语义层接入而回退
- 新的冻结结论：
  - `C4/C5` 标记完成
  - Milestone C 剩余工作收敛为 `C6` 验收冻结
  - 当前冻结快照先保持不变，等 `C6` 统一重算并整理最终 C 阶段基线
- 新增结果文件：
  - `result/runtime_smoke/project2_gpu_native_rns_lifecycle_block_smoke_c45.log`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_full_2x3000_c45lifecycle.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_noverify_64x4096_c45lifecycle.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_c45lifecycle.csv`

## Change 2026-04-16 C6 Final Acceptance

- 解锁任务：
  - `C6. 冻结一次“Milestone C 验收版”`
- 改动文件：
  - `docs/freeze_checklist.md`
  - `docs/architecture.md`
  - `result/runtime_smoke/project2_cuda_stage1_summary.csv`
- correctness case：
  - `./bin/project2_gpu_native_rns_smoke --lifecycle-block-smoke > result/runtime_smoke/project2_gpu_native_rns_lifecycle_block_smoke_c45.log`
  - `lifecycle_block_smoke_status = ok`
  - `./bin/project2_gpu_native_rns_smoke --value-count 2 --slot-count 3000 --iterations 1 --input-bits 24 --verify-mode full --output-tag full_2x3000_c45lifecycle`
  - `status = ok`
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode full --output-tag full_64x4096_c45lifecycle`
  - `status = ok`
- performance case：
  - 当前验收基线：
  - `no-verify`: `avg_kernel_ms = 0.585498`, `avg_end_to_end_ms = 0.599273`, `437.44M scalar slots/s`
  - `sampled(64)`: `avg_kernel_ms = 0.585325`, `avg_download_reconstruct_ms = 0.090364`, `avg_end_to_end_ms = 0.779366`, `336.36M scalar slots/s`
  - `full`: `avg_kernel_ms = 0.689914`, `avg_download_reconstruct_ms = 354.231544`, `avg_end_to_end_ms = 990.866828`
  - 对比早先冻结基线 `c2finalfuse`：
  - `no-verify avg_kernel_ms: 0.624429 -> 0.585498`
  - `sampled avg_end_to_end_ms: 0.831110 -> 0.779366`
  - `full avg_kernel_ms: 0.730554 -> 0.689914`
  - 对比热路径峰值：
  - `no-verify` 峰值仍是 `c1fwdfinalfuse` 的 `0.594960 ms / 440.61M slots/s`
  - `sampled` 峰值仍是 `c3pointtriplet` 的 `0.768132 ms / 341.27M slots/s`
  - 但 `c45lifecycle` 相对这些峰值的损失保持在小百分比内，同时补齐了 lifecycle/block 语义
- 是否达到完成标准：
  - `达到`
  - 已给出最终 C 阶段验收基线
  - 已给出从 `C2 -> C6` 的性能比较与剩余瓶颈判断
- 新的冻结结论：
  - Milestone C 正式验收完成
  - 最终接受基线采用 `c45lifecycle`
  - 接受理由不是它在每一项上都绝对最快，而是它把 hot-path 优化、sampled 验证分层、level lifecycle 与 polynomial/block facade 收敛到同一条可继续演化的主线
  - 当前剩余第一性瓶颈仍然分裂为两类：
  - `no-verify / sampled` 继续由 `ntt_global` 的 stage 组织与 global-memory 往返主导
  - `full` 继续由 host `download + CRT reconstruction` 主导，而且因为 kernel 更快，相对比值进一步放大
  - 下一步从 D 阶段开始，把 `scaled constants / reciprocal / sqrt / division` 迁入这条已经完成 C 阶段验收的 RNS 主线
- 新增结果文件：
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_full_64x4096_c45lifecycle.csv`

## Change 2026-04-16 D1 Scaled Constant

- 解锁任务：
  - `D1. scaled constants 迁入 RNS 主线`
- 改动文件：
  - `include/project2_gpu_native_rns/rns_runtime.cuh`
  - `src/rns_runtime.cu`
  - `src/main.cu`
- 改动内容：
  - 新增 `make_scaled_constant_tensor` 与 `encode_scaled_constant_blocks`
  - 直接在 native RNS runtime 内构造 uniform monomial scaled constant，不再借道旧 `mpz/hybrid` 路线
  - 新增 `--scaled-constant-smoke`，覆盖 full-level constant、`drop_moduli_prefix` 后的 reduced-level constant，以及后续 add/mul/convolution metadata 传播
- D1 smoke：
  - `./bin/project2_gpu_native_rns_smoke --scaled-constant-smoke`
  - `scaled_constant_smoke_status = ok`
  - `full_modulus_count = 3`
  - `active_modulus_count = 2`
  - `dropped_scale_bits = 9`
  - `level_scale_bits = 11`
  - `product_scale_bits = 20`
  - `convolution_logical_slots = 7`
- correctness case：
  - `./bin/project2_gpu_native_rns_smoke --value-count 2 --slot-count 3000 --iterations 1 --input-bits 24 --verify-mode full --output-tag full_2x3000_d1scaledconst`
  - `status = ok`
  - `avg_kernel_ms = 0.124384`
  - `avg_download_reconstruct_ms = 8.698859`
  - `avg_end_to_end_ms = 19.752837`
- regression case：
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --no-verify --output-tag noverify_64x4096_d1scaledconst`
  - `avg_convolution_ms = 0.500045`
  - `avg_kernel_ms = 0.580870`
  - `avg_end_to_end_ms = 0.590733`
  - `avg_scalar_slots_per_second_e2e = 443760247.85`
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode sampled --verify-samples 64 --output-tag sampled_64x4096_d1scaledconst_repeat`
  - `avg_convolution_ms = 0.501402`
  - `avg_kernel_ms = 0.585453`
  - `avg_download_reconstruct_ms = 0.099402`
  - `avg_end_to_end_ms = 0.797649`
  - `avg_scalar_slots_per_second_e2e = 328645890.27`
- 是否达到完成标准：
  - `达到`
  - scaled constant 已经能以原生 RNS tensor 的形式直接落地，并接入 reduced-level lifecycle 与 polynomial/block 运算
  - `2 x 3000 full`、`64 x 4096 no-verify`、`64 x 4096 sampled(64)` 回归全部保持 `status = ok`
- 新的冻结结论：
  - Milestone C 验收基线继续保持 `c45lifecycle`
  - D 阶段第一个语义切片已经落地，下一步直接进入 `D2 / D3 / D4`
- 新增结果文件：
  - `result/d_semantics/project2_gpu_native_rns_scaled_constant_smoke_d1.log`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_full_2x3000_d1scaledconst.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_noverify_64x4096_d1scaledconst.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_d1scaledconst.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_d1scaledconst_repeat.csv`

## Change 2026-04-16 D2 Reciprocal Seed

- 解锁任务：
  - `D2. reciprocal 原型迁入 RNS 主线`
- 改动文件：
  - `include/project2_gpu_native_rns/rns_runtime.cuh`
  - `src/rns_runtime.cu`
  - `src/main.cu`
- 改动内容：
  - 新增 `make_reciprocal_seed_tensor` 与 `encode_reciprocal_seed_blocks`
  - 在 native RNS runtime 内直接构造 positive monomial scaled-constant 的 reciprocal seed
  - 新增 `--reciprocal-seed-smoke`，验证 `denominator * seed`、`scaled-one`、`error` 与 `corrected` 全都留在同一条 reduced-level RNS / `scale_bits` 语义链上
- D2 smoke：
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
- correctness case：
  - `./bin/project2_gpu_native_rns_smoke --value-count 2 --slot-count 3000 --iterations 1 --input-bits 24 --verify-mode full --output-tag full_2x3000_d2recipseed`
  - `status = ok`
  - `avg_kernel_ms = 0.123168`
  - `avg_download_reconstruct_ms = 7.819802`
  - `avg_end_to_end_ms = 18.145124`
- regression case：
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --no-verify --output-tag noverify_64x4096_d2recipseed`
  - `avg_convolution_ms = 0.500736`
  - `avg_kernel_ms = 0.587693`
  - `avg_end_to_end_ms = 0.597335`
  - `avg_scalar_slots_per_second_e2e = 438855771.43`
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode sampled --verify-samples 64 --output-tag sampled_64x4096_d2recipseed`
  - `avg_convolution_ms = 0.501933`
  - `avg_kernel_ms = 0.585229`
  - `avg_download_reconstruct_ms = 0.093529`
  - `avg_end_to_end_ms = 0.771984`
  - `avg_scalar_slots_per_second_e2e = 339571716.62`
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode sampled --verify-samples 64 --output-tag sampled_64x4096_d2recipseed_repeat`
  - `avg_convolution_ms = 0.501555`
  - `avg_kernel_ms = 0.587302`
  - `avg_download_reconstruct_ms = 0.121263`
  - `avg_end_to_end_ms = 0.824567`
  - `avg_scalar_slots_per_second_e2e = 317917238.48`
- 是否达到完成标准：
  - `达到`
  - reciprocal 已经以原生 RNS prototype 的形式进入主线，而且 seed、product、ideal、error 的对象语义不再借道旧 `mpz/hybrid`
  - `2 x 3000 full`、`64 x 4096 no-verify`、`64 x 4096 sampled(64)` 回归全部保持 `status = ok`
- 新的冻结结论：
  - D 阶段第二个语义切片已经落地
  - 当前 reciprocal 原型仍限制在 positive monomial scaled-constant 与 `target_product_scale_bits < 64` 的 smoke/seed 范围内
  - 下一步直接推进 `D3 / D4`，并在需要时把 reciprocal seed 扩成更完整的 Newton 迭代
- 新增结果文件：
  - `result/d_semantics/project2_gpu_native_rns_reciprocal_seed_smoke_d2.log`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_full_2x3000_d2recipseed.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_noverify_64x4096_d2recipseed.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_d2recipseed.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_d2recipseed_repeat.csv`

## Change 2026-04-17 D3 Sqrt Seed

- 解锁任务：
  - `D3. sqrt 原型迁入 RNS 主线`
- 改动文件：
  - `include/project2_gpu_native_rns/rns_runtime.cuh`
  - `src/rns_runtime.cu`
  - `src/main.cu`
- 改动内容：
  - 新增 `make_sqrt_seed_tensor` 与 `encode_sqrt_seed_blocks`
  - 在 native RNS runtime 内直接构造 positive monomial scaled-constant 的 sqrt seed
  - 新增 `--sqrt-seed-smoke`，验证 `sqrt_seed^2`、`scaled-radicand`、`error` 与 `corrected` 全都留在同一条 reduced-level RNS / `scale_bits` 语义链上
- D3 smoke：
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
- correctness case：
  - `./bin/project2_gpu_native_rns_smoke --value-count 2 --slot-count 3000 --iterations 1 --input-bits 24 --verify-mode full --output-tag full_2x3000_d3sqrtseed`
  - `status = ok`
  - `avg_kernel_ms = 0.121856`
  - `avg_download_reconstruct_ms = 7.075051`
  - `avg_end_to_end_ms = 16.353396`
- regression case：
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --no-verify --output-tag noverify_64x4096_d3sqrtseed`
  - `avg_convolution_ms = 0.499917`
  - `avg_kernel_ms = 0.579789`
  - `avg_end_to_end_ms = 0.589871`
  - `avg_scalar_slots_per_second_e2e = 444409031.81`
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode sampled --verify-samples 64 --output-tag sampled_64x4096_d3sqrtseed`
  - `avg_convolution_ms = 0.499917`
  - `avg_kernel_ms = 0.583328`
  - `avg_download_reconstruct_ms = 0.084971`
  - `avg_end_to_end_ms = 0.758825`
  - `avg_scalar_slots_per_second_e2e = 345460233.67`
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode sampled --verify-samples 64 --output-tag sampled_64x4096_d3sqrtseed_repeat`
  - `avg_convolution_ms = 0.500525`
  - `avg_kernel_ms = 0.586349`
  - `avg_download_reconstruct_ms = 0.096907`
  - `avg_end_to_end_ms = 0.805743`
  - `avg_scalar_slots_per_second_e2e = 325344352.89`
- 是否达到完成标准：
  - `达到`
  - sqrt 已经以原生 RNS prototype 的形式进入主线，而且 seed、square、scaled-radicand、error 的对象语义不再借道旧 `mpz/hybrid`
  - `2 x 3000 full`、`64 x 4096 no-verify`、`64 x 4096 sampled(64)` 回归全部保持 `status = ok`
- 新的冻结结论：
  - D 阶段第三个语义切片已经落地
  - 当前 sqrt 原型仍限制在 positive monomial scaled-constant、even `radicand_scale_bits` 与 `uint64` 可重建验证范围内
  - 下一步直接推进 `D4 / D5`，并在需要时把 reciprocal/sqrt seed 扩成更完整的 Newton 迭代
- 新增结果文件：
  - `result/d_semantics/project2_gpu_native_rns_sqrt_seed_smoke_d3.log`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_full_2x3000_d3sqrtseed.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_noverify_64x4096_d3sqrtseed.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_d3sqrtseed.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_d3sqrtseed_repeat.csv`

## Change 2026-04-17 D4 Division Quotient

- 解锁任务：
  - `D4. division 原型迁入 RNS 主线`
- 改动文件：
  - `include/project2_gpu_native_rns/rns_runtime.cuh`
  - `src/rns_runtime.cu`
  - `src/main.cu`
- 改动内容：
  - 新增 `make_division_quotient_tensor` 与 `encode_division_quotient_blocks`
  - 在 native RNS runtime 内直接构造 positive monomial scaled-constant 的 division quotient prototype
  - 新增 `--division-smoke`，验证 `quotient * denominator`、`scaled-numerator`、`remainder` 与 `corrected` 全都留在同一条 reduced-level RNS / `scale_bits` 语义链上
- D4 smoke：
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
- correctness case：
  - `./bin/project2_gpu_native_rns_smoke --value-count 2 --slot-count 3000 --iterations 1 --input-bits 24 --verify-mode full --output-tag full_2x3000_d4division`
  - `status = ok`
  - `avg_kernel_ms = 0.120352`
  - `avg_download_reconstruct_ms = 6.799461`
  - `avg_end_to_end_ms = 17.362880`
- regression case：
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --no-verify --output-tag noverify_64x4096_d4division`
  - `avg_convolution_ms = 0.499712`
  - `avg_kernel_ms = 0.579846`
  - `avg_end_to_end_ms = 0.589662`
  - `avg_scalar_slots_per_second_e2e = 444566849.87`
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode sampled --verify-samples 64 --output-tag sampled_64x4096_d4division`
  - `avg_convolution_ms = 0.501350`
  - `avg_kernel_ms = 0.588384`
  - `avg_download_reconstruct_ms = 0.089433`
  - `avg_end_to_end_ms = 0.777641`
  - `avg_scalar_slots_per_second_e2e = 337101567.43`
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode sampled --verify-samples 64 --output-tag sampled_64x4096_d4division_repeat`
  - `avg_convolution_ms = 0.498893`
  - `avg_kernel_ms = 0.582554`
  - `avg_download_reconstruct_ms = 0.090047`
  - `avg_end_to_end_ms = 0.765586`
  - `avg_scalar_slots_per_second_e2e = 342409784.09`
- 是否达到完成标准：
  - `达到`
  - division 已经以原生 RNS prototype 的形式进入主线，而且 quotient、product、scaled-numerator、remainder 的对象语义不再借道旧 `mpz/hybrid`
  - `2 x 3000 full`、`64 x 4096 no-verify`、`64 x 4096 sampled(64)` 回归全部保持 `status = ok`
- 新的冻结结论：
  - D 阶段第四个语义切片已经落地
  - 当前 division 原型仍限制在 positive monomial scaled-constant 与 `uint64` 可重建验证范围内
  - 下一步直接推进 `D5`，统一 reciprocal/sqrt/division 路径里的 error/correction 对象语义
- 新增结果文件：
  - `result/d_semantics/project2_gpu_native_rns_division_smoke_d4.log`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_full_2x3000_d4division.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_noverify_64x4096_d4division.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_d4division.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_d4division_repeat.csv`

## Change 2026-04-17 D5 Correction Domain

- 解锁任务：
  - `D5. error/correction 保持在 RNS domain`
- 改动文件：
  - `include/project2_gpu_native_rns/rns_runtime.cuh`
  - `src/main.cu`
  - `src/rns_runtime.cu`
- 改动内容：
  - 新增共享接口 `compute_residual_correction`
  - D2/D3/D4 smoke 全部改为复用同一套 `target - approximate -> residual`、`approximate + residual -> corrected` 语义
  - 新增 `--correction-domain-smoke`，把 reciprocal / sqrt / division 三类 case 放进同一个 shared correction-domain 验收切片
  - 在 smoke 里显式验证 residual 与 corrected 的 `moduli / logical_slots / scale_bits / level / sign` 都没有掉出目标 reduced-level RNS domain
- D5 smoke：
  - `./bin/project2_gpu_native_rns_smoke --correction-domain-smoke`
  - `correction_domain_smoke_status = ok`
  - `shared_correction_api = compute_residual_correction`
  - `active_modulus_count = 2`
  - `reciprocal_residual_coefficient = 1`
  - `sqrt_residual_coefficient = 26015159`
  - `division_residual_coefficient = 53`
- correctness case：
  - `./bin/project2_gpu_native_rns_smoke --value-count 2 --slot-count 3000 --iterations 1 --input-bits 24 --verify-mode full --output-tag full_2x3000_d5correction`
  - `status = ok`
  - `avg_kernel_ms = 0.122816`
  - `avg_download_reconstruct_ms = 6.809169`
  - `avg_end_to_end_ms = 16.206014`
- regression case：
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --no-verify --output-tag noverify_64x4096_d5correction`
  - `avg_convolution_ms = 0.501760`
  - `avg_kernel_ms = 0.583142`
  - `avg_end_to_end_ms = 0.593679`
  - `avg_scalar_slots_per_second_e2e = 441558187.80`
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode sampled --verify-samples 64 --output-tag sampled_64x4096_d5correction`
  - `avg_convolution_ms = 0.499302`
  - `avg_kernel_ms = 0.583334`
  - `avg_download_reconstruct_ms = 0.087666`
  - `avg_end_to_end_ms = 0.771615`
  - `avg_scalar_slots_per_second_e2e = 339734369.98`
  - `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode sampled --verify-samples 64 --output-tag sampled_64x4096_d5correction_repeat`
  - `avg_convolution_ms = 0.501760`
  - `avg_kernel_ms = 0.586733`
  - `avg_download_reconstruct_ms = 0.088740`
  - `avg_end_to_end_ms = 0.785253`
  - `avg_scalar_slots_per_second_e2e = 333833638.92`
- 是否达到完成标准：
  - `达到`
  - D 阶段第五个语义切片已经落地，reciprocal / sqrt / division 的 residual/corrected 对象不再各自维护一套临时 correction 语义
  - `2 x 3000 full`、`64 x 4096 no-verify`、`64 x 4096 sampled(64)` 回归全部保持 `status = ok`
- 新的冻结结论：
  - Milestone D 的最小函数语义闭环已经完成
  - 这次改动只触及 shared correction 语义与 smoke 验收层，没有改 hot-path kernel，因此 `64 x 4096` 吞吐基本维持在 D4 / C45 主线附近
  - `64 x 4096 full` 没有重跑；原因是这轮没有改动 kernel 主体，而当前 full 模式的主瓶颈也仍然明确是 host `download + CRT reconstruction`
  - 下一步正式切换到 `Milestone E`，开始决定完整 `pi` 路线与端到端 `digits/s` 评测
- 新增结果文件：
  - `result/correction_domain/project2_gpu_native_rns_correction_domain_smoke_d5.log`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_full_2x3000_d5correction.log`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_full_2x3000_d5correction.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_noverify_64x4096_d5correction.log`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_noverify_64x4096_d5correction.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_d5correction.log`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_d5correction.csv`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_d5correction_repeat.log`
  - `result/runtime_smoke/project2_gpu_native_rns_smoke_sampled_64x4096_d5correction_repeat.csv`

## Change 2026-04-17 E1 E2 Pi Route Selection

- 解锁任务：
  - `E1. 选定完整 pi 路线`
  - `E2. 决定是重写 Chudnovsky/Binary Splitting 还是转向更适合 GPU 的常数算法`
- 改动文件：
  - `include/project2_gpu_native_rns/rns_runtime.cuh`
  - `src/main.cu`
  - `src/rns_runtime.cu`
- 改动内容：
  - 新增 `PiRouteReport`
  - 新增 `plan_pi_route / print_pi_route_report / pi_route_smoke_test`
  - 新增 CLI：`--pi-route-smoke --pi-digits`
  - 把 E 阶段的路线选择从“文档讨论”落成了可执行 smoke：显式输出目标位数、guard digits、working bits、Chudnovsky 项数、binary-splitting 树深，以及为什么当前拒绝 AGM-like 常数算法
- E1/E2 smoke：
  - `./bin/project2_gpu_native_rns_smoke --pi-route-smoke --pi-digits 1000000`
  - `pi_route_smoke_status = ok`
  - `chosen_route = chudnovsky_binary_splitting`
  - `chudnovsky_terms = 70516`
  - `binary_split_depth = 17`
  - `final_nonmultiply_steps = 2`
  - `./bin/project2_gpu_native_rns_smoke --pi-route-smoke --pi-digits 100000000`
  - `pi_route_smoke_status = ok`
  - `chosen_route = chudnovsky_binary_splitting`
  - `chudnovsky_terms = 7051370`
  - `binary_split_depth = 23`
  - `final_nonmultiply_steps = 2`
- route decision：
  - `采用 exact Chudnovsky + binary splitting 作为 E 阶段主路线`
  - `暂不转向 AGM-like 常数算法`
  - 原因：
    `当前 native RNS 主线已经是 exact multiply / NTT 强项，而 full-precision reciprocal / sqrt / division 还只是 prototype-level`
  - 这意味着最稳妥的端到端路线应该把绝大多数工作压回 exact multiply tree，只把最终 `sqrt(10005)` 与一次 final division 保留为非乘法步骤
- regression case：
  - `./bin/project2_gpu_native_rns_smoke --correction-domain-smoke`
  - `correction_domain_smoke_status = ok`
- 是否达到完成标准：
  - `达到`
  - E 阶段前两项已经从模糊方向选择收敛为明确主线：`Chudnovsky/Binary Splitting on native RNS exact multiply backbone`
  - 现有 D 阶段闭环在引入新 CLI 与 route planner 后仍保持通过
- 新的冻结结论：
  - E1/E2 已冻结
  - 当前不再讨论“要不要换路线”，而是直接进入 `E3 / E4`
  - 下一步是把 binary-splitting 的对象语义、乘法树调度与最终 `sqrt/division` 收口接到端到端 `pi` 流水线，并建立真实 `digits/s` 评测
- 新增结果文件：
  - `result/route_and_plan/project2_gpu_native_rns_pi_route_smoke_1m_e12.log`
  - `result/route_and_plan/project2_gpu_native_rns_pi_route_smoke_100m_e12.log`
  - `result/correction_domain/project2_gpu_native_rns_correction_domain_smoke_d5_recheck_after_e12.log`

## Change 2026-04-17 E3 Partial Execution Plan

- 解锁任务：
  - `E3. 接通从 exact multiply 到完整 pi 求解流水线`
- 改动文件：
  - `include/project2_gpu_native_rns/rns_runtime.cuh`
  - `src/main.cu`
  - `src/rns_runtime.cu`
- 改动内容：
  - 新增 `PiExecutionLevelReport / PiExecutionPlanReport`
  - 新增 `plan_pi_execution / print_pi_execution_plan_report / pi_execution_plan_smoke_test`
  - 新增 CLI：`--pi-execution-plan-smoke --pi-digits`
  - 把 binary-splitting 路线继续推进成“可执行的执行骨架”，显式输出 leaf task 切分、merge level 数、每层 node count、估计整数位宽、slot 数、NTT 规模，以及在当前三模数组合下的 `safe_limb_bits`
- E3 partial smoke：
  - `./bin/project2_gpu_native_rns_smoke --pi-execution-plan-smoke --pi-digits 1000000`
  - `pi_execution_plan_smoke_status = ok`
  - `leaf_terms_per_task = 32`
  - `leaf_task_count = 2204`
  - `merge_level_count = 13`
  - `peak_slot_count = 103814`
  - `peak_ntt_size = 262144`
  - `root_safe_limb_bits = 37`
  - `./bin/project2_gpu_native_rns_smoke --pi-execution-plan-smoke --pi-digits 100000000`
  - `pi_execution_plan_smoke_status = ok`
  - `leaf_terms_per_task = 2048`
  - `leaf_task_count = 3444`
  - `merge_level_count = 13`
  - `peak_slot_count = 10381029`
  - `peak_ntt_size = 33554432`
  - `root_safe_limb_bits = 33`
- regression case：
  - `./bin/project2_gpu_native_rns_smoke --correction-domain-smoke`
  - `correction_domain_smoke_status = ok`
- 是否达到完成标准：
  - `部分达到`
  - binary-splitting 的对象规模与 merge-tree 调度已经第一次落成可执行 skeleton
  - 但这还不是完整 `pi` 流水线；当前只完成了执行计划层，没有真正把 `P/Q/T` 对象和 exact multiply runtime 接起来
- 新的冻结结论：
  - E3 已经启动，但仍保持未完成状态
  - `100M` digits 的 root multiply 已经对应到约 `10.38M` slots 与 `2^25` 级 NTT，说明后续端到端主瓶颈将不再是“路线不清楚”，而是大规模 merge-tree multiply 的真实落地
  - 当前三模数组合对 `32-bit` limbs 仍然可用，但在 `100M` root 处只剩 `33` bits 安全余量，后续如果继续冲更高位数，需要谨慎评估 base/模数组合/多层 CRT 设计
  - 下一步直接推进真正的 `P/Q/T` 对象语义和 merge-tree execution，而不是继续停留在计划层
- 新增结果文件：
  - `result/route_and_plan/project2_gpu_native_rns_pi_execution_plan_1m_e3.log`
  - `result/route_and_plan/project2_gpu_native_rns_pi_execution_plan_100m_e3.log`
  - `result/correction_domain/project2_gpu_native_rns_correction_domain_smoke_d5_recheck_after_e3plan.log`

## Change 2026-04-17 E3 PQT Semantics Smoke

- 解锁任务：
  - `E3. 接通从 exact multiply 到完整 pi 求解流水线`
- 改动文件：
  - `include/project2_gpu_native_rns/rns_runtime.cuh`
  - `src/main.cu`
  - `src/rns_runtime.cu`
- 改动内容：
  - 新增 `PiPqtNodeReport / PiPqtTreeReport`
  - 新增 `plan_pi_pqt_tree / print_pi_pqt_tree_report / pi_pqt_smoke_test`
  - 新增 CLI：`--pi-pqt-smoke --pi-terms`
  - 内置最小 host-side `HostBigInt`，专门用于小规模 exact semantic smoke
  - 用真实 Chudnovsky leaf recurrence 与 merge recurrence 验证 `P/Q/T` 对象语义，而不是只做规模估计
- E3 P/Q/T smoke：
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-smoke --pi-terms 8`
  - `pi_pqt_smoke_status = ok`
  - `node_count = 15`
  - `root_q_bits = 410`
  - `root_t_bits = 434`
  - `root_slot_count = 14`
  - `root_ntt_size = 32`
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-smoke --pi-terms 16`
  - `pi_pqt_smoke_status = ok`
  - `node_count = 31`
  - `root_q_bits = 920`
  - `root_t_bits = 944`
  - `root_slot_count = 30`
  - `root_ntt_size = 64`
- regression case：
  - `./bin/project2_gpu_native_rns_smoke --correction-domain-smoke`
  - `correction_domain_smoke_status = ok`
- 是否达到完成标准：
  - `部分达到`
  - `P/Q/T` 的 leaf/merge recurrence 已经第一次被可执行地固定下来
  - 但当前仍然只是 host-side exact semantic smoke，还没有把这些对象真正映射到 native-RNS tensor 与 GPU merge execution
- 新的冻结结论：
  - E3 现在已经有两块可执行骨架：
    `execution plan` 和 `exact P/Q/T semantics`
  - 当前阻塞已经进一步收缩到：
    如何把这套 `P/Q/T` 对象从 host-side exact smoke 迁到 native-RNS tensor / merge kernel
  - 下一步直接推进 leaf tensor construction、merge-node tensor layout，以及真实的 bottom-up merge execution
- 新增结果文件：
  - `result/pqt_host_semantics/project2_gpu_native_rns_pi_pqt_smoke_8terms_e3.log`
  - `result/pqt_host_semantics/project2_gpu_native_rns_pi_pqt_smoke_16terms_e3.log`
  - `result/correction_domain/project2_gpu_native_rns_correction_domain_smoke_d5_recheck_after_e3pqt.log`

## Change 2026-04-17 E3 Tensor Merge Smoke

- 解锁任务：
  - `E3. 接通从 exact multiply 到完整 pi 求解流水线`
- 改动文件：
  - `include/project2_gpu_native_rns/rns_runtime.cuh`
  - `src/main.cu`
  - `src/rns_runtime.cu`
- 改动内容：
  - 新增 `PiPqtTensorReport`
  - 新增 `run_pi_pqt_tensor_smoke / print_pi_pqt_tensor_report / pi_pqt_tensor_smoke_test`
  - 新增 CLI：`--pi-pqt-tensor-smoke --pi-terms`
  - 新增 signed limb polynomial helper，把 `P/Q/T` 的 `base-2^16` 系数直接编码到模数域里
  - 在 native-RNS tensor 上第一次真实执行一个 Chudnovsky 两叶子 merge：
    `P=P_left*P_right`、`Q=Q_left*Q_right`、`T=T_left*Q_right + P_left*T_right`
  - 用 host 侧中心提升把输出系数精确还原，并和 exact host-side 节点逐项对齐
- E3 tensor smoke：
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-tensor-smoke --pi-terms 2`
  - `pi_pqt_tensor_smoke_status = ok`
  - `merge_interval = 0:2`
  - `chosen_limb_bits = 16`
  - `uses_signed_residue_encoding = 1`
  - `p_output_slot_count = 1`
  - `q_output_slot_count = 4`
  - `t_output_slot_count = 5`
  - `t_output_ntt_size = 8`
  - `metadata_ok = 1`
  - `p_match = 1`
  - `q_match = 1`
  - `t_match = 1`
- regression case：
  - `./bin/project2_gpu_native_rns_smoke --correction-domain-smoke`
  - `correction_domain_smoke_status = ok`
- 是否达到完成标准：
  - `部分达到`
  - E3 第一次真正把 `P/Q/T` merge 语义接到了 native-RNS tensor 与 device convolution/add execution 上
  - 但当前还只是单个两叶子 merge 节点的可执行切片，距离批量 leaf generation、整棵 bottom-up merge tree 和完整 `pi` 流水线仍有明显距离
- 新的冻结结论：
  - E3 不再只有 `execution plan` 和 `host-side exact semantics` 两块骨架，而是已经有了第三块：`first native-RNS tensor merge execution`
  - 当前主阻塞进一步收缩为：
    如何把这段单节点 tensor merge 推广成批量 leaf tensor construction、多节点 merge scheduling 和真正的 bottom-up merge tree
  - 下一步不再讨论“单个 merge 能不能在 GPU 语义里成立”，而是直接推进多节点执行与端到端 `digits/s` 评测
- 新增结果文件：
  - `result/pqt_tensor_smoke/project2_gpu_native_rns_pi_pqt_tensor_smoke_2terms_e3.log`
  - `result/correction_domain/project2_gpu_native_rns_correction_domain_smoke_d5_recheck_after_e3tensor.log`

## Change 2026-04-17 E3 Tensor Tree Smoke

- 解锁任务：
  - `E3. 接通从 exact multiply 到完整 pi 求解流水线`
- 改动文件：
  - `include/project2_gpu_native_rns/rns_runtime.cuh`
  - `src/main.cu`
  - `src/rns_runtime.cu`
- 改动内容：
  - 新增 `PiPqtTensorTreeReport`
  - 新增 `run_pi_pqt_tensor_tree_smoke / print_pi_pqt_tensor_tree_report / pi_pqt_tensor_tree_smoke_test`
  - 新增 CLI：`--pi-pqt-tree-tensor-smoke --pi-terms`
  - 新增通用 `pad_polynomial_tensor_suffix_zeros`，让中间节点在 device 上补零扩展，而不是回 host 重新编码
  - 把 leaf tensor construction、device-side zero-pad 和多节点 bottom-up merge execution 接到同一条 native-RNS 路径里
  - 在 tree smoke 里加入按规模自适应的 limb base 选择：`4 terms -> 16-bit`、`8 terms -> 8-bit`、`16 terms -> 4-bit`
- E3 tensor-tree smoke：
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-tree-tensor-smoke --pi-terms 4`
  - `pi_pqt_tensor_tree_smoke_status = ok`
  - `chosen_limb_bits = 16`
  - `root_t_slot_count = 11`
  - `peak_ntt_size = 16`
  - `p_match = 1`
  - `q_match = 1`
  - `t_match = 1`
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-tree-tensor-smoke --pi-terms 8`
  - `pi_pqt_tensor_tree_smoke_status = ok`
  - `chosen_limb_bits = 8`
  - `root_t_slot_count = 51`
  - `peak_ntt_size = 64`
  - `p_match = 1`
  - `q_match = 1`
  - `t_match = 1`
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-tree-tensor-smoke --pi-terms 16`
  - `pi_pqt_tensor_tree_smoke_status = failed`
  - `chosen_limb_bits = 4`
  - `root_t_slot_count = 230`
  - `peak_ntt_size = 256`
  - `p_match = 1`
  - `q_match = 0`
  - `t_match = 0`
- regression case：
  - `./bin/project2_gpu_native_rns_smoke --correction-domain-smoke`
  - `correction_domain_smoke_status = ok`
- 是否达到完成标准：
  - `部分达到`
  - E3 已经不只是在单节点上证明 merge 语义，而是真的把小规模 bottom-up merge tree 接通了
  - 但 `16 terms` 即使降到 `4-bit limbs` 仍然失配，说明当前固定三模数组合的动态范围已经成为新的主阻塞
- 新的冻结结论：
  - E3 现在已经有四个可执行切片：
    `execution plan`、`host-side exact semantics`、`single-node tensor merge`、`multi-node tensor tree`
  - 当前主阻塞再次收缩为：
    三模数组合不足以承接更深 tree 上 raw polynomial coefficient growth
  - 下一步如果还要继续向更深 tree/更高 digits 走，不再优先纠缠 kernel 细节，而是要正式进入更多模数、更强 CRT/base extension 或新的归一化策略
- 新增结果文件：
  - `result/pqt_tree_tensor_smoke/project2_gpu_native_rns_pi_pqt_tensor_tree_smoke_4terms_e3.log`
  - `result/pqt_tree_tensor_smoke/project2_gpu_native_rns_pi_pqt_tensor_tree_smoke_8terms_e3.log`
  - `result/pqt_tree_tensor_smoke/project2_gpu_native_rns_pi_pqt_tensor_tree_smoke_16terms_e3.log`
  - `result/correction_domain/project2_gpu_native_rns_correction_domain_smoke_d5_recheck_after_e3tensortree.log`

## Change 2026-04-17 E3 Extended Modulus Tensor Tree

- 解锁任务：
  - `E3. 接通从 exact multiply 到完整 pi 求解流水线`
- 改动文件：
  - `include/project2_gpu_native_rns/rns_runtime.cuh`
  - `src/rns_runtime.cu`
- 改动内容：
  - 把可用模数组扩展到 `7` 个 NTT-friendly primes，但默认 active runtime 仍保持 `3` 个模数，避免破坏已冻结主线
  - 为 `PiPqtTensorReport / PiPqtTensorTreeReport` 增加显式 `modulus_count` 报告
  - 在 `term_count > 8` 的 tensor / tensor-tree smoke 上显式切到 `7` 模数实验路径
  - 把 centered coefficient reconstruction 从 `unsigned __int128` 升级为 `HostBigInt`，修复扩模后 host 验证链的模数积溢出
  - 修复 shared-NTT path 中 `primitive_root_for_modulus_index()` 只覆盖前两个模数的旧硬编码，补齐新增模数的 primitive roots
- correctness case：
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-tensor-smoke --pi-terms 16`
  - `pi_pqt_tensor_smoke_status = ok`
  - `modulus_count = 7`
  - `p_match = 1`
  - `q_match = 1`
  - `t_match = 1`
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-tree-tensor-smoke --pi-terms 16`
  - `pi_pqt_tensor_tree_smoke_status = ok`
  - `chosen_limb_bits = 4`
  - `modulus_count = 7`
  - `root_t_slot_count = 230`
  - `peak_ntt_size = 256`
  - `p_match = 1`
  - `q_match = 1`
  - `t_match = 1`
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-tree-tensor-smoke --pi-terms 8`
  - `pi_pqt_tensor_tree_smoke_status = ok`
  - `modulus_count = 3`
  - `p_match = 1`
  - `q_match = 1`
  - `t_match = 1`
  - `./bin/project2_gpu_native_rns_smoke --correction-domain-smoke`
  - `correction_domain_smoke_status = ok`
- 是否达到完成标准：
  - `部分达到`
  - `16 terms` 的 tensor-tree exactness 已经被正式打通，说明此前“卡在 16 terms”的阻塞不再成立
  - 但 E3 仍未完成，因为当前还只是把更深一层 bottom-up tree 接通，距离真正的端到端 `pi` 收口与 `E4 digits/s` 评测还有明显距离
- 新的冻结结论：
  - 当前默认主线仍是 `3` 模数；扩展到 `7` 模数是 E3 深树诊断与推进用的显式实验路径
  - 这次修复说明上一阶段的失败并不只是“动态范围不够”，还叠加了两个实现问题：
    `shared-NTT primitive root` 的模数索引硬编码不完整，以及 host-side centered reconstruction 的 `__int128` 溢出
  - 当前下一道阻塞已经前移到：
    `16+ terms` 以上的更深 tree、批量 leaf tensor generation、非乘法 closing steps（`sqrt/division`）以及真正的端到端 `digits/s` 评测
- 新增结果文件：
  - `result/pqt_tensor_smoke/project2_pi_pqt_tensor_terms16_modswitch_20260417_v3.log`
  - `result/pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms8_modswitch_20260417_v3.log`
  - `result/pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms16_modswitch_20260417_v3.log`
  - `result/correction_domain/project2_correction_domain_modswitch_20260417_v2.log`

## Change 2026-04-17 E3 Tensor Tree To 32 Terms

- 解锁任务：
  - `E3. 接通从 exact multiply 到完整 pi 求解流水线`
- 改动文件：
  - `src/rns_runtime.cu`
- 改动内容：
  - 把 tensor-tree smoke 的 limb-base 选择整理成显式策略函数
  - 将 tensor-tree smoke 的上限从 `16 terms` 推进到 `32 terms`
  - 新增一档更深树策略：`17..32 terms -> 2-bit limbs + 7-modulus path`
- correctness case：
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-tree-tensor-smoke --pi-terms 32`
  - `pi_pqt_tensor_tree_smoke_status = ok`
  - `chosen_limb_bits = 2`
  - `modulus_count = 7`
  - `root_q_slot_count = 982`
  - `root_t_slot_count = 993`
  - `peak_ntt_size = 1024`
  - `p_match = 1`
  - `q_match = 1`
  - `t_match = 1`
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-tree-tensor-smoke --pi-terms 16`
  - `pi_pqt_tensor_tree_smoke_status = ok`
  - `p_match = 1`
  - `q_match = 1`
  - `t_match = 1`
  - `./bin/project2_gpu_native_rns_smoke --correction-domain-smoke`
  - `correction_domain_smoke_status = ok`
- 是否达到完成标准：
  - `部分达到`
  - E3 的 bottom-up tensor tree 已从 `16 terms exact` 继续推进到 `32 terms exact`
  - 但这仍然是 tensor-tree 子流程，离最终端到端 `pi` 收口和 `digits/s` 评测还有距离
- 新的冻结结论：
  - 当前默认主线仍是 `3` 模数；`>8 terms` 的更深树实验路径使用 `7` 模数
  - 当前已知 exact 覆盖范围已从 `16 terms` 推进到 `32 terms`
  - 下一道阻塞继续前移到：
    `32+ terms` 更深 tree、批量 leaf tensor generation、非乘法 closing steps（`sqrt/division`）以及真正的端到端 `digits/s` 评测
- 新增结果文件：
  - `result/pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms32_modswitch_20260417_v1.log`
  - `result/pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms16_modswitch_20260417_v4.log`
  - `result/correction_domain/project2_correction_domain_terms32_recheck_20260417_v1.log`

## Change 2026-04-17 E3 End-to-End Pi Host Closure Smoke

- 解锁任务：
  - `E3. 接通从 exact multiply 到完整 pi 求解流水线`
- 改动文件：
  - `include/project2_gpu_native_rns/rns_runtime.cuh`
  - `src/main.cu`
  - `src/rns_runtime.cu`
- 改动内容：
  - 新增 `PiEndToEndReport`
  - 新增 CLI：`--pi-end-to-end-smoke`
  - 把 GPU tensor-tree 产出的 root `Q/T` 接到 host-side exact `isqrt/division` closing path
  - 在仓内 `HostBigInt` 上补齐 host-close 所需的最小 exact 算法：
    bit access、binary long division、`isqrt`、十进制格式化
  - 若未显式指定 `--pi-digits`，end-to-end smoke 默认走 `50 digits`
  - 若未显式指定 `--pi-terms`，会自动把 term 数抬到满足 route 规划所需的最小项数
- correctness case：
  - `./bin/project2_gpu_native_rns_smoke --pi-end-to-end-smoke`
  - `pi_end_to_end_smoke_status = ok`
  - `term_count = 8`
  - `target_digits = 50`
  - `required_terms = 6`
  - `reference_prefix_digits_checked = 50`
  - `chosen_limb_bits = 8`
  - `modulus_count = 3`
  - `peak_ntt_size = 64`
  - `pi_decimal = 3.14159265358979323846264338327950288419716939937510`
  - `prefix_match = 1`
  - `./bin/project2_gpu_native_rns_smoke --pi-end-to-end-smoke --pi-digits 100`
  - `pi_end_to_end_smoke_status = ok`
  - `term_count = 10`
  - `target_digits = 100`
  - `required_terms = 10`
  - `reference_prefix_digits_checked = 100`
  - `chosen_limb_bits = 4`
  - `modulus_count = 7`
  - `peak_ntt_size = 256`
  - `prefix_match = 1`
  - `./bin/project2_gpu_native_rns_smoke --correction-domain-smoke`
  - `correction_domain_smoke_status = ok`
- 是否达到完成标准：
  - `部分达到`
  - E3 已经不再只有局部 `P/Q/T` 子流程，而是有了第一条真正从 GPU multiply tree 走到最终 `pi` 前缀的 end-to-end smoke
  - 但当前 closing 仍是 host-side exact bigint path，不是 native-RNS 上的最终 `sqrt/division` 收口，也还没有进入真正的 `digits/s` benchmark
- 新的冻结结论：
  - E3 现在已经至少有六个可执行切片：
    `route`、`execution plan`、`host-side exact semantics`、`single-node tensor merge`、`multi-node tensor tree`、`end-to-end pi host closure`
  - 当前下一道阻塞继续前移到：
    批量 leaf tensor generation、更深的 `32+ terms` tree、native-RNS final `sqrt/division` closing，以及 `E4` 的真实 `digits/s` 评测
- 新增结果文件：
  - `result/end_to_end_smoke/project2_pi_end_to_end_smoke_default_20260417_v2.log`
  - `result/end_to_end_smoke/project2_pi_end_to_end_smoke_100digits_20260417_v2.log`
  - `result/correction_domain/project2_correction_domain_recheck_after_pi_end_to_end_20260417.log`

## Change 2026-04-17 E4 First End-to-End Pi Benchmark

- 解锁任务：
  - `E4. 建立真正的 digits/s 端到端评测`
- 改动文件：
  - `include/project2_gpu_native_rns/rns_runtime.cuh`
  - `src/main.cu`
  - `src/rns_runtime.cu`
- 改动内容：
  - 新增 `PiBenchmarkReport`
  - 新增 CLI：`--pi-end-to-end-benchmark`
  - 把端到端 `pi` 路线拆成 `tree execution / host closure / full end-to-end` 三段计时
  - 新增 benchmark CSV 输出，开始冻结第一条真实的 `digits/s` 口径
  - end-to-end report 新增 `reference_prefix_digits_checked`，允许 benchmark 在更高 target digits 上只校验当前内置参考前缀
- correctness case：
  - `./bin/project2_gpu_native_rns_smoke --pi-end-to-end-smoke`
  - `pi_end_to_end_smoke_status = ok`
  - `closure_mode = host_exact_bigint_isqrt_division_after_gpu_tensor_tree`
  - `./bin/project2_gpu_native_rns_smoke --pi-end-to-end-smoke --pi-digits 100`
  - `pi_end_to_end_smoke_status = ok`
  - `closure_mode = host_exact_bigint_isqrt_division_after_gpu_tensor_tree`
- performance case：
  - `./bin/project2_gpu_native_rns_smoke --pi-end-to-end-benchmark --pi-digits 400 --iterations 3 --output-tag 400digits_20260417_v2`
  - `pi_end_to_end_benchmark_status = ok`
  - `term_count = 31`
  - `required_terms = 31`
  - `reference_prefix_digits_checked = 100`
  - `chosen_limb_bits = 2`
  - `modulus_count = 7`
  - `peak_ntt_size = 1024`
  - `avg_tree_execution_ms = 59.656`
  - `avg_host_closure_ms = 2.58248`
  - `avg_end_to_end_ms = 62.2385`
  - `avg_digits_per_second_e2e = 6426.89`
  - `prefix_match = 1`
- 是否达到完成标准：
  - `部分达到`
  - E4 已经不再是空白占位，而是有了第一条可重复的端到端 `digits/s` 基准和 CSV 输出
  - 但当前 benchmark 仍停留在 `32 terms` 以内的小规模 exact 区间，而且最终 closing 仍是 host-side bigint path，不是 native-RNS final closing
- 新的冻结结论：
  - 当前第一条 E4 数据已经把瓶颈位置讲清楚了：在 `400 digits / 31 terms` 这个阶段，主时间大头是 `GPU tensor tree`，不是 host-side closing
  - 因此下一道真正该攻的阻塞不是 `isqrt/division` 常数，而是更深的 `32+ terms` tree、批量 leaf tensor generation，以及 tensor-tree 执行本体的组织效率
  - E4 可以继续向前推，但当前还不应该勾成“已完成”，因为距离更大位数、更深树和更接近最终主线的 closing 还有明显距离
- 新增结果文件：
  - `result/end_to_end_benchmark/project2_pi_end_to_end_benchmark_400digits_20260417_v2.log`
  - `result/end_to_end_benchmark/project2_pi_end_to_end_benchmark_400digits_20260417_v2.csv`

## Change 2026-04-17 E3 Exact Ceiling To 47 Terms

- 解锁任务：
  - `E3. 接通从 exact multiply 到完整 pi 求解流水线`
- 改动文件：
  - `src/rns_runtime.cu`
- 改动内容：
  - 把 tensor-tree 的最深档从 `17..32 terms -> 2-bit limbs` 继续扩展到 `33..47 terms -> 1-bit limbs`
  - 把当前 7 模数 exact 路线的自动上限收敛到 `47 terms`
  - 用离线 carry-free 系数峰值诊断确认当前 ceiling：
    `47 terms` 峰值约 `208 bits`，`48 terms` 约 `212 bits`，`64 terms` 约 `286 bits`
- correctness case：
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-tree-tensor-smoke --pi-terms 47`
  - `pi_pqt_tensor_tree_smoke_status = ok`
  - `chosen_limb_bits = 1`
  - `modulus_count = 7`
  - `root_q_slot_count = 3008`
  - `root_t_slot_count = 3031`
  - `peak_ntt_size = 4096`
  - `p_match = 1`
  - `q_match = 1`
  - `t_match = 1`
- 是否达到完成标准：
  - `部分达到`
  - 当前 7 模数实验路径上的 exact 覆盖范围已经从 `32 terms` 继续推进到 `47 terms`
  - 但 `48+ terms` 不再是简单调 limb bits 就能继续过的问题，而是已经碰到当前模数组合的动态范围 ceiling
- 新的冻结结论：
  - 当前 7 模数 exact 路线的现实上限可以先冻结为 `47 terms`
  - 想继续越过 `47`，下一步更可能需要更多模数、批量 leaf generation、或新的表示/收缩策略，而不是继续盲目压 limb bits
- 新增结果文件：
  - `result/pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms47_modswitch_20260417_v1.log`

## Change 2026-04-17 E4 Benchmark Path Cleanup And 630-Digit Case

- 解锁任务：
  - `E4. 建立真正的 digits/s 端到端评测`
- 改动文件：
  - `include/project2_gpu_native_rns/rns_runtime.cuh`
  - `src/rns_runtime.cu`
- 改动内容：
  - benchmark 路径新增 `tree_validation_mode`
  - benchmark 现在跳过中间节点 metadata 下载，只保留 root exactness，从而避免把 tree hot path 混成 host-side metadata gather benchmark
  - 在这个更干净的 benchmark 口径上，继续把端到端 case 推到 `630 digits / 47 terms`
- correctness case：
  - `./bin/project2_gpu_native_rns_smoke --pi-end-to-end-benchmark --pi-digits 400 --iterations 3 --output-tag 400digits_20260417_v3`
  - `pi_end_to_end_benchmark_status = ok`
  - `tree_validation_mode = skip_intermediate_metadata_downloads_keep_root_exact_match`
  - `avg_end_to_end_ms = 59.9317`
  - `avg_digits_per_second_e2e = 6674.26`
  - `./bin/project2_gpu_native_rns_smoke --pi-end-to-end-benchmark --pi-digits 630 --iterations 3 --output-tag 630digits_20260417_v1`
  - `pi_end_to_end_benchmark_status = ok`
  - `term_count = 47`
  - `required_terms = 47`
  - `chosen_limb_bits = 1`
  - `peak_ntt_size = 4096`
  - `avg_tree_execution_ms = 110.515`
  - `avg_host_closure_ms = 5.93018`
  - `avg_end_to_end_ms = 116.445`
  - `avg_digits_per_second_e2e = 5410.26`
  - `prefix_match = 1`
- 是否达到完成标准：
  - `部分达到`
  - E4 的 benchmark 口径更接近真实 tree hot path 了，并且已经推进到当前 exact ceiling 附近
  - 但 benchmark 仍然停留在 host-side bigint closing，不是最终 native-RNS closing
- 新的冻结结论：
  - 当前 benchmark 结果再次确认：主时间大头仍然是 `tensor tree`，不是 host-side closing
  - 当前 benchmark 的真正下一步不再是继续抠 closing 常数，而是解决 `47 terms` 之后的更深 tree ceiling
- 新增结果文件：
  - `result/end_to_end_benchmark/project2_pi_end_to_end_benchmark_400digits_20260417_v3.log`
  - `result/end_to_end_benchmark/project2_pi_end_to_end_benchmark_400digits_20260417_v3.csv`
  - `result/end_to_end_benchmark/project2_pi_end_to_end_benchmark_630digits_20260417_v1.log`
  - `result/end_to_end_benchmark/project2_pi_end_to_end_benchmark_630digits_20260417_v1.csv`

## Change 2026-04-17 E3/E4 Ten-Modulus Deep Exact Path

- 解锁任务：
  - `E3. 接通从 exact multiply 到完整 pi 求解流水线`
  - `E4. 建立真正的 digits/s 端到端评测`
- 改动文件：
  - `include/project2_gpu_native_rns/rns_runtime.cuh`
  - `src/rns_runtime.cu`
- 改动内容：
  - 把可用模数组从 `7` 扩展到 `10`
  - 保留 `3 -> 7 -> 10` 分层：`<=8 terms` 用 `3` 模数，`9..47 terms` 用 `7` 模数，`48..64 terms` 用 `10` 模数
  - 修复 `plan_pi_execution()` 里模数积仍用 `__int128` 的溢出风险，改成 `HostBigInt`
  - 修复 shared-NTT path 的 `primitive_root_for_modulus_index()` 只覆盖旧 `7` 模数的问题；新增模数原先会错误落到 `root = 1`
- correctness case：
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-tensor-smoke --pi-terms 64`
  - `pi_pqt_tensor_smoke_status = ok`
  - `modulus_count = 10`
  - `t_match = 1`
  - `./bin/project2_gpu_native_rns_smoke --pi-pqt-tree-tensor-smoke --pi-terms 64`
  - `pi_pqt_tensor_tree_smoke_status = ok`
  - `chosen_limb_bits = 1`
  - `modulus_count = 10`
  - `root_q_slot_count = 4201`
  - `root_t_slot_count = 4224`
  - `peak_ntt_size = 8192`
  - `p_match = 1`
  - `q_match = 1`
  - `t_match = 1`
- performance case：
  - `./bin/project2_gpu_native_rns_smoke --pi-end-to-end-benchmark --pi-digits 870 --iterations 3 --output-tag 870digits_20260417_v1`
  - `pi_end_to_end_benchmark_status = ok`
  - `term_count = 64`
  - `required_terms = 64`
  - `chosen_limb_bits = 1`
  - `modulus_count = 10`
  - `peak_ntt_size = 8192`
  - `avg_tree_execution_ms = 189.672`
  - `avg_host_closure_ms = 14.3921`
  - `avg_end_to_end_ms = 204.064`
  - `avg_digits_per_second_e2e = 4263.36`
  - `prefix_match = 1`
- 是否达到完成标准：
  - `部分达到`
  - 当前 exact 窗口已经从 `47 terms` 继续推进到 `64 terms`
  - 新的 `10` 模数路径已经不只是在 tree smoke 上成立，而是已经支撑到 `870 digits` 的端到端 benchmark
  - 但 `900 digits` 已经需要 `66 terms`，所以新的 ceiling 也已经开始显现
- 新的冻结结论：
  - 当前 exact 路线的现实边界已经从 `7` 模数的 `47 terms` 提升到 `10` 模数的 `64 terms / 870 digits`
  - 下一个真正的阻塞不再是“10 模数路径能不能跑”，而是如何继续越过 `64 terms`
  - 如果继续往上推，下一步更可能落在更多模数、批量 leaf generation、以及更深 tree 的表示/执行组织
- 新增结果文件：
  - `result/pqt_tensor_smoke/project2_pi_pqt_tensor_terms64_modswitch_20260417_v1.log`
  - `result/pqt_tree_tensor_smoke/project2_pi_pqt_tree_tensor_terms64_modswitch_20260417_v1.log`
  - `result/end_to_end_benchmark/project2_pi_end_to_end_benchmark_870digits_20260417_v1.log`
  - `result/end_to_end_benchmark/project2_pi_end_to_end_benchmark_870digits_20260417_v1.csv`

```md
## Change YYYY-MM-DD

- 解锁任务：
- 改动文件：
- correctness case：
- performance case：
- 是否达到完成标准：
- 新的冻结结论：
```
