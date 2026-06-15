# Project 2 GPU-Native Reboot

## 目标

这份记录对应 `HW/05/project2_gpu_native_rns/` 这个新子项目。它不是对旧 `binary16 limb + Torch extension` 路线的小修，而是一次明确的架构重启：

1. 不再把整数语义建立在 `trim / compare / carry / borrow` 上。
2. 不再让 Python/Torch 负责热路径里的大整数语义调度。
3. 直接把基础表示切到 `RNS/CRT + device-resident metadata`。

## 新子项目结构

- `project2_gpu_native_rns/README.md`
- `project2_gpu_native_rns/docs/architecture.md`
- `project2_gpu_native_rns/Makefile`
- `project2_gpu_native_rns/include/project2_gpu_native_rns/rns_runtime.cuh`
- `project2_gpu_native_rns/src/rns_runtime.cu`
- `project2_gpu_native_rns/src/main.cu`

## 当前已经落地的第一阶段

当前新分支已经实现了一个真正独立的原生 CUDA 表示层闭环：

1. `DeviceRnsTensor`
   - 数据布局：`residues[modulus][value][slot]`
   - 元数据：`sign / logical_slots / scale_bits / level`
2. GPU kernel
   - `encode_u64`
   - `pointwise_add`
   - `pointwise_sub`
   - `pointwise_mul`
   - `pairwise_convolution`
     - `ntt_shared`
     - `ntt_global`
3. host 侧 exact verification
   - 使用 `Garner / CRT` 从 residue tensor 精确重建标量

这意味着新分支已经不是“只有一份想法文档”，而是已经有了可编译、可运行、可验对的运行时核心。

## 本轮 smoke test

执行命令：

```bash
make -C project2_gpu_native_rns run
```

结果文件：

- `result/project2_gpu_native_rns_smoke.log`
- `result/project2_gpu_native_rns_smoke.csv`
- `result/project2_gpu_native_rns_smoke_noverify_256x32.log`
- `result/project2_gpu_native_rns_smoke_noverify_256x32.csv`

本轮实测摘要：

| 指标 | 数值 |
| --- | --- |
| 布局 | `residues[modulus][value][slot]` |
| `modulus_count` | `3` |
| `value_count` | `256` |
| `slot_count` | `4` |
| `convolution_algorithm` | `ntt_shared` |
| `convolution_ntt_size` | `8` |
| `input_bits` | `30` |
| `verification_enabled` | `1` |
| 已验证标量槽位数 | `1024` |
| 已验证卷积系数数 | `1792` |
| 冷启动 kernel 时间 | `337.311676 ms` |
| 热身后平均 encode 时间 | `0.046784 ms` |
| 热身后平均 pointwise 时间 | `0.022490 ms` |
| 热身后平均 convolution 时间 | `0.165069 ms` |
| 热身后平均 kernel 时间 | `0.234342 ms` |
| 热身后平均 download + CRT 时间 | `1.486925 ms` |
| 热身后平均端到端时间 | `1.739813 ms` |
| 热身后平均标量槽位吞吐率 | `588568.93 slots/s` |
| 热身后平均卷积系数吞吐率 | `10856079.45 coeffs/s` |
| 热身后平均 residue 写出吞吐率 | `88485909.21 residue-values/s` |
| 热身后平均 download/kernel 比值 | `6.35x` |
| 状态 | `ok` |

额外的纯 GPU benchmark 样例：

| 指标 | 数值 |
| --- | --- |
| 命令 | `./bin/project2_gpu_native_rns_smoke --value-count 256 --slot-count 32 --iterations 10 --input-bits 28 --no-verify --output-tag noverify_256x32` |
| `verification_enabled` | `0` |
| `convolution_ntt_size` | `64` |
| 热身后平均 kernel 时间 | `0.515018 ms` |
| 热身后平均端到端时间 | `0.526167 ms` |
| 热身后平均标量槽位吞吐率 | `15569189.58 slots/s` |
| 热身后平均 residue 写出吞吐率 | `332540092.57 residue-values/s` |

大规模 `ntt_global` full correctness 样例：

| 指标 | 数值 |
| --- | --- |
| 命令 | `./bin/project2_gpu_native_rns_smoke --value-count 2 --slot-count 3000 --iterations 1 --input-bits 24 --verify-mode full --output-tag full_2x3000` |
| `convolution_algorithm` | `ntt_global` |
| `convolution_ntt_size` | `8192` |
| `verification_enabled` | `1` |
| `verification_mode` | `full` |
| `checked_scalar_slots` | `6000` |
| `checked_convolution_coefficients` | `11998` |
| 热身后平均 kernel 时间 | `0.253792 ms` |
| 热身后平均 download + CRT 时间 | `8.457942 ms` |
| 热身后平均端到端时间 | `19.076323 ms` |
| 状态 | `ok` |

同一规模的 sampled verification 样例：

| 指标 | 数值 |
| --- | --- |
| 命令 | `./bin/project2_gpu_native_rns_smoke --value-count 2 --slot-count 3000 --iterations 1 --input-bits 24 --verify-mode sampled --verify-samples 64 --output-tag sampled_2x3000` |
| `verification_enabled` | `1` |
| `verification_mode` | `sampled` |
| `verification_sample_count` | `64` |
| `checked_scalar_slots` | `64` |
| `checked_convolution_coefficients` | `64` |
| 热身后平均 kernel 时间 | `0.163808 ms` |
| 热身后平均 download + CRT 时间 | `0.108521 ms` |
| 热身后平均端到端时间 | `0.343013 ms` |
| 状态 | `ok` |

大规模 `ntt_global` 纯 GPU benchmark：

| 指标 | 数值 |
| --- | --- |
| 命令 | `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --no-verify --output-tag noverify_64x4096_c45lifecycle` |
| `convolution_algorithm` | `ntt_global` |
| `convolution_ntt_size` | `8192` |
| `verification_enabled` | `0` |
| `input_staging_mode` | `device_reuse` |
| `one_time_input_upload_ms` | `0.908806` |
| `verification_mode` | `none` |
| 热身后平均 encode 时间 | `0.030106 ms` |
| 热身后平均 convolution 时间 | `0.503104 ms` |
| 热身后平均 kernel 时间 | `0.585498 ms` |
| 热身后平均端到端时间 | `0.599273 ms` |
| 热身后平均标量槽位吞吐率 | `437436986.11 slots/s` |
| 热身后平均 residue 写出吞吐率 | `9401971836.37 residue-values/s` |

大规模 `ntt_global` sampled verification benchmark：

| 指标 | 数值 |
| --- | --- |
| 命令 | `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode sampled --verify-samples 64 --output-tag sampled_64x4096_c45lifecycle` |
| `verification_enabled` | `1` |
| `input_staging_mode` | `device_reuse` |
| `one_time_input_upload_ms` | `0.880518` |
| `verification_mode` | `sampled` |
| `verification_sample_count` | `64` |
| `checked_scalar_slots` | `64` |
| `checked_convolution_coefficients` | `64` |
| 热身后平均 encode 时间 | `0.033005 ms` |
| 热身后平均 convolution 时间 | `0.500941 ms` |
| 热身后平均 kernel 时间 | `0.585325 ms` |
| 热身后平均 download + CRT 时间 | `0.090364 ms` |
| 热身后平均端到端时间 | `0.779366 ms` |
| 热身后平均标量槽位吞吐率 | `336355620.52 slots/s` |
| 热身后平均 download/kernel 比值 | `0.154383x` |

大规模 `ntt_global` full verification benchmark：

| 指标 | 数值 |
| --- | --- |
| 命令 | `./bin/project2_gpu_native_rns_smoke --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --verify-mode full --output-tag full_64x4096_c45lifecycle` |
| `verification_enabled` | `1` |
| `input_staging_mode` | `device_reuse` |
| `one_time_input_upload_ms` | `0.941674` |
| `verification_mode` | `full` |
| `checked_scalar_slots` | `262144` |
| `checked_convolution_coefficients` | `524224` |
| 热身后平均 convolution 时间 | `0.503155 ms` |
| 热身后平均 kernel 时间 | `0.689914 ms` |
| 热身后平均 download + CRT 时间 | `354.231544 ms` |
| 热身后平均端到端时间 | `990.866828 ms` |
| 热身后平均标量槽位吞吐率 | `264560.27 slots/s` |
| 热身后平均 download/kernel 比值 | `513.44x` |

reduced-level lifecycle/block smoke：

| 指标 | 数值 |
| --- | --- |
| 命令 | `./bin/project2_gpu_native_rns_smoke --lifecycle-block-smoke` |
| `lifecycle_block_smoke_status` | `ok` |
| `active_modulus_count` | `2` |
| `product_scale_bits` | `18` |
| `convolution_logical_slots` | `15` |

D1 scaled-constant smoke：

| 指标 | 数值 |
| --- | --- |
| 命令 | `./bin/project2_gpu_native_rns_smoke --scaled-constant-smoke` |
| 结果文件 | `../result/project2_gpu_native_rns_scaled_constant_smoke_d1.log` |
| `scaled_constant_smoke_status` | `ok` |
| `full_modulus_count` | `3` |
| `active_modulus_count` | `2` |
| `dropped_scale_bits` | `9` |
| `level_scale_bits` | `11` |
| `product_scale_bits` | `20` |
| `convolution_logical_slots` | `7` |

D2 reciprocal-seed smoke：

| 指标 | 数值 |
| --- | --- |
| 命令 | `./bin/project2_gpu_native_rns_smoke --reciprocal-seed-smoke` |
| 结果文件 | `../result/project2_gpu_native_rns_reciprocal_seed_smoke_d2.log` |
| `reciprocal_seed_smoke_status` | `ok` |
| `denominator_coefficient` | `13` |
| `reciprocal_seed_coefficient` | `1290555` |
| `product_coefficient` | `16777215` |
| `ideal_scaled_one_coefficient` | `16777216` |
| `error_coefficient` | `1` |
| `denominator_scale_bits` | `7` |
| `reciprocal_scale_bits` | `17` |
| `target_product_scale_bits` | `24` |

D3 sqrt-seed smoke：

| 指标 | 数值 |
| --- | --- |
| 命令 | `./bin/project2_gpu_native_rns_smoke --sqrt-seed-smoke` |
| 结果文件 | `../result/project2_gpu_native_rns_sqrt_seed_smoke_d3.log` |
| `sqrt_seed_smoke_status` | `ok` |
| `radicand_coefficient` | `10005` |
| `sqrt_seed_coefficient` | `104883811` |
| `square_coefficient` | `11000613809883721` |
| `target_scaled_radicand_coefficient` | `11000613835898880` |
| `error_coefficient` | `26015159` |
| `radicand_scale_bits` | `0` |
| `sqrt_scale_bits` | `20` |
| `target_square_scale_bits` | `40` |

D4 division smoke：

| 指标 | 数值 |
| --- | --- |
| 命令 | `./bin/project2_gpu_native_rns_smoke --division-smoke` |
| 结果文件 | `../result/project2_gpu_native_rns_division_smoke_d4.log` |
| `division_smoke_status` | `ok` |
| `numerator_coefficient` | `355` |
| `denominator_coefficient` | `113` |
| `quotient_coefficient` | `1647099` |
| `product_coefficient` | `186122187` |
| `scaled_numerator_coefficient` | `186122240` |
| `remainder_coefficient` | `53` |
| `numerator_scale_bits` | `5` |
| `denominator_scale_bits` | `3` |
| `quotient_scale_bits` | `21` |
| `target_product_scale_bits` | `24` |

D5 correction-domain smoke：

| 指标 | 数值 |
| --- | --- |
| 命令 | `./bin/project2_gpu_native_rns_smoke --correction-domain-smoke` |
| 结果文件 | `../result/project2_gpu_native_rns_correction_domain_smoke_d5.log` |
| `correction_domain_smoke_status` | `ok` |
| `shared_correction_api` | `compute_residual_correction` |
| `active_modulus_count` | `2` |
| `reciprocal_residual_coefficient` | `1` |
| `sqrt_residual_coefficient` | `26015159` |
| `division_residual_coefficient` | `53` |

这条 smoke 把 D2/D3/D4 三条路径统一收口到一个共享 correction 接口上：`compute_residual_correction(target, approximate, residual, corrected)`。当前验收会同时检查 reciprocal、sqrt、division 三类 case 的 residual 与 corrected 都没有掉出 target 所在的 reduced-level RNS / `scale_bits` 语义域，因此 D 阶段关于 error/correction 的最小闭环已经成立。

E1/E2 pi-route smoke：

| 指标 | 数值 |
| --- | --- |
| 命令 | `./bin/project2_gpu_native_rns_smoke --pi-route-smoke --pi-digits 1000000` |
| 结果文件 | `../result/project2_gpu_native_rns_pi_route_smoke_1m_e12.log` |
| `chosen_route` | `chudnovsky_binary_splitting` |
| `preferred_multiply_backbone` | `exact_multiply_tree_on_native_rns_ntt` |
| `1M chudnovsky_terms` | `70516` |
| `1M binary_split_depth` | `17` |
| 命令 | `./bin/project2_gpu_native_rns_smoke --pi-route-smoke --pi-digits 100000000` |
| 结果文件 | `../result/project2_gpu_native_rns_pi_route_smoke_100m_e12.log` |
| `100M chudnovsky_terms` | `7051370` |
| `100M binary_split_depth` | `23` |
| `rejected_route` | `agm_like_constant_algorithm` |

这条 smoke 把 E 阶段前两项落成了可执行结论：当前完整 `pi` 路线固定为 `exact Chudnovsky + binary splitting`。原因不是 AGM-like 常数算法“不好”，而是当前 native RNS 主线已经有 exact multiply / NTT 的强项，而 repeated full-precision reciprocal / sqrt / division 还只是 prototype-level；在这个阶段，把大头工作压回 multiply tree 才是最现实的极限推进路线。

E3 partial pi-execution-plan smoke：

| 指标 | 数值 |
| --- | --- |
| 命令 | `./bin/project2_gpu_native_rns_smoke --pi-execution-plan-smoke --pi-digits 1000000` |
| 结果文件 | `../result/project2_gpu_native_rns_pi_execution_plan_1m_e3.log` |
| `1M leaf_terms_per_task` | `32` |
| `1M leaf_task_count` | `2204` |
| `1M peak_ntt_size` | `262144` |
| 命令 | `./bin/project2_gpu_native_rns_smoke --pi-execution-plan-smoke --pi-digits 100000000` |
| 结果文件 | `../result/project2_gpu_native_rns_pi_execution_plan_100m_e3.log` |
| `100M leaf_terms_per_task` | `2048` |
| `100M leaf_task_count` | `3444` |
| `100M peak_slot_count` | `10381029` |
| `100M peak_ntt_size` | `33554432` |
| `100M root_safe_limb_bits` | `33` |

这条 smoke 是 E3 的第一块执行骨架。它还没有真的把 `P/Q/T` 对象算出来，但已经把 binary-splitting 的 merge-tree 结构、每层 exact multiply 的规模，以及当前三模数组合下的 `32-bit limbs` 安全余量显式打印出来了。对后续工程推进来说，这比继续抽象谈路线更有价值，因为它已经把 `100M` 位数下的峰值 NTT 规模明确压到了台面上。

E3 P/Q/T semantics smoke：

| 指标 | 数值 |
| --- | --- |
| 命令 | `./bin/project2_gpu_native_rns_smoke --pi-pqt-smoke --pi-terms 8` |
| 结果文件 | `../result/project2_gpu_native_rns_pi_pqt_smoke_8terms_e3.log` |
| `8 terms node_count` | `15` |
| `8 terms root_t_bits` | `434` |
| `8 terms root_ntt_size` | `32` |
| 命令 | `./bin/project2_gpu_native_rns_smoke --pi-pqt-smoke --pi-terms 16` |
| 结果文件 | `../result/project2_gpu_native_rns_pi_pqt_smoke_16terms_e3.log` |
| `16 terms node_count` | `31` |
| `16 terms root_t_bits` | `944` |
| `16 terms root_ntt_size` | `64` |
| `merge_formula` | `T_left*Q_right + P_left*T_right` |

这条 smoke 是 E3 的第二块骨架。它用一个最小 host-side exact big-int，把 Chudnovsky 的 `P/Q/T` leaf/merge recurrence 直接跑起来，目的不是替代 GPU，而是把对象语义先钉死。这样后面把 leaf nodes 和 merge nodes 映射到 native-RNS tensor 时，语义层就不需要再来回试错。

E3 P/Q/T tensor merge smoke：

| 指标 | 数值 |
| --- | --- |
| 命令 | `./bin/project2_gpu_native_rns_smoke --pi-pqt-tensor-smoke --pi-terms 2` |
| 结果文件 | `../result/project2_gpu_native_rns_pi_pqt_tensor_smoke_2terms_e3.log` |
| `merge_interval` | `0:2` |
| `chosen_limb_bits` | `16` |
| `uses_signed_residue_encoding` | `1` |
| `p_output_slot_count` | `1` |
| `q_output_slot_count` | `4` |
| `t_output_slot_count` | `5` |
| `t_output_ntt_size` | `8` |
| `metadata_ok` | `1` |
| `p_match / q_match / t_match` | `1 / 1 / 1` |

这条 smoke 是 E3 的第三块骨架。它第一次真的把一个 Chudnovsky 两叶子 merge 节点搬到 native-RNS tensor 上执行，而不再只是在 host 上做 exact semantics。当前实现把 `P/Q/T` 编码成 `base-2^16` 的 signed limb polynomial，把负系数直接映射到模数域里，再用现有 convolution/add kernel 跑出 `P`、`Q` 和 `T` 的 merge recurrence，最后通过中心提升把系数精确还原并和 host-side exact node 对齐。这说明当前 runtime 已经能承接第一段真实的 `P/Q/T` merge 语义，下一步的阻塞就不再是“能不能算一个 merge”，而是如何把它批量化成 bottom-up merge tree。

E3 P/Q/T tensor tree smoke：

| 指标 | 数值 |
| --- | --- |
| 命令 | `./bin/project2_gpu_native_rns_smoke --pi-pqt-tree-tensor-smoke --pi-terms 4` |
| 结果文件 | `../result/project2_gpu_native_rns_pi_pqt_tensor_tree_smoke_4terms_e3.log` |
| `4 terms chosen_limb_bits` | `16` |
| `4 terms root_t_slot_count` | `11` |
| `4 terms peak_ntt_size` | `16` |
| `4 terms p/q/t match` | `1 / 1 / 1` |
| 命令 | `./bin/project2_gpu_native_rns_smoke --pi-pqt-tree-tensor-smoke --pi-terms 8` |
| 结果文件 | `../result/project2_gpu_native_rns_pi_pqt_tensor_tree_smoke_8terms_e3.log` |
| `8 terms chosen_limb_bits` | `8` |
| `8 terms root_t_slot_count` | `51` |
| `8 terms peak_ntt_size` | `64` |
| `8 terms p/q/t match` | `1 / 1 / 1` |
| 命令 | `./bin/project2_gpu_native_rns_smoke --pi-pqt-tree-tensor-smoke --pi-terms 16` |
| 结果文件 | `../result/project2_gpu_native_rns_pi_pqt_tensor_tree_smoke_16terms_e3.log` |
| `16 terms chosen_limb_bits` | `4` |
| `16 terms modulus_count` | `7` |
| `16 terms root_t_slot_count` | `230` |
| `16 terms peak_ntt_size` | `256` |
| `16 terms p/q/t match` | `1 / 1 / 1` |
| 命令 | `./bin/project2_gpu_native_rns_smoke --pi-pqt-tree-tensor-smoke --pi-terms 32` |
| 结果文件 | `../result/project2_pi_pqt_tree_tensor_terms32_modswitch_20260417_v1.log` |
| `32 terms chosen_limb_bits` | `2` |
| `32 terms modulus_count` | `7` |
| `32 terms root_t_slot_count` | `993` |
| `32 terms peak_ntt_size` | `1024` |
| `32 terms p/q/t match` | `1 / 1 / 1` |
| 命令 | `./bin/project2_gpu_native_rns_smoke --pi-pqt-tree-tensor-smoke --pi-terms 47` |
| 结果文件 | `../result/project2_pi_pqt_tree_tensor_terms47_modswitch_20260417_v1.log` |
| `47 terms chosen_limb_bits` | `1` |
| `47 terms modulus_count` | `7` |
| `47 terms root_t_slot_count` | `3031` |
| `47 terms peak_ntt_size` | `4096` |
| `47 terms p/q/t match` | `1 / 1 / 1` |
| 命令 | `./bin/project2_gpu_native_rns_smoke --pi-pqt-tree-tensor-smoke --pi-terms 64` |
| 结果文件 | `../result/project2_pi_pqt_tree_tensor_terms64_modswitch_20260417_v1.log` |
| `64 terms chosen_limb_bits` | `1` |
| `64 terms modulus_count` | `10` |
| `64 terms root_t_slot_count` | `4224` |
| `64 terms peak_ntt_size` | `8192` |
| `64 terms p/q/t match` | `1 / 1 / 1` |

这条 smoke 是 E3 的第四块骨架。它把 leaf tensor construction、device-side zero-pad 和多节点 bottom-up merge execution 真正串起来了，因此它给出的结论比单节点 merge 更硬。最新一轮扩模之后，默认主线仍保持 `3` 模数，而更深树诊断会先切到 `7` 模数实验路径，再在更深一层切到 `10` 模数实验路径；对应的 CRT 动态范围也从约 `90.47 bits` 提升到约 `212.35 bits`，再进一步提升到约 `302.73 bits`。在补齐 shared-NTT path 对新增模数的 primitive-root 映射、把 host 侧 centered reconstruction 从 `__int128` 升级到任意精度 `HostBigInt`、并修掉 `plan_pi_execution()` 里模数积统计也还停在 `__int128` 的问题之后，`16 terms` 在 `4-bit limbs` 下可以保持 exact，`32 terms` 在 `2-bit limbs` 下保持 exact，`47 terms` 在 `1-bit limbs + 7 moduli` 下保持 exact，而现在 `64 terms` 也已经在 `1-bit limbs + 10 moduli` 下保持 exact。于是当前阻塞又一次前移：它不再是“如何越过 47 terms”，而是已经变成了“如何越过当前 10 模数路径的 `64 terms` ceiling”。

E3 end-to-end `pi` host-closure smoke：

| 指标 | 数值 |
| --- | --- |
| 命令 | `./bin/project2_gpu_native_rns_smoke --pi-end-to-end-smoke` |
| 结果文件 | `../result/project2_pi_end_to_end_smoke_default_20260417_v2.log` |
| `50 digits term_count` | `8` |
| `50 digits required_terms` | `6` |
| `50 digits reference_prefix_digits_checked` | `50` |
| `50 digits chosen_limb_bits` | `8` |
| `50 digits modulus_count` | `3` |
| `50 digits peak_ntt_size` | `64` |
| `50 digits prefix_match` | `1` |
| 命令 | `./bin/project2_gpu_native_rns_smoke --pi-end-to-end-smoke --pi-digits 100` |
| 结果文件 | `../result/project2_pi_end_to_end_smoke_100digits_20260417_v2.log` |
| `100 digits term_count` | `10` |
| `100 digits required_terms` | `10` |
| `100 digits reference_prefix_digits_checked` | `100` |
| `100 digits chosen_limb_bits` | `4` |
| `100 digits modulus_count` | `7` |
| `100 digits peak_ntt_size` | `256` |
| `100 digits prefix_match` | `1` |

这条 smoke 是 E3 的第五块骨架。它第一次不再只验证 `P/Q/T` 的局部 exactness，而是真的把 GPU tensor-tree 产出的 root `Q/T` 接到了最终 Chudnovsky 收口上。当前 closing 仍然是 host-side exact path：在仓内 `HostBigInt` 上补齐最小 `isqrt/division`，算出 `floor(pi * 10^digits)`，再和内置参考前缀逐位对齐。它还不是最终想要的 native-RNS `sqrt/division` 收口，但它已经把“这条主线能不能完整吐出一段对的 `pi`”这件事变成了可执行结论。

E4 first end-to-end `pi digits/s` benchmark：

| 指标 | 数值 |
| --- | --- |
| 命令 | `./bin/project2_gpu_native_rns_smoke --pi-end-to-end-benchmark --pi-digits 400 --iterations 3 --output-tag 400digits_20260417_v3` |
| 结果文件 | `../result/project2_pi_end_to_end_benchmark_400digits_20260417_v3.log` |
| 结果文件 | `../result/project2_pi_end_to_end_benchmark_400digits_20260417_v3.csv` |
| `term_count` | `31` |
| `required_terms` | `31` |
| `reference_prefix_digits_checked` | `100` |
| `chosen_limb_bits` | `2` |
| `modulus_count` | `7` |
| `peak_ntt_size` | `1024` |
| `tree_validation_mode` | `skip_intermediate_metadata_downloads_keep_root_exact_match` |
| `avg_tree_execution_ms` | `57.2298` |
| `avg_host_closure_ms` | `2.70191` |
| `avg_end_to_end_ms` | `59.9317` |
| `avg_digits_per_second_e2e` | `6674.26` |
| `400 digits prefix_match` | `1` |
| 命令 | `./bin/project2_gpu_native_rns_smoke --pi-end-to-end-benchmark --pi-digits 630 --iterations 3 --output-tag 630digits_20260417_v1` |
| 结果文件 | `../result/project2_pi_end_to_end_benchmark_630digits_20260417_v1.log` |
| 结果文件 | `../result/project2_pi_end_to_end_benchmark_630digits_20260417_v1.csv` |
| `630 digits term_count` | `47` |
| `630 digits required_terms` | `47` |
| `630 digits chosen_limb_bits` | `1` |
| `630 digits modulus_count` | `7` |
| `630 digits peak_ntt_size` | `4096` |
| `630 digits avg_tree_execution_ms` | `110.515` |
| `630 digits avg_host_closure_ms` | `5.93018` |
| `630 digits avg_end_to_end_ms` | `116.445` |
| `630 digits avg_digits_per_second_e2e` | `5410.26` |
| `630 digits prefix_match` | `1` |
| 命令 | `./bin/project2_gpu_native_rns_smoke --pi-end-to-end-benchmark --pi-digits 870 --iterations 3 --output-tag 870digits_20260417_v1` |
| 结果文件 | `../result/project2_pi_end_to_end_benchmark_870digits_20260417_v1.log` |
| 结果文件 | `../result/project2_pi_end_to_end_benchmark_870digits_20260417_v1.csv` |
| `870 digits term_count` | `64` |
| `870 digits required_terms` | `64` |
| `870 digits chosen_limb_bits` | `1` |
| `870 digits modulus_count` | `10` |
| `870 digits peak_ntt_size` | `8192` |
| `870 digits avg_tree_execution_ms` | `189.672` |
| `870 digits avg_host_closure_ms` | `14.3921` |
| `870 digits avg_end_to_end_ms` | `204.064` |
| `870 digits avg_digits_per_second_e2e` | `4263.36` |
| `870 digits prefix_match` | `1` |

这条 benchmark 是 E4 的第一块骨架，而且现在已经不只是一条起步数据。先前的 benchmark harness 已经补上了更干净的 tree 计时口径：跳过中间节点 metadata 下载，只保留 root exactness，因此 `400 digits / 31 terms` 的端到端结果提升到约 `6.67k digits/s`。随后又把 case 推到 `630 digits / 47 terms`，在旧 7 模数 exact ceiling 附近拿到约 `5.41k digits/s`；而随着 10 模数深树路径打通，现在又把窗口继续推进到 `870 digits / 64 terms`，拿到约 `4.26k digits/s`。这三组数据共同说明：当前时间大头始终还是 `tensor tree`，而不是 host-side closing；所以下一步真正值得继续攻的，是如何越过 `64 terms` ceiling、以及如何把更深 tree 的 leaf generation 和 merge scheduling 组织得更好。

## 当前意义

这条 aggressive reboot 路线的意义不在于它现在就比旧主线更快，而在于它第一次把问题拆到了更底层也更正确的地方：

1. 先重建表示层。
2. 再重建乘法与函数语义。
3. 最后才谈完整 `pi` 求解。

和旧路线相比，它当前最重要的价值是：

1. 不再被 carry/trim 语义直接束缚。
2. 为后续 `NTT / CRT / residue-domain reciprocal/sqrt/division` 铺好了数据布局。
3. 把“是否值得为极限性能重启架构”这件事，从讨论推进成了真正的代码起点。

同时，这轮数据也给出了一个很重要的阶段性判断：

1. 这条分支已经不再是“只有 pointwise ops 的 runtime”，而是已经真正接上了 `NTT` 卷积路径。
2. 现在验证链已经拆成 `none / sampled / full` 三档；`validated_*` 继续表示整批 workload 大小，`checked_*` 则表示本次真的核查了多少点。
3. `sampled` 模式已经可以把验证成本从主曲线上大幅剥离出来。以 `64 x 4096` 为例，`full` 的 `download + CRT` 仍有 `354.23 ms`，而 `sampled(64)` 只有 `0.090 ms`。
4. measured loop 现在会复用预上传到 device 的输入，因此 `avg_encode_ms` 只表示循环内的 residue encode；一次性的 host->device 准备成本则单独记录成 `one_time_input_upload_ms`，大约是 `1 ms` 级别。
5. 这使得当前 C 阶段验收版的 `sampled` benchmark 端到端吞吐来到约 `336.4M slots/s`，而 `no-verify` 达到约 `437.4M slots/s`。
6. 这条最终接受的 C 阶段主线不只把卷积 hot-path 压到 `0.50 ms` 级，也已经把 lifecycle/block 语义变成真实接口：可变 `modulus_count` 分配、`drop_moduli_prefix`、以及 formal polynomial/block facade 都已经能在 reduced-level smoke 中独立通过。
7. `ntt_global` 已经在 `8192` padded size 上同时通过 `full`、`sampled` 和 reduced-level lifecycle/block correctness 路径；不过也正因为 kernel 已经继续变快，`full` 路径里 host reconstruction 的相对主导程度现在更夸张，`download/kernel` 比值达到约 `513x`。
8. 在很小的 `slot_count` 上，NTT 常数会比朴素卷积更重，所以这一步的价值主要是“打通算法路线并开始跨规模”，不是立刻在最小 smoke 上赢回时间。

## 下一步

下一阶段最合理的动作是：

1. D1 到 D5 已经把 `scaled constants / reciprocal seed / sqrt seed / division quotient / shared correction domain` 迁入这条已经完成 C 阶段验收的 RNS 主线，Milestone D 的最小语义闭环已经完成。
2. E1/E2 也已经完成并冻结：完整 `pi` 路线明确选为 `exact Chudnovsky + binary splitting`。
3. E3 现在已经同时有了 execution-plan 骨架、exact `P/Q/T` semantic smoke、第一段 native-RNS tensor merge smoke、小规模 multi-node tensor tree smoke，以及第一条 end-to-end `pi` host-closure smoke。
4. E4 也已经继续推进：`400 digits / 31 terms` 的新口径结果约 `6.67k digits/s`，`630 digits / 47 terms` 的 7 模数 ceiling case 约 `5.41k digits/s`，而新的 `870 digits / 64 terms` case 约 `4.26k digits/s`。
5. 下一步不再是泛泛地说“继续冲 32+ terms”，因为当前 exact 路线的现实上限已经从 `47 terms` 继续推进到 `64 terms / 870 digits`；真正的下一步是决定如何越过这个新 ceiling，包括更多模数、批量 leaf generation，以及更深 tree 的表示/执行组织。
