# Aggressive GPU-Native Architecture

## 目标

这条分支的目标不是“在旧实现上再 GPU 化一点”，而是建立一个真正原生的 GPU 多精度框架，使下面三件事第一次能够在架构上成立：

1. 数据表示是 GPU-first，而不是 host-first。
2. 数学语义是 GPU-resident，而不是仅把 buffer 放在显存。
3. 主乘法、scale、reciprocal、sqrt、division 最终都可以在同一表示层内闭环。

当前执行层面的冻结边界、任务顺序和变更控制，统一见：

- `docs/freeze_checklist.md`

## 核心选择

### 1. 表示层：RNS tensor

基础对象定义为：

$$
\texttt{residues}[m][v][s]
$$

其中：

- $m$ 是模数通道
- $v$ 是独立值或 batch 项
- $s$ 是 slot/系数位置

与旧 `base-2^{16}` limb 路线相比，这个表示刻意牺牲了“看起来像整数”的直观性，换来：

1. carry-free pointwise 运算
2. 适合批处理
3. 为后续 NTT/CRT 乘法做好布局准备

### 2. 元数据：device-resident side channels

每个 value 同步携带：

- `sign`
- `logical_slots`
- `scale_bits`
- `level`

这四个字段的意义是：

1. `sign`
   - 保留符号域，不在热路径里重新推导
2. `logical_slots`
   - 记录当前逻辑长度，而不是每次 `trim`
3. `scale_bits`
   - 把后续 reciprocal/sqrt/division 的缩放信息显式化
4. `level`
   - 为 RNS base extension / modulus dropping / CRT pipeline 预留层级控制

### 3. 算法层：先乘法框架，后完整 pi

这条分支的顺序会与旧项目不同：

1. 先做原生表示与乘法框架
2. 再做 reciprocal / sqrt / division
3. 最后才接完整 $\pi$ 求解器

原因很简单：如果表示层和算子层没有闭环，直接冲 `pi` 只会再次陷入 host fallback。

## 建议实施里程碑

### Milestone A

目标：原生 RNS runtime 闭环

- `DeviceRnsTensor`
- GPU encode
- GPU pointwise add/sub/mul
- host CRT reconstruction

状态：本轮已完成。

### Milestone B

目标：polynomial/block 路径成为主路径

- 真正使用 `slot_count > 1`
- block 级 pointwise ops
- block metadata propagation
- residue-domain convolution plan

状态：已推进到第二步。当前已经有：

- `allocate_polynomial_block_tensor / encode_polynomial_blocks / add_polynomial_blocks / multiply_polynomial_blocks / convolve_polynomial_blocks`
- `slot_count > 1` 的 `pairwise_convolution`
- 对应的 convolution metadata merge
- 基于单 block shared memory 的 radix-2 NTT 卷积原型
- 超出 shared-memory 阈值时自动切换的 staged global-memory NTT 路径
- `--no-verify` 纯 GPU benchmark 模式

当前限制：

1. staged global-memory NTT 已能跑过 `8192` padded size，而且已经补上第一轮关键吞吐优化：
   - stage twiddle constant-memory 预计算
   - forward `lhs/rhs` 双路 stage 合并 kernel
   - 持久 workspace 复用
   - `lhs/rhs` 双路 pad + bit-reverse 预处理 kernel
   - `encode_u64` 前端持久 upload workspace + 双路 encode/metadata kernel
   - `length <= 8192` 的 device-resident stage twiddle cache
   - 但还没有进入更激进的 device-side CRT/lazy reconstruction 与完整函数语义迁移
2. 在很小的 `slot_count` 上，NTT 常数会明显大于朴素卷积。
3. 它适合证明“新的表示层已经真的能跑 NTT 并开始跨过 shared-memory 极限”，但还不是最终高吞吐 exact multiply 层。

### Milestone C

目标：GPU-native exact multiply

- 多模数 `NTT`
- CRT recombine / lazy reconstruction
- level management
- 乘法 benchmark 与旧 GPU FFT/limb 路线对比

状态：已经开始真正落地，而且验证链也从单一的 “full host reconstruct” 拆成了三档：

- `verify_mode=none`
  只看 device-resident 热路径吞吐
- `verify_mode=sampled`
  只抽样拉回少量 residue 并做 host CRT 校验
- `verify_mode=full`
  保留全量下载与全量精确重建，作为严格 correctness 基线

当前这一阶段已经落下六刀：

1. `length <= 8192` 的 stage twiddle schedule 变成一次构建、常驻 device 的 cache，`ntt_global` 不再每轮重传 stage 表。
2. smoke runtime 引入 `sampled` 验证模式，把 benchmark 主线从 “每次都被 host reconstruction 拖住” 里解耦出来。
3. `ntt_global` 的 transform pointwise multiply 与 inverse 前的 bit-reverse 重排已经合并，少掉了一次 transform-wide 的全局内存重排 kernel。
4. measured loop 现在会复用预上传到 device 的输入标量，`avg_encode_ms` 只统计循环内的 residue encode + metadata 初始化，而一次性准备成本改由 `one_time_input_upload_ms` 单独记录。
5. global-NTT 路径中 `length <= 1024` 的初始 stages 已经迁入 shared-memory tile 融合执行，forward dual / inverse single 都先在 tile 内吃掉前几层，再回到大 stage 的 global-memory 路径。
6. inverse 路径的 final stage 与 scale-out 也已经合并；当 `inverse_start_length == ntt_size` 时，会直接把最后一层 butterfly 的结果缩放后写入输出 tensor，少掉一轮额外的 full-prefix 读写。

此外，当前主线已经补上两块语义层闭环：

1. `allocate_device_tensor(..., modulus_count)`、`set_uniform_tensor_metadata` 与 `drop_moduli_prefix` 已把 level / modulus lifecycle 从占位字段推进成真实接口。
2. `slot_count > 1` 的 polynomial/block 路径现在有了正式 facade，不再只靠 smoke 内部隐式走 `pairwise_convolution`。

### Milestone D

目标：GPU-native function pipeline

- scaled constants
- reciprocal
- sqrt
- division
- error/correction 仍保持在 RNS domain

状态：D1 到 D5 都已落地。当前已经有 `make_scaled_constant_tensor / encode_scaled_constant_blocks`、`make_reciprocal_seed_tensor / encode_reciprocal_seed_blocks`、`make_sqrt_seed_tensor / encode_sqrt_seed_blocks`、`make_division_quotient_tensor / encode_division_quotient_blocks`，以及共享的 `compute_residual_correction` 与 `--correction-domain-smoke`。这意味着 native RNS runtime 不只能够分别构造 scaled constant、reciprocal seed、sqrt seed 与 division quotient prototype，也已经把 reciprocal / sqrt / division 三条路径里的 residual/corrected 对象统一到同一套 reduced-level RNS / `scale_bits` correction 语义里；Milestone D 的最小函数语义闭环已经完成，下一步正式进入完整 `pi` 路线选择与端到端评测。

### Milestone E

目标：完整的 $\pi$ 路线选择

当前选择：

1. 主路线固定为 `exact Chudnovsky + binary splitting`
2. 乘法骨架固定为 `native RNS exact multiply / NTT tree`
3. 最终只保留一次 `sqrt(10005)` 与一次 final division 作为非乘法高精度步骤
4. 暂不转向 AGM-like 常数算法

状态：E1/E2 已完成。当前新增了 `PiRouteReport` 与 `--pi-route-smoke --pi-digits`，并用可执行 smoke 把路线选择冻结下来：`1M` digits 需要约 `70516` 个 Chudnovsky 项、binary-splitting 深度约 `17`；`100M` digits 需要约 `7051370` 个项、树深约 `23`。之所以不选 AGM-like 路线，不是因为它没有理论价值，而是因为当前 native RNS 主线的现实强项是 exact multiply / NTT，而 repeated full-precision reciprocal / sqrt / division 仍处在 prototype 阶段；在这个阶段，选择把大头工作压回 multiply tree 才是最稳的极限路线。

E3 当前已经推进到第一块 execution skeleton：新增了 `PiExecutionPlanReport` 与 `--pi-execution-plan-smoke --pi-digits`，把 Chudnovsky/Binary-Splitting 进一步展开成 batched bottom-up merge tree。这个计划层不是最终流水线，但它第一次给出了工程上真正需要的规模感知：`1M` digits 的 root 约对应 `103814` slots 与 `262144` 点 NTT；`100M` digits 的 root 约对应 `10381029` slots 与 `33554432` 点 NTT。更关键的是，当前三模数组合的动态范围约 `91` bits，因此在 `100M` root 上对 `32-bit limbs` 仍然可行，但安全余量只剩约 `33` bits，这已经把后续更高位数的 base/模数组合问题提前暴露出来了。

E3 还补上了第二块更贴近对象语义的 smoke：新增 `PiPqtTreeReport` 与 `--pi-pqt-smoke --pi-terms`，在 host-side exact arithmetic 下直接验证 Chudnovsky 的 leaf recurrence 与 merge recurrence。它的意义不是性能，而是把 `P/Q/T` 这三个对象第一次从抽象名词变成了明确可执行的节点语义：`leaf(0)` 特判、`leaf(k>0)` 的 `P/Q/T` 构造，以及 `T_left*Q_right + P_left*T_right` 这条 merge 公式都已经被固定下来。

在这之后，E3 的第三块切片也已经落地：新增 `PiPqtTensorReport` 与 `--pi-pqt-tensor-smoke --pi-terms`，把一个真实的两叶子 Chudnovsky merge 节点直接搬到 native-RNS tensor 上执行。当前做法不是重新发明一套 GPU big-int，而是把 `P/Q/T` 先编码成 `base-2^16` 的 signed limb polynomial，把负系数直接映射到模数域里，再复用现有 convolution/add kernel 算出 `P=P_left*P_right`、`Q=Q_left*Q_right`、`T=T_left*Q_right + P_left*T_right`，最后在 host 端通过中心提升把系数精确还原。它还不是完整的 merge tree，但它第一次证明了：当前 runtime 不只会做 plan 和 host-side semantics，而是真的已经能承接一个 exact `P/Q/T` merge 节点的 device execution。到这一步为止，E3 已经同时有了“规模骨架”“对象语义骨架”和“第一段真实 tensor merge execution”。

E3 的第四块切片也已经补上：新增 `PiPqtTensorTreeReport` 与 `--pi-pqt-tree-tensor-smoke --pi-terms`，把 leaf tensor construction、device-side zero-pad 与多节点 bottom-up merge execution 接到同一条 native-RNS 路径里。这里最关键的不是 smoke 本身，而是它第一次把“base/模数组合是否够用”这件事从推测变成了可执行诊断。最初的三模数组合只有约 `91` bits CRT 动态范围，因此 `4 terms` 在 `16-bit limbs` 下可以保持 exact，`8 terms` 必须把 limb base 主动降到 `8 bits` 才继续保持 exact，而 `16 terms` 即使再降到 `4-bit limbs` 也会失配。随后这条线继续向前推进：可用模数组先被扩展到 `7` 个 NTT-friendly primes，深树 smoke 在 `term_count > 8` 时显式切到 `7` 模数实验路径，CRT 动态范围提升到约 `212` bits；再往前一刀，又把可用模数组扩展到 `10` 个，并把深树路径分成 `3 -> 7 -> 10` 三层。期间也暴露出并修掉了几个真正的实现问题：host 侧 centered reconstruction 还停留在 `__int128`，`plan_pi_execution()` 里的模数积统计也仍在用 `__int128`，以及 shared-NTT path 中 `primitive_root_for_modulus_index()` 只覆盖旧的 `7` 模数，导致新增模数错误落到 `root = 1`。把这几层都补齐后，`16 terms` 在 `4-bit limbs` 下重新回到 exact，`32 terms` 在 `2-bit limbs` 下保持 exact，`47 terms` 在 `1-bit limbs + 7 moduli` 下保持 exact，而现在 `64 terms` 也已经在 `1-bit limbs + 10 moduli` 下保持 exact。于是当前阶段的真实阻塞又一次前移：它不再是“如何越过 47 terms”，而是很具体地变成了“如何越过当前 10 模数路径的 `64 terms` ceiling”。

E3 的第五块切片也已经出现：新增 `PiEndToEndReport` 与 `--pi-end-to-end-smoke`，把 GPU tensor-tree 产出的 root `Q/T` 真正接到了最终 Chudnovsky 收口上。当前实现仍然把最后的 `sqrt/division` 保持在 host 侧，而且是刻意用仓内 `HostBigInt` 补齐最小 exact `isqrt/division` 路径，而不是重新引入旧 hybrid fallback；这样做的目的不是宣称“最终 closing 已经完成”，而是先证明当前主线已经可以从 GPU multiply tree 一路走到正确的 `pi` 前缀。现阶段这条 smoke 已经在 `50 digits` 和 `100 digits` 两个 case 上通过，说明 E3 不再只是局部语义切片的集合，而是真正有了一条可执行的 end-to-end `pi` 闭环。

E4 现在也已经不只是一块起步 benchmark：新增 `PiBenchmarkReport` 与 `--pi-end-to-end-benchmark` 之后，这条端到端 `pi` 路线已经能正式拆成 `tree execution / host closure / full end-to-end` 三段计时，并输出 CSV。随后又把 benchmark 路径里的中间节点 metadata 下载从 hot path 里拔掉，只保留 root exactness，这让 benchmark 更接近真正想看的 tree 执行本体。在这个更干净的口径上，`400 digits / 31 terms / 3 iterations` 给出约 `6674 digits/s`，更贴近旧 7 模数 ceiling 的 `630 digits / 47 terms / 3 iterations` 给出约 `5410 digits/s`，而随着 10 模数深树路径打通，新的 `870 digits / 64 terms / 3 iterations` 也已经给出约 `4263 digits/s`。这组数据最重要的意义不是绝对速度，而是它继续确认：当前阶段的时间大头依旧钉死在 `GPU tensor tree` 本体，而不是 host-side bigint closing；换句话说，现阶段要继续追的主阻塞已经从“哪里慢”收敛成了“如何越过 64-term ceiling 并把更深 tree 组织起来”。

## 当前最重要的取舍

这条 aggressive 路线的价值不在于“短期内立刻比当前 hybrid 更快”，而在于：

1. 它第一次把表示层问题正面解决。
2. 它让后续优化不再被 carry/trim/host fallback 这些旧语义反复拖回去。
3. 它更像一个真正能继续扩展的研究项目，而不是 stage-1/stage-2 原型的自然延长。

当前 smoke 的最新阶段拆解给了更明确的信号：

1. `64 x 4096` 的 `ntt_global --no-verify` 在当前 C 阶段验收版 `c45lifecycle` 下达到 `avg_kernel_ms = 0.585498 ms`、`avg_end_to_end_ms = 0.599273 ms`，对应约 `437.44M scalar slots/s`。
2. 同一规模切到 `verify_mode=sampled --verify-samples 64` 后，`avg_download_reconstruct_ms = 0.090364 ms`，`avg_end_to_end_ms = 0.779366 ms`，仍保留约 `336.36M scalar slots/s`。
3. 同一规模切到 `verify_mode=full` 后，`avg_download_reconstruct_ms` 仍高达 `354.231544 ms`，而由于 kernel 已进一步压低，`avg_download_over_kernel_ratio` 升到约 `513.44x`，端到端吞吐仍只有约 `265k scalar slots/s`。
4. `2 x 3000` 的 correctness case 在 `full` 和 `sampled(64)` 两种模式下都保持 `status = ok`，说明验证分层没有破坏 `8192` padded size 的正确性。
5. 这意味着当前 C 阶段的第一性瓶颈已经被明确拆开：
   - `no-verify / sampled` 看的是 device hot path
   - `full` 看的是严格 correctness 基线
6. 所以这一阶段最真实的阻塞，不再是 “不知道慢在哪里”，而是已经定位到：
   - 研究吞吐时，主瓶颈在 NTT/pointwise/encode 本身
   - 研究严格闭环 exactness 时，主瓶颈仍是 host download + CRT reconstruction
7. 在 forward final-stage 融合、pointwise sampled triplet gather 与 lifecycle/block 语义接入之后，当前 C 阶段验收版把 `64 x 4096 sampled(64)` 的 `avg_convolution_ms` 压到 `0.500941 ms`、`avg_kernel_ms` 压到 `0.585325 ms`；这说明当前 hot path 里，stage 组织和全局内存往返仍是最值得继续深挖的部分，而 sampled 辅路径也已经被进一步削薄。

Milestone C 现在已经完成验收，而 Milestone D 的五个最小语义切片 `scaled constants / reciprocal seed prototype / sqrt seed prototype / division quotient prototype / shared correction domain` 也已经全部迁入主线。Milestone E 的前两步也已经完成：完整 `pi` 路线已经明确收敛到 `exact Chudnovsky + binary splitting`。接下来不再回头讨论路线切换，而是直接进入 E3/E4：把 binary-splitting 的对象语义、乘法树调度、最终 `sqrt/division` 收口和真实 `digits/s` 评测接到这条 native RNS 主线上；如果之后继续深挖函数层，也应该是在这套共享 correction API 上把 reciprocal/sqrt/division prototype 扩成更完整的 Newton 迭代。
