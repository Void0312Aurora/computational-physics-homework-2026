# Next-Gen CPU Architecture

这份文档定义 `HW/05/project2_pi/` 的下一代纯 CPU 架构。它必须是一次真正的 clean-slate 重启，而不是把 `gmp_backend.cpp` 的思想换个壳继续搬过来。

## 立场

这条新架构先把边界说死：

1. 不继续扩展 `gmp_backend.cpp`
2. 不继续扩展 `gmp_backend_levelpool.cpp`
3. 不在运行时计算路径里使用 `mpz_t`
4. 不在运行时计算路径里使用 `mpn_*`
5. 不允许“先算到最后再桥接回 GMP 做 sqrt/div”
6. 不再给旧主线追加新的 `parallel_mode`

换句话说，新架构不是“旧 GMP 路线的下一版”，而是“新的 CPU exact engine”。

## 当前已落地内容

当前已经开始落地 `Phase 1` 骨架，代码位于：

- `project2_pi/nextgen_cpu/`

本轮已经实现：

- `PageArena`
- `BigHandle`
- `TripleSlot`
- canonical block add/mul/affine-merge smoke
- `canonical 32-bit block -> residue inject -> cached multi-modulus NTT -> direct CRT carry -> canonical` 乘法 smoke
- 支持跨 level arena 读输入 / 写输出的乘法接口
- destructive product tree prototype
- 独立可编译目标：`project2_nextgen_cpu_smoke`
- 独立 benchmark 目标：`project2_nextgen_cpu_benchmark`
- 独立 tree benchmark 目标：`project2_nextgen_cpu_tree_benchmark`

当前 smoke 结果记录在：

- `result/project2_nextgen_cpu_smoke.log`

当前 isolated multiply benchmark 结果记录在：

- `result/project2_nextgen_cpu_multiply_benchmark.csv`
- `result/project2_nextgen_cpu_multiply_benchmark.log`
- `result/project2_nextgen_cpu_multiply_benchmark_extended.csv`
- `result/project2_nextgen_cpu_multiply_benchmark_extended.log`
- `result/project2_nextgen_cpu_tree_benchmark.csv`
- `result/project2_nextgen_cpu_tree_benchmark.log`

这一阶段的目标不是性能，而是先把下面三件事变成代码事实：

1. 新目录中的运行时计算路径不依赖 `mpz_t`
2. 新目录中的运行时计算路径不依赖 `mpn_*`
3. `(P,Q,T)` 已经从“拥有内存的对象树”切换成 handle + arena 的基本语义
4. 新目录已经开始拥有自己的 CPU 侧 `RNS/NTT` 乘法域，而不再只是 canonical block 骨架

也就是说，clean-slate 现在已经从文档进入代码骨架阶段了。

## 当前 benchmark 结论

当前已经补了两轮 isolated multiply benchmark，用固定随机种子比较：

- canonical schoolbook multiply
- `RNS/NTT` multiply + CRT 重建

当前 quick benchmark 仍覆盖 `8 / 16 / 32 / 64 / 128 / 256` blocks；在完成一轮结构性重构之后，`256` blocks 的结果已经更新为：

- schoolbook 平均约 `2.99985e-4 s`
- `RNS/NTT` 平均约 `8.44737e-4 s`
- 当前 `speedup_rns_vs_schoolbook` 约 `0.355`

同时 extended benchmark 继续把规模推到 `512 / 1024 / 2048 / 4096` blocks，结果显示 crossover 已经出现：

- `512` blocks：`speedup_rns_vs_schoolbook` 约 `0.673`
- `1024` blocks：`speedup_rns_vs_schoolbook` 约 `1.282`
- `2048` blocks：`speedup_rns_vs_schoolbook` 约 `2.381`
- `4096` blocks：`speedup_rns_vs_schoolbook` 约 `4.109`

同时，本轮已经把乘法核推进到 destructive product tree prototype。默认 tree benchmark 覆盖：

- `32 x 64` blocks：adaptive `RNS` 约 `0.980x`
- `32 x 128` blocks：adaptive `RNS` 约 `1.462x`
- `64 x 64` blocks：adaptive `RNS` 约 `1.405x`
- `64 x 128` blocks：adaptive `RNS` 约 `2.152x`
- `64 x 256` blocks：adaptive `RNS` 约 `3.477x`

这说明当前 phase 的状态很明确：

1. 新引擎的跨域乘法链条已经打通
2. correctness 已经不再依赖旧 GMP
3. `RNS/NTT` 已经开始从“正确的第一版”进入“中大规模能赢”的阶段，并且这个优势已经能延续到 tree workload
4. 但 current tree prototype 的 `peak_live_blocks` 仍和 schoolbook 同量级，说明我们还没有拿到真正的 transform-resident 上层表示收益

这轮带来跨越的关键，不是继续调参数，而是直接改了乘法主路径的表示与执行方式：

- 把 `RNS` 输入从 `base-2^16` 改成直接复用 canonical `32-bit block`
- 给每个 `ntt_size` 引入 cached plan，预计算 bit-reversal、stage roots、`inv_n`
- 引入 `RnsWorkspace`，去掉逐次乘法里的 scratch 重新分配
- 把 CRT 常量移出逐系数循环，并直接 carry 回 canonical block
- 把 multiply kernel 改成支持跨 arena 源/目标，从而能真的按 level 生命周期组织 destructive tree

因此下一步的重点不该再是“是否继续验证正确性”，而是：

- 继续把 `RNS/NTT` 域推进到真正的 transform-resident reduction tree，而不是只在每次 merge 时临时进出
- 在 butterflies 内引入更激进的 lazy reduction / redundant range
- 继续减少 canonical 与 transform 域之间的来回切换

## 为什么必须更激进

旧 CPU 路线已经把结论说透了：

- `frontier` 说明只改 merge 顺序会制造新的串行热点
- `levelpool` 说明只改对象复用只能拿到低个位数百分比收益
- `mpn` benchmark 说明只把高层 `mpz_mul` 换成 `mpn_mul_n` 也不会自动带来数量级突破

所以如果目标真的是明显更快，而不是再拿 `3%~8%`，那就不能再围绕 `mpz/mpn` 生态继续缝补。

## 新架构的核心方向

下一代要走的是：

- `page/block engine`
- `redundant arithmetic`
- `transform-resident large multiply`
- `destructive reduction`
- `custom reciprocal / custom sqrt`

也就是：

1. 小规模阶段用自定义 block 表示。
2. 大规模乘法进入多模数 `RNS/NTT` 域。
3. reduction tree 的上层尽量保持在 transform-friendly 表示里，而不是频繁回 canonical 大整数。
4. 最终倒数、开方、除法也留在同一套 engine 里完成。

这才算“新起炉灶”。

## 不再接受的旧思路

以下思路在新架构里全部视为否决项：

- `Triple(mpz_t p, q, t)` 作为热路径基本单位
- 节点级对象创建/清理
- `std::vector<Triple>` 整轮重建
- 最终阶段回旧 GMP 后端“先跑通再说”
- 用 `parallel_mode=` 继续堆变体
- 先把旧主线对象树保留，再在旁边局部替换一两个 kernel

这些做法的共同问题是：它们看起来像重构，实际上仍然受旧表示层牵制。

## 选定的新基本模型

### 1. 双域精确整数引擎

下一代 CPU backend 采用双域模型：

- `Canonical Block Domain`
  - 用固定 radix 的 block/page 存储精确整数
  - 负责 leaf 生成、局部加减、小规模乘法、格式化前的最终归一
- `RNS/NTT Domain`
  - 用多模数 residue 表示大整数
  - 负责大规模乘法和 reduction tree 的上层主力计算

核心不是“有两个表示”本身，而是：

- 大整数一旦足够大，就不再继续停留在 canonical limb 对象里做乘法
- 而是提升到 `RNS/NTT` 域，把大乘法批量消化掉

这比“arena + mpn”更激进，也更有希望拿到结构性收益。

### 2. destructive reduction tree

merge 不再是：

- 构造新节点
- `left * right`
- 写回新的拥有者对象

而是：

- 每层有自己的输出 page arena
- `merge_into(dst_slot, left_slot, right_slot, workspace)`
- 完成一层后，整层输入 arena 直接回收

也就是说，新架构按“层生命周期”管理内存，而不是按“节点生命周期”管理内存。

### 3. delayed carry / redundant blocks

canonical block 域不应该每做一次加减乘就立刻完全归一化。

要允许：

- signed blocks
- bounded redundant range
- 局部 merge 后延迟 carry
- 在需要跨域转换或最终输出时再集中归一

如果还像传统 `mpz_t` 那样每步都紧绷着 canonical form，性能空间会被提前吃掉。

## 建议的数据结构

### `PageArena`

职责：

- 管理大块连续 page
- 每个 page 存固定数量的 blocks
- 支持 bump allocate / whole-level recycle

建议字段：

- `block_t* base`
- `page_count`
- `page_size_blocks`
- `cursor`

### `BigHandle`

职责：

- 只描述整数在 arena 中的位置和元信息
- 不拥有内存，不做 RAII 风格的单对象析构

建议字段：

- `page_id`
- `offset_blocks`
- `used_blocks`
- `capacity_blocks`
- `sign`
- `domain`

其中 `domain in {canonical, rns}`

### `TripleSlot`

职责：

- 对应 `(P,Q,T)`，但不再是三个对象，而是三个 handle

建议注意：

- `P/Q/T` 的容量策略必须分开
- `T` 不应再被迫和 `P/Q` 走同一种 sizing 逻辑

### `LevelPool`

职责：

- 管理 reduction tree 某一层的全部 `TripleSlot`
- 按层整体分配、整体回收

### `ThreadWorkspace`

职责：

- 保存每个线程的局部 scratch
- 包括：
  - small-mul scratch
  - carry scratch
  - NTT temporary buffers
  - CRT / base conversion scratch

## 算术内核策略

### 小规模

小规模先用：

- Comba / schoolbook
- Karatsuba
- Toom-Cook

这些全部针对 `Canonical Block Domain` 自己实现。

### 大规模

大规模必须转入：

- 多模数 `RNS/NTT`
- SIMD/AVX2 批处理 butterflies
- 固定 scratch 复用

重点不是“也能做 FFT”，而是：

- reduction tree 的高层不能再回到传统大整数对象乘法
- 必须让上层主乘法长期停留在 transform-friendly 域

### 跨域转换

转换只发生在明确边界：

- leaf canonical -> upper-level rns
- rns -> canonical for final normalize / output

要避免：

- 每一层 merge 都来回转域
- 每次乘法后立即回 canonical

## 对 Chudnovsky 的专门化

下一代不建议继续把 binary splitting 看成“通用三元组树”。

更激进的做法是把它拆成三个专门部件：

1. `P-tree`
   - 小因子乘积树
2. `Q-tree`
   - 主乘法成本最大的积树
3. `T affine engine`
   - 专门处理 `T1 * Q2 + P1 * T2`

也就是说，`T` 的处理逻辑不应只是“和 P/Q 一样再来一遍乘法”，而要按它的 affine 结构单独设计 kernel 和 scratch 策略。

这是比“继续维护通用 Triple 对象”更激进、也更合理的方向。

## 并行模型

新的并行模型不再是“旧对象树 + OpenMP”。

建议采用：

1. leaf phase
   - 静态分块
   - 每线程独占输入区间和 workspace
2. level reduction phase
   - 每层并行
   - 每任务只写自己负责的输出 slot
3. transform batches
   - 同规模乘法按 batch 组织
   - 共享 twiddle/cache/scratch

并行的目标不只是“多线程都在干活”，而是：

- 没有共享输出对象竞争
- 没有节点级 heap 行为
- 没有跨层残留垃圾内存

## 最终阶段

这里是新架构必须比我上一版更强硬的地方：

- 最终倒数
- 最终开方
- 最终除法

都不再允许桥接回旧 GMP 计算路径。

新架构必须自己提供：

- reciprocal engine
- sqrt engine
- division / normalization engine

GMP 最多只允许作为：

- 离线 verifier
- 单元测试参考答案生成器

而不是 next-gen runtime 的一部分。

## 目录建议

建议明确新建独立目录：

- `project2_pi/nextgen_cpu/`

并拆成这些模块：

- `nextgen_cpu/page_arena.hpp`
- `nextgen_cpu/block_handle.hpp`
- `nextgen_cpu/triple_slot.hpp`
- `nextgen_cpu/canonical_smallmul.cpp`
- `nextgen_cpu/canonical_carry.cpp`
- `nextgen_cpu/rns_domain.cpp`
- `nextgen_cpu/ntt_kernels.cpp`
- `nextgen_cpu/base_convert.cpp`
- `nextgen_cpu/p_tree.cpp`
- `nextgen_cpu/q_tree.cpp`
- `nextgen_cpu/t_affine_engine.cpp`
- `nextgen_cpu/reciprocal.cpp`
- `nextgen_cpu/sqrt.cpp`
- `nextgen_cpu/final_division.cpp`
- `nextgen_cpu/main.cpp`

## 分阶段落地计划

### Phase 0

- 冻结旧基线
- 不再修改 `gmp_backend.cpp` / `gmp_backend_levelpool.cpp` 的性能路径

### Phase 1

- 建立 `nextgen_cpu/` 目录
- 落地 `PageArena`、`BigHandle`、`TripleSlot`
- 要求：
  - `nextgen_cpu/*` 中不出现 `mpz_t`
  - `nextgen_cpu/*` 中不出现 `mpn_*`

### Phase 2

- 完成 canonical block 域：
  - add/sub
  - carry normalize
  - small mul
- 完成小位数 exact smoke

### Phase 3

- 完成 `RNS/NTT` 大乘法域
- 完成 canonical <-> rns base conversion
- 做 isolated multiply benchmarks

### Phase 4

- 完成 `P-tree / Q-tree / T affine engine`
- 跑 `1M / 10M` 端到端

### Phase 5

- 完成 custom reciprocal / custom sqrt / final division
- 不允许 runtime 回落到旧 GMP
- 跑 `50M / 100M`

### Phase 6

- 与 frozen baseline 正面对照
- 如果收益仍然只有低个位数百分比，则继续下沉到：
  - 更激进的 truncated arithmetic
  - 更激进的 transform-resident `T` pipeline

## 验收标准

新架构是否合格，不看“是不是终于跑通了”，而看下面这些硬标准：

1. 运行时计算路径中不出现 `mpz_t`
2. 运行时计算路径中不出现 `mpn_*`
3. reduction tree 的上层主乘法停留在 `RNS/NTT` 域
4. 最终 reciprocal / sqrt / division 不回旧 GMP
5. 相比 frozen baseline，必须争取明显不是个位数的收益

建议先用这些目标压自己：

- `50M` 至少追求 `> 1.5x`
- `100M` 至少追求 `> 1.3x`

如果连这个量级都摸不到，就说明新架构还不够新。

## 当前决策

当前正式决策应该是：

- 旧 CPU 主线已经结束
- 旧 CPU 主线只剩基线价值
- 下一步不是“继续优化 `gmp_backend.cpp`”
- 下一步是“开始实现一个新的 CPU exact engine”
