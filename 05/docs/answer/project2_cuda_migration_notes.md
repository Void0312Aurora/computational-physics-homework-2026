# Project 2 CUDA 迁移阶段记录

## 记录目的

这份记录不是最终提交文档的定稿，而是为后续补入 `docs/answer/answer.md` 准备的阶段性素材。重点记录三件事：

1. 目前已经完成了哪些从混合 `CPU/GPU` 向更彻底 `CUDA` 路线迁移的工作。
2. 这些工作对应了哪些真实文件和测试结果。
3. 当前“能跑通”和“真正提速”之间还差在哪里。

## 先前阶段的主线进展

在进入 `full-CUDA` 路线之前，`Project 2` 已经沿着混合精确计算主线完成了若干轮真实优化：

1. 把最终大整数除法从纯 `mpz //` 推进到 `newton-chunk-gpu-seed-prototype`，显著压低了 `final_division_seconds`。
2. 把 $\sqrt{10005}$ 的高精度构造从 `CPU isqrt` 推到 `chunk-gpu-rsqrt-prototype`，把 `sqrt_compute_seconds` 从原先的十秒级压低到三秒级附近。
3. 重新调优 `binary splitting` 的 `chunk_terms / bs_leaf_terms`，把 `partial_generation_seconds` 明显压低。
4. 把 `merge_tree` 中每次 GPU 乘法前无条件 `empty_cache()` 的路径拿掉，并增加 `trim_chunks()` 热路径快分支，把 `merge_tree_seconds` 继续往下压。

截至这一步，当前最佳 `100M` 位完整 `pi` 实测结果来自：

- `result/project2_gpu_hybrid_mergeopt_trimfast_100m.csv`
- 总时间 `15.693548 s`
- 吞吐率 `6.372045 M digits/s`

这已经明显快于更早的 `19.03 s / 5.25M digits/s`，但仍然没有达到期望中的 `30M/s` 量级。因此，后续工作不再是继续做混合路径微调，而是正式开启 `CUDA` 迁移线。

## 本轮新增的 full-CUDA 基建

本轮新增了三层基础设施：

1. `cuda/project2_chunk_ops.cpp`
2. `cuda/project2_chunk_ops.cu`
3. `project2_cuda_chunk_ops.py`

它们组成了一个可热加载的 `torch cpp_extension`，当前已经支持在 `base = 2^16` little-endian limb 表示上运行这些 CUDA 算子：

- `trim`
- `compare`
- `add`
- `sub`
- `mul_small`
- `div2`

这些算子目前仍然属于 `stage-1 prototype`：

- 每个 kernel 还是单 block / 单线程风格
- 有若干长度和比较结果需要从 GPU 回读到 host
- 正确性优先于性能

也就是说，它们的意义主要是“把全 CUDA limb runtime 主干立起来”，而不是立刻冲高吞吐。

## 新增的 full-CUDA 原型入口

新增脚本：

- `project2_gpu_pi_full_cuda.py`

它当前承担的是 `stage-1 full-CUDA runtime smoke benchmark`，而不是最终的纯 GPU 全流程 `pi` 实现。它会在同一个 GPU limb 表示上跑：

- `add`
- `sub`
- `mul_small`
- `div2`
- `compare`
- `GPU FFT multiply`

对应的 smoke 结果文件：

- `result/project2_full_cuda_prototype_smoke.csv`

这份 smoke 的主要意义是确认：

1. `CUDA extension` 能在本机编译通过。
2. 新算子和现有 `GpuFftMultiplier` 可以在同一个 CUDA 数据表示上共存。
3. 后续把 `merge / division / sqrt` 逐步换到这一套 runtime 上是可行的。

## 新增的端到端接入测试

为了不让 `full-CUDA` 路线停留在“只有单个算子 benchmark”，本轮还把新的 chunk 算术后端以可切换方式接进了现有主流水线：

- `project2_gpu_pi_hybrid.py` 新增 `--chunk-arith-backend`
- 可选值：`python`、`cuda-ext`、`auto`

其中 `cuda-ext` 会在 `binary16 + CUDA` 路线下，把部分 GPU chunk 算术切到新 extension：

- `trim_chunks`
- `compare_abs_chunks`
- `add_abs_chunks`
- `sub_abs_chunks`
- `mul_abs_chunks_small`
- `div_chunks_by_two`

这样做的目的不是宣称“已经完全纯 GPU”，而是先完成一件更重要的事：让新的 CUDA limb runtime 真正跑进完整 `pi` 主流水线，并给出真实端到端结果。

## 本轮真实测试结果

### 1. CUDA chunk 算子正确性

进行了 20 组随机 case 的 CPU 参考对照，结果为：

- `random_cuda_chunk_ops_ok`

这说明当前 `trim / compare / add / sub / mul_small / div2` 的 CUDA 实现，在随机小规模 limb case 上和 CPU 参考结果一致。

### 2. full-CUDA runtime smoke

文件：

- `result/project2_full_cuda_prototype_smoke.csv`

命令：

```bash
python project2_gpu_pi_full_cuda.py --chunk-length 4096 --repeats 1 --csv result/project2_full_cuda_prototype_smoke.csv
```

结果摘要：

- `chunk_length=4096`
- `add_seconds=0.010876`
- `sub_seconds=0.000565`
- `mul_small_seconds=0.000410`
- `div2_seconds=0.000416`
- `compare_seconds=0.000131`
- `fft_mul_seconds=0.140290`

### 3. 主流水线端到端 smoke

文件：

- `result/project2_gpu_hybrid_chunkarith_python_smoke.csv`
- `result/project2_gpu_hybrid_chunkarith_cudaext_smoke.csv`

命令：

```bash
python project2_gpu_pi_hybrid.py \
  --digits-list 100000,500000,1000000 \
  --gpu-stages merge-and-final \
  --gpu-chunk-format binary16 \
  --sqrt-mode chunk-gpu-rsqrt-prototype \
  --division-mode newton-chunk-gpu-seed-prototype \
  --chunk-arith-backend python \
  --csv result/project2_gpu_hybrid_chunkarith_python_smoke.csv \
  --output ''

python project2_gpu_pi_hybrid.py \
  --digits-list 100000,500000,1000000 \
  --gpu-stages merge-and-final \
  --gpu-chunk-format binary16 \
  --sqrt-mode chunk-gpu-rsqrt-prototype \
  --division-mode newton-chunk-gpu-seed-prototype \
  --chunk-arith-backend cuda-ext \
  --csv result/project2_gpu_hybrid_chunkarith_cudaext_smoke.csv \
  --output ''
```

端到端结果对比如下：

| 位数 | `python` backend | `cuda-ext` backend | 备注 |
| ---: | ---: | ---: | --- |
| 100000 | `0.499983 s` | `0.492219 s` | 两者都 `prefix_ok=True` |
| 500000 | `0.309613 s` | `0.620889 s` | `cuda-ext` 明显更慢 |
| 1000000 | `0.656236 s` | `0.945293 s` | `cuda-ext` 仍更慢 |
| 5000000 | `2.810535 s` | `4.657391 s` | 中等规模下差距进一步拉大 |

这说明本轮最重要的事实不是“已经提速”，而是：

1. 新的 CUDA chunk 算术已经成功进入完整 `pi` 主流水线。
2. 目前它在真实端到端路径上仍然偏慢。
3. 慢的原因和实现形态一致：当前 kernel 还属于单线程串行 carry/borrow 原型，并且有多次 host 标量回读。

`5M` 位那组结果还额外说明了一个细节：`cuda-ext` 路线下 `gpu_backend_total_seconds` 反而低于 `python` 路线，但总时间更差。这意味着当前真正拖后腿的并不是 FFT 主乘法，而是新引入的 stage-1 limb 算术和相关 correction 路径。这一点对于后续优化方向非常关键。

## 当前判断

到目前为止，`full-CUDA` 迁移已经完成了“可编译、可验证、可接入主流程”的第一阶段，但还没有完成“性能上胜过原有张量实现”的第二阶段。

这并不矛盾，反而是正常的阶段结论：

1. 先把 GPU limb runtime 建起来，证明主架构是可行的。
2. 再逐步把原型 kernel 改成真正并行的 scan/carry/borrow 路线。
3. 最后再谈数量级性能目标。

如果直接跳过第 1 步，后面很容易陷入“目标很大，但主路径没有落地”的空转。

## 本轮对 bottleneck segment 的修复

在 `stage-1` 路线能跑进主流程之后，又继续针对真正的慢段做了拆分 profiling。结果表明：

1. 并不是所有 chunk 算术都慢。
2. 真正拖后腿的是大尺寸 `add / sub / mul_small / compare`。
3. 它们慢的根源不是 CUDA 本身，而是当前 `project2_chunk_ops.cu` 还是单线程 kernel，并且部分结果会回读到 host。

在 `5M` 位 case 上，强制 `cuda-ext` 时最重的几段大致是：

- `add_abs_base65536`
- `mul_small_base65536`
- `sub_abs_base65536`

进一步对不同 chunk 长度做微基准后，可以看到一个很清晰的分界：

1. 小尺寸 chunk 下，`cuda-ext` 仍然有价值。
2. 到了 `32768` 甚至更大的 chunk，现有单线程 extension 已经明显慢于原本的 `torch` 张量实现。

因此当前最有效的修复不是“继续强推全部走 extension”，而是引入分流策略：

1. `--chunk-arith-backend python`：全部使用原来的张量实现。
2. `--chunk-arith-backend cuda-ext`：全部强制走 extension，便于继续研究纯 CUDA 路线。
3. `--chunk-arith-backend auto`：只让小 chunk 走 `cuda-ext`，大 chunk 自动回退到原本更快的张量实现。

经过阈值 sweep，当前代码把 `PROJECT2_CUDA_CHUNK_ARITH_SMALL_LEN` 固定为 `65536`。这一步并不意味着“已经完成 full CUDA”，但它确实解决了当前最现实的 bottleneck segment：避免让单线程 extension 去接管数百万到上千万 chunk 的大整数后处理。

## bottleneck segment 修复后的真实结果

文件：

- `result/project2_gpu_hybrid_chunkarith_auto_final.csv`

命令：

```bash
python project2_gpu_pi_hybrid.py \
  --digits-list 5000000,20000000,100000000 \
  --gpu-stages merge-and-final \
  --gpu-chunk-format binary16 \
  --sqrt-mode chunk-gpu-rsqrt-prototype \
  --division-mode newton-chunk-gpu-seed-prototype \
  --chunk-arith-backend auto \
  --csv result/project2_gpu_hybrid_chunkarith_auto_final.csv \
  --output ''
```

结果如下：

| 位数 | 时间 | 吞吐率 |
| ---: | ---: | ---: |
| 5000000 | `1.456719 s` | `3.432371 M digits/s` |
| 20000000 | `3.001980 s` | `6.662269 M digits/s` |
| 100000000 | `15.417480 s` | `6.486144 M digits/s` |

和此前的代表性结果相比：

1. `5M` 位从 `2.810535 s` 压到 `1.456719 s`。
2. `20M` 位从 `4.166239 s` 压到 `3.001980 s`。
3. `100M` 位从 `15.693548 s` 压到 `15.417480 s`。

这说明“瓶颈段分流”是有效的，而且效果主要体现在中等规模位数上；`100M` 位虽然改善没有 `5M / 20M` 那么剧烈，但仍然取得了新的最佳结果。

## 本轮继续压 FFT 热路径后的结果

在把 chunk 算术的慢段分流掉之后，再继续往下看，`100M` 位 case 中仍然有一个很明确的大头：

1. `gpu_backend_total_seconds` 仍然在 `8s+`。
2. 其中真正的大头是大量 GPU-resident exact multiply，而不是 Python 外层调度。
3. 因此下一刀不再是继续改 `cuda-ext` limb kernel，而是直接改 `project2_gpu_fft_backend.py` 的热路径。

这轮做的两个改动比较克制，但都直接瞄准了大整数 FFT 乘法主干：

1. 去掉 `multiply_chunks()` 里对输入 tensor 的无条件 `copy=True`，只在 dtype 或 contiguous 性不满足时才做规范化复制。
2. 给混合主线的 hot path 增加 `assume_nonzero_trimmed=True`，从而把每次乘法前的整向量 `count_nonzero()` 扫描，换成对已 trim 操作数的便宜零判断。

对应代码位置：

- `project2_gpu_fft_backend.py`
- `project2_gpu_pi_hybrid.py`

为了避免把冷启动和热态结果混在一起，这轮比较采用和上一轮 `auto_final` 相同的顺序：

```bash
python project2_gpu_pi_hybrid.py \
  --digits-list 5000000,20000000,100000000 \
  --gpu-stages merge-and-final \
  --gpu-chunk-format binary16 \
  --sqrt-mode chunk-gpu-rsqrt-prototype \
  --division-mode newton-chunk-gpu-seed-prototype \
  --chunk-arith-backend auto \
  --csv result/project2_gpu_hybrid_fftopt_final.csv \
  --output ''
```

结果如下：

| 位数 | 时间 | 吞吐率 |
| ---: | ---: | ---: |
| 5000000 | `1.490251 s` | `3.355139 M digits/s` |
| 20000000 | `3.023205 s` | `6.615496 M digits/s` |
| 100000000 | `15.168054 s` | `6.592804 M digits/s` |

和上一轮 `project2_gpu_hybrid_chunkarith_auto_final.csv` 相比：

1. `5M / 20M` 基本属于持平附近的小幅波动，没有出现新的数量级提升。
2. `100M` 位从 `15.417480 s` 继续压到 `15.168054 s`，新增约 `0.249 s` 的改进。
3. `100M` 位吞吐率从 `6.486144 M digits/s` 提高到 `6.592804 M digits/s`。
4. `100M` 位 `gpu_backend_total_seconds` 从 `8.314041 s` 降到 `8.220095 s`，`gpu_kernel_seconds` 从 `6.083703 s` 降到 `6.016804 s`。
5. `100M` 位 `sqrt_compute_seconds` 也从 `2.822942 s` 降到 `2.715814 s`，说明这组改动不只影响 merge tree，也影响了 reciprocal-sqrt 原型里的大乘法热段。

因此，这轮优化的实际结论比较清楚：

1. 它确实进一步压缩了当前混合主线里最大的 GPU multiply 热段。
2. 收益主要体现在超大位数 case，而不是小中位数。
3. 这还不是“再提一个数量级”的变化，但它证明当前 `100M` 路线上仍然存在可以继续从 FFT 主后端里挖出来的真实空间。

## 本轮把 reciprocal / sqrt bootstrap 拔出 CPU `mpz/mpfr` 主线

在继续朝更彻底的 GPU 迁移推进时，下一步先处理的是 active 主线中最显眼的两个 CPU bootstrap 点：

1. `reciprocal_chunks_gpu_seed_prototype()` 之前会把高位 prefix 拉回 CPU，转成 `mpz` 再做 reciprocal seed。
2. `reciprocal_sqrt_constant_gpu_seed_prototype()` 之前会调用 `reciprocal_sqrt_constant_bootstrap()`，本质上还是 `mpfr/mpz` 高精度 bootstrap。

这两段虽然只发生在 bootstrap / seed 阶段，但它们会把主线重新拖回 CPU 高精度路径，不符合继续往 GPU 迁移的目标。

因此本轮改成了新的做法：

1. 用 `torch.float64` 在 GPU 上只读取最高若干个 chunk，构造一个带上界修正的归一化高位近似。
2. 用这个近似直接生成一个“保守偏低”的 reciprocal / reciprocal-sqrt 小精度种子。
3. 不再假装它一开始就有 `1024 / 4096` chunk 的真实精度，而是从小精度开始，通过 Newton doubling 逐轮扩展精度。

这里的关键不是“直接造一个巨大精度的假种子”，而是让 bootstrap 的精度增长规律重新和 Newton 迭代本身一致。前者虽然能跑通，但会导致最终 correction 巨大；后者虽然多出若干轮小规模 doubling，但可以把 active 主线里的 CPU correction fallback 重新压回去。

本轮之后，active 主线：

- `sqrt_mode=chunk-gpu-rsqrt-prototype`
- `division_mode=newton-chunk-gpu-seed-prototype`

已经不再依赖 CPU `mpz/mpfr` 做 bootstrap。

## 本轮对 `compare/effective_length` 的 CUDA 并行化补丁

把 bootstrap 拔到 GPU 之后，又继续处理了 `cuda/project2_chunk_ops.cu` 中最容易先动手的两个串行热点：

1. `effective_length_kernel`
2. `compare_abs_kernel`

它们此前仍然是 `<<<1,1>>>` 风格，会在 GPU 上单线程扫完整个 limb 数组。现在已经改成了：

1. `effective_length`：block-stride 扫描 + block 内 max reduction + global `atomicMax`
2. `compare`：并行查找最高不同 limb，再用 packed index/sign 做全局归约

在此基础上，`auto` 路线对 `compare` 单独放宽了 extension 的启用阈值，而没有去提前放宽 `add/sub/mul_small` 这些仍未真正并行 carry/borrow 化的 op。

## active 主线迁移后的真实结果

为了公平比较，仍然采用热态顺序跑：

```bash
python project2_gpu_pi_hybrid.py \
  --digits-list 5000000,20000000,100000000 \
  --gpu-stages merge-and-final \
  --gpu-chunk-format binary16 \
  --sqrt-mode chunk-gpu-rsqrt-prototype \
  --division-mode newton-chunk-gpu-seed-prototype \
  --chunk-arith-backend auto \
  --csv result/project2_gpu_seedstart_comparewide_final.csv \
  --output ''
```

结果如下：

| 位数 | 时间 | 吞吐率 |
| ---: | ---: | ---: |
| 5000000 | `1.682008 s` | `2.972637 M digits/s` |
| 20000000 | `3.192574 s` | `6.264538 M digits/s` |
| 100000000 | `15.192064 s` | `6.582384 M digits/s` |

更重要的是，这组结果对应的 active 主线统计里：

1. `newton_cpu_correction_used=False`
2. `newton_chunk_fastpath_used=True`
3. `prefix_ok=True`

也就是说，当前主线已经在不依赖 CPU `mpz/mpfr` bootstrap 的前提下，重新回到了正确的 GPU-seed + chunk fastpath 轨道。

和上一轮偏向“纯性能最优”的 `project2_gpu_hybrid_fftopt_final.csv` 相比：

1. `100M` 位从 `15.168054 s` 变成 `15.192064 s`，只慢了约 `0.024 s`。
2. 吞吐率从 `6.592804 M digits/s` 变成 `6.582384 M digits/s`，基本属于近似持平。
3. 这说明 active 主线虽然完成了实质性的 bootstrap 迁移，但目前还没有为此付出明显的数量级性能代价。

这点很重要，因为它意味着：

1. 我们已经不只是“把 CPU bootstrap 换了个位置”，而是真的把 active 主线里的这两个高精度构造点迁到 GPU/torch 侧。
2. 当前下一步就可以更放心地继续啃 `cuda/project2_chunk_ops.cu` 里的真正大头：`add / sub / mul_small / div2` 的并行 carry/borrow scan。

## 本轮继续推进 full-CUDA kernel 本体

在把 active 主线里的 bootstrap 拔出 CPU 路径之后，本轮没有再回头改 hybrid 算法结构，而是直接继续推进 `full-CUDA` runtime 本体：

- `cuda/project2_chunk_ops.cu`
- `project2_gpu_pi_full_cuda.py`

这轮真正动到的 kernel 有：

1. `effective_length`
2. `compare`
3. `add`
4. `sub`
5. `div2`

其中：

1. `effective_length` 现在是 block-stride 扫描 + block 内 max reduction。
2. `compare` 现在并行查找最高不同 limb，再做 packed result 归约。
3. `add` / `sub` 不再是单线程 carry/borrow 串行循环，而是基于 `generate / propagate` 状态做 GPU exclusive scan，再并行收尾。
4. `div2` 也不再串行从高位往低位扫，而是按相邻 limb 并行计算：
   - `out[i] = (a[i] >> 1) + ((a[i+1] & 1) << 15)`

仍然还没有完成并行化的主要算子是：

1. `mul_small`

它目前还是单线程 kernel，所以 full-CUDA limb runtime 还没有完全摆脱 prototype 性质。但和之前相比，已经不是“除了 FFT 之外其余 limb 算术都还是串行 bring-up”的状态了。

### 本轮正确性验证

做了 80 组随机 limb case 的 GPU/CPU 对照，覆盖：

1. `add`
2. `sub`
3. `compare`
4. `div2`

结果为：

- `random_cuda_chunk_ops_v2_ok`

说明这轮引入的 scan 型 `add/sub` 和并行 `div2` 至少在随机中等规模 case 上已经和 CPU 参考一致。

### 本轮 full-CUDA runtime benchmark

文件：

- `result/project2_full_cuda_prototype_smoke_v3.csv`
- `result/project2_full_cuda_prototype_262144_v3.csv`

命令：

```bash
python project2_gpu_pi_full_cuda.py \
  --chunk-length 4096 \
  --repeats 1 \
  --csv result/project2_full_cuda_prototype_smoke_v3.csv

python project2_gpu_pi_full_cuda.py \
  --chunk-length 262144 \
  --repeats 1 \
  --csv result/project2_full_cuda_prototype_262144_v3.csv
```

结果摘要：

| chunk length | add | sub | mul_small | div2 | compare | fft_mul | total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | `0.010246 s` | `0.000559 s` | `0.000428 s` | `0.000155 s` | `0.000169 s` | `0.118363 s` | `0.129919 s` |
| 262144 | `0.009232 s` | `0.000465 s` | `0.019399 s` | `0.000335 s` | `0.000192 s` | `0.120029 s` | `0.149651 s` |

从这组结果至少可以读出两个事实：

1. 新并行化的 `div2` 在 full-CUDA runtime 中已经非常轻，说明这条 kernel 的串行依赖确实被消掉了。
2. 当前 full-CUDA runtime 里新的最明显短板已经收缩到 `mul_small`，这也和它仍然是单线程 kernel 的实现状态一致。

## 本轮继续把 `mul_small` 也改成 full-CUDA 并行 kernel

在上一轮之后，`full-CUDA limb runtime` 里最明显的剩余串行大头已经收缩到了：

1. `mul_small`

因此本轮继续直接修改：

- `cuda/project2_chunk_ops.cu`

新的实现不再用单线程 carry 链，而是分成三步：

1. 并行计算每个 limb 的原始乘积 `a[i] * multiplier`
2. 在 GPU 上并行抽取 `carry / remainder`
3. 在 GPU 上把上一位 carry 右移一格再并行加回，循环直到没有新的 carry

这里的外层 still 有一个很短的 host 控制循环，用来判断 carry 是否还存在；但真正的 limb 乘积与归一化都已经在 CUDA kernel 内完成。也就是说，它已经不再是“单线程 GPU 串行大整数乘法”。

### 本轮正确性验证

在上一轮随机对照的基础上，把 `mul_small` 也纳入验证，结果为：

- `random_cuda_chunk_ops_v3_ok`

这说明当前：

1. `add`
2. `sub`
3. `compare`
4. `div2`
5. `mul_small`

这五个 full-CUDA limb 算子在随机 case 上都已经和 CPU 参考一致。

### 本轮 full-CUDA runtime benchmark

文件：

- `result/project2_full_cuda_prototype_smoke_v4.csv`
- `result/project2_full_cuda_prototype_262144_v4.csv`

命令：

```bash
python project2_gpu_pi_full_cuda.py \
  --chunk-length 4096 \
  --repeats 1 \
  --csv result/project2_full_cuda_prototype_smoke_v4.csv

python project2_gpu_pi_full_cuda.py \
  --chunk-length 262144 \
  --repeats 1 \
  --csv result/project2_full_cuda_prototype_262144_v4.csv
```

结果摘要：

| chunk length | add | sub | mul_small | div2 | compare | fft_mul | total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | `0.012032 s` | `0.000590 s` | `0.000427 s` | `0.000141 s` | `0.000153 s` | `0.145909 s` | `0.159253 s` |
| 262144 | `0.011393 s` | `0.000535 s` | `0.000714 s` | `0.000187 s` | `0.000219 s` | `0.148018 s` | `0.161065 s` |

和上一轮 `v3` 相比，最关键的变化不是总时间，而是 `mul_small` 自身：

1. 在 `chunk_length=262144` 时，`mul_small` 从 `0.019399 s` 降到 `0.000714 s`。
2. 这意味着单个 `mul_small` kernel 本身已经取得了数量级上的改进。
3. 但 full-CUDA prototype 的 `total_seconds` 还混着一次大 FFT multiply，因此总时间会被 FFT 波动明显影响，不能简单把 `total` 直接当成这轮 `mul_small` 改动的收益。

因此这轮的正确表述应该是：

1. `mul_small` 这个 full-CUDA limb kernel 已经从串行实现推进到了并行实现。
2. 内核级性能改善是明确存在的。
3. 但 full-CUDA runtime 的端到端总时间还没有同步出现同样幅度的下降，因为它暂时仍被 `fft_mul_seconds` 主导。

## 下一步最值得做的事

后续如果继续推进，优先级应该是：

1. 把 `project2_chunk_ops.cu` 里的单线程 `add/sub/mul_small/div2/trim` 改成并行前缀扫描风格。
2. 避免频繁把“有效长度”和“比较结果”从 GPU 回传到 CPU，把长度元数据尽量保存在 GPU runtime 内。
3. 继续把 `sqrt / reciprocal / division` 中的 chunk 算术逐步替换到这套 CUDA limb runtime。
4. 再往前推进 `partial_generation` 的 GPU 友好表示，最终减少甚至消除 `mpz -> chunk` 的大规模中间转换。

## 建议后续补入正式答案文档的位置

这份记录后续可以拆分补入 `docs/answer/answer.md` 的两个位置：

1. `文件说明`：
   - 增加 `project2_cuda_chunk_ops.py`
   - 增加 `cuda/project2_chunk_ops.cpp/.cu`
   - 增加 `project2_gpu_pi_full_cuda.py`
   - 增加新的 smoke benchmark CSV
2. `Project 2` 的扩展讨论或附录：
   - 说明 `full-CUDA stage-1` 已经完成 runtime 落地和端到端 smoke
   - 明确指出当前瓶颈是 kernel 仍偏原型化，尚未进入真正的高吞吐 scan 实现

## 本轮 v5：收缩 runtime 同步点，并把 full-CUDA 入口接成真实 pipeline

这轮工作的目标不再是继续围着 hybrid 主线做边缘微调，而是把 `full-CUDA` 迁移线往前推两步：

1. 在 `cuda/project2_chunk_ops.cu` 里继续压缩 runtime 的 host 同步点。
2. 把 `project2_gpu_pi_full_cuda.py` 从“只有 chunk 微基准”的壳子，推进成真正会算出 `pi` 的 stage-2 smoke pipeline。

### 1. runtime v5 改动

本轮和 `v4` 相比，最直接的两处改动是：

1. `mul_small` 不再每做一次 carry pass 就回 host 读一次 flag，而是按组执行多次 carry pass 后再统一检查，从而减少 runtime 里的同步频率。
2. `GpuFftMultiplier.multiply_chunks()` 新增 `a_length / b_length`，这样后续 full-CUDA 流水线即使持有带零尾巴的 limb buffer，也可以只按有效 limb 长度做 FFT，不必被 padding 放大工作量。

对应的 `v5` runtime benchmark 文件：

- `result/project2_full_cuda_prototype_smoke_v5.csv`
- `result/project2_full_cuda_prototype_262144_v5.csv`

和上一轮 `v4` 的直接对比如下：

| chunk length | `v4 total` | `v5 total` | `v4 mul_small` | `v5 mul_small` |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | `0.159253 s` | `0.127888 s` | `0.000427 s` | `0.000329 s` |
| 262144 | `0.161065 s` | `0.140322 s` | `0.000714 s` | `0.000626 s` |

这里需要如实解释：

1. `mul_small` 的同步点确实继续缩小了，内核级时间也进一步下降。
2. `total_seconds` 同时还会受到一次 FFT multiply 波动的影响，所以不能把全部降幅都简单归因到 `mul_small`。
3. 但和 `v4` 相比，`v5` 至少已经给出一个更一致的方向：runtime 自身不是在退步，而是在继续收缩。

### 2. full-CUDA stage-2 pipeline smoke 已经接通

本轮最重要的新事实不是某个孤立 kernel 更快，而是：

1. `project2_gpu_pi_full_cuda.py` 现在新增了 `--mode pipeline`。
2. 这条路径会强制走：
   - `merge-and-final`
   - `binary16`
   - `chunk-gpu-rsqrt-prototype`
   - `newton-chunk-gpu-seed-prototype`
   - `cuda-ext`
3. 也就是说，`full-CUDA` 入口已经不再只是“单个 limb 算子 + FFT”的拼盘，而是能真实算出完整 `pi` smoke。

本轮新增的端到端结果文件：

- `result/project2_full_cuda_pipeline_smoke_10k_v1.csv`
- `result/project2_full_cuda_pipeline_smoke_100k_v1.csv`
- `result/project2_full_cuda_pipeline_smoke_1m_v1.csv`

结果摘要：

| 位数 | 时间 | 吞吐率 | `prefix_ok` |
| ---: | ---: | ---: | --- |
| 10000 | `1.065326 s` | `9386.80 digits/s` | `True` |
| 100000 | `0.745308 s` | `134172.66 digits/s` | `True` |
| 1000000 | `1.178811 s` | `848312.42 digits/s` | `True` |

这三组数据虽然还远远谈不上高吞吐，但它们已经回答了一个更关键的问题：`full-CUDA` 入口是否真的能接成完整 `pi` 链路。答案现在是“能，而且前缀校验已经通过”。

### 3. stage-2 pipeline 里现在真正的大头在哪里

从 `10k / 100k / 1M` 这三组 smoke 可以直接看出：

1. `merge_tree_seconds` 只占几毫秒级，说明“GPU merge 乘法本身”目前并不是这条 smoke 的最大问题。
2. 当前端到端大头反而主要落在：
   - `sqrt_compute_seconds`
   - `final_division_seconds`
   - `newton_reciprocal_seconds`
3. 到 `1M` 位时，`partial_generation_seconds` 也开始抬头，说明再往上推时，CPU 生成 partials 仍会重新变成结构性瓶颈。

以 `1M` 位为例：

1. 总时间是 `1.178811 s`。
2. 其中 `gpu_backend_total_seconds = 0.317275 s`，真正的 `gpu_kernel_seconds = 0.082987 s`。
3. 这说明当前 `full-CUDA stage-2` 还没有被“纯乘法内核”拖死，而是更多被 bootstrap、Newton 迭代和 CPU partial generation 这些外围阶段拉住。

### 4. 当前阶段的结论

截至这一轮，`full-CUDA` 迁移线已经从“stage-1 runtime smoke”推进到了“stage-2 pipeline smoke”：

1. CUDA limb runtime 继续保持随机正确性：
   - `random_cuda_chunk_ops_v5_ok`
2. `project2_gpu_pi_full_cuda.py --mode pipeline` 已经能算出完整 `pi` smoke，并且 `prefix_ok=True`。
3. 当前最值得继续追的方向已经比上一轮更具体了：
   - 继续减少 `sqrt / reciprocal / division` 路径里的 host 回读和 correction 开销
   - 把长度元数据进一步留在 GPU runtime 内，而不是频繁回 host 决策
   - 再往前才是考虑把 `partial_generation` 往 GPU 友好表示迁移

换句话说，现在已经不是“full-CUDA 路线只有零散 kernel”，而是“完整 pipeline 已经接通，但真正的大头已经清楚暴露出来了”。

## 本轮继续推进：把 `sqrt` 比例尺幂搬上 GPU，并清掉 Newton 常量减法里的 host 标量回读

在上一轮 stage-2 pipeline smoke 之后，最值得继续追的一段已经比较明确：

1. `sqrt_compute_seconds` 仍然长期占大头。
2. 虽然 `sqrt(10005)` 的主迭代已经在 chunk/GPU 路线上，但 `10^digits` 这个比例尺仍然先在 CPU `mpz` 上构造，再转成 `binary16` chunks。
3. `reciprocal / rsqrt` 的 Newton 路径里，`sub_scaled_constant_minus_chunks()` 仍然有 `.item()` 级别的 host 标量回读。

因此这轮继续做了两件更贴近“full-CUDA”目标的事：

### 1. 用 GPU chunk 二进制幂构造 `10^digits`

在 `project2_gpu_pi_hybrid.py` 里新增：

- `small_chunk_integer()`
- `pow_small_integer_chunks_gpu()`

它们让 `compute_sqrt_term_chunks_gpu_prototype()` 不再依赖：

```python
scale_mpz = gmpy2.mpz(10) ** total_digits
scale_chunks = mpz_to_chunk_integer(scale_mpz, ...)
```

而是改成在 `binary16` chunk 表示上，直接通过 GPU merge 乘法做二进制幂：

$$
10^n = \prod_{k=0}^{\lfloor \log_2 n \rfloor} (10^{2^k})^{b_k}
$$

这一步的意义不只是“也许会更快”，更重要的是：

1. `sqrt` 路径里又少了一块 CPU 大整数构造。
2. full-CUDA pipeline 里 `10^digits` 这种基础比例尺已经能以 GPU chunk 形式原生生成。
3. 后续如果继续推进完全 GPU-resident 的 `sqrt / reciprocal / division`，这块基础设施可以复用。

对应 correctness check：

- `pow10_gpu_chunks_ok`

说明当前 GPU chunk 版 `10^n` 构造在测试的多组指数上和 `gmpy2.mpz(10) ** n` 完全一致。

### 2. 用 chunk compare/sub 替换手写 borrow 路径

原先 `sub_scaled_constant_minus_chunks()` 自己手写了低位 borrow 传播，并在关键位置做：

- `low_nonzero_prefix[-1].item()`
- `padded[scale_chunks].item()`

这会把 Newton 迭代里一部分决策重新拉回 host。

本轮把它改成直接走已有的 chunk 算术路径：

1. 先构造 `coeff * base^scale`
2. 再用 `compare_abs_chunks()` / `sub_abs_chunks()` 完成精确差值

对应 correctness check：

- `sub_scaled_constant_minus_chunks_v2_ok`

这一步的直接效果是：

1. `reciprocal / rsqrt` 的常量减法路径不再自己维护一套和主 chunk 算术栈平行的 borrow 逻辑。
2. 相关决策更多复用已经接进 CUDA extension 的 compare/sub 通路。
3. 这让 Newton 迭代的行为更统一，也减少了 host 标量回读点。

### 3. 新的 full-CUDA pipeline smoke 结果

对应新的结果文件：

- `result/project2_full_cuda_pipeline_smoke_100k_v3.csv`
- `result/project2_full_cuda_pipeline_smoke_1m_v3.csv`

和此前 `v1` 的对比如下：

| 位数 | `v1 time` | `v3 time` | `v1 digits/s` | `v3 digits/s` |
| ---: | ---: | ---: | ---: | ---: |
| 100000 | `0.745308 s` | `0.687382 s` | `134172.66` | `145479.45` |
| 1000000 | `1.178811 s` | `1.153276 s` | `848312.42` | `867095.00` |

两点需要一起看：

1. 这轮确实把 `100k / 1M` 两档 full-CUDA pipeline smoke 都继续往前推了一点。
2. 但 `sqrt_scale_seconds` 现在被单独量化出来之后，也能看到 GPU `10^digits` 构造本身并不便宜：
   - `100k` 位约 `0.401766 s`
   - `1M` 位约 `0.408406 s`

这说明当前这一步的价值更多是“把 CPU 路径拔掉并把结构迁到 GPU-resident 表示”，而不是“立刻带来数量级提速”。

### 4. 当前新的判断

到这一轮为止，full-CUDA 路线已经比上一轮更完整了一层：

1. `sqrt` 路径里的比例尺幂 `10^digits` 已经不必经由 CPU `mpz` 生成。
2. `Newton` 常量减法路径也更一致地回到了 chunk compare/sub 栈中。
3. 端到端 smoke 继续保持 `prefix_ok=True`，而且在 `100k / 1M` 上都有小幅正向改善。

但新的 profiling 也把下一阶段的重点说得更清楚了：

1. `sqrt_scale_seconds` 现在已经被暴露为一个独立的大块。
2. 如果继续追 full-CUDA，下一刀更值得考虑的是：
   - 减少构造 `10^digits` 所需的大乘法次数
   - 让比例尺幂可以跨 `sqrt / division` 复用，而不是每次重新建
   - 再往后才是把 `partial_generation` 往 GPU 友好表示继续推进

## 对上一轮“速度似乎更糟”的复核与修正

在继续往下推之前，专门对上一轮新增的 `5^n << n bits` 比例尺构造做了复核。结论是：这个方向在当前实现里并没有稳定赢过原来的 GPU `10^n` 直幂，因此不应该保留在默认热路径。

做了三类检查：

1. isolated scale benchmark：
   - `pow_small_integer_chunks_gpu(10, n)`
   - `build_decimal_scale_chunks_gpu(n)`，也就是 `5^n << n bits`
2. full pipeline smoke 对比：
   - `result/project2_full_cuda_pipeline_smoke_100k_v4.csv`
   - 与此前 `v3` 对照
3. correctness：
   - `shift_left_bits_chunks_ok`
   - `decimal_scale_gpu_chunks_ok`

复核结果说明：

1. `5^n << n bits` 在 isolated benchmark 下并没有给出稳定优势。
2. 放到完整 pipeline 里时，它还会带来明显波动；例如 `100k` 的一次 `v4` 结果只有 `126469 digits/s`，比 `v3` 更差。
3. 因此这条路径目前只适合作为研究备选，不适合继续作为默认热路径。

基于这点，本轮已经把 `compute_sqrt_term_chunks_gpu_prototype()` 的默认比例尺构造切回 GPU `10^n` 直幂，同时保留：

1. `sqrt_scale_seconds` 的单独统计
2. `shift_left_bits_chunks()` 和 `build_decimal_scale_chunks_gpu()` 这些辅助设施，方便后面继续做 A/B

恢复默认热路径后，新的 smoke 文件：

- `result/project2_full_cuda_pipeline_smoke_100k_v5.csv`

结果为：

| 位数 | 时间 | 吞吐率 | `prefix_ok` |
| ---: | ---: | ---: | --- |
| 100000 | `0.672712 s` | `148652.03` | `True` |

这说明用户的判断是对的：上一刀确实不该直接留在默认路径里。当前代码已经据此修正回更快、更稳的版本。
