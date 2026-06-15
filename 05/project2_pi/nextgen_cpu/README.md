# Next-Gen CPU Phase 2

这里存放 `HW/05/project2_pi` 的新一代纯 CPU exact engine 骨架。

当前阶段已经落地的是：

- `PageArena`
- `BigHandle`
- `TripleSlot`
- canonical block add/mul/affine-merge smoke
- `canonical 32-bit block -> residue inject -> cached multi-modulus NTT -> direct CRT carry -> canonical` 乘法 smoke
- destructive product tree prototype

当前目录已经不只是 correctness 骨架，还完成了一轮表示层重构：

- `RNS` 输入直接复用 canonical `32-bit block`
- `NTT` 计划按 `ntt_size` 缓存
- scratch buffer 通过 `RnsWorkspace` 复用
- CRT 常量只预计算一次，并直接 carry 回 canonical blocks
- multiply kernel 已支持跨 level arena 输入/输出

当前 quick/extended benchmark 显示：

- `256` blocks 时，`speedup_rns_vs_schoolbook` 约 `0.355`
- `1024` blocks 时，`speedup_rns_vs_schoolbook` 约 `1.282`
- `4096` blocks 时，`speedup_rns_vs_schoolbook` 约 `4.109`

当前 tree benchmark 还显示：

- `32 x 128` blocks workload：adaptive `RNS` product tree 约 `1.462x`
- `64 x 128` blocks workload：adaptive `RNS` product tree 约 `2.152x`
- `64 x 256` blocks workload：adaptive `RNS` product tree 约 `3.477x`

也就是说，新目录现在已经开始在 tree workload 上出现结构性收益。但由于 `peak_live_blocks` 还没有下降，接下来真正要继续展开的是：

1. 把 `RNS/NTT` 乘法从“每次 merge 临时进出”推进到真正 transform-resident 的 reduction tree 主路径
2. 把更激进的 lazy reduction / redundant butterfly 引进来
3. 继续压低 canonical 与 transform domain 之间的转换频率
