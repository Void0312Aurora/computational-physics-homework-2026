# Project 2 Hybrid Refactor Summary

| profile | digits | seconds | digits_per_second | gpu_peak_gb | budget_cpu_calls | oom_cpu_calls | status | note | source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| old-fast-auto-window | 140000000 |  |  |  |  |  | failed | pre-refactor limit-window benchmark hit CUDA OOM at 140M digits | project2_hybrid_extremes_limit_window.md |
| fast-auto-140m | 140000000 | 22.490386 | 6224881.99 | 18.239 | 0 | 0 | ok | merge_gpu_calls=146,final_gpu_calls=3,merge_frontier_max_nodes=10 | project2_gpu_hybrid_fastauto_140m_after_refactor.csv |
| fast-auto-160m | 160000000 | 55.498087 | 2882982.24 | 17.882 | 0 | 3 | ok | merge_gpu_calls=153,final_gpu_calls=2,merge_frontier_max_nodes=11 | project2_gpu_hybrid_fastauto_160m_after_refactor.csv |
| high-digits-100m | 100000000 | 22.414052 | 4461486.96 | 16.243 | 1 | 0 | ok | merge_gpu_calls=177,final_gpu_calls=2,merge_frontier_max_nodes=14 | project2_gpu_hybrid_highdigits_100m_after_refactor.csv |
| high-digits-140m | 140000000 | 79.793957 | 1754518.83 | 15.708 | 7 | 0 | ok | merge_gpu_calls=142,final_gpu_calls=0,merge_frontier_max_nodes=10 | project2_gpu_hybrid_highdigits_140m_after_refactor.csv |

## Key Takeaways

- 流式首轮 chunk 转换和逐层相邻配对归并，把旧的 `140M` OOM 窗口重新压回可运行区间。
- `fast-auto` 在不设显存预算时，`160M` digits 仍能跑通；其中有 `3` 次乘法在真正触发 CUDA OOM 后自动回退到 CPU。
- `high-digits-auto` 用 `12 GB` 预算主动把超大乘法留给 CPU，`140M` digits 也能稳定跑通，但吞吐明显低于不设预算的主线。
