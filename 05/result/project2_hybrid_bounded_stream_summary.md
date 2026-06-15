# Project 2 Bounded-Stream Summary

## Smoke Validation

在同一套 `fast-auto` 参数下，仅切换 `stream_partials`，`100000` 位和 `1000000` 位输出完全一致，前缀校验均通过。

| digits | `--no-stream-partials` / s | `--stream-partials` / s | 结果 |
| --- | ---: | ---: | --- |
| 100000 | 0.711035 | 0.752738 | 完全一致 |
| 1000000 | 0.743476 | 0.788436 | 完全一致 |

## High-Digit Results

| 配置 | digits | seconds | digits/s | `gpu_peak_gb` | `merge_frontier_max_nodes` | `budget_cpu_calls` | 自动除法 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `bounded-stream-auto` | 100000000 | 43.540514 | 2.30e+06 | 8.651 | 4 | 5 | `mpz-div` |
| `bounded-stream-auto` | 140000000 | 96.484432 | 1.45e+06 | 8.387 | 4 | 10 | `mpz-div` |
| `bounded-stream-auto` | 160000000 | 112.662978 | 1.42e+06 | 8.717 | 4 | 10 | `mpz-div` |

## Takeaways

- `merge_frontier_max_nodes` 始终只有 `4`，说明流式 partial 生成加 frontier merge 已把 merge live-set 压到了小常数级。
- `gpu_peak_gb` 在 `100M` 到 `160M` 间稳定在约 `8.4~8.7 GiB`，说明当前方案已经把 GPU FFT 工作集压进了接近常数的窗口，但这不是整条流水线对位数严格不变的全局显存上界。
- 速度代价非常明显：`100M` 只有 `high-digits-auto` 的约 `51%` 吞吐，`140M` 只有 `fast-auto` 的约 `24%`、`high-digits-auto` 的约 `62%`，`160M` 也只剩 `fast-auto` 的约 `45%`。
- `140M` 和 `160M` 的吞吐只比纯 CPU 基线高约 `11%`，因此在严格显存预算下，`bounded-stream-auto` 更适合作为“保证能跑完”的工程 profile，而不是新的速度主线。
