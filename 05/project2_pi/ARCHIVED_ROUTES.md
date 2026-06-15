# Archived Routes In `project2_pi`

这份说明只标记“仍保留源码，但不再继续作为主线优化对象”的路线。

## Current Winner

当前 `project2_pi` 的主线是：

- `python -m project2_pi.gpu_pi_hybrid --profile fast-auto`

在 `2026-04-20` 的本地 matrix 中，这条路线在 `10,000,000` digits 上达到约 `5.25e6 digits/s`。

## Archived Prototype

### `gpu_pi_full_cuda.py`

这个入口仍然保留，原因是它对 `chunk-arithmetic / rsqrt / newton-division` 的研究过程有参考价值，但它不再作为当前机器上的主线。

归档依据：

- `full_cuda_pipeline` 在 `100,000` digits 上约为 `4.29e5 digits/s`
- 同期主线 `gpu_hybrid_merge_fast_auto` 在 `10,000,000` digits 上约为 `5.25e6 digits/s`
- `full_cuda` 仍然更像是“把更多阶段搬到 GPU 上的验证原型”，而不是当前最优的端到端吞吐实现

因此：

- 保留源码
- 不再作为默认入口
- 只在需要验证某个 GPU-only 原型假设时手动使用
