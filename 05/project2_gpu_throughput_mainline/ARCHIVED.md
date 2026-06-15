# Archived Route

`project2_gpu_throughput_mainline` 已经转为归档分支。

它保留了很多很有价值的 grouped planner / exact-moduli / staged-closure 研究工作，但不再是当前工作区里继续追求最高 `pi digits/s` 的主线。

## Why Archived

- 项目自身 README 已标明这条线在 `2026-04-18` 冻结
- 本地 `2026-04-20` 复测中，这条路线在 `2500` digits 的 end-to-end smoke 上 steady-state 约为 `6.68e4 digits/s`
- 当前主线 `project2_pi.gpu_pi_hybrid --profile fast-auto` 在 `10,000,000` digits 上约为 `5.25e6 digits/s`
- 这条线虽然在架构研究上很强，但在当前机器和当前目标下，继续投入不如集中火力优化 hybrid 主线

## Current Status

- 保留源码、README、结果和冻结快照
- 不再作为默认入口
- 只在需要回看 throughput-first 架构实验时手动使用

如果当前目标是“更快地算 `pi`”，请回到 `HW/05/project2_pi/` 主线。
