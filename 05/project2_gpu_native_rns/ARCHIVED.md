# Archived Route

`project2_gpu_native_rns` 已经转为归档分支。

原因不是实现无效，而是它在当前工作区里的定位已经从“继续追求最高 `pi digits/s` 的主线”变成了“保存原生 RNS 架构试验和结果的研究记录”。

## Why Archived

- 本地 `2026-04-20` 复测中，这条路线在 `870` digits 的 end-to-end benchmark 上约为 `6.12e3 digits/s`
- 当前验证 ceiling 仍主要停在 `sub-1k digits` 到 `低千位` 级别
- 相比之下，当前主线 `project2_pi.gpu_pi_hybrid --profile fast-auto` 在 `10,000,000` digits 上约为 `5.25e6 digits/s`

## Current Status

- 保留源码、结果和文档
- 不再作为默认开发入口
- 只作为 RNS / CRT / GPU-native exact semantics 的参考档案

如果只是想继续提升当前机器上的 `pi` 计算速度，请回到 `HW/05/project2_pi/` 主线。
