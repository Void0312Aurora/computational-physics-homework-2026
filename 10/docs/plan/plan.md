# Plan

## Scope

- Target folder: `HW/10`
- Problem source:
  - `HW/10/docs/problems/image.png`
  - `HW/10/docs/problems/image_1.png`
  - `HW/10/docs/problems/image_2.png`
  - `HW/10/docs/problems/image_3.png`

## Problem Summary

- Chinese translation:
  - Problem 1: 对若干内置或自选随机数生成器进行随机性的卡方检验。将三维单位立方体分成 `N` 个箱子，抽取许多 `P` 个三元组，若随机性良好，每个箱子的频数应接近 $n=P/N$，典型波动约为 $(P/N)^{1/2}$；题图给出示例 `N=100, P=1000`。
  - Problem 2: 做一个“最小”统计检验：检验样本均值是否接近 0.5。要求测试内置随机数生成器和自己写的生成器。
  - Problem 3: 写出自己的 quasi-random number generator，并对其进行卡方随机性检验和“最小”样本均值统计检验。
  - Problem 4: 重做 Homework 9 Problem 2，即用传统接受-拒绝法计算 $\pi$，分别使用自己的 pseudo-random number generator 和 quasi-random number generator，并评论哪一种结果更好以及原因。
- Required outputs:
  - 随机数生成器的卡方检验结果、重复检验统计、样本均值检验结果。
  - 自写 quasi-random 序列的检验结果。
  - 接受-拒绝法估计 $\pi$ 的收敛表、点分布图和误差图。
  - `docs/answer/answer.md`、`answer.docx`、`answer.pdf`。
- Numerical or implementation constraints:
  - 卡方检验需要明确自由度、理论均值和 p 值判据。
  - 对 quasi-random 序列，卡方统计量可能过小，这表示序列比随机采样更均匀，不应简单解释为普通随机性检验中的“典型随机样本”。
  - 接受-拒绝法中 pseudo-random 误差遵循 $O(N^{-1/2})$ 的 Monte Carlo 量级；quasi-random 低差异序列常能得到更小误差，但不满足独立随机样本的二项标准差模型。

## Approach

- Language/tool choice:
  - 使用 `Python` 完成随机数生成器、统计检验、CSV 输出、图像绘制和文档导出。
- Core algorithm or script plan:
  - 实现线性同余生成器
    $x_{k+1}=(a x_k+c)\bmod m$，
    并测试 `PCG64`、Park-Miller 型 LCG、自题图参数得到的若干 LCG。
  - Problem 1:
    - 将 $[0,1)^3$ 分成 $5^3=125$ 个箱子。
    - 每轮抽取 $P=10000$ 个三元组，计算
      $$
      \chi^2=\sum_{i=1}^{N}\frac{(O_i-E)^2}{E},\qquad E=P/N.
      $$
    - 对每个生成器重复 200 轮，与自由度 $N-1=124$ 的卡方分布比较。
  - Problem 2:
    - 对每个生成器取 $20000$ 个样本，计算运行均值。
    - 用
      $$
      z=\frac{\bar x-1/2}{\sqrt{1/(12M)}}
      $$
      做正态近似检验。
  - Problem 3:
    - 自写 Halton 低差异序列。三维卡方检验使用基数 $(2,3,5)$，样本均值检验使用一维基数 2。
  - Problem 4:
    - 自写 pseudo-random 生成器使用 Park-Miller 型 LCG。
    - quasi-random 生成器使用二维 Halton 序列，基数 $(2,3)$。
    - 对 $[0,1]^2$ 采样，统计 $x^2+y^2\le 1$ 的比例，以 $\hat\pi=4M/N$ 估计 $\pi$。
  - `docs/answer/answer.md` 采用统一报告结构：
    - 罗马数字一级标题表示大问；
    - 每个小问单独成节；
    - 每个小问内部使用 `待求问题`、`解决方式`、`问题答案`、`理解`；
    - 正文只保留中文题目陈述，不保留英文原题。
- Output files to create:
  - `scripts/hw10_random_tests.py`
  - `Makefile`
  - `result/problem1_chi_square_summary.csv`
  - `result/problem1_chi_square_repeats.csv`
  - `result/problem1_chi_square_distribution.png`
  - `result/problem1_example_bin_counts.png`
  - `result/problem2_sample_mean_summary.csv`
  - `result/problem2_sample_mean.png`
  - `result/problem3_quasi_tests.csv`
  - `result/problem3_halton_counts.png`
  - `result/problem3_halton_sample_mean.png`
  - `result/problem4_pi_acceptance_rejection.csv`
  - `result/problem4_pi_convergence.png`
  - `result/problem4_points.png`
  - `result/temp-01.log`
  - `result/hw10_summary.json`
  - `docs/answer/answer.md`
  - `docs/answer/answer.docx`
  - `docs/answer/answer.pdf`

## Testing

- Commands to run:
  - `make run`
  - `make docs`
- Expected checks:
  - `PCG64` 和 Park-Miller 型 LCG 的三维卡方统计量应落在自由度 124 的合理区间附近。
  - 低质量 LCG 应在卡方检验或样本均值检验中显著失败。
  - Halton 序列的箱子计数应非常均匀，表现为过小的卡方统计量；样本均值接近 0.5。
  - 接受-拒绝法中 quasi-random 序列的 $\pi$ 估计误差应整体小于 pseudo-random LCG。
  - 导出的 PDF 中图像、公式、中文和表格应能正常显示。

## Risks

- Known numerical pitfalls:
  - p 值为 0 的结果是双精度浮点下的数值下溢或极端小概率表示，含义是显著拒绝均匀独立假设。
  - 卡方检验只检验被分箱后的三维均匀性，不能证明生成器完全随机。
  - 样本均值检验过于弱，某些坏生成器仍可能通过该检验。
  - quasi-random 序列不是独立随机样本，因此传统随机性检验的解释需要单独说明。
- Toolchain or environment concerns:
  - 文档导出依赖 `pypandoc`、`pandoc` 和 `xelatex`。
  - 图像较多，渲染后需要检查 PDF 是否正确嵌入图像。
