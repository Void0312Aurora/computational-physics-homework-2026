# Plan

## Scope

- Target folder: `HW/11`
- Problem source:
  - `HW/11/docs/problems/image.png`
  - `HW/11/docs/problems/image_2.png`
  - `HW/11/docs/problems/image_3.png`
  - `HW/11/docs/problems/image_4.png`
  - `HW/11/docs/problems/image_5.png`
  - `HW/11/docs/problems/image_6.png`

## Problem Summary

- Chinese translation:
  - Problem 1: 使用传统 Monte Carlo 方法计算积分 $I=\int_0^1 e^x dx$。复现题图中 MC 估计量的分布，并给出最终积分结果与误差估计。
  - Problem 2: 对 $I=\int_0^1 dx/(\sqrt{x}+x)$ 使用重要性抽样。均匀抽样会导致近似对数发散的无限方差估计量；改用 $g(x)=\alpha/\sqrt{x}$ 型密度，使 $f(x)/g(x)$ 具有有限方差。
  - Problem 3: 对同一个奇异积分使用分层抽样。比较简单 MC、固定每层样本数 $n_i=10$ 的分层抽样，以及在 $k=5000,\ p_i=1/5000$ 时的正则化 Neyman 分层分配，并评论与 Problem 2 的重要性抽样差异。
  - Problem 4: 使用自写伪随机数生成器和准随机数生成器，用简单 MC 计算 $I=\int_0^1 dx/(\sqrt{x}+x)$。拟合误差随样本数 $N$ 的幂律指数，并解释伪随机与准随机的差别。解析结果为 $I=2\ln 2$。
  - Problem 5: 使用 hit-and-miss 方法积分振荡函数 $I=\int_0^\pi 2\sin(2\sqrt{\pi^2-x^2})dx$。若被积函数变号，需自行加常数使其非负，再扣除常数贡献。
  - Problem 6: 实现 Monte Carlo 方法估计任意维数 $N$、半径 $R$ 的球体积，并计算 $R=1,\ N=20$，尝试更高维如 25 维或以上，同时与解析公式比较。
- Required outputs:
  - 传统 MC 估计分布图、最终估计和标准误差。
  - 重要性抽样、分层抽样、伪随机与准随机积分结果及误差比较。
  - hit-and-miss 积分的点图、收敛图和最终误差。
  - 20 维和更高维球体积的 MC 估计、解析值和相对误差。
  - `docs/answer/answer.md`、`answer.docx`、`answer.pdf`。
- Numerical or implementation constraints:
  - 对奇异积分需要避免直接使用含 $x=0$ 的样本点。
  - 重要性抽样密度 $g(x)=1/(2\sqrt{x})$ 可由变换 $x=u^2$ 采样，并使权重简化为 $2/(1+\sqrt{x})$。
  - 分层抽样的最优样本分配需要按层内标准差近似 $n_i\propto p_i\sigma_i$，并处理整数取整后总样本数不变的问题。
  - 高维球体积的传统超立方体接受-拒绝法在高维下接受率极低，需要报告这种退化，并用径向-角向/重要性思路作为对照。

## Approach

- Language/tool choice:
  - 使用 `Python` 完成 Monte Carlo 实验、低差异序列、CSV/JSON 输出、图像绘制和文档导出。
- Core algorithm or script plan:
  - Problem 1:
    - 对 $e^x$ 在 $[0,1]$ 上均匀采样。
    - 重复多轮实验，绘制 $N=1000, 10000, 100000, 1000000, 10000000$ 时估计量分布，并叠加正态近似。
    - 额外绘制 $e-1$ 附近的局部放大图，观察高采样数下估计量分布向 Dirac delta 收缩的过程。
    - 用独立大样本给出最终估计、标准误差和相对误差。
  - Problem 2:
    - 取 $g(x)=1/(2\sqrt{x})$，采样 $x=u^2$。
    - 权重为 $w(x)=f(x)/g(x)=2/(1+\sqrt{x})$。
    - 计算均值、样本标准差、标准误差，并与解析值 $2\ln 2$ 对照。
  - Problem 3:
    - 简单 MC 使用均匀分布但避开端点。
    - 固定分层：$k=5000,\ n_i=10$，每层均匀取样。
    - 正则化 Neyman 分层：第一层 `[0,1/5000]` 含端点奇异性且层内二阶矩发散，先解析计算该层积分；其余层用解析二阶矩计算层内 $\sigma_i$，再按 $n_i\propto p_i\sigma_i$ 分配总样本数，与固定分层使用相近总样本量。
    - 用独立重复估计各方法的方差。
  - Problem 4:
    - 自写 Park-Miller LCG 作为伪随机数生成器。
    - 自写 Halton 基数 2 序列作为准随机数生成器。
    - 在多个 $N$ 下计算绝对误差，拟合 $\log(error)=a+b\log(N)$ 中的指数 $b$。
  - Problem 5:
    - 函数 $h(x)=2\sin(2\sqrt{\pi^2-x^2})$，取常数 $C=2$，使 $h(x)+C\ge 0$。
    - 在矩形 $[0,\pi]\times[0,4]$ 内做 hit-and-miss，估计 $\int_0^\pi(h(x)+2)dx$ 后减去 $2\pi$，并用命中比例的二项方差给出标准误差。
    - 使用高精度数值积分作为参考值。
  - Problem 6:
    - 传统方法：在 $[-R,R]^d$ 超立方体中均匀采样，统计 $\sum x_i^2\le R^2$。
    - 对照方法：利用 $d$ 维单位球体积满足 $V_d=\pi^{d/2}R^d/\Gamma(d/2+1)$，并用径向积分 $V_d=S_{d-1}\int_0^R r^{d-1}dr$ 的一维 MC 检查该公式；同时报告传统超立方体法的期望命中数和理论相对标准误差。
- Output files to create:
  - `scripts/hw11_monte_carlo.py`
  - `Makefile`
  - `result/temp-01.log`
  - `result/hw11_summary.json`
  - `result/problem1_distribution.csv`
  - `result/problem1_mc_distribution.png`
  - `result/problem1_mc_distribution_zoom.png`
  - `result/problem2_importance.csv`
  - `result/problem2_weight_distribution.png`
  - `result/problem3_stratified.csv`
  - `result/problem3_variance_comparison.png`
  - `result/problem4_convergence.csv`
  - `result/problem4_convergence.png`
  - `result/problem5_hitmiss.csv`
  - `result/problem5_hitmiss_points.png`
  - `result/problem5_convergence.png`
  - `result/problem6_sphere.csv`
  - `result/problem6_sphere_volumes.png`
  - `docs/answer/answer.md`
  - `docs/answer/answer.docx`
  - `docs/answer/answer.pdf`

## Testing

- Commands to run:
  - `make run`
  - `make docs`
- Expected checks:
  - Problem 1 的 MC 估计量分布中心应接近 $e-1$，宽度随 $N^{-1/2}$ 缩小。
  - Problem 2 的重要性抽样估计应接近 $2\ln2$，误差按有限方差标准误差估算。
  - Problem 3 中固定分层和正则化 Neyman 分层应显著降低简单均匀 MC 的方差；正则化 Neyman 分层的重复标准差应与理论预测标准差相近，并重点给靠近 $x=0$ 的剩余层更多样本。
  - Problem 4 中伪随机误差拟合指数应接近 $-1/2$，Halton 准随机误差通常更陡，但会受函数端点奇异性影响，不必机械等于 $-1$。
  - Problem 5 的 hit-and-miss 结果应接近数值积分参考值，题图给出的约 $-2.1095$ 作为定性参考。
  - Problem 6 的传统超立方体方法在 20 维及以上会因接受率极低而不稳定；期望命中数应解释零命中的原因，解析值和径向 MC 对照应在标准误差范围内一致。
  - 导出的 PDF 中图像、公式、中文和表格应能正常显示。

## Risks

- Known numerical pitfalls:
  - $1/(\sqrt{x}+x)$ 在 $x=0$ 有可积奇异性，均匀 MC 的方差发散，样本标准误差可能低估尾部风险。
  - 重要性抽样的标准误差只对所选 $g(x)$ 的有限方差权重有效。
  - 分层抽样若直接包含 $x=0$ 所在层，层内二阶矩会发散；需解析处理第一层，再对剩余层做 Neyman 分配。
  - 拟合收敛指数时，误差可能偶然接近零，需用多种样本量和重复统计降低偶然性。
  - 高维球体积传统 MC 的接受概率随维数急剧下降，25 维以上可能出现零命中。
- Toolchain or environment concerns:
  - 文档导出依赖 `pypandoc`、`pandoc` 和 `xelatex`。
  - 若 PDF 渲染缺少中文字体，需要改用系统存在的 CJK 字体。
