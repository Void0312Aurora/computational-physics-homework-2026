# Plan

## Scope

- Target folder: `HW/09`
- Problem source: six image files under `HW/09/docs/problems/`

## Problem Summary

- Chinese translation:
  - Problem 1: 用传统 Monte Carlo 方法模拟 Buffon 投针问题并估计 $\pi$。在固定 $N$ 下改变 $x=b/a$ 讨论影响结果的因素；改变 $N$ 观察能得到的最高精度；解释计算机为什么只能达到这种精度；可选地用长针公式考察 $x=a/b=1,2,\ldots,14$ 时能否正确估计 $\pi$ 并作图。
  - Problem 2: 用传统接受-拒绝法，即单位正方形中四分之一圆面积比 $A_{\rm circle}/A_{\rm square}=\pi/4$，计算 $\pi$。改变 $N$ 观察精度并解释计算机精度限制。
  - Problem 3: 给定 Harvard、Yale、Dartmouth 三校之间父子就读概率组成的 Markov 链，求 Harvard 校友的孙子就读 Harvard 的概率；再将 Harvard 校友儿子总是去 Harvard 的假设加入后重新求该概率。
  - Problem 4: 老鼠在六个房间迷宫中随机选择出口移动。要求给出转移矩阵，说明链不可约但非非周期，求平稳分布；房间 5 为陷阱且从房间 1 出发时，求首次到达房间 5 的期望步数；求回到房间 1 的期望时间。
  - Problem 5: 在间距为 $a,b$ 的矩形网格上投长度 $\ell<\min(a,b)$ 的针，验证 $P(A)=2\ell/(\pi a)$、$P(B)=2\ell/(\pi b)$、$P(A\cap B)=\ell^2/(\pi ab)$，并用传统 Monte Carlo 方法分别由三式估计 $\pi$，比较三种方法，建议绘制方法 (c) 的结果。
- Required outputs:
  - Monte Carlo 估计的日志、CSV 表格和收敛图。
  - Markov 链转移矩阵、平稳分布、首达时间和返回时间的解析计算结果，并用模拟结果交叉核对。
  - 最终 `docs/answer/answer.md`、`answer.docx` 和 `answer.pdf`。
- Numerical or implementation constraints:
  - Monte Carlo 误差主要是随机采样误差，标准差按二项分布传播估计，约随 $N^{-1/2}$ 下降。
  - Buffon 长针情形需要使用分段解析公式，不能继续套用短针公式。
  - Markov 链题需要给出解析解，而模拟只作为验证。

## Approach

- Language/tool choice:
  - 使用 `Python` 进行 Monte Carlo 模拟、线性方程求解、CSV/JSON 汇总、绘图和文档导出。
- Core algorithm or script plan:
  - `scripts/hw09_analysis.py` 固定随机种子 `20260513`，统一生成所有数值结果。
  - Problem 1:
    - 短针情形设直线间距 $a=1$、针长比 $x=b/a\le 1$，模拟针中心到最近平行线距离和针角度，统计穿线事件。
    - 固定 $N=10^6$ 改变 $x=0.1,0.2,\ldots,1.0$；固定 $x=1$ 改变 $N$；可选长针部分使用 $P(x)$ 的分段公式估计 $\pi$。
  - Problem 2:
    - 在 $[0,1]^2$ 上均匀采点，统计 $x^2+y^2\le 1$ 的点数，以 $\hat\pi=4M/N$ 估计 $\pi$。
  - Problem 3:
    - 写出三状态转移矩阵 $P$，用 $P^2_{HH}$ 求两代后的 Harvard 概率；修改第一行后重新计算。
  - Problem 4:
    - 将迷宫视作无向图，按每个房间门数均匀转移得到 $P$。
    - 用 $\pi P=\pi$、$\sum_i\pi_i=1$ 解平稳分布。
    - 用首达时间方程 $h_i=1+\sum_jP_{ij}h_j$ 求 $E_1[T_5]$，并用 Kac 公式 $E_1[T_1^+]=1/\pi_1$ 求回到房间 1 的期望时间。
  - Problem 5:
    - 设 $a=1$、$b=1.5$、$\ell=0.8$，满足 $\ell<\min(a,b)$。
    - 随机生成针中心到最近竖线、横线距离和角度，分别统计 $A$、$B$、$A\cap B$，由三种公式反解 $\pi$ 并作图。
- Output files to create:
  - `scripts/hw09_analysis.py`
  - `Makefile`
  - `result/*.csv`, `result/*.png`, `result/temp-01.log`, `result/hw09_summary.json`
  - `docs/answer/render_docs.py`
  - `docs/answer/answer.md`, `answer.docx`, `answer.pdf`

## Testing

- Commands to run:
  - `make run`
  - `make docs`
- Expected checks:
  - Monte Carlo 估计值随 $N$ 增大整体接近 $\pi$，误差量级与二项标准差相符。
  - Buffon 短针固定 $N$ 时，$x$ 越大交叉概率越高，理论标准差越小。
  - Problem 3 的解析结果为两步转移矩阵中的相应元素。
  - Problem 4 的平稳分布与各房间度数成正比，首达时间与返回时间解析值能被独立模拟近似验证。
  - Problem 5 三种网格方法均收敛到 $\pi$，但交叉事件概率较小，方法 (c) 的方差相对较大。
  - PDF 中中文、公式、表格和图像能正常渲染。

## Risks

- Known numerical pitfalls:
  - Monte Carlo 结果具有随机波动，单次绝对误差不一定随 $N$ 单调下降，应按统计误差解释。
  - 若事件概率过低，例如 Buffon 短针的很小 $x$ 或网格交叉事件，命中数少会显著放大 $\pi$ 的反解误差。
  - 浮点舍入误差和伪随机数质量不是本任务的主误差源；在当前样本量下主要限制是 $O(N^{-1/2})$ 的采样误差。
- Toolchain or environment concerns:
  - 文档导出依赖 `pypandoc`、`pandoc`、`xelatex` 和 CJK 字体；若 pandoc 不在 PATH，渲染脚本会尝试使用 `pypandoc` 获取本地版本。
  - 生成图像后需要检查 PDF 中图像是否正确嵌入，避免只在 Markdown 中存在引用。
