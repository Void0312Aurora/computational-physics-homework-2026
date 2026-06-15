# Plan

## Scope

- Target folder: `HW/08`
- Problem source: `HW/08/docs/problems/image.png`, `HW/08/docs/problems/image_1.png`

## Problem Summary

- Chinese translation:
  - Problem 1: 对给出的四组二维数据分别进行直线拟合，并讨论各组数据的拟合效果是好还是坏；同时绘制这些拟合图。
  - Problem 2: 给出放射性衰变计数数据 $(t_i, N_i)$，在考虑数据点不确定度的前提下求参数 $A$ 与 $\tau$，并绘制拟合结果。
- Required outputs:
  - 四组数据的拟合参数、残差诊断与可视化图像
  - 放射性衰减参数估计、参数不确定度、拟合图像
  - 原始日志、CSV 结果、最终 `answer.md/.docx/.pdf`
- Numerical or implementation constraints:
  - Problem 1 需要分别对四组数据做最小二乘直线拟合，并解释为什么仅看回归系数不足以判断拟合质量。
  - Problem 2 需要“taking into account the uncertainties”，因此应显式引入计数型数据的不确定度模型。

## Approach

- Language/tool choice:
  - `Python` 负责数值拟合、统计汇总、绘图与文档导出。
- Core algorithm or script plan:
  - Problem 1:
    - 读取四组数据，分别用最小二乘法拟合 $y=mx+b$。
    - 计算斜率、截距、$R^2$、残差标准差与杠杆异常点特征。
    - 绘制四联图，对比散点与拟合直线，突出异常结构。
  - Problem 2:
    - 假设放射性计数满足 Poisson 统计，不确定度取 $\sigma_i=\sqrt{N_i}$。
    - 用加权非线性最小二乘拟合 $N(t)=A\exp(-t/\tau)$。
    - 同时给出线性化加权拟合作为交叉核对，比较两者参数与误差。
    - 绘制带误差棒的数据点和拟合曲线。
- Output files to create:
  - `scripts/hw08_analysis.py`
  - `Makefile`
  - `docs/answer/render_docs.py`
  - `result/*.csv`, `result/*.png`, `result/temp-01.log`
  - `docs/answer/answer.md`, `docs/answer/answer.docx`, `docs/answer/answer.pdf`

## Testing

- Commands to run:
  - `make run`
  - `make docs`
- Expected checks:
  - 所有结果文件成功生成。
  - Problem 1 四组数据的回归参数接近 Anscombe quartet 的经典结果，并能从图中看出拟合优劣差异。
  - Problem 2 的指数衰减参数在非线性加权拟合与对数线性加权拟合之间保持一致到统计误差量级。
  - 导出的 PDF 中图像、公式和中文章节均正常显示。

## Risks

- Known numerical pitfalls:
  - Problem 2 的对数线性化会改变误差模型，因此不能把线性化结果当作唯一答案，只能作为核对。
  - 计数较小时 Poisson 误差与高斯近似可能有轻微偏差，但本题计数均为正且量级足够，可作为课程作业近似。
- Toolchain or environment concerns:
  - `pandoc` 可能不在 PATH，需要渲染脚本兼容自动下载。
  - PDF 导出依赖 `xelatex` 与系统字体，渲染后需要检查图像是否嵌入成功。
