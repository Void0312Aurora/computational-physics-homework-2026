# Plan

## Scope

- Target folder: `HW/03`
- Problem source:
  - `HW/03/docs/problem/image.png`
  - `HW/03/docs/problem/image_2.png`
  - `HW/03/docs/problem/image_3.png`
  - `HW/03/docs/problem/image_4.png`
  - `HW/03/docs/problem/image_5.png`

## Problem Summary

- Chinese translation:
  - Problem 1: 研究二次方程 `x^2 - b x + 1 = 0` 的两个根，尤其是小根 `x2=(b-r)/2`（其中 `r=sqrt(b^2-4)`）在 `b` 很大时的相消误差；对 `b=100` 以及更多更大的 `b` 做实验并列表，再把 `x2` 有理化为 `2/(b+r)` 后重做，并解释误差为何变化。
  - Problem 2: 分别用 single、double、quad 精度计算 `(x-1)^10`、它的多项式展开式，以及展开式的 Horner 形式；在 `x in [0.7, 1.3]` 上放大观察这些曲线，并画出展开式相对误差的分布，再说明 Horner 法带来的变化与原因。
  - Problem 3: 求半精度浮点数的 machine precision、range 等指标，并给出一个能明显体现 roundoff error 严重性的例子。
- Required outputs:
  - 数值实验程序
  - 问题 1 到 3 的结果表、原始日志与图像
  - `docs/answer/answer.md`
  - `docs/answer/answer.docx` 与 `docs/answer/answer.pdf`
- Numerical or implementation constraints:
  - Problem 1 与 Problem 2 都涉及不同浮点精度的真实比较，不能把 `long double` 当作 true quad 使用
  - Problem 2 的参考值需要高于 `__float128` 的精度，避免把“quad 自己对自己”当成真值
  - Problem 3 需要确认本机 `_Float16` 的实际位布局与舍入行为

## Approach

- Language/tool choice:
  - 使用 `C` 负责 `float`、`double`、`__float128`、`_Float16` 的原始数值实验
  - 使用 `Python` 负责高精度参考值、误差统计、绘图与答案文档生成
- Core algorithm or script plan:
  - Problem 1: 对多个 `b` 值分别计算标准公式与有理化公式的小根，输出原始结果 CSV；用高精度参考值计算相对误差并画对数图
  - Problem 2: 在 `[0.7, 1.3]` 上离散采样，分别用 direct / expanded / Horner 三种形式与三种精度求值；再用高精度参考值统计相对误差并生成曲线图
  - Problem 3: 用 `_Float16` 的显式逐步舍入实验估计 machine epsilon、最小正规数、最小非正规数与最大有限数；再用一个包含严重相消的表达式展示半精度 roundoff error
  - `plot_results.py` 读取原始 CSV，生成所有图表、汇总 CSV 与 `docs/answer/answer.md`
  - `docs/answer/render_docs.py` 使用 `pypandoc` 导出 `docx/pdf`，并生成兼容性占位 `answer.doc`
- Output files to create:
  - `scripts/problem1.c`
  - `scripts/problem2.c`
  - `scripts/problem3.c`
  - `plot_results.py`
  - `Makefile`
  - `README.md`
  - `result/temp-01.log`
  - `result/problem1_roots.csv`
  - `result/problem1_summary.csv`
  - `result/problem1_relative_error.png`
  - `result/problem2_values.csv`
  - `result/problem2_summary.csv`
  - `result/problem2_zoom.png`
  - `result/problem2_relative_error.png`
  - `result/problem3_half_metrics.csv`
  - `result/problem3_roundoff.csv`
  - `result/problem3_roundoff.png`
  - `docs/answer/answer.md`
  - `docs/answer/render_docs.py`

## Testing

- Commands to run:
  - `make docs`
- Expected checks:
  - `scripts/problem1.c`、`scripts/problem2.c`、`scripts/problem3.c` 均可独立编译运行，并把原始 CSV 与日志写入 `result/`
  - Problem 1 的标准公式相对误差随 `b` 增大明显恶化，而有理化公式保持稳定
  - Problem 2 的 naive expanded 在 `x=1` 附近误差显著放大，Horner 法有明显改善
  - Problem 3 的 `_Float16` 指标与 binary16 预期一致，roundoff 示例出现明显失真
  - `answer.docx`、`answer.pdf` 与占位 `answer.doc` 成功生成

## Risks

- Known numerical pitfalls:
  - `long double` 在本机是 80-bit extended 存在 16-byte 存储，不等于 binary128，必须显式用 `__float128`
  - Problem 2 在 `x=1` 处真值为 0，相对误差需要避开除以 0
  - `_Float16` 表达式若不显式逐步舍入，可能被编译器提升到更高精度求值
- Toolchain or environment concerns:
  - `gcc` 需要链接 `-lquadmath`
  - 绘图与文档导出依赖 `.venv` 中的 `matplotlib` 与 `pypandoc`
  - PDF 导出依赖本机 `xelatex`
