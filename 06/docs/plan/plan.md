# Plan

## Scope

- Target folder: `HW/06`
- Problem source: `docs/problem/image.png`, `image_2.png`, `image_3.png`, `image_4.png`, `image_5.png`, `image_6.png`

## Problem Summary

- Chinese translation:
  - Problem 1 要求分别推导 `y=f(x)` 的二阶、三阶、四阶、五阶导数的向前差分、向后差分与中心差分公式。
  - Problem 2 给出 `f(x)=x e^x` 在 `x=1.8,1.9,2.0,2.1,2.2` 的表格值，要求用本讲义中所有适用公式近似 `f'(2.0)`，与精确值比较，并继续测试 `h=0.1,0.01,0.001,0.0001,...` 时哪种公式最好，以及为什么 `h` 过小时反而变差。
  - Problem 3 给出 `f(x)=x exp(x^2)` 在同样节点上的表格值，要求用所有适用公式近似 `f''(2.0)`，与精确值比较，并继续测试不同 `h` 下的最佳公式与误差反弹现象。
  - Problem 4 从前向差分定义出发，要求实现通用导数程序；对 `f(x)=x(x-1)` 在 `x=1` 处测试一组 `delta`；复现 `f(x)=sin x` 在 `x=0.5` 的 `log10-log10` 误差图；再依据课堂误差模型估计最佳步长 `h_best`。
- Required outputs:
  - symbolic derivative formulas for Problem 1
  - numeric tables, error scans, and figures for Problems 2 to 4
  - `docs/answer/answer.md`, `answer.docx`, and `answer.pdf`
- Numerical or implementation constraints:
  - the provided tables are rounded to 6 decimal places, so the `h=0.1` results should be interpreted as tabulated-data approximations
  - the report must explain why smaller `h` eventually worsens the answer when cancellation and roundoff dominate

## Approach

- Language/tool choice: `Python 3.13 + sympy + numpy + matplotlib + pypandoc`
- Core algorithm or script plan:
  - use Taylor expansion and finite-difference moment conditions to derive Problem 1 formulas symbolically
  - evaluate lecture-applicable first- and second-derivative stencils on the given `h=0.1` tables
  - sweep `h` over logarithmic grids for Problems 2 and 3 to study truncation-versus-roundoff tradeoffs
  - generate the requested Problem 4 forward-difference tables and the `sin(x)` log-log error model
- Output files to create:
  - `solution.py`
  - `scripts/hw06_analysis.py`
  - `result/problem1_formulas.md`
  - `result/*.csv`
  - `result/*.png`
  - `result/hw06_summary.json`
  - `docs/answer/render_docs.py`
  - `docs/answer/answer.md`

## Testing

- Commands to run:
  - `make run`
  - `make docs`
- Expected checks:
  - `result/temp-01.log` records the latest computation
  - `result/temp-02.log` records the latest document-render step
  - all CSV and PNG artifacts are regenerated
  - `docs/answer/answer.md` matches the latest numbers in `result/hw06_summary.json`
  - `answer.docx` and `answer.pdf` render successfully with embedded figures

## Risks

- Known numerical pitfalls:
  - very small `h` or `delta` causes subtractive cancellation and amplifies roundoff in both first- and second-derivative formulas
  - second-derivative formulas are especially sensitive because the numerator is divided by `h^2`
- Toolchain or environment concerns:
  - `pandoc` is not guaranteed to be on `PATH`, so the render script should fall back to `pypandoc`'s downloader when necessary
  - figure readability must be checked after rendering so that logarithmic axes and legends remain legible in the PDF
