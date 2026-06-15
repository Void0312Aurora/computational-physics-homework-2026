# Plan

## Scope

- Target folder: `HW/07`
- Problem source: `HW/07/docs/problem/*.png`
- Required outputs: runnable numerical scripts, raw result logs and tables under `result/`, figures for the report, and final homework report artifacts under `docs/answer/`.

## Problem Summary

- Problem 1: 根据 Debye 固体热容公式，给定铝样品体积 `1000 cm^3`、原子数密度 $\rho=6.022\times 10^{28}\,\mathrm{m^{-3}}$、Debye 温度 $\theta_D=428\,\mathrm{K}$，用 Simpson 法计算 $C_V(T)$，并绘制 $T=5\,\mathrm{K}$ 到 $500\,\mathrm{K}$ 的热容曲线。
- Problem 2: 对
  $$
  I=\int_0^1\sin^2(\sqrt{100x})\,dx
  $$
  分别使用自适应梯形法和自适应 Simpson 法达到 $\epsilon=10^{-10}$ 的精度，输出每次加倍切片数后的积分估计和误差估计，并解释 Simpson 法为什么应当更快达到精度。
- Problem 3: 对
  $$
  I=\int_0^1\frac{dx}{1+x^2}=\frac{\pi}{4}
  $$
  用梯形法、Simpson 法和 Romberg 积分分别达到 $10^{-2}$ 到 $10^{-12}$ 的精度，并用误差对步长 $h$ 的拟合检验误差阶关系；额外将误差图横轴延长到 $h=2^{-28}$，观察舍入误差平台与 Romberg 外推末端浮动。
- Problem 4: 对
  $$
  \int_{-1}^{1}\sqrt{1-x^2}\,dx,\qquad
  \int_0^\pi \sin^2\theta\,d\theta,\qquad
  \int_0^\infty \frac{dx}{(1+x)\sqrt{x}}
  $$
  编写积分程序；对含端点奇性或无穷区间的问题使用变量代换得到稳定的有限区间积分。
- Problem 5: 计算单位半径 $n$ 维超球体积随维数变化的关系。解析参考值仍按题面公式计算，但数值部分改为真正的多维直角坐标积分：直接在超立方体上积分单位超球的指示函数，不使用 Monte Carlo、递推关系、Gamma 积分化简或一维降维；同时以相对误差不超过 `10%` 为阈值，探讨该方法的可计算维数上限。

## Approach

- Language/tool choice: 使用 Python 3，配合 `numpy`、`matplotlib` 和标准库 `math` 完成数值积分、拟合、制图和结果导出。
- Core algorithm plan:
  - 复合 Simpson 法、复合梯形法采用统一函数实现。
  - 自适应梯形法通过切片数加倍并复用新增中点函数值；误差估计用相邻两级梯形结果差的三分之一。
  - 自适应 Simpson 法使用相邻两级梯形结果组合 $S_i=(4T_i-T_{i-1})/3$，误差估计用相邻两级 Simpson 结果差的十五分之一。
  - Romberg 积分从复合梯形序列出发做 Richardson 外推。
  - Problem 4 的奇性积分使用 $x=\sin\theta$、$x=\tan^2\theta$ 等代换避免直接处理不可导端点或无穷上限。
  - Problem 5 采用真正的多维自适应直角坐标积分器 `scipy.integrate.cubature`，并仅利用坐标轴对称性：把积分域从 $[-1,1]^n$ 缩到第一卦限 $[0,1]^n$，再把积分结果乘以 $2^n$。
  - Problem 5 从低维开始逐维测试，把每一维的数值结果与闭式公式比较；一旦相对误差超过 `10%`，就把此前一维记为该算法在当前参数下的可计算维数上限，并至少验证到越界后的下一维以确认交叉点。
- Output files to create:
  - `scripts/hw07_integrals.py`
  - `Makefile`
  - `result/temp-01.log`
  - `result/problem*_*.csv`
  - `result/problem*_*.png`
  - `docs/answer/answer.md`
  - `docs/answer/render_docs.py`
  - `docs/answer/answer.docx`
  - `docs/answer/answer.pdf`

## Testing

- Commands to run:
  - `python3 scripts/hw07_integrals.py`
  - `make run`
  - `make docs`
- Expected checks:
  - Problem 1 的高温极限接近 Dulong-Petit 极限 $3V\rho k_B$。
  - Problem 2 的积分值应接近题面提示的 $0.45$，并满足误差阈值 $10^{-10}$。
  - Problem 3 的三种方法都应在可达到范围内满足指定精度；梯形法拟合阶应接近 2，Simpson 法因本题主误差项抵消应接近 6；延长横轴后应能观察到机器精度平台和 Romberg 外推的舍入误差影响。
  - Problem 4 的三项积分应分别与 $\pi/2$、$\pi/2$、$\pi$ 一致。
  - Problem 5 的解析体积应在低维阶段于 $n=5$ 左右达到峰值；真正的多维直角坐标积分误差会随维数快速增长，在当前实现中应当在某个中低维处越过 `10%` 阈值，因此需要实测到首次越界的维数为止。

## Risks

- Known numerical pitfalls:
  - Debye 积分在 $x=0$ 处为可去奇点，需使用极限值避免 `0/0`。
  - Problem 4 的端点奇性若直接积分会降低收敛阶，变量代换后再积分更可靠。
  - Problem 3 横轴延长后，误差不会无限下降；Romberg 积分在接近机器精度后会受舍入误差影响，误差阶拟合不能按固定幂律机械解释。
  - 真正的多维直角坐标积分会遭遇明显的维数灾难；即使允许利用对称性并使用自适应 cubature，随着维数增加，分裂次数和误差估计仍会迅速恶化，并最终越过允许误差阈值。
- Toolchain concerns:
  - PDF 导出依赖 `pypandoc`、pandoc 与 XeLaTeX；若系统无全局 pandoc，则渲染脚本会尝试使用 `pypandoc` 下载或定位本地 pandoc。
