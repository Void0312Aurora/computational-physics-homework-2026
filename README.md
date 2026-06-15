本文档考虑对有关的作业流程进行标准化。

首先是项目标准的标准化，每一子项目服从以下的结构特征：
Program/docs/problems
            /answers
            /plans
        /scripts
        /results
        /data(&alternative component)

其中scripts需要考虑如下组织方法：scripts下脚本按照题目组织，每一题目来构建一个单独的实现脚本，不过可以构造单独的模块来构造基类或者共享类/模块，这是之前的工作尚未提及或者标准化的步骤。

reslut下有关结果也按照题目分割而非扁平化堆积，

answer的组织参照HW/EX Workflow的有关组织规则，以下给出一个示范，当然也可以参照00下的 Markdown 示例：

后续报告默认收敛到 Markdown/Pandoc 链路：以 `docs/answer/answer.md` 作为报告源文件，再按需要导出 PDF 或 docx。
`00/docs/answer/latex/` 和 `13/docs/answer/latex/` 下的原生 LaTeX 样例已经冻结，仅作为历史试验保留，不再作为新作业模板套用，也不继续跟随后续 Markdown 规则演化。

---
title: "Homework XX Report"
subtitle: "占位"
author: "占位"
date: "2026-xx-xx"
geometry: margin=1in
fontsize: 11pt
header-includes:
  - \usepackage{amsmath}
  - \usepackage{booktabs}
  - \usepackage{float}
  - \floatplacement{figure}{H}
  - \renewcommand{\figurename}{图}
  - \renewcommand{\tablename}{表}
---

| ![Portrait](profile.jpg){ width=20% } |

<!--如果为小组作业，需要考虑将三名小组成员的profile都添加进入-->

|:--:|

| 项目 | 内容 |
|:--|:--|
| 源题编号 | `Homework xx` |
<!--题源编号和作业的编号和子项目编号对齐，不受到其他参数影响，比如PPT内编号-->
| 学生姓名 | 占位 |
| 报告主题 | 占位 |
| 实验环境 | `占位` |
| 统一单位 | 除特别说明外，取 $占位$ |
| 随机种子 | `占位` |

封面表只保留面向报告读者的作业信息；不要在第一页写 `作业目录`、脚本路径、提交包结构或“本文件不替代主报告”之类的仓库内部说明。

\newpage

# I. XX {-}

## Problem 1：XX

相关脚本：

- [本地 scripts/problemX.py](../../scripts/problemX.py)
- [GitHub scripts/problemX.py](https://github.com/<owner>/<repo>/blob/main/XX/scripts/problemX.py)

### 待求问题

占位

### 解决方式

占位

<!--可以辅以有关的伪代码块，算法图等辅助性内容说明-->

### 问题答案

占位

### 分析

占位

## Problem 2：XX

......

如果 Markdown/Pandoc 构建链路启用了 `--number-sections`，同时又使用 `# I. XX {-}`
这类不编号的大问视觉分组，需要在 Pandoc 参数中加入
`HW/docs/ref/fix_unnumbered_group_counters.lua`。否则 Pandoc 会把下一层
`## Problem ...` 解释为“第 0 章下的小节”，从而产生 `0.1`、`0.1.1` 这样的多余前缀。

对于作业的组织或者打包，遵从Cp Pack skill给出的参照，这里不予以示例了。
