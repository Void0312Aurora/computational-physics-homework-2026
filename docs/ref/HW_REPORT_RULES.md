# HW Report Rules

These rules apply only to `HW/` tasks. `EX/` keeps the existing compact packaging and report style.

## Reference Inputs

- Format reference: `HW/docs/ref/HW15_白博臣 何骐多 夏营.pdf`
- Available portrait asset: `HW/docs/ref/pic.jpg`
- Additional portrait assets for group reports: `HW/docs/ref/pic_2.jpg`, `HW/docs/ref/pic_3.jpg`

The reference PDF is useful mainly for its overall homework-report feel:

- a recognizable cover or leading page
- clear `Problem`-by-`Problem` organization
- figures placed close to the corresponding explanation

It should not be copied mechanically. In particular, the long source-code listings in the sample are not a required part of our `HW` reports.

## What Stays The Same

- The submission folder contract still stays:
  - `docs/answer.pdf`
  - `scripts/`
  - `results/`
- Full runnable code still belongs in `scripts/` or the task root source files.
- Raw outputs and generated artifacts still belong in `result/` or `results/`.

## What Changes For HW

### 1. Report Tone

`HW` reports should look like a course homework report, not just a repository inventory.

- Prefer a cover page or leading block with:
  - homework number such as `HW05`
  - author name
  - portrait photo(s) when available
- If the user states that the report is a group assignment, default to a three-person cover layout unless the user requests otherwise.
- When `pic.jpg`, `pic_2.jpg`, and `pic_3.jpg` are all available, use all three portraits on the cover for group reports.
- For multi-author work, one photo per author is required when the portrait assets are available.
- For single-author work, one centered portrait is enough.

### 2. Body Structure

After the cover, organize the body by the original problem numbering:

- `待求问题`
- `解决方式`
- `问题答案`
- `理解`

Keep `Problem 1`, `Problem 1(a)`, `Problem 2`, and similar numbering visible all the way through the report.

- If the original problem contains subquestions such as `(a)`, `(b)`, `(c)` or `(1)`, `(2)`, `(3)`, they must be expanded explicitly inside `题目陈述`、`解决方案`、`问题答案` and `讨论和扩展`.
- Do not merge several subquestions into one generic paragraph after the problem has been expanded.
- If automatic section numbering is enabled, do not write heading text such as `第一题` or `Problem 1` again inside the title itself.
- Prefer headings like `# L1 点的平衡方程与数值求解`, letting the renderer display the section number automatically.
- If the report instead uses unnumbered top-level visual groups such as `# I. ... {-}` and numbered `## Problem ...` headings below them, add `HW/docs/ref/fix_unnumbered_group_counters.lua` as a Pandoc Lua filter. Otherwise Pandoc will render the lower headings as `0.1`, `0.1.1`, and so on.
- If the original problem contains `(a)`, `(b)` or similar subparts, keep those subparts explicit inside the corresponding chapter.
- If a subpart asks for a derivation or proof, write the derivation itself; do not replace it with a one-line summary.

### 3. Code Presentation

Do not paste full source files into the report by default.

Instead, use one of these forms when explaining the implementation:

- algorithm design diagram
- logic flow chart
- concise pseudocode
- key formulas
- small annotated code excerpts only when a few lines are being discussed directly

The full implementation stays in `scripts/`. The report should explain the method and justify the result, not duplicate the entire codebase.

### 4. Figures And Tables

- Put important plots, tables, and diagrams near the paragraph that explains them.
- Do not let a discussion paragraph finish on one page and push the cited figure to a later page unless the layout is unavoidable.
- If pagination separates the explanation and the figure, resize the figure or reorder the local text block.
- If a figure is cited as evidence, embed it in the pdf instead of only listing its filename.
- Fix crowding, overlap, clipped labels, and unreadable legends before packaging.

### 5. Writing Style

- Use formal academic Chinese rather than conversational phrasing.
- Do not include internal decision notes such as why the report was organized in a certain way.
- Give equations, derivations, assumptions, and numerical evidence directly.

## Suggested HW Answer Layout

```md
---
title: "HWXX Report"
date: "YYYY-MM-DD"
toc-title: "目录"
mainfont: "Noto Serif CJK SC"
monofont: "Noto Sans Mono CJK SC"
geometry: margin=1in
fontsize: 11pt
---

# HWXX

姓名

![证件照](assets/profile.jpg){ width=28% }

\newpage

# 说明

1. 本文档对应 `HW/XX`。
2. 完整代码保留在 `scripts/`，正文只展示解释问题所需的算法图、逻辑图、伪代码或关键公式。
3. 所有数值和图像来自本机真实运行。

# 待求问题

# 解决方式

## Problem 1 算法设计

# 问题答案

# 理解
```

## Portrait Asset Notes

- If `HW/docs/ref/pic.jpg` is reused directly, make sure the render step can resolve it.
- For a three-person group cover, either copy `pic.jpg`, `pic_2.jpg`, and `pic_3.jpg` into the task-local assets folder or include `HW/docs/ref/` in the render resource path.
- The safer choice is to copy the portrait into a task-local folder such as `docs/answer/assets/`.
- If the render script uses pandoc `--resource-path`, include the task-local assets folder or the shared `HW/docs/ref/` path explicitly.

## Practical Rule Of Thumb

When unsure, use this split:

- `scripts/` stores the full implementation
- `results/` stores raw outputs and generated figures
- `answer.pdf` tells the story of the method, the evidence, and the conclusions
