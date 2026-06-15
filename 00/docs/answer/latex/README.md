# HW00 Native LaTeX V2 Prototype

> Frozen historical prototype. Do not use this directory as the template for new
> homework reports. Future report work should converge on the Markdown/Pandoc
> source in `HW/00/docs/answer/answer.md`.

This directory is a second report-format prototype for `HW/00`.
It keeps the lightweight content and structure of `../answer.md`, but writes the
report directly in LaTeX for tighter control over tables, formulas, figure
placement, and reusable section macros.

It does not replace the Markdown/Pandoc report in `HW/00/docs/answer/`.

## Build

```bash
make -C HW/00/docs/answer/latex
```

The generated PDF is:

```text
HW/00/docs/answer/latex/answer.pdf
```

## Scope

- Demonstrate a native LaTeX cover block.
- Preserve the `待求问题` / `解决方式` / `问题答案` / `分析` section rhythm.
- Keep top-level Roman headings as visual groups, while problems use `1`, `2`
  and inner blocks use `1.1`, `1.2` instead of Pandoc's incidental `0.` prefix.
- Demonstrate `\scriptlink{problemX.py}` links from each problem to the local
  `HW/00/scripts/` implementation files.
- Preserve the final state of the native LaTeX experiment without promoting it
  as a future homework template.
- Reuse the portrait asset from `../assets/profile.jpg`.
