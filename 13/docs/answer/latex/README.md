# Native LaTeX Report Prototype

This directory is an experiment for writing the `HW/13` report directly in LaTeX.
It does not replace the Markdown/Pandoc report in `docs/answer/`.

## Build

```bash
make -C HW/13/docs/answer/latex
```

The generated PDF is:

```text
HW/13/docs/answer/latex/answer.pdf
```

## Purpose

- Test direct control over figure placement with `figure[H]`.
- Test native algorithm blocks with `algorithm2e`.
- Keep top-level Roman headings as visual groups, while problems use `1`, `2`
  and inner blocks use `1.1`, `1.2` instead of Pandoc's incidental `0.` prefix.
- Keep the same numerical artifacts under `HW/13/result/`.
- Compare maintainability against the current Markdown source.
