# LaTeX Report

This directory contains a frozen classroom-report LaTeX snapshot synchronized
from `../answer.md`. The original Markdown, DOCX, and PDF files in
`docs/answer/` are kept unchanged.

## Build

```bash
make -C HW/04/docs/answer/latex
```

The generated PDF is:

```text
HW/04/docs/answer/latex/answer.pdf
```

## Typography

- A4 page with 2.7 cm side margins and 2.5 cm top/bottom margins.
- Small-four body text, first-line indentation, and 1.36 line stretch.
- Compact heading spacing, small table/figure captions, and framed small code blocks.

## Source Drift Check

`make` first verifies that `source_manifest.json` still matches both
`../answer.md` and the frozen `answer.tex` snapshot. After intentionally
updating both files and confirming they are synchronized, refresh the recorded
hashes with:

```bash
make -C HW/04/docs/answer/latex refresh-source-manifest
```
