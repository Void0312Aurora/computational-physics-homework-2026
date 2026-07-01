# LaTeX Report

This directory contains the classroom-report LaTeX version used for the final
PDF layout. It follows the indented HW00 sample more closely than the default
Pandoc PDF.

## Build

```bash
make -C 07/docs/answer/latex
```

The generated PDF is:

```text
07/docs/answer/latex/answer.pdf
```

The root `07/Makefile` copies this PDF to `07/docs/answer/answer.pdf`
after building the regular Markdown exports.

## Typography

- A4 page with 2.7 cm side margins and 2.5 cm top/bottom margins.
- Small-four body text, first-line indentation, and 1.36 line stretch.
- Compact heading spacing, small table/figure captions, and framed small code blocks.
