# HW/03

This directory contains a mixed `C + Python` workflow for the three numerical-analysis
questions shown in `docs/problem/`.

## Files

- `solution.c`: raw experiments in `float`, `double`, `__float128`, and `_Float16`
- `plot_results.py`: reads CSV outputs, computes high-precision references, draws plots,
  and regenerates `docs/answer/answer.md`
- `docs/answer/render_docs.py`: exports `answer.docx` / `answer.pdf`
- `Makefile`: reproducible entry point

## Usage

```bash
make docs
```

This command rebuilds the executable, refreshes `result/*`, regenerates the answer
report, and exports the document files.
