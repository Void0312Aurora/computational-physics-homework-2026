# HW/03

This directory contains a mixed `C + Python` workflow for the three numerical-analysis
questions shown in `docs/problem/`.

## Files

- `scripts/problem1.c`: standalone root-cancellation experiment
- `scripts/problem2.c`: standalone polynomial-evaluation experiment
- `scripts/problem3.c`: standalone half-precision roundoff experiment
- `plot_results.py`: reads CSV outputs, computes high-precision references, draws plots,
  and regenerates `docs/answer/answer.md`
- `docs/answer/render_docs.py`: exports `answer.docx` / `answer.pdf`
- `Makefile`: reproducible entry point

## Usage

```bash
make docs
```

This command runs the three problem scripts, refreshes `result/*`, regenerates the answer
figures/tables, preserves the formatted `docs/answer/answer.md` source, and exports
the document files.

To rebuild the normalized submission folder and tar archive:

```bash
make package
```

The LaTeX report version follows the `HW/00/docs/answer/latex` layout and can be
rebuilt with:

```bash
make -C docs/answer/latex
```
