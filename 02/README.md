# HW/02 Notes

## Scope

This directory contains a single C reference implementation for all five homework problems:

- Gregory-Leibniz summation order
- Machin's formula for pi
- Endianness detection
- Machine epsilon detection
- A recurrence that should converge to pi but becomes unstable

## Why C here

The homework statements explicitly ask for C programs in Problems 3 and 4, and Problem 2 explicitly points to `quadmath.h`. Using one C executable across all five problems keeps the numerical behavior consistent and matches the assignment style.

## Files

- `solution.c`: main implementation
- `Makefile`: build, run, and document-export entry point
- `result/temp-01.log`: raw output captured from the executable
- `docs/answer/answer.md`: summary report source
- `docs/answer/render_docs.py`: exports `docx` and `pdf`

## Build and run

```bash
cd HW/02
make
./hw02
```

To refresh the report outputs:

```bash
cd HW/02
make docs
```
