# Problem 5 Direct Cartesian Core Scripts

This directory keeps only the final reproducible line used for HW/07 Problem 5.

The reported numerical results are produced by strict direct Cartesian midpoint
tensor enumeration in the first orthant:

- no Monte Carlo or quasi-Monte Carlo sampling;
- no closed-form volume formula for producing the estimate;
- no volume recurrence;
- no bitset/operator result path;
- no dynamic-programming result path;
- coordinate sign symmetry is used only to reduce `[-1,1]^n` to `[0,1]^n`;
- the coefficient predictor is used only to choose candidate grid parameters.

## Core Commands

Build the strict q23456 CPU kernel:

```bash
make
```

Regenerate q23456 coefficient-screening rows:

```bash
python3 run_q23456_coefficient_search.py --dimensions 27,28,29,30,31
```

Select the compact under-30 candidate set:

```bash
python3 run_q23456_select_under30_results.py --dimensions 27,28,29,30
```

Run or resume the actual strict direct Cartesian rows through 30 dimensions:

```bash
python3 run_q23456_selected_under30_direct.py --dimensions 27,28,29,30
```

The 30-dimensional row was computed with:

```bash
python3 run_q23456_selected_under30_direct.py --dimensions 30 --tail-dimension 10 --prefix-chunk-points 128 --batch-prefixes 16
```

Benchmark low-risk layout choices:

```bash
python3 run_q23456_strict_layout_optimization_tests.py
python3 run_q23456_strict_layout_repeat_checks.py
```

Run the engineering ablation suite and regenerate its figure:

```bash
python3 run_q23456_ablation_tests.py --repeats 3
```

Regenerate the Problem 5(b) formula-reference trend plot:

```bash
python3 run_problem5_volume_trend_plot.py
```

## Core Files

- `problem5_core.py`: shared CSV, reference-volume, point-count, coefficient
  predictor, and build helpers.
- `direct_tensor_midpoint_orthant_mixed_q26_tile_batch_avx2.c`: strict q23456
  first-orthant midpoint tensor enumerator.
- `direct_tensor_midpoint_ablation_kernel.c`: scalar proxy kernel for engineering
  ablation tests.
- `run_q23456_coefficient_search.py`: coefficient-count parameter screening.
- `run_q23456_select_under30_results.py`: compact candidate selection.
- `run_q23456_selected_under30_direct.py`: final strict direct Cartesian run.
- `run_q23456_ablation_tests.py`: engineering ablation tests and speedup figure.
- `run_problem5_volume_trend_plot.py`: Problem 5(b) volume-vs-dimension trend
  plot from the problem formula, with actual direct rows overlaid.
- `run_q23456_strict_layout_optimization_tests.py`: single-run low-risk layout
  benchmark.
- `run_q23456_strict_layout_repeat_checks.py`: repeated layout sanity check.

## Key Results

Actual direct rows are in:

```text
results/q23456_selected_under30_direct_results.csv
```

The completed rows are:

| n | pattern | relative error | count runtime |
|---:|:--|---:|---:|
| 27 | `2^1 3^16 4^7 5^0 6^3` | `1.990582e-02` | `100.7609 s` |
| 28 | `2^1 3^16 4^7 5^2 6^2` | `1.588363e-02` | `457.9538 s` |
| 29 | `2^1 3^14 4^12 5^0 6^2` | `1.797700e-02` | `2068.7067 s` |
| 30 | `2^1 3^19 4^1 5^9 6^0` | `1.171641e-02` | `6681.2057 s` |

The practical 31-dimensional count-time estimate remains about `9-10 h` under
the strict direct path, depending on which q23456 candidate is selected.
