# Problem 5 Direct Cartesian Note

## Method Boundary

The numerical estimate is based on direct Cartesian tensor midpoint integration
of the indicator function

$$
V_n=\int_{[-1,1]^n}\mathbf 1_{\sum_i x_i^2\le 1}\,dx.
$$

The implementation uses sign symmetry,

$$
V_n=2^n\int_{[0,1]^n}\mathbf 1_{\sum_i x_i^2\le 1}\,dx,
$$

and then enumerates first-orthant tensor midpoint points. The first-orthant grid
may use mixed per-axis resolutions `q=2,3,4,5,6`; a pattern such as
`2^1 3^16 4^7 5^2 6^2` means one axis with two midpoint intervals, sixteen axes
with three midpoint intervals, and so on.

The final direct rows do not use Monte Carlo, quasi-Monte Carlo, closed-form
volume evaluation, recurrence, dynamic programming, bitset counting, sparse
grids, or boundary pruning. The coefficient-count predictor is retained only as
a parameter-selection tool.

## Algorithm

For one axis with first-orthant interval count `q`, midpoint coordinates are

$$
x_k=\frac{2k+1}{2q},\qquad k=0,\ldots,q-1.
$$

To avoid floating-point drift inside the point test, the C kernel uses the
equivalent integer comparison under the common denominator `60`:

$$
\sum_i \left((2k_i+1)\frac{60}{q_i}\right)^2\le 120^2.
$$

The optimized direct kernel splits axes into a prefix and tail:

1. build the tail table of integer squared-radius sums;
2. encode tail sums by lossless rank bytes for strict `rank8` comparison;
3. for every prefix point, compute the remaining radius threshold;
4. compare that threshold against every tail rank vector with AVX2;
5. accumulate the inside-point count and multiply by the mixed-grid cell volume.

This is still a full prefix-tail comparison: every prefix-tail pair is tested.
The forbidden bitset path was not used for final numerical results.

The final direct runs used ascending axis order, so the tail is the last axes
after laying out all `q=2`, then `q=3`, `q=4`, `q=5`, and `q=6` axes. The
rank8 table stores one byte per tail tensor point; tail sums larger than
`120^2` are saturated to one outside sentinel before ranking, which is exact for
all prefix thresholds.

| n | tail dimension | tail pattern | tail points | rank8 bytes | saturated rank values |
|---:|---:|:--|---:|---:|---:|
| 27 | 8 | `4^5 6^3` | `221184` | `221184` | `54` |
| 28 | 8 | `4^4 5^2 6^2` | `230400` | `230400` | `213` |
| 29 | 8 | `4^6 6^2` | `147456` | `147456` | `45` |
| 30 | 10 | `4^1 5^9` | `7812500` | `7812500` | `32` |

## Engineering Ablation

`run_q23456_ablation_tests.py` runs proxy ablation tests for the implementation
choices used by the strict direct path. It builds both the optimized AVX2 kernel
and `direct_tensor_midpoint_ablation_kernel`, then writes:

```text
results/q23456_engineering_ablation_raw.csv
results/q23456_engineering_ablation_summary.csv
results/q23456_engineering_ablation_speedups.png
```

All ablation pairs check that the inside-point count is unchanged. The measured
speedups in the current summary are:

| optimization | speedup |
|:--|---:|
| sign symmetry point-count reduction | `1024.000000` |
| integer/incremental radius update | `54.339480` |
| precomputed tail table | `8.852083` |
| scalar prefix batching | `2.431771` |
| rank8 tail storage | `2.938240` |
| AVX2 prefix batching | `2.812923` |
| AVX2 unrolled comparison | `36.246700` |
| OpenMP 88-thread execution | `34.233108` |
| layout repeat-check tuning | `1.156986` |
| batch32 guardrail | `0.698746` |

## Prediction And Hyperparameter Selection

`run_q23456_coefficient_search.py` computes the exact coefficient count for a
candidate mixed midpoint grid. This is fast because it counts how many tensor
midpoints would be inside, but it is not used as the submitted numerical result.
Its role is to screen candidate patterns before spending hours on strict direct
enumeration.

For a candidate count vector `c=(c2,c3,c4,c5,c6)`, the predictor uses the
integer midpoint contributions

```text
a(q,k) = ((2*k + 1) * 60 / q)^2
```

and computes the coefficient distribution of

```text
F_c(z) = product_q (sum_k z^a(q,k))^c_q .
```

The candidate inside count is the exact coefficient sum for exponents
`s <= 120^2`. The candidate error is then

```text
abs(2^n * inside / product(q_i) - reference_volume(n)) / reference_volume(n)
```

where `reference_volume` is the formula supplied in the problem statement. This
coefficient count is deliberately kept out of the submitted result path: using
it as the answer would no longer be strict point-by-point Cartesian
enumeration. It is only a parameter-selection oracle.

The compact selection rule for `n=27,28,29,30` was:

```text
among q2-containing q23456 candidates,
choose the minimum total point count subject to predicted relative error <= 2%
```

The selected rows were then run through
`run_q23456_selected_under30_direct.py`, which invokes the strict C kernel and
records the actual direct inside counts.

## Volume Trend Plot

`run_problem5_volume_trend_plot.py` produces the Problem 5(b) trend plot:

```text
result/problem5_volume_trend_formula.csv
result/problem5_volume_trend_formula.png
```

This plot uses the formula stated in the problem only as a reference curve for
the dimension trend. It overlays the actual strict direct rows for `n=27..30`.
It is not used to generate the submitted direct numerical estimates in
Problem 5(a).

## Results

Actual direct output:

```text
results/q23456_selected_under30_direct_results.csv
```

| n | pattern | total first-orthant points | inside | estimate | relative error | count runtime | throughput |
|---:|:--|---:|---:|---:|---:|---:|---:|
| 27 | `2^1 3^16 4^7 5^0 6^3` | `304679870005248` | 516 | `2.273085768574309967e-04` | `1.990582e-02` | `100.7609 s` | `3.023791e12` points/s |
| 28 | `2^1 3^16 4^7 5^2 6^2` | `1269499458355200` | 487 | `1.029760715623896702e-04` | `1.588363e-02` | `457.9538 s` | `2.772113e12` points/s |
| 29 | `2^1 3^14 4^12 5^0 6^2` | `5777633090469888` | 529 | `4.915589273338612715e-05` | `1.797700e-02` | `2068.7067 s` | `2.792872e12` points/s |
| 30 | `2^1 3^19 4^1 5^9 6^0` | `18160335421875000` | 375 | `2.217212263133558743e-05` | `1.171641e-02` | `6681.2057 s` | `2.718122e12` points/s |

All four direct rows are below the 10 percent relative-error threshold.

## Low-Risk Layout Optimization

The strict layout benchmark only changed execution layout:

- tail dimension;
- prefix chunk size;
- ascending or descending axis order;
- `batch_prefixes=16` versus `32`.

The repeated checks showed that `batch_prefixes=32` is slower on this machine.
For the cheapest 31-dimensional candidate style, `asc_tail10_c512_b16` and
`desc_tail12_c128_b16` improved the proxy median rate to about `2.88e12`
points/s. For the predicted-below-2-percent candidate style, the original
`batch16` layout remained the safest choice.

Updated 31-dimensional strict count-time estimates from repeated median proxy
rates:

| target | selected layout basis | estimated count time |
|:--|:--|---:|
| `2^1 3^15 4^12 5^1 6^2` | `asc_tail10_c512_b16` proxy median | `8.336 h` |
| `2^1 3^16 4^10 5^1 6^3` | `asc_tail9_c128_b16` proxy median | `9.656 h` |
