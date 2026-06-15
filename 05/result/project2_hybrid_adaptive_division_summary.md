# Project 2 Hybrid Adaptive Division Summary

## Raw Runs

| profile | digits | seconds | digits_per_second | division_mode | resolved_division_mode | gpu_peak_gb | budget_cpu_calls | oom_cpu_calls | prefix_ok | source |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| fast-auto-140m-baseline | 140000000 | 22.490386 | 6224881.99 | newton-chunk-gpu-seed-prototype | newton-chunk-gpu-seed-prototype | 18.239 | 0 | 0 | True | project2_gpu_hybrid_fastauto_140m_after_refactor.csv |
| fast-auto-160m-baseline | 160000000 | 55.498087 | 2882982.24 | newton-chunk-gpu-seed-prototype | newton-chunk-gpu-seed-prototype | 17.882 | 0 | 3 | True | project2_gpu_hybrid_fastauto_160m_after_refactor.csv |
| high-digits-100m-baseline | 100000000 | 22.414052 | 4461486.96 | newton-chunk-gpu-seed-prototype | newton-chunk-gpu-seed-prototype | 16.243 | 1 | 0 | True | project2_gpu_hybrid_highdigits_100m_after_refactor.csv |
| high-digits-140m-baseline | 140000000 | 79.793957 | 1754518.83 | newton-chunk-gpu-seed-prototype | newton-chunk-gpu-seed-prototype | 15.708 | 7 | 0 | True | project2_gpu_hybrid_highdigits_140m_after_refactor.csv |
| fast-auto-140m-auto-div | 140000000 | 22.849448 | 6127062.70 | auto | newton-chunk-gpu-seed-prototype | 18.239 | 0 | 0 | True | project2_gpu_hybrid_fastauto_140m_auto_div_seq.csv |
| fast-auto-160m-auto-div | 160000000 | 50.259581 | 3183472.63 | auto | mpz-div | 16.633 | 0 | 1 | True | project2_gpu_hybrid_fastauto_160m_auto_div_seq.csv |
| high-digits-100m-auto-div | 100000000 | 22.366066 | 4471059.06 | auto | newton-chunk-gpu-seed-prototype | 16.243 | 1 | 0 | True | project2_gpu_hybrid_highdigits_100m_auto_div_seq.csv |
| high-digits-140m-auto-div | 140000000 | 59.662100 | 2346548.30 | auto | mpz-div | 15.708 | 3 | 0 | True | project2_gpu_hybrid_highdigits_140m_auto_div_seq.csv |

## Before vs After

| profile_family | digits | before_seconds | after_seconds | speedup_ratio | after_division | after_budget_cpu_calls | after_oom_cpu_calls |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| fast-auto | 140000000 | 22.490386 | 22.849448 | 0.984 | newton-chunk-gpu-seed-prototype | 0 | 0 |
| fast-auto | 160000000 | 55.498087 | 50.259581 | 1.104 | mpz-div | 0 | 1 |
| high-digits | 100000000 | 22.414052 | 22.366066 | 1.002 | newton-chunk-gpu-seed-prototype | 1 | 0 |
| high-digits | 140000000 | 79.793957 | 59.662100 | 1.337 | mpz-div | 3 | 0 |

## Key Takeaways

- `fast-auto` 在 `140M` 位保持 `newton-chunk-gpu-seed-prototype`，性能基本不变。
- `fast-auto` 在 `160M` 位自动切到 `mpz-div`，把总时间从约 `55.50s` 降到约 `50.26s`。
- `high-digits-auto` 在 `100M` 位保持 `newton-chunk-gpu-seed-prototype`，避免了对中档位数的回退。
- `high-digits-auto` 在 `140M` 位自动切到 `mpz-div`，把总时间从约 `79.79s` 降到约 `59.66s`。
