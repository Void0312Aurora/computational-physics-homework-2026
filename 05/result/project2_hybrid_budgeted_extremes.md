# CPU / Hybrid Extreme Exploration

| route | target_digits | seconds | digits_per_second | prefix_ok | status | stop_reason | notes |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| hybrid-fast-auto | 100000000 | 14.637568 | 6831736.02 | True | ok |  | workers=32,chunk_terms=262144,gpu_stages=merge-and-final,sqrt=chunk-gpu-rsqrt-prototype,div=newton-chunk-gpu-seed-prototype,chunk_backend=auto,gpu_budget_gb=None |
| hybrid-fast-auto | 140000000 | 22.370206 | 6258324.11 | True | ok |  | workers=32,chunk_terms=524288,gpu_stages=merge-and-final,sqrt=chunk-gpu-rsqrt-prototype,div=newton-chunk-gpu-seed-prototype,chunk_backend=auto,gpu_budget_gb=None |
| hybrid-high-digits-auto | 100000000 | 21.761306 | 4595312.47 | True | ok |  | workers=32,chunk_terms=262144,gpu_stages=merge-and-final,sqrt=chunk-gpu-rsqrt-prototype,div=newton-chunk-gpu-seed-prototype,chunk_backend=auto,gpu_budget_gb=12.0 |
| hybrid-high-digits-auto | 140000000 | 79.195793 | 1767770.67 | True | ok |  | workers=32,chunk_terms=524288,gpu_stages=merge-and-final,sqrt=chunk-gpu-rsqrt-prototype,div=newton-chunk-gpu-seed-prototype,chunk_backend=auto,gpu_budget_gb=12.0 |
