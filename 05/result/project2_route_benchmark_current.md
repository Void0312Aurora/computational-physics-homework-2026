# Project2 Pi Benchmark Matrix

| route | group | target_digits | seconds | digits_per_second | prefix_ok | status | notes |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| python_gmp_bridge | full_scale | 10000000 | 4.754297 | 2103360.18 | True | ok | workers=32,chunk_terms=131072 |
| cpp_gmp_openmp | full_scale | 10000000 | 3.826070 | 2613650.00 | True | ok | threads=44,chunk_terms=131072,leaf_terms=8,task_terms=131072,parallel_mode=chunked |
| cpp_gmp_levelpool | full_scale | 10000000 | 3.822220 | 2616280.00 | True | ok | threads=44,chunk_terms=131072,leaf_terms=8,representation=levelpool |
| cpp_gmp_openmp_frontier | full_scale | 10000000 | 4.058650 | 2463880.00 | True | ok | threads=44,chunk_terms=131072,leaf_terms=8,task_terms=131072,parallel_mode=frontier |
| gpu_hybrid_legacy_default | full_scale | 10000000 | 4.897437 | 2041884.26 | True | ok | workers=32,chunk_terms=32768,gpu_stages=final-only,sqrt=mpz-isqrt,div=mpz-div,chunk_backend=python,gpu_budget_gb=None |
| gpu_hybrid_merge_fast_python | full_scale | 10000000 | 1.960125 | 5101714.87 | True | ok | workers=32,chunk_terms=32768,gpu_stages=merge-and-final,sqrt=chunk-gpu-rsqrt-prototype,div=auto,chunk_backend=python,gpu_budget_gb=None |
| gpu_hybrid_merge_fast_auto | full_scale | 10000000 | 1.991292 | 5021864.16 | True | ok | workers=32,chunk_terms=32768,gpu_stages=merge-and-final,sqrt=chunk-gpu-rsqrt-prototype,div=auto,chunk_backend=auto,gpu_budget_gb=None |
| full_cuda_pipeline | prototype_full_pi | 100000 | 0.222649 | 449138.09 | True | ok | workers=32,chunk_terms=8192,gpu_calls=79 |
| gpu_native_rns_end_to_end | prototype_full_pi | 870 | 0.141455 | 6150.36 | True | ok | native-rns benchmark ceiling is still sub-1k digits |
| gpu_throughput_mainline_end_to_end | prototype_full_pi | 2500 | 0.036482 | 68527.10 | True | ok | throughput-mainline is frozen and only validated on low-thousands digits |
