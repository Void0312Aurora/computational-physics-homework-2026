# CPU / Hybrid Extreme Exploration

| route | target_digits | seconds | digits_per_second | prefix_ok | status | stop_reason | notes |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| cpu-cpp | 20000000 | 9.916140 | 2016910.00 | True | ok |  | threads=44,chunk_terms=262144 |
| cpu-cpp | 50000000 | 28.883100 | 1731110.00 | True | ok |  | threads=44,chunk_terms=524288 |
| cpu-cpp | 100000000 | 65.292800 | 1531560.00 | True | ok |  | threads=44,chunk_terms=1048576 |
| cpu-python | 20000000 | 10.913369 | 1832614.74 | True | ok |  | workers=32,chunk_terms=262144 |
| cpu-python | 50000000 | 31.070347 | 1609251.42 | True | ok |  | workers=32,chunk_terms=524288 |
| cpu-python | 100000000 | 73.665812 | 1357481.81 | True | ok |  | workers=32,chunk_terms=524288 |
| hybrid-fast-auto | 20000000 | 3.152494 | 6344184.14 | True | ok |  | workers=32,chunk_terms=65536,gpu_stages=merge-and-final,sqrt=chunk-gpu-rsqrt-prototype,div=newton-chunk-gpu-seed-prototype,chunk_backend=auto |
| hybrid-fast-auto | 50000000 | 7.175674 | 6967986.33 | True | ok |  | workers=32,chunk_terms=131072,gpu_stages=merge-and-final,sqrt=chunk-gpu-rsqrt-prototype,div=newton-chunk-gpu-seed-prototype,chunk_backend=auto |
| hybrid-fast-auto | 100000000 | 14.454642 | 6918192.69 | True | ok |  | workers=32,chunk_terms=262144,gpu_stages=merge-and-final,sqrt=chunk-gpu-rsqrt-prototype,div=newton-chunk-gpu-seed-prototype,chunk_backend=auto |
