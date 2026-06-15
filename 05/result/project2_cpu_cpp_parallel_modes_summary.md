# Project 2 CPU Parallel Modes Summary

| route | mode | digits | repeats | avg_seconds | best_seconds | avg_digits_per_second | best_digits_per_second | threads | chunk_terms | leaf_terms | task_terms | prefix_ok | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| cpp_gmp_openmp | chunked | 10000000 | 2 | 4.236705 | 3.993300 | 2368145.00 | 2504200.00 | 44 | 131072 | 8 | 131072 | True | ok |
  samples_seconds=4.480110;3.993300; samples_digits_per_second=2232090.00;2504200.00
| cpp_gmp_openmp_tasks | tasks | 10000000 | 2 | 4.222150 | 4.124040 | 2369740.00 | 2424800.00 | 44 | 131072 | 8 | 131072 | True | ok |
  samples_seconds=4.320260;4.124040; samples_digits_per_second=2314680.00;2424800.00
| cpp_gmp_openmp_frontier | frontier | 10000000 | 2 | 4.394320 | 4.390320 | 2275670.00 | 2277740.00 | 44 | 131072 | 8 | 131072 | True | ok |
  samples_seconds=4.398320;4.390320; samples_digits_per_second=2273600.00;2277740.00
| cpp_gmp_openmp | chunked | 50000000 | 2 | 22.996300 | 22.970000 | 2174270.00 | 2176760.00 | 44 | 131072 | 8 | 131072 | True | ok |
  samples_seconds=22.970000;23.022600; samples_digits_per_second=2176760.00;2171780.00
| cpp_gmp_openmp_tasks | tasks | 50000000 | 2 | 23.649050 | 23.271500 | 2114790.00 | 2148550.00 | 44 | 131072 | 8 | 131072 | True | ok |
  samples_seconds=24.026600;23.271500; samples_digits_per_second=2081030.00;2148550.00
| cpp_gmp_openmp_frontier | frontier | 50000000 | 2 | 38.056800 | 36.900000 | 1315040.00 | 1355010.00 | 44 | 131072 | 8 | 131072 | True | ok |
  samples_seconds=36.900000;39.213600; samples_digits_per_second=1355010.00;1275070.00
| cpp_gmp_openmp | chunked | 100000000 | 2 | 60.214950 | 59.392200 | 1661025.00 | 1683720.00 | 44 | 131072 | 8 | 131072 | True | ok |
  samples_seconds=59.392200;61.037700; samples_digits_per_second=1683720.00;1638330.00
| cpp_gmp_openmp_tasks | tasks | 100000000 | 2 | 58.586950 | 58.331000 | 1706895.00 | 1714350.00 | 44 | 131072 | 8 | 131072 | True | ok |
  samples_seconds=58.331000;58.842900; samples_digits_per_second=1714350.00;1699440.00
| cpp_gmp_openmp_frontier | frontier | 100000000 | 2 | 137.560500 | 134.733000 | 727260.50 | 742207.00 | 44 | 131072 | 8 | 131072 | True | ok |
  samples_seconds=140.388000;134.733000; samples_digits_per_second=712314.00;742207.00
