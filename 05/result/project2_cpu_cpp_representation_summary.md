# Project 2 CPU Representation Refactor Summary

| route | digits | repeats | avg_seconds | best_seconds | avg_digits_per_second | best_digits_per_second | threads | chunk_terms | leaf_terms | task_terms | prefix_ok | status | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| cpp_gmp_openmp | 1000000 | 1 | 0.400567 | 0.400567 | 2496460.00 | 2496460.00 | 44 | 131072 | 8 | 131072 | True | ok | threads=44,chunk_terms=131072,leaf_terms=8,task_terms=131072,parallel_mode=chunked |
  samples_seconds=0.400567; samples_digits_per_second=2496460.00
| cpp_gmp_openmp_tasks | 1000000 | 1 | 0.393262 | 0.393262 | 2542840.00 | 2542840.00 | 44 | 131072 | 8 | 131072 | True | ok | threads=44,chunk_terms=131072,leaf_terms=8,task_terms=131072,parallel_mode=tasks |
  samples_seconds=0.393262; samples_digits_per_second=2542840.00
| cpp_gmp_levelpool | 1000000 | 1 | 0.402067 | 0.402067 | 2487150.00 | 2487150.00 | 44 | 131072 | 8 | 131072 | True | ok | threads=44,chunk_terms=131072,leaf_terms=8,representation=levelpool |
  samples_seconds=0.402067; samples_digits_per_second=2487150.00
| cpp_gmp_openmp | 10000000 | 1 | 3.882740 | 3.882740 | 2575500.00 | 2575500.00 | 44 | 131072 | 8 | 131072 | True | ok | threads=44,chunk_terms=131072,leaf_terms=8,task_terms=131072,parallel_mode=chunked |
  samples_seconds=3.882740; samples_digits_per_second=2575500.00
| cpp_gmp_openmp_tasks | 10000000 | 1 | 4.011880 | 4.011880 | 2492600.00 | 2492600.00 | 44 | 131072 | 8 | 131072 | True | ok | threads=44,chunk_terms=131072,leaf_terms=8,task_terms=131072,parallel_mode=tasks |
  samples_seconds=4.011880; samples_digits_per_second=2492600.00
| cpp_gmp_levelpool | 10000000 | 1 | 4.043400 | 4.043400 | 2473170.00 | 2473170.00 | 44 | 131072 | 8 | 131072 | True | ok | threads=44,chunk_terms=131072,leaf_terms=8,representation=levelpool |
  samples_seconds=4.043400; samples_digits_per_second=2473170.00
| cpp_gmp_openmp | 50000000 | 2 | 21.855500 | 21.675400 | 2287910.00 | 2306760.00 | 44 | 131072 | 8 | 131072 | True | ok | threads=44,chunk_terms=131072,leaf_terms=8,task_terms=131072,parallel_mode=chunked |
  samples_seconds=21.675400;22.035600; samples_digits_per_second=2306760.00;2269060.00
| cpp_gmp_openmp_tasks | 50000000 | 2 | 22.029400 | 21.594400 | 2270580.00 | 2315420.00 | 44 | 131072 | 8 | 131072 | True | ok | threads=44,chunk_terms=131072,leaf_terms=8,task_terms=131072,parallel_mode=tasks |
  samples_seconds=22.464400;21.594400; samples_digits_per_second=2225740.00;2315420.00
| cpp_gmp_levelpool | 50000000 | 2 | 21.653100 | 21.575900 | 2309170.00 | 2317400.00 | 44 | 131072 | 8 | 131072 | True | ok | threads=44,chunk_terms=131072,leaf_terms=8,representation=levelpool |
  samples_seconds=21.575900;21.730300; samples_digits_per_second=2317400.00;2300940.00
| cpp_gmp_openmp | 100000000 | 2 | 49.357100 | 48.132500 | 2027300.00 | 2077600.00 | 44 | 131072 | 8 | 131072 | True | ok | threads=44,chunk_terms=131072,leaf_terms=8,task_terms=131072,parallel_mode=chunked |
  samples_seconds=50.581700;48.132500; samples_digits_per_second=1977000.00;2077600.00
| cpp_gmp_openmp_tasks | 100000000 | 2 | 49.245750 | 47.874900 | 2032210.00 | 2088780.00 | 44 | 131072 | 8 | 131072 | True | ok | threads=44,chunk_terms=131072,leaf_terms=8,task_terms=131072,parallel_mode=tasks |
  samples_seconds=50.616600;47.874900; samples_digits_per_second=1975640.00;2088780.00
| cpp_gmp_levelpool | 100000000 | 2 | 51.637600 | 50.976900 | 1936890.00 | 1961670.00 | 44 | 131072 | 8 | 131072 | True | ok | threads=44,chunk_terms=131072,leaf_terms=8,representation=levelpool |
  samples_seconds=50.976900;52.298300; samples_digits_per_second=1961670.00;1912110.00
