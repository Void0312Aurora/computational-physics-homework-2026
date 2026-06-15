# Project 2 CPU Task-Tree Summary

| route | mode | digits | repeats | avg_seconds | best_seconds | avg_digits_per_second | best_digits_per_second | threads | chunk_terms | leaf_terms | task_terms | prefix_ok | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| cpp_gmp_openmp | chunked | 10000000 | 2 | 4.744045 | 4.723260 | 2107945.00 | 2117180.00 | 44 | 131072 | 8 | 131072 | True | ok |
  samples_seconds=4.723260;4.764830; samples_digits_per_second=2117180.00;2098710.00
| cpp_gmp_openmp_tasks | tasks | 10000000 | 2 | 4.714915 | 4.696020 | 2120960.00 | 2129460.00 | 44 | 131072 | 8 | 131072 | True | ok |
  samples_seconds=4.733810;4.696020; samples_digits_per_second=2112460.00;2129460.00
| cpp_gmp_openmp | chunked | 50000000 | 4 | 24.763475 | 22.645300 | 2031050.00 | 2207970.00 | 44 | 131072 | 8 | 131072 | True | ok |
  samples_seconds=26.231300;27.039200;23.138100;22.645300; samples_digits_per_second=1906120.00;1849170.00;2160940.00;2207970.00
| cpp_gmp_openmp_tasks | tasks | 50000000 | 4 | 23.622125 | 22.491200 | 2124332.50 | 2223090.00 | 44 | 131072 | 8 | 131072 | True | ok |
  samples_seconds=26.100500;22.491200;22.608800;23.288000; samples_digits_per_second=1915680.00;2223090.00;2211530.00;2147030.00
| cpp_gmp_openmp | chunked | 100000000 | 2 | 50.624150 | 50.610700 | 1975345.00 | 1975870.00 | 44 | 131072 | 8 | 131072 | True | ok |
  samples_seconds=50.637600;50.610700; samples_digits_per_second=1974820.00;1975870.00
| cpp_gmp_openmp_tasks | tasks | 100000000 | 2 | 49.825550 | 49.531400 | 2007070.00 | 2018920.00 | 44 | 131072 | 8 | 131072 | True | ok |
  samples_seconds=49.531400;50.119700; samples_digits_per_second=2018920.00;1995220.00
