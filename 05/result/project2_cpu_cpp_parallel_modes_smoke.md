# Project 2 CPU Parallel Modes Summary

| route | mode | digits | repeats | avg_seconds | median_seconds | iqr_seconds | best_seconds | avg_digits_per_second | median_digits_per_second | iqr_digits_per_second | best_digits_per_second | threads | chunk_terms | leaf_terms | task_terms | prefix_ok | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| cpp_gmp_openmp | chunked | 100000 | 5 | 0.028208 | 0.031236 | 0.006668 | 0.021617 | 3632376.00 | 3201450.00 | 846260.00 | 4625910.00 | 4 | 8192 | 8 | 8192 | True | ok |
  samples_seconds=0.031601;0.031236;0.031652;0.024933;0.021617; samples_digits_per_second=3164450.00;3201450.00;3159360.00;4010710.00;4625910.00; run_index_samples=1;3;5;7;9; execution_order=1:sample1;3:sample2;5:sample3;7:sample4;9:sample5; order_strategy=interleave; loadavg=3.42,2.99,1.75->3.31,2.97,1.75
| cpp_gmp_openmp_tasks | tasks | 100000 | 5 | 0.027939 | 0.027095 | 0.005258 | 0.025111 | 3609382.00 | 3690750.00 | 669750.00 | 3982300.00 | 4 | 8192 | 8 | 8192 | True | ok |
  samples_seconds=0.027095;0.025514;0.030773;0.025111;0.031202; samples_digits_per_second=3690750.00;3919350.00;3249600.00;3982300.00;3204910.00; run_index_samples=2;4;6;8;10; execution_order=2:sample1;4:sample2;6:sample3;8:sample4;10:sample5; order_strategy=interleave; loadavg=3.42,2.99,1.75->3.31,2.97,1.75
