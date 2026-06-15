# CPU / Hybrid Extreme Exploration

| route | target_digits | seconds | digits_per_second | prefix_ok | status | stop_reason | notes |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| hybrid-fast-auto | 200000000 | 0.000000 | 0.00 | False | failed | exception | CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 23.55 GiB of which 1.65 GiB is free. Process 5906 has 293.38 MiB memory in use. Process 343080 has 9.54 MiB memory in use. Including non-PyTorch memory, this process has 21.22 GiB memory in use. Of the allocated memory 20.04 GiB is allocated by PyTorch, and 903.34 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) |
