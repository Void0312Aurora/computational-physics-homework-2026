.PHONY: all clean

CC ?= gcc
CFLAGS ?= -O3 -march=native -std=c11 -Wall -Wextra -pedantic
LDFLAGS ?= -lm
OPENMP_CFLAGS ?= -fopenmp
OPENMP_LDFLAGS ?= -fopenmp
AVX2_CFLAGS ?= -mavx2

all: direct_tensor_midpoint_orthant_mixed_q26_tile_batch_avx2 direct_tensor_midpoint_ablation_kernel

direct_tensor_midpoint_orthant_mixed_q26_tile_batch_avx2: direct_tensor_midpoint_orthant_mixed_q26_tile_batch_avx2.c
	$(CC) $(CFLAGS) $(OPENMP_CFLAGS) $(AVX2_CFLAGS) $< -o $@ $(LDFLAGS) $(OPENMP_LDFLAGS)

direct_tensor_midpoint_ablation_kernel: direct_tensor_midpoint_ablation_kernel.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f direct_tensor_midpoint_orthant_mixed_q26_tile_batch_avx2 direct_tensor_midpoint_ablation_kernel
