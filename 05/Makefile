PYTHON ?= python
CXX ?= g++
CXXFLAGS ?= -O3 -march=native -flto -fopenmp -std=c++20
PROJECT2_PI_DIR := project2_pi
PROJECT2_PI_BIN_DIR := $(PROJECT2_PI_DIR)/bin

GMPY2_INCLUDE := $(shell $(PYTHON) -c "from pathlib import Path; import gmpy2; print(Path(gmpy2.__file__).resolve().parent)")
GMPY2_LIBDIR := $(shell $(PYTHON) -c "from pathlib import Path; import gmpy2; print(Path(gmpy2.__file__).resolve().parent.parent / 'gmpy2.libs')")
GMPY2_GMP_SONAME := $(shell basename $(firstword $(wildcard $(GMPY2_LIBDIR)/libgmp*.so*)))

.PHONY: run docs problem1 problem2 problem3 problem4_project2 benchmark_pi benchmark_extremes benchmark_cpu_tasktree benchmark_cpu_parallel_modes benchmark_cpu_representation cpp_backend cpp_levelpool_backend nextgen_cpu_backend nextgen_cpu_smoke benchmark_nextgen_cpu benchmark_nextgen_cpu_extended benchmark_nextgen_cpu_tree mpn_benchmark fft_gpu_benchmark gpu_pi_hybrid gpu_pi_hybrid_legacy gpu_pi_full_cuda gpu_native_rns gpu_throughput_archive

run:
	$(PYTHON) scripts/solution.py

docs: run
	$(PYTHON) docs/answer/render_docs.py

problem1:
	$(PYTHON) scripts/problem1.py

problem2:
	$(PYTHON) scripts/problem2.py

problem3:
	$(PYTHON) scripts/problem3.py

problem4_project2:
	$(PYTHON) scripts/problem4_project2.py

benchmark_pi:
	$(PYTHON) -m project2_pi.benchmark_matrix

benchmark_extremes:
	$(PYTHON) -m project2_pi.benchmark_extremes

benchmark_cpu_tasktree:
	$(PYTHON) -m project2_pi.benchmark_cpu_tasktree

benchmark_cpu_parallel_modes:
	$(PYTHON) -m project2_pi.benchmark_cpu_tasktree

benchmark_cpu_representation:
	$(PYTHON) -m project2_pi.benchmark_cpu_representation

cpp_backend: $(PROJECT2_PI_BIN_DIR)/project2_gmp_backend
cpp_levelpool_backend: $(PROJECT2_PI_BIN_DIR)/project2_gmp_levelpool_backend
nextgen_cpu_backend: $(PROJECT2_PI_BIN_DIR)/project2_nextgen_cpu_smoke

nextgen_cpu_smoke: $(PROJECT2_PI_BIN_DIR)/project2_nextgen_cpu_smoke
	$(PROJECT2_PI_BIN_DIR)/project2_nextgen_cpu_smoke | tee result/project2_nextgen_cpu_smoke.log

benchmark_nextgen_cpu: $(PROJECT2_PI_BIN_DIR)/project2_nextgen_cpu_benchmark
	$(PROJECT2_PI_BIN_DIR)/project2_nextgen_cpu_benchmark --csv result/project2_nextgen_cpu_multiply_benchmark.csv | tee result/project2_nextgen_cpu_multiply_benchmark.log

benchmark_nextgen_cpu_extended: $(PROJECT2_PI_BIN_DIR)/project2_nextgen_cpu_benchmark
	$(PROJECT2_PI_BIN_DIR)/project2_nextgen_cpu_benchmark --sizes-blocks 256,512,1024,2048,4096 --csv result/project2_nextgen_cpu_multiply_benchmark_extended.csv | tee result/project2_nextgen_cpu_multiply_benchmark_extended.log

benchmark_nextgen_cpu_tree: $(PROJECT2_PI_BIN_DIR)/project2_nextgen_cpu_tree_benchmark
	$(PROJECT2_PI_BIN_DIR)/project2_nextgen_cpu_tree_benchmark --csv result/project2_nextgen_cpu_tree_benchmark.csv | tee result/project2_nextgen_cpu_tree_benchmark.log

mpn_benchmark: $(PROJECT2_PI_BIN_DIR)/project2_mpn_benchmark

fft_gpu_benchmark:
	$(PYTHON) -m project2_pi.fft_gpu_benchmark

gpu_pi_hybrid:
	$(PYTHON) -m project2_pi.gpu_pi_hybrid --profile fast-auto

gpu_pi_hybrid_legacy:
	$(PYTHON) -m project2_pi.gpu_pi_hybrid --profile legacy-default

gpu_pi_full_cuda:
	@printf '%s\n' 'Archived prototype route. See HW/05/project2_pi/ARCHIVED_ROUTES.md.'
	$(PYTHON) -m project2_pi.gpu_pi_full_cuda

gpu_native_rns:
	@printf '%s\n' 'Archived prototype route. See HW/05/project2_gpu_native_rns/ARCHIVED.md.'
	$(MAKE) -C project2_gpu_native_rns run

gpu_throughput_archive:
	@printf '%s\n' 'Archived prototype route. See HW/05/project2_gpu_throughput_mainline/ARCHIVED.md.'
	$(MAKE) -C project2_gpu_throughput_mainline piendtoend2500

$(PROJECT2_PI_BIN_DIR):
	mkdir -p $@

$(PROJECT2_PI_BIN_DIR)/project2_gmp_backend: $(PROJECT2_PI_DIR)/gmp_backend.cpp | $(PROJECT2_PI_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(GMPY2_INCLUDE) $< -L$(GMPY2_LIBDIR) -Wl,-rpath,$(GMPY2_LIBDIR) -l:$(GMPY2_GMP_SONAME) -o $@

$(PROJECT2_PI_BIN_DIR)/project2_gmp_levelpool_backend: $(PROJECT2_PI_DIR)/gmp_backend_levelpool.cpp | $(PROJECT2_PI_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(GMPY2_INCLUDE) $< -L$(GMPY2_LIBDIR) -Wl,-rpath,$(GMPY2_LIBDIR) -l:$(GMPY2_GMP_SONAME) -o $@

$(PROJECT2_PI_BIN_DIR)/project2_nextgen_cpu_smoke: $(PROJECT2_PI_DIR)/nextgen_cpu/main.cpp $(PROJECT2_PI_DIR)/nextgen_cpu/types.hpp $(PROJECT2_PI_DIR)/nextgen_cpu/page_arena.hpp $(PROJECT2_PI_DIR)/nextgen_cpu/canonical_ops.hpp $(PROJECT2_PI_DIR)/nextgen_cpu/base_convert.hpp $(PROJECT2_PI_DIR)/nextgen_cpu/rns_ntt.hpp $(PROJECT2_PI_DIR)/nextgen_cpu/product_tree.hpp | $(PROJECT2_PI_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(PROJECT2_PI_DIR)/nextgen_cpu $(PROJECT2_PI_DIR)/nextgen_cpu/main.cpp -o $@

$(PROJECT2_PI_BIN_DIR)/project2_nextgen_cpu_benchmark: $(PROJECT2_PI_DIR)/nextgen_cpu/benchmark.cpp $(PROJECT2_PI_DIR)/nextgen_cpu/types.hpp $(PROJECT2_PI_DIR)/nextgen_cpu/page_arena.hpp $(PROJECT2_PI_DIR)/nextgen_cpu/canonical_ops.hpp $(PROJECT2_PI_DIR)/nextgen_cpu/base_convert.hpp $(PROJECT2_PI_DIR)/nextgen_cpu/rns_ntt.hpp | $(PROJECT2_PI_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(PROJECT2_PI_DIR)/nextgen_cpu $(PROJECT2_PI_DIR)/nextgen_cpu/benchmark.cpp -o $@

$(PROJECT2_PI_BIN_DIR)/project2_nextgen_cpu_tree_benchmark: $(PROJECT2_PI_DIR)/nextgen_cpu/tree_benchmark.cpp $(PROJECT2_PI_DIR)/nextgen_cpu/product_tree.hpp $(PROJECT2_PI_DIR)/nextgen_cpu/types.hpp $(PROJECT2_PI_DIR)/nextgen_cpu/page_arena.hpp $(PROJECT2_PI_DIR)/nextgen_cpu/canonical_ops.hpp $(PROJECT2_PI_DIR)/nextgen_cpu/base_convert.hpp $(PROJECT2_PI_DIR)/nextgen_cpu/rns_ntt.hpp | $(PROJECT2_PI_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(PROJECT2_PI_DIR)/nextgen_cpu $(PROJECT2_PI_DIR)/nextgen_cpu/tree_benchmark.cpp -o $@

$(PROJECT2_PI_BIN_DIR)/project2_mpn_benchmark: $(PROJECT2_PI_DIR)/mpn_benchmark.cpp | $(PROJECT2_PI_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(GMPY2_INCLUDE) $< -L$(GMPY2_LIBDIR) -Wl,-rpath,$(GMPY2_LIBDIR) -l:$(GMPY2_GMP_SONAME) -o $@
