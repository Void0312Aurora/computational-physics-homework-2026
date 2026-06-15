PYTHON := python3
VENV_PYTHON := /home/void0312/Cpy/.venv/bin/python
TARGET := scripts/solution.py
GPU_TARGET := scripts/gpu_ultra_fractal.py
TRITON_BENCH := scripts/triton_newton_bench.py
DOUBLEFLOAT_PROTO := scripts/doublefloat_newton_prototype.py
TRITON_DOUBLEFLOAT_PROTO := scripts/triton_doublefloat_prototype.py
SPARSE_VALIDATE := scripts/sparse_reference_validation.py
FRONTIER_TARGET := scripts/problem1_frontier_curves.py
CPU_PROFILE_TARGET := scripts/problem1_cpu_profile.py
CPU_ULTRA_SRC := scripts/problem1_cpu_ultra.c
CPU_ULTRA_BIN := problem1_cpu_ultra
CPU_ULTRA_BENCH := scripts/problem1_cpu_ultra_bench.py
CPU_ULTRA_RENDER := scripts/problem1_cpu_ultra_render.py
DOC_RENDER := docs/answer/render_docs.py
GPU_ARGS ?=
TRITON_ARGS ?=
DOUBLEFLOAT_ARGS ?=
TRITON_DOUBLEFLOAT_ARGS ?=
SPARSE_VALIDATE_ARGS ?=
FRONTIER_ARGS ?=
CPU_PROFILE_ARGS ?=
CPU_ULTRA_ARGS ?=
CPU_ULTRA_BENCH_ARGS ?=
CPU_ULTRA_RENDER_ARGS ?=

.PHONY: run docs problem1 problem2 problem3 problem4 problem5 gpu-ultra triton-bench doublefloat-proto triton-doublefloat-proto sparse-validate frontier-curves cpu-profile cpu-ultra cpu-ultra-bench cpu-ultra-render clean

run:
	mkdir -p result
	$(PYTHON) $(TARGET)

docs:
	mkdir -p result
	$(PYTHON) $(TARGET)
	$(VENV_PYTHON) $(DOC_RENDER)

problem1:
	$(PYTHON) scripts/problem1.py

problem2:
	$(PYTHON) scripts/problem2.py

problem3:
	$(PYTHON) scripts/problem3.py

problem4:
	$(PYTHON) scripts/problem4.py

problem5:
	$(PYTHON) scripts/problem5.py

gpu-ultra:
	mkdir -p result
	$(PYTHON) $(GPU_TARGET) $(GPU_ARGS)

triton-bench:
	mkdir -p result
	$(PYTHON) $(TRITON_BENCH) $(TRITON_ARGS)

doublefloat-proto:
	mkdir -p result
	$(PYTHON) $(DOUBLEFLOAT_PROTO) $(DOUBLEFLOAT_ARGS)

triton-doublefloat-proto:
	mkdir -p result
	$(PYTHON) $(TRITON_DOUBLEFLOAT_PROTO) $(TRITON_DOUBLEFLOAT_ARGS)

sparse-validate:
	mkdir -p result
	$(PYTHON) $(SPARSE_VALIDATE) $(SPARSE_VALIDATE_ARGS)

frontier-curves:
	mkdir -p result
	$(PYTHON) $(FRONTIER_TARGET) $(FRONTIER_ARGS)

cpu-profile:
	mkdir -p result
	$(PYTHON) $(CPU_PROFILE_TARGET) $(CPU_PROFILE_ARGS)

$(CPU_ULTRA_BIN): $(CPU_ULTRA_SRC)
	gcc -O3 -march=native -ffast-math -fopenmp -o $(CPU_ULTRA_BIN) $(CPU_ULTRA_SRC) -lm

cpu-ultra: $(CPU_ULTRA_BIN)
	mkdir -p result
	./$(CPU_ULTRA_BIN) $(CPU_ULTRA_ARGS)

cpu-ultra-bench: $(CPU_ULTRA_BIN)
	mkdir -p result
	$(PYTHON) $(CPU_ULTRA_BENCH) $(CPU_ULTRA_BENCH_ARGS)

cpu-ultra-render: $(CPU_ULTRA_BIN)
	mkdir -p result
	$(PYTHON) $(CPU_ULTRA_RENDER) $(CPU_ULTRA_RENDER_ARGS)

clean:
	rm -rf result
	rm -f docs/answer/answer.docx docs/answer/answer.pdf
	rm -f $(CPU_ULTRA_BIN)
