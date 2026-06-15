NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++20
LDFLAGS ?= -lcufft

TARGET := bin/project2_gpu_throughput_mainline
SRC := src/main.cu src/runtime.cu
INCLUDES := -Iinclude

.PHONY: all build run plan pointwise pairpack persistentreduce fftbackbone fftbridge groupedfftbridge groupedlevelplanner groupedlevelplannermultilimb groupedlevelplannerexact32 groupedlevelplannerexactmoduli groupedlevelplannerexactmodulipfactor groupedlevelplannerexactmodulipq groupedlevelplannerexactmodulipqt piendtoend piendtoendscaled piendtoend1000 piendtoend2000 piendtoend2500 groupedlevelplannersplit63 clean

all: build

build: $(TARGET)

$(TARGET): $(SRC) include/project2_gpu_throughput_mainline/runtime.cuh
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SRC) -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

plan: $(TARGET)
	./$(TARGET) --print-plan

pointwise: $(TARGET)
	./$(TARGET) --pointwise-add-smoke --batch-count 4096 --slot-count 4096 --modulus-count 8 --iterations 20

pairpack: $(TARGET)
	./$(TARGET) --pair-pack-smoke --merge-count 2048 --slot-count 4096 --modulus-count 8 --iterations 20

persistentreduce: $(TARGET)
	./$(TARGET) --persistent-level-reduce-smoke --node-count 4096 --slot-count 4096 --modulus-count 8 --iterations 20

fftbackbone: $(TARGET)
	./$(TARGET) --batched-fft-smoke --fft-batch-count 2048 --fft-length 4096 --iterations 20

fftbridge: $(TARGET)
	./$(TARGET) --residue-fft-bridge-smoke --node-count 4096 --slot-count 4096 --modulus-count 8 --bridge-modulus-index 0 --packing-mask 63 --iterations 20

groupedfftbridge: $(TARGET)
	./$(TARGET) --grouped-residue-fft-bridge-smoke --node-count 4096 --slot-count 4096 --modulus-count 8 --packing-mask 63 --iterations 20

groupedlevelplanner: $(TARGET)
	./$(TARGET) --grouped-level-planner-smoke --node-count 4096 --slot-count 4096 --modulus-count 8 --packing-mask 15 --iterations 20

groupedlevelplannermultilimb: $(TARGET)
	./$(TARGET) --grouped-level-planner-multilimb-smoke --node-count 4096 --slot-count 4096 --modulus-count 8 --packing-mask 4095 --iterations 20

groupedlevelplannerexact32: $(TARGET)
	./$(TARGET) --grouped-level-planner-multilimb-smoke --node-count 4096 --slot-count 4096 --modulus-count 8 --packing-mask 4294967295 --iterations 3

groupedlevelplannerexactmoduli: $(TARGET)
	./$(TARGET) --grouped-level-planner-exact-moduli-smoke --node-count 4096 --slot-count 4096 --modulus-count 8 --packing-mask 4294967295 --iterations 3

groupedlevelplannerexactmodulipfactor: $(TARGET)
	./$(TARGET) --grouped-level-planner-exact-moduli-pfactor-smoke --node-count 4096 --slot-count 4096 --modulus-count 8 --packing-mask 4294967295 --iterations 3

groupedlevelplannerexactmodulipq: $(TARGET)
	./$(TARGET) --grouped-level-planner-exact-moduli-pq-smoke --node-count 4096 --slot-count 4096 --modulus-count 8 --packing-mask 4294967295 --iterations 3

groupedlevelplannerexactmodulipqt: $(TARGET)
	./$(TARGET) --grouped-level-planner-exact-moduli-pqt-smoke --node-count 4096 --slot-count 4096 --modulus-count 8 --packing-mask 4294967295 --iterations 3

piendtoend: $(TARGET)
	./$(TARGET) --pi-end-to-end-smoke --term-count 16 --slot-count 256 --modulus-count 10 --target-digits 80 --warmup 1 --iterations 3

piendtoendscaled: $(TARGET)
	./$(TARGET) --pi-end-to-end-smoke --term-count 128 --slot-count 1024 --modulus-count 10 --target-digits 80 --warmup 1 --iterations 3

piendtoend1000: $(TARGET)
	./$(TARGET) --pi-end-to-end-smoke --term-count 128 --slot-count 1024 --modulus-count 10 --target-digits 1000 --warmup 1 --iterations 3

piendtoend2000: $(TARGET)
	./$(TARGET) --pi-end-to-end-smoke --term-count 256 --slot-count 2048 --modulus-count 10 --target-digits 2000 --warmup 1 --iterations 3

piendtoend2500: $(TARGET)
	./$(TARGET) --pi-end-to-end-smoke --term-count 256 --slot-count 2048 --modulus-count 10 --target-digits 2500 --warmup 1 --iterations 3

groupedlevelplannersplit63: $(TARGET)
	./$(TARGET) --grouped-level-planner-split-mask63-smoke --node-count 4096 --slot-count 4096 --modulus-count 8 --packing-mask 63 --iterations 20

clean:
	rm -f $(TARGET)
