NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++20

TARGET := bin/project2_gpu_native_rns_smoke
SRC := src/main.cu src/rns_runtime.cu
INCLUDES := -Iinclude

.PHONY: all build run run-noverify run-global-noverify clean

all: build

build: $(TARGET)

$(TARGET): $(SRC) include/project2_gpu_native_rns/rns_runtime.cuh
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SRC) -o $@

run: $(TARGET)
	./$(TARGET)

run-noverify: $(TARGET)
	./$(TARGET) --value-count 256 --slot-count 32 --iterations 10 --input-bits 28 --no-verify --output-tag noverify_256x32

run-global-noverify: $(TARGET)
	./$(TARGET) --value-count 64 --slot-count 4096 --iterations 5 --input-bits 24 --no-verify --output-tag noverify_64x4096

clean:
	rm -f $(TARGET)
