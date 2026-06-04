# PTX Micro-Benchmark Suite Makefile
# Targets RTX 4050 (sm_89), falls back to sm_75

CUDA_PATH ?= /usr/local/cuda
NVCC      := $(CUDA_PATH)/bin/nvcc
ARCH      := -gencode arch=compute_75,code=sm_75 -gencode arch=compute_89,code=sm_89
CXXFLAGS  := -O3 -std=c++17 --expt-relaxed-constexpr
PTXFLAGS  := -Xptxas -v  # Show register usage

BENCHMARKS := bench_hash bench_dot bench_softmax bench_search bench_embed bench_svd
BINS       := $(addprefix build/,$(BENCHMARKS))

.PHONY: all bench clean analyze

all: $(BINS)

build/%: src/%.cu
	@mkdir -p build
	$(NVCC) $(ARCH) $(CXXFLAGS) $(PTXFLAGS) -o $@ $<

bench: all
	@mkdir -p results
	@for bin in $(BINS); do \
		echo "=== Running $$bin ==="; \
		./build/$$bin 2>&1 | tee results/$${bin}.json; \
		echo ""; \
	done

analyze: bench
	cd analysis && python3 analyze.py

clean:
	rm -rf build results/*.json
