# PTX Micro-Benchmark Suite

Measure GPU performance at the PTX instruction level for operations that matter to **lever-runner** and **tile-compiler**.

## Why PTX?

CUDA C is productive but hides the silicon. Hand-written PTX assembly gives direct control over:
- Register allocation and pressure
- Warp-level primitives (shuffle, reduce)
- Async memory copy pipelines
- Instruction scheduling and latency hiding

This suite quantifies the gap between naive, optimized, and PTX-level implementations.

## Target Hardware

**RTX 4050 (Ada Lovelace, sm_89):**
- 24 SMs × 128 CUDA cores = 3,072 total cores
- 6 GB GDDR6, 256-bit bus, ~256 GB/s bandwidth
- 48 KB shared memory per SM
- 16 MB L2 cache

## Benchmarks

| Benchmark | Operation | Relevance |
|-----------|-----------|-----------|
| `bench_hash` | BLAKE2b throughput | lever-runner hashing |
| `bench_dot` | Dot product (64d–1024d) | tile-compiler similarity |
| `bench_softmax` | Softmax (warp-level) | tile-compiler attention |
| `bench_search` | Vector search (end-to-end) | lever-runner retrieval |
| `bench_embed` | Embedding generation | tile-compiler encoding |
| `bench_svd` | Power iteration SVD | tile-compiler decomposition |

## Three Implementations Per Benchmark

1. **Naive CUDA C** — straightforward, readable
2. **Optimized CUDA C** — shared memory, warp primitives, launch tuning
3. **Hand-written PTX** — direct instruction control

## Metrics

- **Throughput** (ops/sec)
- **Latency** (µs per operation)
- **Occupancy** (% SM capacity)
- **Memory bandwidth** achieved

## Build & Run

```bash
make all          # Build all benchmarks
make bench        # Run all benchmarks
make analyze      # Generate comparison tables
make clean
```

Requires CUDA Toolkit ≥ 11.0, sm_89 capable GPU (falls back to sm_75 for older GPUs).

## Results Format

Results are written to `results/` as JSON, one file per benchmark. The `analysis/analyze.py` script parses all results and generates comparison tables.

## Project Structure

```
ptx-bench/
├── Makefile
├── src/
│   ├── bench_hash.cu
│   ├── bench_dot.cu
│   ├── bench_softmax.cu
│   ├── bench_search.cu
│   ├── bench_embed.cu
│   └── bench_svd.cu
├── ptx/
│   ├── blake2b.ptx
│   ├── warp_dot.ptx
│   ├── warp_softmax.ptx
│   ├── warp_reduce.ptx
│   └── README.md
├── results/
├── analysis/
│   └── analyze.py
└── README.md
```
