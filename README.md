# PTX Bench

**PTX Bench** is a benchmark suite measuring GPU kernel performance across three implementation tiers: naive CUDA C++, warp-optimized CUDA, and hand-written PTX (Parallel Thread Execution) inline assembly. It targets the critical operations in ternary neural networks: dot products, softmax, embeddings, hashing, vector search, and SVD.

## Why It Matters

The gap between naive and optimized GPU code can exceed 10×. For ternary networks — where weights are {-1, 0, +1} and multiply-accumulate reduces to sign-flip-and-add — the theoretical throughput is enormous, but realizing it requires careful kernel engineering. PTX Bench answers: how much performance is left on the table? By comparing three implementation depths for each operation, developers can quantify the optimization ceiling and decide where engineering effort is best spent. Benchmarks run on consumer GPUs (RTX 4050) to data-center GPUs (A100), making results relevant for both edge and cloud deployment decisions.

## How It Works

### Three-Tier Benchmarking

Each operation is implemented at three levels:

1. **Naive**: Straightforward translation of the math to CUDA. Each thread does the obvious computation. No memory coalescing, no shared memory, no warp-level primitives.

2. **Optimized**: Uses warp shuffles for reduction, `float4` vectorized loads, FMA (fused multiply-add) instructions, loop unrolling, and shared memory tiling. This is what a competent CUDA engineer would write.

3. **PTX Assembly**: Hand-written inline PTX using `asm!("...")`. Explicit register allocation, instruction scheduling, and warp-level primitives (`redux.sync.add`, `dp4a` for int8 dot products).

### Key Operations Benchmarked

| Kernel | Math | Ternary Optimization |
|--------|------|---------------------|
| `bench_dot` | y = Σ aᵢbᵢ | Replace multiply with conditional negate |
| `bench_softmax` | σᵢ = eˣⁱ / Σeˣʲ | Ternary-aware exponent approximation |
| `bench_embed` | Embedding lookup | 1-bit weights → 32× compression |
| `bench_hash` | BLAKE2b / FNV | Ternary hash for content addressing |
| `bench_search` | Cosine similarity top-K | Popcount for ternary dot product |
| `bench_svd` | Matrix decomposition | Ternary-structured decomposition |

### Performance Metrics

Each benchmark reports:
- **Throughput**: operations/second (GOP/s for arithmetic, GB/s for memory)
- **Latency**: μs per kernel invocation
- **Occupancy**: achieved % of theoretical warp occupancy
- **Memory bandwidth**: % of peak DRAM bandwidth utilized

### Roofline Analysis

Results are plotted against the roofline model:

```
Attained FLOPs/s = min(Peak FLOPs/s, Peak_BW × Operational_Intensity)
```

where Operational Intensity = FLOPs / bytes loaded. Kernels below the roofline are compute- or memory-bound; those far below have optimization opportunities.

## Quick Start

```bash
# Build (requires CUDA toolkit)
nvcc -O3 -arch=sm_89 src/bench_dot.cu -o bench_dot
nvcc -O3 -arch=sm_89 src/bench_softmax.cu -o bench_softmax
nvcc -O3 -arch=sm_89 src/bench_embed.cu -o bench_embed

# Run dot product benchmark
./bench_dot  # Outputs: naive vs optimized vs PTX timings

# Run all benchmarks
for bench in bench_*; do ./$bench; done
```

Example output:
```
Dot Product (n=10000, dim=384):
  Naive:      1.234 ms  |  3.1 TFLOP/s
  Optimized:  0.187 ms  | 20.5 TFLOP/s  (6.6× speedup)
  PTX:        0.142 ms  | 27.0 TFLOP/s  (8.7× speedup)
```

## API

| File | Kernel | Description |
|------|--------|-------------|
| `bench_dot.cu` | `dot_naive`, `dot_opt`, `dot_ptx` | Batched dot product throughput |
| `bench_softmax.cu` | `softmax_naive`, `softmax_warp` | Softmax over rows |
| `bench_embed.cu` | `embed_lookup`, `embed_batch` | Ternary embedding gather |
| `bench_hash.cu` | `hash_blake2b`, `hash_fnv` | Hash kernel throughput |
| `bench_search.cu` | `search_topk` | Top-K cosine similarity |
| `bench_svd.cu` | `svd_truncated` | Truncated SVD |

## Architecture Notes

PTX Bench measures the raw computational capacity (γ — generation throughput) of the SuperInstance GPU stack. The ternary optimization opportunities (replacing multiply with sign-flip) are the η (elimination) side: by removing floating-point multiplies, we eliminate 75% of the arithmetic cost. The benchmark data feeds directly into the γ + η = C equation by quantifying how much C (competence) each implementation tier achieves per watt-second. See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

1. NVIDIA. (2024). *CUDA C++ Programming Guide*, Version 12.x. — PTX ISA and warp primitives.
2. Williams, S., Waterman, A., & Patterson, D. (2009). "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *Communications of the ACM*, 52(4), 65–76.
3. Gregg, C., & Hazelwood, K. (2011). "Where is the data? Why you cannot debate CPU vs. GPU performance without the answer." *ISPASS*.

## License

MIT
