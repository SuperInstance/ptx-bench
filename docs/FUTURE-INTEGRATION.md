# Future Integration: ptx-bench

## Current State
A PTX micro-benchmark suite measuring GPU performance at the instruction level: BLAKE2b hashing, dot products (64d–1024d), warp-level softmax, and end-to-end vector search. Targeted at RTX 4050 (Ada Lovelace, sm_89). Quantifies the gap between naive, optimized, and hand-written PTX implementations.

## Integration Opportunities

### With ternary-cell GPU simulation
ternary-cell's CellGrid is embarrassingly parallel — 1M cells × 6-phase tick. The PTX benchmarks provide exact performance numbers for each phase: hash benchmark → cell ID hashing for neighbor lookup, dot product → surprise computation (prediction vs perception), softmax → energy redistribution during GC, vector search → nearest-neighbor signaling. On RTX 4050: 3,072 CUDA cores × 240 MHz is sufficient for 1M cells at 100Hz with headroom.

### With forgemaster
The Forgemaster uses PTX benchmarks to validate that its GPU simulation kernels meet performance targets. Before deploying a new kernel to the fleet, the Forgemaster runs it through ptx-bench's methodology: naive → optimized → PTX, measuring each step. If the PTX version doesn't hit target throughput, the kernel isn't deployed.

### With tile-cuda
tile-cuda's kernels (batch_hash, batch_embed, batch_cosine_search, batch_evolve) should be validated against ptx-bench's methodology. The benchmark suite provides the baseline; tile-cuda provides the application kernels.

## Dormant Ideas Now Unlockable
The benchmarks were reference-only with no application target. Now the ternary-cell GPU simulation provides the concrete use case, and the Forgemaster provides the deployment pipeline. Every benchmark directly answers "can we simulate X cells at Y Hz on hardware Z?"

## Potential in Mature Systems
ptx-bench becomes the fleet's GPU performance oracle. When a new GPU joins the fleet (Jetson Orin, RTX 5090, whatever's next), ptx-bench runs automatically and produces a capability profile. The Forgemaster uses this profile to decide how many cells that GPU can handle and which kernel configurations to use.

## Cross-Pollination Ideas
- **cudaclaw-1**: cudaclaw's kernel dispatch should be benchmarked via ptx-bench methodology
- **git-cuda-agent**: Template should include ptx-bench for auto-profiling on new hardware
- **agentic-compiler**: PTX benchmark results feed the compiler's optimization decisions

## Dependencies for Next Steps
- Benchmark ternary-cell tick phases at PTX level
- Automate benchmark runs on fleet hardware via Forgemaster
- Produce per-GPU capability profiles for room sizing
