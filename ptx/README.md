# PTX Optimization Notes

## Register Allocation

Use `.reg .f32 %f<N>` to pre-allocate register banks. This prevents the compiler from spilling to local memory.

```ptx
.reg .f32 %f<64>;   // 64 f32 registers — plenty for 256-dim dot product
.reg .u32 %r<16>;   // Index and counter registers
```

### .maxnreg Directive

Control register pressure for occupancy:
- `__launch_bounds__(256, 2)` → max 32 regs/thread for 2 blocks/SM
- More regs = fewer warps = lower latency hiding
- RTX 4050: 65536 regs/SM, so 256 threads × 32 regs = 2 blocks/SM

## Warp Shuffle (shfl.sync)

The key PTX primitive for warp-level programming:

```ptx
// Butterfly reduction (all-to-all pattern)
shfl.sync.bfly.b32 %dest, %src, %src, delta, 0xffffffff;
```

Patterns:
- **bfly** (butterfly): XOR-based, great for FFT/reductions
- **up**: Upward shift, useful for scan operations
- **down**: Downward shift
- **idx**: Direct lane index, broadcast/gather

Full mask `0xffffffff` ensures all 32 lanes participate.

## Async Copy (cp.async)

Overlap compute and memory transfers on Ada Lovelace:

```ptx
// Async copy from global to shared memory
cp.async.ca.shared.global [shared_ptr], [global_ptr], 16;
cp.async.commit_group;
cp.async.wait_group 0;
```

Pipeline depth: typically 2-3 stages for BLAKE2b, 4-6 for matrix ops.

## Tensor Cores (wmma.mmasync)

For matrix operations (dot products ≥128d, embedding matmuls):

```ptx
// 16×16×16 FP16 matrix multiply-accumulate
wmma.mmasync.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32
    {%d0,...,%d7}, {%a0,...,%a7}, {%b0,...,%b7}, {%c0,...,%c7};
```

Useful when tile-compiler operations map to small matmuls.

## Instruction Scheduling

Ada Lovelace scheduling tips:
1. **Hide latency with independent instructions**: FP32 ADD has 4-cycle latency, but throughput is 1/cycle. Interleave independent chains.
2. **Avoid bank conflicts in shared memory**: Pad arrays to 33 columns for float.
3. **Use predicated execution** (`@%p`) instead of branches for short conditionals.
4. **FMA over MUL+ADD**: `fma.rn.f32 %d, %a, %b, %c` is one instruction, two ops.

## BLAKE2b Specifics

The G function has a critical path of:
- 4 × add.u64 (chain dependencies)
- 4 × xor.b64
- 4 × ror64 (2 shifts + or each)

Total: ~20 instructions on critical path. With proper scheduling, we can overlap
message loads with computation of the previous round.

## Benchmark Tips

1. **Warm up**: Run 10 iterations before timing (GPU frequency ramp-up)
2. **cudaEventRecord**: Use GPU-side timers, not CPU-side
3. **Multiple runs**: Report median, not mean (outliers from OS jitter)
4. **Occupancy**: Query with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
5. **Bandwidth**: Compare achieved vs theoretical (256 GB/s for RTX 4050)
