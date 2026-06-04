// bench_dot.cu — Dot product throughput benchmark
// Three implementations: naive, optimized (warp shuffle), PTX inline assembly

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ---------- NAIVE DOT PRODUCT ----------
// Each thread computes one dot product sequentially
__global__ void dot_naive_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ results,
                                  int n_vectors, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vectors) return;

    float sum = 0.0f;
    const float* va = a + (size_t)idx * dim;
    const float* vb = b + (size_t)idx * dim;

    for (int i = 0; i < dim; i++) {
        sum += va[i] * vb[i];
    }
    results[idx] = sum;
}

// ---------- OPTIMIZED DOT PRODUCT ----------
// Warp shuffle reduction, FMA, vectorized loads (float4)
__global__ void __launch_bounds__(256, 4)
dot_opt_kernel(const float* __restrict__ a,
               const float* __restrict__ b,
               float* __restrict__ results,
               int n_vectors, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vectors) return;

    const float* va = a + (size_t)idx * dim;
    const float* vb = b + (size_t)idx * dim;

    float sum = 0.0f;

    // Vectorized load (float4 = 4 elements at once)
    int i = 0;
    for (; i + 3 < dim; i += 4) {
        float4 fa = *reinterpret_cast<const float4*>(va + i);
        float4 fb = *reinterpret_cast<const float4*>(vb + i);
        sum = fmaf(fa.x, fb.x, sum);
        sum = fmaf(fa.y, fb.y, sum);
        sum = fmaf(fa.z, fb.z, sum);
        sum = fmaf(fa.w, fb.w, sum);
    }
    // Tail
    for (; i < dim; i++) {
        sum = fmaf(va[i], vb[i], sum);
    }

    results[idx] = sum;
}

// ---------- PTX DOT PRODUCT ----------
// Inline PTX for FMA and warp shuffle reduction
// Processes dim/32 elements per lane, reduces across warp

__global__ void __launch_bounds__(256, 4)
dot_ptx_kernel(const float* __restrict__ a,
               const float* __restrict__ b,
               float* __restrict__ results,
               int n_vectors, int dim) {
    // Each warp processes one dot product (32 lanes)
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    if (warp_id >= n_vectors) return;

    const float* va = a + (size_t)warp_id * dim;
    const float* vb = b + (size_t)warp_id * dim;

    float sum = 0.0f;

    // Each lane processes every 32nd element
    for (int i = lane_id; i < dim; i += 32) {
        float av, bv;
        // Inline PTX for global loads + FMA
        asm volatile("ld.global.f32 %0, [%1];" : "=f"(av) : "l"(va + i));
        asm volatile("ld.global.f32 %0, [%1];" : "=f"(bv) : "l"(vb + i));
        asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(sum) : "f"(av), "f"(bv));
    }

    // Warp-level butterfly reduction via inline PTX
    float other;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %1, 16, 0xffffffff;" : "=f"(other) : "f"(sum));
    sum += other;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %1, 8, 0xffffffff;" : "=f"(other) : "f"(sum));
    sum += other;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %1, 4, 0xffffffff;" : "=f"(other) : "f"(sum));
    sum += other;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %1, 2, 0xffffffff;" : "=f"(other) : "f"(sum));
    sum += other;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %1, 1, 0xffffffff;" : "=f"(other) : "f"(sum));
    sum += other;

    if (lane_id == 0) {
        results[warp_id] = sum;
    }
}

// ---------- BENCHMARK RUNNER ----------

void run_benchmark(int n_vectors, int dim, FILE* json_out) {
    size_t vec_bytes = (size_t)n_vectors * dim * sizeof(float);
    size_t res_bytes = (size_t)n_vectors * sizeof(float);

    float *h_a = (float*)malloc(vec_bytes);
    float *h_b = (float*)malloc(vec_bytes);
    for (size_t i = 0; i < (size_t)n_vectors * dim; i++) {
        h_a[i] = 0.1f * (float)(i % 100);
        h_b[i] = 0.2f * (float)(i % 50);
    }

    float *d_a, *d_b, *d_res;
    CUDA_CHECK(cudaMalloc(&d_a, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_res, res_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, vec_bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks_naive = (n_vectors + threads - 1) / threads;
    int blocks_ptx = (n_vectors * 32 + threads - 1) / threads;
    int warmup = 10, runs = 50;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup all kernels
    for (int i = 0; i < warmup; i++) {
        dot_naive_kernel<<<blocks_naive, threads>>>(d_a, d_b, d_res, n_vectors, dim);
        dot_opt_kernel<<<blocks_naive, threads>>>(d_a, d_b, d_res, n_vectors, dim);
        dot_ptx_kernel<<<blocks_ptx, threads>>>(d_a, d_b, d_res, n_vectors, dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto bench_kernel = [&](auto kernel, int blks) -> float {
        float ms = 0;
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < runs; i++) kernel<<<blks, threads>>>(d_a, d_b, d_res, n_vectors, dim);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms / runs;
    };

    float naive_ms = bench_kernel(dot_naive_kernel<decltype(d_a), decltype(d_b), decltype(d_res), int, int>, blocks_naive);
    // Simplified: just call directly
    float naive_time, opt_time, ptx_time;

    // Naive
    {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < runs; i++)
            dot_naive_kernel<<<blocks_naive, threads>>>(d_a, d_b, d_res, n_vectors, dim);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&naive_time, start, stop));
        naive_time /= runs;
    }
    // Optimized
    {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < runs; i++)
            dot_opt_kernel<<<blocks_naive, threads>>>(d_a, d_b, d_res, n_vectors, dim);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&opt_time, start, stop));
        opt_time /= runs;
    }
    // PTX
    {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < runs; i++)
            dot_ptx_kernel<<<blocks_ptx, threads>>>(d_a, d_b, d_res, n_vectors, dim);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ptx_time, start, stop));
        ptx_time /= runs;
    }

    printf("| %6d × %4dd | %8.3f ms | %8.3f ms | %8.3f ms | %5.2fx |\n",
           n_vectors, dim, naive_time, opt_time, ptx_time, naive_time / ptx_time);

    if (json_out) {
        fprintf(json_out, "%s{\"n_vectors\":%d,\"dim\":%d,\"naive_ms\":%.4f,\"opt_ms\":%.4f,\"ptx_ms\":%.4f,\"speedup\":%.2f}",
                ftell(json_out) > 2 ? ",\n" : "[\n", n_vectors, dim, naive_time, opt_time, ptx_time, naive_time/ptx_time);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_res));
    free(h_a);
    free(h_b);
}

int main() {
    printf("=== Dot Product Benchmark ===\n\n");

    FILE* json = fopen("results/bench_dot.json", "w");
    if (json) fprintf(json, "[\n");

    int dims[] = {64, 128, 256, 512, 1024};
    for (int dim : dims) {
        printf("| %15s |     Naive | Optimized |       PTX | Speedup |\n", "");
        printf("|-----------------|-----------|-----------|-----------|---------|\n");
        int scales[] = {1000, 10000, 100000, 1000000};
        for (int n : scales) {
            run_benchmark(n, dim, json);
        }
        printf("\n");
    }

    if (json) {
        fprintf(json, "\n]\n");
        fclose(json);
        printf("Results saved to results/bench_dot.json\n");
    }

    return 0;
}
