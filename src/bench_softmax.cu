// bench_softmax.cu — Softmax throughput benchmark
// Three implementations: naive, optimized (warp primitives), PTX inline assembly

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

constexpr float NEG_INF = -1e30f;

// ---------- NAIVE SOFTMAX ----------
// Single-threaded per row, straightforward implementation
__global__ void softmax_naive_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int n_rows, int row_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows) return;

    const float* row = input + (size_t)idx * row_len;
    float* out = output + (size_t)idx * row_len;

    // Find max
    float max_val = NEG_INF;
    for (int i = 0; i < row_len; i++) {
        max_val = fmaxf(max_val, row[i]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < row_len; i++) {
        out[i] = expf(row[i] - max_val);
        sum += out[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < row_len; i++) {
        out[i] *= inv_sum;
    }
}

// ---------- OPTIMIZED SOFTMAX ----------
// One warp per row (for row_len ≤ 32), warp shuffle for max/reduce
__global__ void __launch_bounds__(256, 4)
softmax_opt_kernel(const float* __restrict__ input,
                   float* __restrict__ output,
                   int n_rows, int row_len) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    if (warp_id >= n_rows) return;

    const float* row = input + (size_t)warp_id * row_len;
    float* out = output + (size_t)warp_id * row_len;

    // Load lane value
    float val = (lane_id < row_len) ? row[lane_id] : NEG_INF;

    // Warp max reduction
    float max_val = val;
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, max_val, offset);
        max_val = fmaxf(max_val, other);
    }
    // Broadcast max to all lanes
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    // Compute exp(x - max)
    float exp_val = (lane_id < row_len) ? expf(val - max_val) : 0.0f;

    // Warp sum reduction
    float sum = exp_val;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    sum = __shfl_sync(0xffffffff, sum, 0);

    // Normalize and store
    if (lane_id < row_len) {
        out[lane_id] = exp_val / sum;
    }
}

// ---------- PTX SOFTMAX ----------
// Inline PTX for exp approximation and warp shuffle
__global__ void __launch_bounds__(256, 4)
softmax_ptx_kernel(const float* __restrict__ input,
                   float* __restrict__ output,
                   int n_rows, int row_len) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    if (warp_id >= n_rows) return;

    const float* row = input + (size_t)warp_id * row_len;
    float* out = output + (size_t)warp_id * row_len;

    // Load
    float val = (lane_id < row_len) ? row[lane_id] : NEG_INF;

    // PTX warp max via shfl.bfly
    float max_val = val;
    float other;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %1, 16, 0xffffffff;" : "=f"(other) : "f"(max_val));
    max_val = fmaxf(max_val, other);
    asm volatile("shfl.sync.bfly.b32 %0, %1, %1, 8, 0xffffffff;" : "=f"(other) : "f"(max_val));
    max_val = fmaxf(max_val, other);
    asm volatile("shfl.sync.bfly.b32 %0, %1, %1, 4, 0xffffffff;" : "=f"(other) : "f"(max_val));
    max_val = fmaxf(max_val, other);
    asm volatile("shfl.sync.bfly.b32 %0, %1, %1, 2, 0xffffffff;" : "=f"(other) : "f"(max_val));
    max_val = fmaxf(max_val, other);
    asm volatile("shfl.sync.bfly.b32 %0, %1, %1, 1, 0xffffffff;" : "=f"(other) : "f"(max_val));
    max_val = fmaxf(max_val, other);

    // Broadcast max from lane 0
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0xffffffff;" : "=f"(max_val) : "f"(max_val));

    // PTX exp approximation: ex2.approx(x / ln2)
    float shifted = val - max_val;
    float ln2_inv = 1.44269504089f;
    float exp_arg;
    asm volatile("mul.f32 %0, %1, %2;" : "=f"(exp_arg) : "f"(shifted), "f"(ln2_inv));
    float exp_val;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(exp_val) : "f"(exp_arg));
    if (lane_id >= row_len) exp_val = 0.0f;

    // Warp sum via shfl.bfly
    float sum = exp_val;
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
    // Broadcast sum
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0xffffffff;" : "=f"(sum) : "f"(sum));

    // Normalize
    if (lane_id < row_len) {
        float result;
        asm volatile("div.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(exp_val), "f"(sum));
        out[lane_id] = result;
    }
}

// ---------- BENCHMARK RUNNER ----------

void run_benchmark(int n_rows, int row_len, FILE* json_out) {
    size_t data_bytes = (size_t)n_rows * row_len * sizeof(float);

    float *h_in = (float*)malloc(data_bytes);
    float *h_out = (float*)malloc(data_bytes);
    for (size_t i = 0; i < (size_t)n_rows * row_len; i++) {
        h_in[i] = 0.01f * ((float)(i % 200) - 100.0f);
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, data_bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks_naive = (n_rows + threads - 1) / threads;
    int blocks_warp = (n_rows * 32 + threads - 1) / threads;
    int warmup = 10, runs = 50;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        softmax_naive_kernel<<<blocks_naive, threads>>>(d_in, d_out, n_rows, row_len);
        softmax_opt_kernel<<<blocks_warp, threads>>>(d_in, d_out, n_rows, row_len);
        softmax_ptx_kernel<<<blocks_warp, threads>>>(d_in, d_out, n_rows, row_len);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto time_kernel = [&](auto kernel, int blks) -> float {
        float ms;
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < runs; i++) kernel<<<blks, threads>>>(d_in, d_out, n_rows, row_len);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms / runs;
    };

    float naive_ms = time_kernel(softmax_naive_kernel<decltype(d_in), decltype(d_out), int, int>, blocks_naive);
    float opt_ms = time_kernel(softmax_opt_kernel<decltype(d_in), decltype(d_out), int, int>, blocks_warp);
    float ptx_ms = time_kernel(softmax_ptx_kernel<decltype(d_in), decltype(d_out), int, int>, blocks_warp);

    // Simplified direct calls
    float t1, t2, t3;
    {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < runs; i++)
            softmax_naive_kernel<<<blocks_naive, threads>>>(d_in, d_out, n_rows, row_len);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&t1, start, stop));
        t1 /= runs;
    }
    {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < runs; i++)
            softmax_opt_kernel<<<blocks_warp, threads>>>(d_in, d_out, n_rows, row_len);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&t2, start, stop));
        t2 /= runs;
    }
    {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < runs; i++)
            softmax_ptx_kernel<<<blocks_warp, threads>>>(d_in, d_out, n_rows, row_len);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&t3, start, stop));
        t3 /= runs;
    }

    printf("| %8d × %2d | %8.3f ms | %8.3f ms | %8.3f ms | %5.2fx |\n",
           n_rows, row_len, t1, t2, t3, t1 / t3);

    if (json_out) {
        fprintf(json_out, "%s{\"n_rows\":%d,\"row_len\":%d,\"naive_ms\":%.4f,\"opt_ms\":%.4f,\"ptx_ms\":%.4f,\"speedup\":%.2f}",
                ftell(json_out) > 2 ? ",\n" : "[\n", n_rows, row_len, t1, t2, t3, t1/t3);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
}

int main() {
    printf("=== Softmax Benchmark ===\n\n");

    FILE* json = fopen("results/bench_softmax.json", "w");
    if (json) fprintf(json, "[\n");

    printf("| %15s |     Naive | Optimized |       PTX | Speedup |\n", "");
    printf("|-----------------|-----------|-----------|-----------|---------|\n");

    int row_lens[] = {8, 16, 32};
    int scales[] = {1000, 10000, 100000, 1000000};

    for (int rl : row_lens) {
        for (int n : scales) {
            run_benchmark(n, rl, json);
        }
    }

    if (json) {
        fprintf(json, "\n]\n");
        fclose(json);
        printf("\nResults saved to results/bench_softmax.json\n");
    }

    return 0;
}
