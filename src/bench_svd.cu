// bench_svd.cu — Power iteration SVD benchmark
// Three implementations: naive, optimized (cublas), PTX inline assembly for hot loops

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

// ---------- NAIVE POWER ITERATION SVD ----------
// Simple matrix-vector multiply for power iteration
// A^T * A * v → dominant eigenvector of A^T*A = right singular vector

__global__ void matvec_naive_kernel(const float* __restrict__ A,
                                     const float* __restrict__ v,
                                     float* __restrict__ out,
                                     int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    float sum = 0.0f;
    for (int c = 0; c < cols; c++) {
        sum += A[(size_t)row * cols + c] * v[c];
    }
    out[row] = sum;
}

__global__ void normalize_kernel(float* __restrict__ v, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += v[i] * v[i];
    }
    float inv_norm = 1.0f / sqrtf(sum);
    for (int i = 0; i < n; i++) {
        v[i] *= inv_norm;
    }
}

// ---------- OPTIMIZED POWER ITERATION ----------
// Shared memory tiling for matvec

__global__ void __launch_bounds__(256, 2)
matvec_opt_kernel(const float* __restrict__ A,
                  const float* __restrict__ v,
                  float* __restrict__ out,
                  int rows, int cols) {
    __shared__ float s_v[256];

    int tid = threadIdx.x;
    int row = blockIdx.x;

    // Load v into shared memory cooperatively
    for (int i = tid; i < cols; i += blockDim.x) {
        s_v[i] = v[i];
    }
    __syncthreads();

    // Each thread handles one row (or multiple if needed)
    for (int r = row; r < rows; r += gridDim.x) {
        float sum = 0.0f;
        const float* a_row = A + (size_t)r * cols;

        int c = 0;
        for (; c + 3 < cols; c += 4) {
            float4 av = *reinterpret_cast<const float4*>(a_row + c);
            float4 sv = *reinterpret_cast<float4*>(s_v + c);
            sum = fmaf(av.x, sv.x, sum);
            sum = fmaf(av.y, sv.y, sum);
            sum = fmaf(av.z, sv.z, sum);
            sum = fmaf(av.w, sv.w, sum);
        }
        for (; c < cols; c++) {
            sum = fmaf(a_row[c], s_v[c], sum);
        }
        out[r] = sum;
    }
}

// ---------- PTX POWER ITERATION ----------
// Inline PTX for the critical dot product in matvec

__global__ void __launch_bounds__(256, 2)
matvec_ptx_kernel(const float* __restrict__ A,
                  const float* __restrict__ v,
                  float* __restrict__ out,
                  int rows, int cols) {
    __shared__ float s_v[256];

    int tid = threadIdx.x;

    // Load v into shared with PTX
    for (int i = tid; i < cols; i += blockDim.x) {
        float val;
        asm volatile("ld.global.f32 %0, [%1];" : "=f"(val) : "l"(v + i));
        s_v[i] = val;
    }
    __syncthreads();

    for (int r = blockIdx.x; r < rows; r += gridDim.x) {
        const float* a_row = A + (size_t)r * cols;
        float sum = 0.0f;

        for (int c = 0; c < cols; c++) {
            float av, sv = s_v[c];
            asm volatile("ld.global.f32 %0, [%1];" : "=f"(av) : "l"(a_row + c));
            asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(sum) : "f"(av), "f"(sv));
        }
        out[r] = sum;
    }
}

// ---------- BENCHMARK RUNNER ----------

void run_svd_benchmark(int rows, int cols, int iterations, int version, FILE* json_out) {
    // version: 0=naive, 1=opt, 2=ptx
    size_t a_bytes = (size_t)rows * cols * sizeof(float);
    size_t v_bytes = (size_t)cols * sizeof(float);
    size_t o_bytes = (size_t)rows * sizeof(float);

    float *h_A = (float*)malloc(a_bytes);
    float *h_v = (float*)malloc(v_bytes);
    for (size_t i = 0; i < (size_t)rows * cols; i++) h_A[i] = 0.01f * (float)(i % 100);
    for (size_t i = 0; i < (size_t)cols; i++) h_v[i] = 1.0f / sqrtf((float)cols);

    float *d_A, *d_v, *d_tmp, *d_out;
    CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_v, v_bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp, (size_t)std::max(rows, cols) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, o_bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, v_bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks_mv = std::min(rows, 65536);
    int warmup = 3, runs = 10;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int w = 0; w < warmup; w++) {
        for (int it = 0; it < iterations; it++) {
            // v = A^T * (A * v) simplified: just A*v for benchmark
            if (version == 0) matvec_naive_kernel<<<(rows+threads-1)/threads, threads>>>(d_A, d_v, d_out, rows, cols);
            else if (version == 1) matvec_opt_kernel<<<blocks_mv, threads>>>(d_A, d_v, d_out, rows, cols);
            else matvec_ptx_kernel<<<blocks_mv, threads>>>(d_A, d_v, d_out, rows, cols);
            // Swap v and out (simplified)
            float* tmp = d_v; d_v = d_out; d_out = tmp;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float elapsed;
    CUDA_CHECK(cudaEventRecord(start));
    for (int w = 0; w < runs; w++) {
        for (int it = 0; it < iterations; it++) {
            if (version == 0) matvec_naive_kernel<<<(rows+threads-1)/threads, threads>>>(d_A, d_v, d_out, rows, cols);
            else if (version == 1) matvec_opt_kernel<<<blocks_mv, threads>>>(d_A, d_v, d_out, rows, cols);
            else matvec_ptx_kernel<<<blocks_mv, threads>>>(d_A, d_v, d_out, rows, cols);
            float* tmp = d_v; d_v = d_out; d_out = tmp;
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= runs;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_out));
    free(h_A);
    free(h_v);

    // Store in json_out (called per-version, caller aggregates)
    if (json_out) {
        fprintf(json_out, "%.4f", elapsed);
    }
}

void run_comparison(int rows, int cols, int iterations, FILE* json_out) {
    float naive_ms, opt_ms, ptx_ms;

    FILE* f1 = tmpfile(); run_svd_benchmark(rows, cols, iterations, 0, f1);
    rewind(f1); fscanf(f1, "%f", &naive_ms); fclose(f1);

    FILE* f2 = tmpfile(); run_svd_benchmark(rows, cols, iterations, 1, f2);
    rewind(f2); fscanf(f2, "%f", &opt_ms); fclose(f2);

    FILE* f3 = tmpfile(); run_svd_benchmark(rows, cols, iterations, 2, f3);
    rewind(f3); fscanf(f3, "%f", &ptx_ms); fclose(f3);

    printf("| %5d × %4d (%2d it) | %8.3f ms | %8.3f ms | %8.3f ms | %5.2fx |\n",
           rows, cols, iterations, naive_ms, opt_ms, ptx_ms, naive_ms / ptx_ms);

    if (json_out) {
        fprintf(json_out, "%s{\"rows\":%d,\"cols\":%d,\"iterations\":%d,\"naive_ms\":%.4f,\"opt_ms\":%.4f,\"ptx_ms\":%.4f,\"speedup\":%.2f}",
                ftell(json_out) > 2 ? ",\n" : "[\n", rows, cols, iterations, naive_ms, opt_ms, ptx_ms, naive_ms/ptx_ms);
    }
}

int main() {
    printf("=== Power Iteration SVD Benchmark ===\n\n");

    FILE* json = fopen("results/bench_svd.json", "w");
    if (json) fprintf(json, "[\n");

    printf("| %25s |     Naive | Optimized |       PTX | Speedup |\n", "");
    printf("|---------------------------|-----------|-----------|-----------|---------|\n");

    run_comparison(256, 64, 20, json);
    run_comparison(512, 64, 20, json);
    run_comparison(1024, 64, 20, json);
    run_comparison(1024, 128, 20, json);
    run_comparison(4096, 64, 20, json);
    run_comparison(4096, 128, 20, json);

    if (json) {
        fprintf(json, "\n]\n");
        fclose(json);
        printf("\nResults saved to results/bench_svd.json\n");
    }

    return 0;
}
