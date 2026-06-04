// bench_search.cu — Vector search end-to-end benchmark
// Three implementations: naive brute-force, optimized (shared memory tiling), PTX inline assembly

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

// ---------- NAIVE VECTOR SEARCH ----------
// Each thread: compute distance of one query to one database vector, write (idx, dist)
// Then find top-K on host side (simplified: just find nearest)

__global__ void search_naive_kernel(const float* __restrict__ queries,
                                     const float* __restrict__ database,
                                     float* __restrict__ distances,
                                     int n_queries, int n_db, int dim) {
    int q = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (q >= n_queries || d >= n_db) return;

    const float* qvec = queries + (size_t)q * dim;
    const float* dvec = database + (size_t)d * dim;

    float dist_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = qvec[i] - dvec[i];
        dist_sq += diff * diff;
    }
    distances[(size_t)q * n_db + d] = dist_sq;
}

// ---------- OPTIMIZED VECTOR SEARCH ----------
// Shared memory tiling: load query into shared, compute against tile of DB

__global__ void __launch_bounds__(256, 2)
search_opt_kernel(const float* __restrict__ queries,
                  const float* __restrict__ database,
                  float* __restrict__ distances,
                  int n_queries, int n_db, int dim) {
    __shared__ float s_query[256];  // Max dim = 256 for shared mem

    int q = blockIdx.x;
    int tid = threadIdx.x;

    // Cooperative load of query vector into shared memory
    for (int i = tid; i < dim; i += blockDim.x) {
        s_query[i] = queries[(size_t)q * dim + i];
    }
    __syncthreads();

    // Each thread handles multiple DB vectors
    for (int d = tid; d < n_db; d += blockDim.x) {
        const float* dvec = database + (size_t)d * dim;
        float dist_sq = 0.0f;

        // Vectorized FMA
        int i = 0;
        for (; i + 3 < dim; i += 4) {
            float4 sv = *reinterpret_cast<const float4*>(s_query + i);
            float4 dv = *reinterpret_cast<const float4*>(dvec + i);
            float d0 = sv.x - dv.x; dist_sq = fmaf(d0, d0, dist_sq);
            float d1 = sv.y - dv.y; dist_sq = fmaf(d1, d1, dist_sq);
            float d2 = sv.z - dv.z; dist_sq = fmaf(d2, d2, dist_sq);
            float d3 = sv.w - dv.w; dist_sq = fmaf(d3, d3, dist_sq);
        }
        for (; i < dim; i++) {
            float diff = s_query[i] - dvec[i];
            dist_sq = fmaf(diff, diff, dist_sq);
        }
        distances[(size_t)q * n_db + d] = dist_sq;
    }
}

// ---------- PTX VECTOR SEARCH ----------
// Inline PTX for loads, FMA, and reduction

__global__ void __launch_bounds__(256, 2)
search_ptx_kernel(const float* __restrict__ queries,
                  const float* __restrict__ database,
                  float* __restrict__ distances,
                  int n_queries, int n_db, int dim) {
    __shared__ float s_query[256];

    int q = blockIdx.x;
    int tid = threadIdx.x;

    for (int i = tid; i < dim; i += blockDim.x) {
        s_query[i] = queries[(size_t)q * dim + i];
    }
    __syncthreads();

    for (int d = tid; d < n_db; d += blockDim.x) {
        const float* dvec = database + (size_t)d * dim;
        float dist_sq = 0.0f;

        for (int i = 0; i < dim; i++) {
            float sv = s_query[i];
            float dv, diff;
            asm volatile("ld.global.f32 %0, [%1];" : "=f"(dv) : "l"(dvec + i));
            asm volatile("sub.f32 %0, %1, %2;" : "=f"(diff) : "f"(sv), "f"(dv));
            asm volatile("fma.rn.f32 %0, %1, %1, %0;" : "+f"(dist_sq) : "f"(diff));
        }
        distances[(size_t)q * n_db + d] = dist_sq;
    }
}

// ---------- BENCHMARK RUNNER ----------

void run_benchmark(int n_queries, int n_db, int dim, FILE* json_out) {
    size_t q_bytes = (size_t)n_queries * dim * sizeof(float);
    size_t db_bytes = (size_t)n_db * dim * sizeof(float);
    size_t dist_bytes = (size_t)n_queries * n_db * sizeof(float);

    float *h_q = (float*)malloc(q_bytes);
    float *h_db = (float*)malloc(db_bytes);
    for (size_t i = 0; i < (size_t)n_queries * dim; i++) h_q[i] = 0.01f * (float)(i % 100);
    for (size_t i = 0; i < (size_t)n_db * dim; i++) h_db[i] = 0.01f * (float)(i % 150);

    float *d_q, *d_db, *d_dist;
    CUDA_CHECK(cudaMalloc(&d_q, q_bytes));
    CUDA_CHECK(cudaMalloc(&d_db, db_bytes));
    CUDA_CHECK(cudaMalloc(&d_dist, dist_bytes));
    CUDA_CHECK(cudaMemcpy(d_q, h_q, q_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_db, h_db, db_bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    dim3 naive_blocks(n_queries, (n_db + threads - 1) / threads);
    int opt_blocks = n_queries;  // 1 block per query
    int warmup = 5, runs = 20;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        search_naive_kernel<<<naive_blocks, threads>>>(d_q, d_db, d_dist, n_queries, n_db, dim);
        search_opt_kernel<<<opt_blocks, threads>>>(d_q, d_db, d_dist, n_queries, n_db, dim);
        search_ptx_kernel<<<opt_blocks, threads>>>(d_q, d_db, d_dist, n_queries, n_db, dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float t1, t2, t3;

    {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < runs; i++)
            search_naive_kernel<<<naive_blocks, threads>>>(d_q, d_db, d_dist, n_queries, n_db, dim);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&t1, start, stop));
        t1 /= runs;
    }
    {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < runs; i++)
            search_opt_kernel<<<opt_blocks, threads>>>(d_q, d_db, d_dist, n_queries, n_db, dim);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&t2, start, stop));
        t2 /= runs;
    }
    {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < runs; i++)
            search_ptx_kernel<<<opt_blocks, threads>>>(d_q, d_db, d_dist, n_queries, n_db, dim);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&t3, start, stop));
        t3 /= runs;
    }

    printf("| %5dQ × %5dD × %3dd | %8.3f ms | %8.3f ms | %8.3f ms | %5.2fx |\n",
           n_queries, n_db, dim, t1, t2, t3, t1 / t3);

    if (json_out) {
        fprintf(json_out, "%s{\"n_queries\":%d,\"n_db\":%d,\"dim\":%d,\"naive_ms\":%.4f,\"opt_ms\":%.4f,\"ptx_ms\":%.4f,\"speedup\":%.2f}",
                ftell(json_out) > 2 ? ",\n" : "[\n", n_queries, n_db, dim, t1, t2, t3, t1/t3);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_db));
    CUDA_CHECK(cudaFree(d_dist));
    free(h_q);
    free(h_db);
}

int main() {
    printf("=== Vector Search Benchmark ===\n\n");

    FILE* json = fopen("results/bench_search.json", "w");
    if (json) fprintf(json, "[\n");

    printf("| %25s |     Naive | Optimized |       PTX | Speedup |\n", "");
    printf("|---------------------------|-----------|-----------|-----------|---------|\n");

    // Different scale combinations
    run_benchmark(100, 10000, 64, json);
    run_benchmark(100, 10000, 128, json);
    run_benchmark(100, 100000, 64, json);
    run_benchmark(1000, 10000, 64, json);
    run_benchmark(1000, 100000, 64, json);
    run_benchmark(1000, 100000, 128, json);

    if (json) {
        fprintf(json, "\n]\n");
        fclose(json);
        printf("\nResults saved to results/bench_search.json\n");
    }

    return 0;
}
