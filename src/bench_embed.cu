// bench_embed.cu — Embedding generation benchmark
// Three implementations: naive, optimized (shared memory + FMA), PTX inline assembly
// Simulates embedding generation as a matrix-vector multiply (lookup + projection)

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

// ---------- NAIVE EMBEDDING ----------
// Simple gather + linear projection
// Each thread handles one output dimension for one token
__global__ void embed_naive_kernel(const int* __restrict__ token_ids,
                                    const float* __restrict__ embed_table,
                                    float* __restrict__ output,
                                    int n_tokens, int embed_dim, int proj_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * proj_dim;
    if (tid >= total) return;

    int token = tid / proj_dim;
    int out_d = tid % proj_dim;

    int tok_id = token_ids[token];
    // Simple projection: output = embed_table[tok_id] (identity, no actual projection matrix)
    // For benchmarking, we do embed_table[tok_id * embed_dim + i] * proj[i * proj_dim + out_d]
    // Simplified: just copy embed + add a weighted sum
    float val = 0.0f;
    const float* emb = embed_table + (size_t)tok_id * embed_dim;
    for (int i = 0; i < embed_dim; i++) {
        val += emb[i] * 0.01f * (float)(out_d + 1);  // Fake projection weight
    }
    output[(size_t)token * proj_dim + out_d] = val;
}

// ---------- OPTIMIZED EMBEDDING ----------
// Shared memory for embedding vector, vectorized ops
__global__ void __launch_bounds__(256, 4)
embed_opt_kernel(const int* __restrict__ token_ids,
                 const float* __restrict__ embed_table,
                 float* __restrict__ output,
                 int n_tokens, int embed_dim, int proj_dim) {
    __shared__ float s_embed[256];  // Shared copy of current token's embedding

    int token = blockIdx.x;
    if (token >= n_tokens) return;

    // Cooperative load of embedding into shared memory
    int tid = threadIdx.x;
    for (int i = tid; i < embed_dim; i += blockDim.x) {
        int tok_id = token_ids[token];
        s_embed[i] = embed_table[(size_t)tok_id * embed_dim + i];
    }
    __syncthreads();

    // Each thread computes one output dimension
    for (int out_d = tid; out_d < proj_dim; out_d += blockDim.x) {
        float val = 0.0f;
        int i = 0;
        for (; i + 3 < embed_dim; i += 4) {
            float4 se = *reinterpret_cast<float4*>(s_embed + i);
            float w0 = 0.01f * (float)(out_d + 1);
            val = fmaf(se.x, w0, val);
            val = fmaf(se.y, w0, val);
            val = fmaf(se.z, w0, val);
            val = fmaf(se.w, w0, val);
        }
        for (; i < embed_dim; i++) {
            val = fmaf(s_embed[i], 0.01f * (float)(out_d + 1), val);
        }
        output[(size_t)token * proj_dim + out_d] = val;
    }
}

// ---------- PTX EMBEDDING ----------
// Inline PTX for loads and FMA
__global__ void __launch_bounds__(256, 4)
embed_ptx_kernel(const int* __restrict__ token_ids,
                 const float* __restrict__ embed_table,
                 float* __restrict__ output,
                 int n_tokens, int embed_dim, int proj_dim) {
    __shared__ float s_embed[256];

    int token = blockIdx.x;
    if (token >= n_tokens) return;

    int tid = threadIdx.x;
    int tok_id;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(tok_id) : "l"(token_ids + token));

    // Load embedding with PTX
    for (int i = tid; i < embed_dim; i += blockDim.x) {
        float val;
        asm volatile("ld.global.f32 %0, [%1];" : "=f"(val) : "l"(embed_table + (size_t)tok_id * embed_dim + i));
        s_embed[i] = val;
    }
    __syncthreads();

    for (int out_d = tid; out_d < proj_dim; out_d += blockDim.x) {
        float val = 0.0f;
        float weight;
        asm volatile("mul.f32 %0, 0f3c23d70a, %1;" : "=f"(weight) : "r"(out_d + 1));
        // 0f3c23d70a ≈ 0.01f

        for (int i = 0; i < embed_dim; i++) {
            float se = s_embed[i];
            asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(val) : "f"(se), "f"(weight));
        }
        float* out_ptr = output + (size_t)token * proj_dim + out_d;
        asm volatile("st.global.f32 [%0], %1;" :: "l"(out_ptr), "f"(val));
    }
}

// ---------- BENCHMARK RUNNER ----------

void run_benchmark(int n_tokens, int embed_dim, int proj_dim, int vocab_size, FILE* json_out) {
    size_t table_bytes = (size_t)vocab_size * embed_dim * sizeof(float);
    size_t out_bytes = (size_t)n_tokens * proj_dim * sizeof(float);

    int *h_ids = (int*)malloc(n_tokens * sizeof(int));
    float *h_table = (float*)malloc(table_bytes);
    for (int i = 0; i < n_tokens; i++) h_ids[i] = i % vocab_size;
    for (size_t i = 0; i < (size_t)vocab_size * embed_dim; i++) h_table[i] = 0.01f * (float)(i % 200);

    int *d_ids;
    float *d_table, *d_out;
    CUDA_CHECK(cudaMalloc(&d_ids, n_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_table, table_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMemcpy(d_ids, h_ids, n_tokens * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_table, h_table, table_bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks_flat = (n_tokens * proj_dim + threads - 1) / threads;
    int blocks_opt = n_tokens;
    int warmup = 10, runs = 50;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        embed_naive_kernel<<<blocks_flat, threads>>>(d_ids, d_table, d_out, n_tokens, embed_dim, proj_dim);
        embed_opt_kernel<<<blocks_opt, threads>>>(d_ids, d_table, d_out, n_tokens, embed_dim, proj_dim);
        embed_ptx_kernel<<<blocks_opt, threads>>>(d_ids, d_table, d_out, n_tokens, embed_dim, proj_dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float t1, t2, t3;
    {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < runs; i++)
            embed_naive_kernel<<<blocks_flat, threads>>>(d_ids, d_table, d_out, n_tokens, embed_dim, proj_dim);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&t1, start, stop));
        t1 /= runs;
    }
    {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < runs; i++)
            embed_opt_kernel<<<blocks_opt, threads>>>(d_ids, d_table, d_out, n_tokens, embed_dim, proj_dim);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&t2, start, stop));
        t2 /= runs;
    }
    {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < runs; i++)
            embed_ptx_kernel<<<blocks_opt, threads>>>(d_ids, d_table, d_out, n_tokens, embed_dim, proj_dim);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&t3, start, stop));
        t3 /= runs;
    }

    printf("| %6d tok × %3d→%3d | %8.3f ms | %8.3f ms | %8.3f ms | %5.2fx |\n",
           n_tokens, embed_dim, proj_dim, t1, t2, t3, t1 / t3);

    if (json_out) {
        fprintf(json_out, "%s{\"n_tokens\":%d,\"embed_dim\":%d,\"proj_dim\":%d,\"naive_ms\":%.4f,\"opt_ms\":%.4f,\"ptx_ms\":%.4f,\"speedup\":%.2f}",
                ftell(json_out) > 2 ? ",\n" : "[\n", n_tokens, embed_dim, proj_dim, t1, t2, t3, t1/t3);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_ids));
    CUDA_CHECK(cudaFree(d_table));
    CUDA_CHECK(cudaFree(d_out));
    free(h_ids);
    free(h_table);
}

int main() {
    printf("=== Embedding Generation Benchmark ===\n\n");

    FILE* json = fopen("results/bench_embed.json", "w");
    if (json) fprintf(json, "[\n");

    printf("| %22s |     Naive | Optimized |       PTX | Speedup |\n", "");
    printf("|------------------------|-----------|-----------|-----------|---------|\n");

    int vocab = 50000;

    run_benchmark(1000, 64, 64, vocab, json);
    run_benchmark(1000, 128, 64, vocab, json);
    run_benchmark(10000, 64, 64, vocab, json);
    run_benchmark(10000, 128, 64, vocab, json);
    run_benchmark(100000, 64, 64, vocab, json);
    run_benchmark(100000, 128, 64, vocab, json);

    if (json) {
        fprintf(json, "\n]\n");
        fclose(json);
        printf("\nResults saved to results/bench_embed.json\n");
    }

    return 0;
}
