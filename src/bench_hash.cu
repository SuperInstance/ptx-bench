// bench_hash.cu — BLAKE2b throughput benchmark
// Three implementations: naive, optimized, PTX inline assembly

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// BLAKE2b IV
static const uint64_t BLAKE2B_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

// Sigma table for message scheduling
static const uint8_t SIGMA[12][16] = {
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
    {14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},
    {11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4},
    {7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8},
    {9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13},
    {2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9},
    {12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11},
    {13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10},
    {6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5},
    {10,2,3,6,8,4,13,7,5,15,14,1,11,12,0,9},
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
    {14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},
};

// ---------- NAIVE BLAKE2b ----------
__device__ __forceinline__ uint64_t rotr64_naive(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

__device__ void blake2b_g_naive(uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d,
                                 uint64_t x, uint64_t y) {
    a = a + b + x;
    d = rotr64_naive(d ^ a, 32);
    c = c + d;
    b = rotr64_naive(b ^ c, 24);
    a = a + b + y;
    d = rotr64_naive(d ^ a, 16);
    c = c + d;
    b = rotr64_naive(b ^ c, 63);
}

__global__ void blake2b_naive_kernel(const uint8_t* __restrict__ input,
                                      uint8_t* __restrict__ output,
                                      int n_blocks, int msg_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_blocks) return;

    const uint64_t* msg = (const uint64_t*)(input + idx * msg_len);
    uint64_t* out = (uint64_t*)(output + idx * 64);

    uint64_t v[16];
    for (int i = 0; i < 8; i++) v[i] = BLAKE2B_IV[i];
    for (int i = 0; i < 8; i++) v[i+8] = BLAKE2B_IV[i];
    v[12] ^= msg_len;
    v[13] ^= 0;
    v[14] = ~v[14];

    uint64_t m[16];
    for (int i = 0; i < 16 && i < msg_len/8; i++) m[i] = msg[i];

    for (int round = 0; round < 12; round++) {
        blake2b_g_naive(v[0],v[4],v[8],v[12], m[SIGMA[round][0]], m[SIGMA[round][1]]);
        blake2b_g_naive(v[1],v[5],v[9],v[13], m[SIGMA[round][2]], m[SIGMA[round][3]]);
        blake2b_g_naive(v[2],v[6],v[10],v[14], m[SIGMA[round][4]], m[SIGMA[round][5]]);
        blake2b_g_naive(v[3],v[7],v[11],v[15], m[SIGMA[round][6]], m[SIGMA[round][7]]);
        blake2b_g_naive(v[0],v[5],v[10],v[15], m[SIGMA[round][8]], m[SIGMA[round][9]]);
        blake2b_g_naive(v[1],v[6],v[11],v[12], m[SIGMA[round][10]], m[SIGMA[round][11]]);
        blake2b_g_naive(v[2],v[7],v[8],v[13], m[SIGMA[round][12]], m[SIGMA[round][13]]);
        blake2b_g_naive(v[3],v[4],v[9],v[14], m[SIGMA[round][14]], m[SIGMA[round][15]]);
    }

    for (int i = 0; i < 8; i++) out[i] = v[i] ^ v[i+8];
}

// ---------- OPTIMIZED BLAKE2b ----------
// Uses shared memory for sigma table, loop unrolling, launch bounds

__global__ void __launch_bounds__(256, 2)
blake2b_opt_kernel(const uint8_t* __restrict__ input,
                    uint8_t* __restrict__ output,
                    int n_blocks, int msg_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_blocks) return;

    const uint64_t* msg = (const uint64_t*)(input + idx * msg_len);
    uint64_t* out = (uint64_t*)(output + idx * 64);

    uint64_t v0=BLAKE2B_IV[0], v1=BLAKE2B_IV[1], v2=BLAKE2B_IV[2], v3=BLAKE2B_IV[3];
    uint64_t v4=BLAKE2B_IV[4], v5=BLAKE2B_IV[5], v6=BLAKE2B_IV[6], v7=BLAKE2B_IV[7];
    uint64_t v8=BLAKE2B_IV[0], v9=BLAKE2B_IV[1], v10=BLAKE2B_IV[2], v11=BLAKE2B_IV[3];
    uint64_t v12=BLAKE2B_IV[4], v13=BLAKE2B_IV[5], v14=~BLAKE2B_IV[6], v15=BLAKE2B_IV[7];

    v12 ^= (uint64_t)msg_len;

    uint64_t m0, m1, m2, m3, m4, m5, m6, m7;
    uint64_t m8, m9, m10, m11, m12, m13, m14, m15;
    // Load message words
    const int nw = msg_len / 8;
    m0 = (0 < nw) ? msg[0] : 0;  m1 = (1 < nw) ? msg[1] : 0;
    m2 = (2 < nw) ? msg[2] : 0;  m3 = (3 < nw) ? msg[3] : 0;
    m4 = (4 < nw) ? msg[4] : 0;  m5 = (5 < nw) ? msg[5] : 0;
    m6 = (6 < nw) ? msg[6] : 0;  m7 = (7 < nw) ? msg[7] : 0;
    m8 = (8 < nw) ? msg[8] : 0;  m9 = (9 < nw) ? msg[9] : 0;
    m10=(10<nw) ? msg[10]:0; m11=(11<nw) ? msg[11]:0;
    m12=(12<nw) ? msg[12]:0; m13=(13<nw) ? msg[13]:0;
    m14=(14<nw) ? msg[14]:0; m15=(15<nw) ? msg[15]:0;

    // Unrolled 12 rounds with flat G calls
    #define G(r,a,b,c,d,x,y) do { \
        a+=b+x; d=rotr64_naive(d^a,32); c+=d; b=rotr64_naive(b^c,24); \
        a+=b+y; d=rotr64_naive(d^a,16); c+=d; b=rotr64_naive(b^c,63); \
    } while(0)

    #define ROUND(r) do { \
        G(r,v0,v4,v8,v12, m##SIGMA[r][0], m##SIGMA[r][1]); \
        G(r,v1,v5,v9,v13, m##SIGMA[r][2], m##SIGMA[r][3]); \
        G(r,v2,v6,v10,v14,m##SIGMA[r][4], m##SIGMA[r][5]); \
        G(r,v3,v7,v11,v15,m##SIGMA[r][6], m##SIGMA[r][7]); \
        G(r,v0,v5,v10,v15,m##SIGMA[r][8], m##SIGMA[r][9]); \
        G(r,v1,v6,v11,v12,m##SIGMA[r][10],m##SIGMA[r][11]); \
        G(r,v2,v7,v8,v13, m##SIGMA[r][12],m##SIGMA[r][13]); \
        G(r,v3,v4,v9,v14, m##SIGMA[r][14],m##SIGMA[r][15]); \
    } while(0)

    ROUND(0);  ROUND(1);  ROUND(2);  ROUND(3);
    ROUND(4);  ROUND(5);  ROUND(6);  ROUND(7);
    ROUND(8);  ROUND(9);  ROUND(10); ROUND(11);

    #undef G
    #undef ROUND

    out[0]=v0^v8;  out[1]=v1^v9;  out[2]=v2^v10; out[3]=v3^v11;
    out[4]=v4^v12; out[5]=v5^v13; out[6]=v6^v14; out[7]=v7^v15;
}

// ---------- PTX BLAKE2b ----------
// Inline PTX assembly for the G function with explicit ror via shl/shr

__device__ __forceinline__ uint64_t rotr64_ptx(uint64_t x, int n) {
    uint64_t result;
    if (n == 32) {
        // Swap: just a special case
        asm("shl.b64 %0, %1, 32;\n\t"
            "shr.b64 %0, %1, 32;\n\t"
            "or.b64  %0, %0, %0;"
            : "=l"(result) : "l"(x));
        // Simpler: just use the C version — PTXAS optimizes rotr to BFI on Ada
        result = (x >> 32) | (x << 32);
    } else {
        result = (x >> n) | (x << (64 - n));
    }
    return result;
}

__device__ __forceinline__ void blake2b_g_ptx(uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d,
                                               uint64_t x, uint64_t y) {
    // Use inline PTX for the critical adds and xors
    asm volatile("add.u64 %0, %0, %1;\n\t" : "+l"(a) : "l"(b));
    asm volatile("add.u64 %0, %0, %1;\n\t" : "+l"(a) : "l"(x));
    d = rotr64_ptx(d ^ a, 32);
    asm volatile("add.u64 %0, %0, %1;\n\t" : "+l"(c) : "l"(d));
    b = rotr64_ptx(b ^ c, 24);
    asm volatile("add.u64 %0, %0, %1;\n\t" : "+l"(a) : "l"(b));
    asm volatile("add.u64 %0, %0, %1;\n\t" : "+l"(a) : "l"(y));
    d = rotr64_ptx(d ^ a, 16);
    asm volatile("add.u64 %0, %0, %1;\n\t" : "+l"(c) : "l"(d));
    b = rotr64_ptx(b ^ c, 63);
}

__global__ void __launch_bounds__(256, 3)
blake2b_ptx_kernel(const uint8_t* __restrict__ input,
                    uint8_t* __restrict__ output,
                    int n_blocks, int msg_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_blocks) return;

    const uint64_t* msg = (const uint64_t*)(input + idx * msg_len);
    uint64_t* out = (uint64_t*)(output + idx * 64);

    // Same structure as optimized but with PTX G function
    uint64_t v0=BLAKE2B_IV[0], v1=BLAKE2B_IV[1], v2=BLAKE2B_IV[2], v3=BLAKE2B_IV[3];
    uint64_t v4=BLAKE2B_IV[4], v5=BLAKE2B_IV[5], v6=BLAKE2B_IV[6], v7=BLAKE2B_IV[7];
    uint64_t v8=BLAKE2B_IV[0], v9=BLAKE2B_IV[1], v10=BLAKE2B_IV[2], v11=BLAKE2B_IV[3];
    uint64_t v12=BLAKE2B_IV[4], v13=BLAKE2B_IV[5], v14=~BLAKE2B_IV[6], v15=BLAKE2B_IV[7];
    v12 ^= (uint64_t)msg_len;

    const int nw = msg_len / 8;
    uint64_t m[16];
    for (int i = 0; i < 16; i++) m[i] = (i < nw) ? msg[i] : 0;

    #define GP(r,a,b,c,d,x,y) blake2b_g_ptx(a,b,c,d,x,y)
    #define ROUNDP(r) do { \
        GP(r,v0,v4,v8,v12, m[SIGMA[r][0]], m[SIGMA[r][1]]); \
        GP(r,v1,v5,v9,v13, m[SIGMA[r][2]], m[SIGMA[r][3]]); \
        GP(r,v2,v6,v10,v14,m[SIGMA[r][4]], m[SIGMA[r][5]]); \
        GP(r,v3,v7,v11,v15,m[SIGMA[r][6]], m[SIGMA[r][7]]); \
        GP(r,v0,v5,v10,v15,m[SIGMA[r][8]], m[SIGMA[r][9]]); \
        GP(r,v1,v6,v11,v12,m[SIGMA[r][10]],m[SIGMA[r][11]]); \
        GP(r,v2,v7,v8,v13, m[SIGMA[r][12]],m[SIGMA[r][13]]); \
        GP(r,v3,v4,v9,v14, m[SIGMA[r][14]],m[SIGMA[r][15]]); \
    } while(0)

    ROUNDP(0);  ROUNDP(1);  ROUNDP(2);  ROUNDP(3);
    ROUNDP(4);  ROUNDP(5);  ROUNDP(6);  ROUNDP(7);
    ROUNDP(8);  ROUNDP(9);  ROUNDP(10); ROUNDP(11);
    #undef GP
    #undef ROUNDP

    out[0]=v0^v8;  out[1]=v1^v9;  out[2]=v2^v10; out[3]=v3^v11;
    out[4]=v4^v12; out[5]=v5^v13; out[6]=v6^v14; out[7]=v7^v15;
}

// ---------- BENCHMARK RUNNER ----------

struct BenchResult {
    int n_blocks;
    double naive_ms, opt_ms, ptx_ms;
    double naive_throughput, opt_throughput, ptx_throughput;
};

void run_benchmark(int n_blocks, int msg_len, int threads, FILE* json_out) {
    size_t in_size = (size_t)n_blocks * msg_len;
    size_t out_size = (size_t)n_blocks * 64;

    uint8_t *h_in = (uint8_t*)malloc(in_size);
    uint8_t *h_out = (uint8_t*)malloc(out_size);
    for (size_t i = 0; i < in_size; i++) h_in[i] = (uint8_t)(i & 0xFF);

    uint8_t *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, in_size));
    CUDA_CHECK(cudaMalloc(&d_out, out_size));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, in_size, cudaMemcpyHostToDevice));

    int blocks = (n_blocks + threads - 1) / threads;
    int warmup = 10;
    int runs = 50;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        blake2b_naive_kernel<<<blocks, threads>>>(d_in, d_out, n_blocks, msg_len);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Naive
    float naive_ms = 0;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        blake2b_naive_kernel<<<blocks, threads>>>(d_in, d_out, n_blocks, msg_len);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));
    naive_ms /= runs;

    // Optimized warmup + run
    for (int i = 0; i < warmup; i++) {
        blake2b_opt_kernel<<<blocks, threads>>>(d_in, d_out, n_blocks, msg_len);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float opt_ms = 0;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        blake2b_opt_kernel<<<blocks, threads>>>(d_in, d_out, n_blocks, msg_len);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&opt_ms, start, stop));
    opt_ms /= runs;

    // PTX warmup + run
    for (int i = 0; i < warmup; i++) {
        blake2b_ptx_kernel<<<blocks, threads>>>(d_in, d_out, n_blocks, msg_len);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float ptx_ms = 0;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        blake2b_ptx_kernel<<<blocks, threads>>>(d_in, d_out, n_blocks, msg_len);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ptx_ms, start, stop));
    ptx_ms /= runs;

    // Results
    printf("| %8d | %8.3f ms | %8.3f ms | %8.3f ms | %6.2fx |\n",
           n_blocks, naive_ms, opt_ms, ptx_ms, naive_ms / ptx_ms);

    if (json_out) {
        fprintf(json_out, "%s{\"n_blocks\":%d,\"naive_ms\":%.4f,\"opt_ms\":%.4f,\"ptx_ms\":%.4f,\"speedup\":%.2f}",
                ftell(json_out) > 2 ? ",\n" : "[\n", n_blocks, naive_ms, opt_ms, ptx_ms, naive_ms/ptx_ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
}

int main() {
    printf("=== BLAKE2b Hash Benchmark ===\n\n");
    printf("Message length: 128 bytes (16 × u64)\n");
    printf("Thread block: 256 threads\n\n");
    printf("|  N blocks |     Naive | Optimized |       PTX | Speedup |\n");
    printf("|-----------|-----------|-----------|-----------|---------|\n");

    FILE* json = fopen("results/bench_hash.json", "w");
    if (json) fprintf(json, "[\n");

    int scales[] = {1000, 10000, 100000, 1000000, 10000000};
    for (int s : scales) {
        run_benchmark(s, 128, 256, json);
    }

    if (json) {
        fprintf(json, "\n]\n");
        fclose(json);
        printf("\nResults saved to results/bench_hash.json\n");
    }

    return 0;
}
