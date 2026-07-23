#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#define N_PAD 132

struct __align__(8) half4 { half x, y, z, w; };
__device__ half4 make_half4(half x, half y, half z, half w) { half4 r={x, y, z, w}; return r; }

struct __align__(16) half8 { half x, y, z, w, a, b, c, d; };
__device__ half8 make_half8(half x, half y, half z, half w, half a, half b, half c, half d) { half8 r={x, y, z, w, a, b, c, d}; return r; }

__device__ void __ldmatrix_a_elems(half8 *regs, half *smem) {
    uint32_t reg0, reg1, reg2, reg3;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(reg0), "=r"(reg1), "=r"(reg2), "=r"(reg3)
        : "l"(__cvta_generic_to_shared(smem))
    );
    uint32_t *addr = reinterpret_cast<uint32_t*>(regs);
    addr[0] = reg0;
    addr[1] = reg1;
    addr[2] = reg2;
    addr[3] = reg3;
}

__device__ void __ldmatrix_b_elems(half4 *regs_lo, half4 *regs_hi, half *smem) {
    uint32_t reg0, reg1, reg2, reg3;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(reg0), "=r"(reg1), "=r"(reg2), "=r"(reg3)
        : "l"(__cvta_generic_to_shared(smem))
    );
    uint32_t *addr_lo = reinterpret_cast<uint32_t*>(regs_lo);
    uint32_t *addr_hi = reinterpret_cast<uint32_t*>(regs_hi);
    addr_lo[0] = reg0;
    addr_lo[1] = reg1;
    addr_hi[0] = reg2;
    addr_hi[1] = reg3;
}

__device__ half4 __WMMA_8_16_16_half_half(half8 a, half4 b, half4 c) {
    int *a_pk = (int *) (&a), *b_pk = (int *) (&b), *c_pk = (int *) (&c);
    asm( "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 { %0, %1 }, { %2, %3, %4, %5 }, { %6, %7 }, { %0, %1 };"
        : "+r"(c_pk[0]), "+r"(c_pk[1]): "r"(a_pk[0]), "r"(a_pk[1]), "r"(a_pk[2]),  "r"(a_pk[3]), "r"(b_pk[0]), "r"(b_pk[1]) );
    return c;
}

extern "C" __global__ void __launch_bounds__(128) wmma_example(half* data0, const half* data1, const half* data2, int N, int K) {
    int grid_m = blockIdx.x;        /* M//64 */
    int grid_n = blockIdx.y;        /* N//128 */
    int threads = threadIdx.x;      /* 128 */
    int wg_m = (threads/64);        // 0 or 1 for 1st and 3rd blocks of b_m=16xb_k=16 vs 2nd and 4th blocks
    int wg_n = (threads/32)%2;      // 0 or 1 for 1st, 3rd, 5th, 7th blocks of b_n=16xb_k=16 vs 2nd, 4th, 6th, 8th blocks - differs from triton
    int wg_threads = threads%32;
    int num_k_blocks = K / 64;

    // load indexes
    size_t global_a_off = ((grid_m * 64) * K) + ((threads %  8) * 8) + ((threads /  8) * K);
    size_t global_b_off = (grid_n * 128)      + ((threads % 16) * 8) + ((threads / 16) * N);

    // swizzled smem store offsets - columns of smem are swizzled
    // here's a link to a description of the triton: https://github.com/triton-lang/triton/discussions/2026#discussioncomment-6746579
    // see also the thunderkittens impl: https://github.com/HazyResearch/ThunderKittens/blob/main/include/types/shared/st.cuh
    size_t store_smem_a_off  = ((threads /  8) *  64) + (((threads * 8) ^ threads) & 56);               // r15
    size_t store_smem_b_off  = ((threads / 16) * 128) + (((threads / 16) * 8) ^ ((threads % 16) * 8));  // r19\

    // ldmatrix indices
    // threads 0-7 are row starts for A, 8-15 for B, 16-23 for C, 24-31 for D
    // [ A | C ]
    // [ - + - ]
    // [ B | D ]

    // swizzled ldmatrix
    size_t load_smem_a_row   = ((wg_m * 16) + (threads % 16)) * 64; // r293
    size_t load_smem_a_phase = (threads / 16) % 2;                  // r4

    size_t load_smem_b_row   = (threads % 16) * 128;                // r299
    size_t load_smem_b_phase = (wg_n * 2) + (((threads / 16) % 2)); // r297 -- this differs from the generated triton kernel (swapped order)

    size_t load_smem_a_0_k_0 = load_smem_a_row + (((load_smem_a_phase +  0) ^ (threads % 8)) * 8); // r38
    size_t load_smem_a_1_k_0 = load_smem_a_0_k_0 + (32 * 64);
    size_t load_smem_b_0_k_0 = load_smem_b_row + (((load_smem_b_phase +  0) ^ (threads % 8)) * 8);
    size_t load_smem_b_1_k_0 = load_smem_b_row + (((load_smem_b_phase +  4) ^ (threads % 8)) * 8);
    size_t load_smem_b_2_k_0 = load_smem_b_row + (((load_smem_b_phase +  8) ^ (threads % 8)) * 8);
    size_t load_smem_b_3_k_0 = load_smem_b_row + (((load_smem_b_phase + 12) ^ (threads % 8)) * 8);

    size_t load_smem_a_0_k_1 = load_smem_a_row + (((load_smem_a_phase +  2) ^ (threads % 8)) * 8); // r58 = r293 + r316;
    size_t load_smem_a_1_k_1 = load_smem_a_0_k_1 + (32 * 64);
    size_t load_smem_b_0_k_1 = load_smem_b_0_k_0 + (16 * 128);
    size_t load_smem_b_1_k_1 = load_smem_b_1_k_0 + (16 * 128);
    size_t load_smem_b_2_k_1 = load_smem_b_2_k_0 + (16 * 128);
    size_t load_smem_b_3_k_1 = load_smem_b_3_k_0 + (16 * 128);

    size_t load_smem_a_0_k_2 = load_smem_a_row + (((load_smem_a_phase +  4) ^ (threads % 8)) * 8); // r59 = r293 + r319;
    size_t load_smem_a_1_k_2 = load_smem_a_0_k_2 + (32 * 64);
    size_t load_smem_b_0_k_2 = load_smem_b_0_k_0 + (32 * 128);
    size_t load_smem_b_1_k_2 = load_smem_b_1_k_0 + (32 * 128);
    size_t load_smem_b_2_k_2 = load_smem_b_2_k_0 + (32 * 128);
    size_t load_smem_b_3_k_2 = load_smem_b_3_k_0 + (32 * 128);

    size_t load_smem_a_0_k_3 = load_smem_a_row + (((load_smem_a_phase +  6) ^ (threads % 8)) * 8); // r60 = r293 + r322;
    size_t load_smem_a_1_k_3 = load_smem_a_0_k_3 + (32 * 64);
    size_t load_smem_b_0_k_3 = load_smem_b_0_k_0 + (48 * 128);
    size_t load_smem_b_1_k_3 = load_smem_b_1_k_0 + (48 * 128);
    size_t load_smem_b_2_k_3 = load_smem_b_2_k_0 + (48 * 128);
    size_t load_smem_b_3_k_3 = load_smem_b_3_k_0 + (48 * 128);

    // create shared mem (A_1 8192 bytes, A_2 8192 bytes, B_1 16384 bytes, B2_16384 bytes)
    __shared__ alignas(16) char smem[49152];

    // create accs (16 WMMAs and 4 output elements each) and zero
    half4 acc_frag_0_0 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_0_1 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_0_2 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_0_3 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_0_4 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_0_5 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_0_6 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_0_7 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_1_0 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_1_1 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_1_2 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_1_3 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_1_4 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_1_5 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_1_6 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_1_7 = make_half4(0.0f,0.0f,0.0f,0.0f);

    // create registers for block A elements (2)
    half8 a_frag_0;
    half8 a_frag_1;

    // create register for block B elements (8)
    half4 b_frag_0;
    half4 b_frag_1;
    half4 b_frag_2;
    half4 b_frag_3;
    half4 b_frag_4;
    half4 b_frag_5;
    half4 b_frag_6;
    half4 b_frag_7;

    half *smem_a_even = (half *)(smem);
    half *smem_a_odd  = (half *)(smem + 8192);
    half *smem_b_even = (half *)(smem + 16384);
    half *smem_b_odd  = (half *)(smem + 32768);

    // https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies

    // start first pre-fetch load A
    __pipeline_memcpy_async(&smem_a_even[store_smem_a_off +  (    0)], &data1[global_a_off + (   0)], 16);
    __pipeline_memcpy_async(&smem_a_even[store_smem_a_off +  (16*64)], &data1[global_a_off + (16*K)], 16);
    __pipeline_memcpy_async(&smem_a_even[store_smem_a_off +  (32*64)], &data1[global_a_off + (32*K)], 16);
    __pipeline_memcpy_async(&smem_a_even[store_smem_a_off +  (48*64)], &data1[global_a_off + (48*K)], 16);

    // start first pre-fetch load B
    __pipeline_memcpy_async(&smem_b_even[store_smem_b_off + (     0)], &data2[global_b_off + (   0)], 16);
    __pipeline_memcpy_async(&smem_b_even[store_smem_b_off + ( 8*128)], &data2[global_b_off + ( 8*N)], 16);
    __pipeline_memcpy_async(&smem_b_even[store_smem_b_off + (16*128)], &data2[global_b_off + (16*N)], 16);
    __pipeline_memcpy_async(&smem_b_even[store_smem_b_off + (24*128)], &data2[global_b_off + (24*N)], 16);
    __pipeline_memcpy_async(&smem_b_even[store_smem_b_off + (32*128)], &data2[global_b_off + (32*N)], 16);
    __pipeline_memcpy_async(&smem_b_even[store_smem_b_off + (40*128)], &data2[global_b_off + (40*N)], 16);
    __pipeline_memcpy_async(&smem_b_even[store_smem_b_off + (48*128)], &data2[global_b_off + (48*N)], 16);
    __pipeline_memcpy_async(&smem_b_even[store_smem_b_off + (56*128)], &data2[global_b_off + (56*N)], 16);
    __pipeline_commit();

    global_a_off += 64;
    global_b_off += 64 * N;
    __syncthreads();

    // start second pre-fetch load A
    __pipeline_memcpy_async(&smem_a_odd[store_smem_a_off +  (    0)], &data1[global_a_off + (   0)], 16);
    __pipeline_memcpy_async(&smem_a_odd[store_smem_a_off +  (16*64)], &data1[global_a_off + (16*K)], 16);
    __pipeline_memcpy_async(&smem_a_odd[store_smem_a_off +  (32*64)], &data1[global_a_off + (32*K)], 16);
    __pipeline_memcpy_async(&smem_a_odd[store_smem_a_off +  (48*64)], &data1[global_a_off + (48*K)], 16);

    // start second pre-fetch load B
    __pipeline_memcpy_async(&smem_b_odd[store_smem_b_off + (     0)], &data2[global_b_off + (   0)], 16);
    __pipeline_memcpy_async(&smem_b_odd[store_smem_b_off + ( 8*128)], &data2[global_b_off + ( 8*N)], 16);
    __pipeline_memcpy_async(&smem_b_odd[store_smem_b_off + (16*128)], &data2[global_b_off + (16*N)], 16);
    __pipeline_memcpy_async(&smem_b_odd[store_smem_b_off + (24*128)], &data2[global_b_off + (24*N)], 16);
    __pipeline_memcpy_async(&smem_b_odd[store_smem_b_off + (32*128)], &data2[global_b_off + (32*N)], 16);
    __pipeline_memcpy_async(&smem_b_odd[store_smem_b_off + (40*128)], &data2[global_b_off + (40*N)], 16);
    __pipeline_memcpy_async(&smem_b_odd[store_smem_b_off + (48*128)], &data2[global_b_off + (48*N)], 16);
    __pipeline_memcpy_async(&smem_b_odd[store_smem_b_off + (56*128)], &data2[global_b_off + (56*N)], 16);
    __pipeline_commit();

    global_a_off += 64;
    global_b_off += 64 * N;

    // wait on needed prefetch value
    __pipeline_wait_prior(0); // TODO: this enables fast iterations, but incorrect results with 1 (it shouldn't)
    __syncthreads();

    for (int block_k = 0; block_k < num_k_blocks; block_k++) {
        // BLOCK_K==4: unroll 4 iterations of ldmatrix/wmma
        half *smem_a_curr = (block_k % 2) ? smem_a_even : smem_a_odd;
        half *smem_b_curr = (block_k % 2) ? smem_b_even : smem_b_odd;

        // first load 16 K elements and 16 WMMAs: BLOCK_M==2 * BLOCK_N==8
        __ldmatrix_a_elems(&a_frag_0,            &smem_a_curr[load_smem_a_0_k_0]);
        __ldmatrix_a_elems(&a_frag_1,            &smem_a_curr[load_smem_a_1_k_0]);
        __ldmatrix_b_elems(&b_frag_0, &b_frag_1, &smem_b_curr[load_smem_b_0_k_0]);
        __ldmatrix_b_elems(&b_frag_2, &b_frag_3, &smem_b_curr[load_smem_b_1_k_0]);
        __ldmatrix_b_elems(&b_frag_4, &b_frag_5, &smem_b_curr[load_smem_b_2_k_0]);
        __ldmatrix_b_elems(&b_frag_6, &b_frag_7, &smem_b_curr[load_smem_b_3_k_0]);
        acc_frag_0_0 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_0, acc_frag_0_0);
        acc_frag_0_1 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_1, acc_frag_0_1);
        acc_frag_0_2 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_2, acc_frag_0_2);
        acc_frag_0_3 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_3, acc_frag_0_3);
        acc_frag_0_4 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_4, acc_frag_0_4);
        acc_frag_0_5 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_5, acc_frag_0_5);
        acc_frag_0_6 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_6, acc_frag_0_6);
        acc_frag_0_7 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_7, acc_frag_0_7);
        acc_frag_1_0 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_0, acc_frag_1_0);
        acc_frag_1_1 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_1, acc_frag_1_1);
        acc_frag_1_2 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_2, acc_frag_1_2);
        acc_frag_1_3 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_3, acc_frag_1_3);
        acc_frag_1_4 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_4, acc_frag_1_4);
        acc_frag_1_5 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_5, acc_frag_1_5);
        acc_frag_1_6 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_6, acc_frag_1_6);
        acc_frag_1_7 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_7, acc_frag_1_7);

        // next 16 K elements
        __ldmatrix_a_elems(&a_frag_0,            &smem_a_curr[load_smem_a_0_k_1]);
        __ldmatrix_a_elems(&a_frag_1,            &smem_a_curr[load_smem_a_1_k_1]);
        __ldmatrix_b_elems(&b_frag_0, &b_frag_1, &smem_b_curr[load_smem_b_0_k_1]);
        __ldmatrix_b_elems(&b_frag_2, &b_frag_3, &smem_b_curr[load_smem_b_1_k_1]);
        __ldmatrix_b_elems(&b_frag_4, &b_frag_5, &smem_b_curr[load_smem_b_2_k_1]);
        __ldmatrix_b_elems(&b_frag_6, &b_frag_7, &smem_b_curr[load_smem_b_3_k_1]);
        acc_frag_0_0 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_0, acc_frag_0_0);
        acc_frag_0_1 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_1, acc_frag_0_1);
        acc_frag_0_2 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_2, acc_frag_0_2);
        acc_frag_0_3 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_3, acc_frag_0_3);
        acc_frag_0_4 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_4, acc_frag_0_4);
        acc_frag_0_5 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_5, acc_frag_0_5);
        acc_frag_0_6 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_6, acc_frag_0_6);
        acc_frag_0_7 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_7, acc_frag_0_7);
        acc_frag_1_0 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_0, acc_frag_1_0);
        acc_frag_1_1 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_1, acc_frag_1_1);
        acc_frag_1_2 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_2, acc_frag_1_2);
        acc_frag_1_3 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_3, acc_frag_1_3);
        acc_frag_1_4 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_4, acc_frag_1_4);
        acc_frag_1_5 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_5, acc_frag_1_5);
        acc_frag_1_6 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_6, acc_frag_1_6);
        acc_frag_1_7 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_7, acc_frag_1_7);

        // next 16 K elements
        __ldmatrix_a_elems(&a_frag_0,            &smem_a_curr[load_smem_a_0_k_2]);
        __ldmatrix_a_elems(&a_frag_1,            &smem_a_curr[load_smem_a_1_k_2]);
        __ldmatrix_b_elems(&b_frag_0, &b_frag_1, &smem_b_curr[load_smem_b_0_k_2]);
        __ldmatrix_b_elems(&b_frag_2, &b_frag_3, &smem_b_curr[load_smem_b_1_k_2]);
        __ldmatrix_b_elems(&b_frag_4, &b_frag_5, &smem_b_curr[load_smem_b_2_k_2]);
        __ldmatrix_b_elems(&b_frag_6, &b_frag_7, &smem_b_curr[load_smem_b_3_k_2]);
        acc_frag_0_0 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_0, acc_frag_0_0);
        acc_frag_0_1 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_1, acc_frag_0_1);
        acc_frag_0_2 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_2, acc_frag_0_2);
        acc_frag_0_3 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_3, acc_frag_0_3);
        acc_frag_0_4 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_4, acc_frag_0_4);
        acc_frag_0_5 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_5, acc_frag_0_5);
        acc_frag_0_6 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_6, acc_frag_0_6);
        acc_frag_0_7 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_7, acc_frag_0_7);
        acc_frag_1_0 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_0, acc_frag_1_0);
        acc_frag_1_1 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_1, acc_frag_1_1);
        acc_frag_1_2 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_2, acc_frag_1_2);
        acc_frag_1_3 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_3, acc_frag_1_3);
        acc_frag_1_4 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_4, acc_frag_1_4);
        acc_frag_1_5 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_5, acc_frag_1_5);
        acc_frag_1_6 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_6, acc_frag_1_6);
        acc_frag_1_7 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_7, acc_frag_1_7);

        // last 16 K elements
        __ldmatrix_a_elems(&a_frag_0,            &smem_a_curr[load_smem_a_0_k_3]);
        __ldmatrix_a_elems(&a_frag_1,            &smem_a_curr[load_smem_a_1_k_3]);
        __ldmatrix_b_elems(&b_frag_0, &b_frag_1, &smem_b_curr[load_smem_b_0_k_3]);
        __ldmatrix_b_elems(&b_frag_2, &b_frag_3, &smem_b_curr[load_smem_b_1_k_3]);
        __ldmatrix_b_elems(&b_frag_4, &b_frag_5, &smem_b_curr[load_smem_b_2_k_3]);
        __ldmatrix_b_elems(&b_frag_6, &b_frag_7, &smem_b_curr[load_smem_b_3_k_3]);
        acc_frag_0_0 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_0, acc_frag_0_0);
        acc_frag_0_1 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_1, acc_frag_0_1);
        acc_frag_0_2 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_2, acc_frag_0_2);
        acc_frag_0_3 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_3, acc_frag_0_3);
        acc_frag_0_4 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_4, acc_frag_0_4);
        acc_frag_0_5 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_5, acc_frag_0_5);
        acc_frag_0_6 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_6, acc_frag_0_6);
        acc_frag_0_7 = __WMMA_8_16_16_half_half(a_frag_0, b_frag_7, acc_frag_0_7);
        acc_frag_1_0 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_0, acc_frag_1_0);
        acc_frag_1_1 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_1, acc_frag_1_1);
        acc_frag_1_2 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_2, acc_frag_1_2);
        acc_frag_1_3 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_3, acc_frag_1_3);
        acc_frag_1_4 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_4, acc_frag_1_4);
        acc_frag_1_5 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_5, acc_frag_1_5);
        acc_frag_1_6 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_6, acc_frag_1_6);
        acc_frag_1_7 = __WMMA_8_16_16_half_half(a_frag_1, b_frag_7, acc_frag_1_7);

        // prefetch next iteration if needed
        __syncthreads();
        if (block_k < (num_k_blocks-2)) {
            __pipeline_memcpy_async(&smem_a_curr[store_smem_a_off +  (    0)], &data1[global_a_off + (   0)], 16);
            __pipeline_memcpy_async(&smem_a_curr[store_smem_a_off +  (16*64)], &data1[global_a_off + (16*K)], 16);
            __pipeline_memcpy_async(&smem_a_curr[store_smem_a_off +  (32*64)], &data1[global_a_off + (32*K)], 16);
            __pipeline_memcpy_async(&smem_a_curr[store_smem_a_off +  (48*64)], &data1[global_a_off + (48*K)], 16);

            __pipeline_memcpy_async(&smem_b_curr[store_smem_b_off + (     0)], &data2[global_b_off + (   0)], 16);
            __pipeline_memcpy_async(&smem_b_curr[store_smem_b_off + ( 8*128)], &data2[global_b_off + ( 8*N)], 16);
            __pipeline_memcpy_async(&smem_b_curr[store_smem_b_off + (16*128)], &data2[global_b_off + (16*N)], 16);
            __pipeline_memcpy_async(&smem_b_curr[store_smem_b_off + (24*128)], &data2[global_b_off + (24*N)], 16);
            __pipeline_memcpy_async(&smem_b_curr[store_smem_b_off + (32*128)], &data2[global_b_off + (32*N)], 16);
            __pipeline_memcpy_async(&smem_b_curr[store_smem_b_off + (40*128)], &data2[global_b_off + (40*N)], 16);
            __pipeline_memcpy_async(&smem_b_curr[store_smem_b_off + (48*128)], &data2[global_b_off + (48*N)], 16);
            __pipeline_memcpy_async(&smem_b_curr[store_smem_b_off + (56*128)], &data2[global_b_off + (56*N)], 16);

            global_a_off += 64;
            global_b_off += 64 * N;
        }
        __pipeline_commit();

        if (block_k < num_k_blocks-1) {
            __pipeline_wait_prior(1);
            __syncthreads();
        }
    }

    // write accumulators to output
    __pipeline_wait_prior(0);
    __syncthreads();

    // // store registers to smem first, then read back to do float4 writes to global
    // float *smem_d = (float *)(smem);
    // size_t smem_d_off = (wg_m * 16 * N_PAD) + (wg_n * 16) + ((wg_threads % 4) * 2) + (((wg_threads / 4) % 8) * N_PAD);
    // smem_d[smem_d_off + 0 + ( 0*8)            ] = acc_frag_0_0.x;
    // smem_d[smem_d_off + 1 + ( 0*8)            ] = acc_frag_0_0.y;
    // smem_d[smem_d_off + 0 + ( 0*8) + (8*N_PAD)] = acc_frag_0_0.z;
    // smem_d[smem_d_off + 1 + ( 0*8) + (8*N_PAD)] = acc_frag_0_0.w;
    // smem_d[smem_d_off + 0 + ( 1*8)            ] = acc_frag_0_1.x;
    // smem_d[smem_d_off + 1 + ( 1*8)            ] = acc_frag_0_1.y;
    // smem_d[smem_d_off + 0 + ( 1*8) + (8*N_PAD)] = acc_frag_0_1.z;
    // smem_d[smem_d_off + 1 + ( 1*8) + (8*N_PAD)] = acc_frag_0_1.w;
    // smem_d[smem_d_off + 0 + ( 4*8)            ] = acc_frag_0_2.x;
    // smem_d[smem_d_off + 1 + ( 4*8)            ] = acc_frag_0_2.y;
    // smem_d[smem_d_off + 0 + ( 4*8) + (8*N_PAD)] = acc_frag_0_2.z;
    // smem_d[smem_d_off + 1 + ( 4*8) + (8*N_PAD)] = acc_frag_0_2.w;
    // smem_d[smem_d_off + 0 + ( 5*8)            ] = acc_frag_0_3.x;
    // smem_d[smem_d_off + 1 + ( 5*8)            ] = acc_frag_0_3.y;
    // smem_d[smem_d_off + 0 + ( 5*8) + (8*N_PAD)] = acc_frag_0_3.z;
    // smem_d[smem_d_off + 1 + ( 5*8) + (8*N_PAD)] = acc_frag_0_3.w;
    // smem_d[smem_d_off + 0 + ( 8*8)            ] = acc_frag_0_4.x;
    // smem_d[smem_d_off + 1 + ( 8*8)            ] = acc_frag_0_4.y;
    // smem_d[smem_d_off + 0 + ( 8*8) + (8*N_PAD)] = acc_frag_0_4.z;
    // smem_d[smem_d_off + 1 + ( 8*8) + (8*N_PAD)] = acc_frag_0_4.w;
    // smem_d[smem_d_off + 0 + ( 9*8)            ] = acc_frag_0_5.x;
    // smem_d[smem_d_off + 1 + ( 9*8)            ] = acc_frag_0_5.y;
    // smem_d[smem_d_off + 0 + ( 9*8) + (8*N_PAD)] = acc_frag_0_5.z;
    // smem_d[smem_d_off + 1 + ( 9*8) + (8*N_PAD)] = acc_frag_0_5.w;
    // smem_d[smem_d_off + 0 + (12*8)            ] = acc_frag_0_6.x;
    // smem_d[smem_d_off + 1 + (12*8)            ] = acc_frag_0_6.y;
    // smem_d[smem_d_off + 0 + (12*8) + (8*N_PAD)] = acc_frag_0_6.z;
    // smem_d[smem_d_off + 1 + (12*8) + (8*N_PAD)] = acc_frag_0_6.w;
    // smem_d[smem_d_off + 0 + (13*8)            ] = acc_frag_0_7.x;
    // smem_d[smem_d_off + 1 + (13*8)            ] = acc_frag_0_7.y;
    // smem_d[smem_d_off + 0 + (13*8) + (8*N_PAD)] = acc_frag_0_7.z;
    // smem_d[smem_d_off + 1 + (13*8) + (8*N_PAD)] = acc_frag_0_7.w;

    // __syncthreads();
    // size_t load_smem_d_off = ((threads % 32) * 4) + ((threads / 32) * N_PAD);
    // float4 d_0_0 = *((float4 *)(smem_d + load_smem_d_off + ( 0 * N_PAD)));
    // float4 d_0_1 = *((float4 *)(smem_d + load_smem_d_off + ( 4 * N_PAD)));
    // float4 d_0_2 = *((float4 *)(smem_d + load_smem_d_off + ( 8 * N_PAD)));
    // float4 d_0_3 = *((float4 *)(smem_d + load_smem_d_off + (12 * N_PAD)));
    // float4 d_0_4 = *((float4 *)(smem_d + load_smem_d_off + (16 * N_PAD)));
    // float4 d_0_5 = *((float4 *)(smem_d + load_smem_d_off + (20 * N_PAD)));
    // float4 d_0_6 = *((float4 *)(smem_d + load_smem_d_off + (24 * N_PAD)));
    // float4 d_0_7 = *((float4 *)(smem_d + load_smem_d_off + (28 * N_PAD)));

    // __syncthreads();
    // smem_d[smem_d_off + 0 + ( 0*8)            ] = acc_frag_1_0.x;
    // smem_d[smem_d_off + 1 + ( 0*8)            ] = acc_frag_1_0.y;
    // smem_d[smem_d_off + 0 + ( 0*8) + (8*N_PAD)] = acc_frag_1_0.z;
    // smem_d[smem_d_off + 1 + ( 0*8) + (8*N_PAD)] = acc_frag_1_0.w;
    // smem_d[smem_d_off + 0 + ( 1*8)            ] = acc_frag_1_1.x;
    // smem_d[smem_d_off + 1 + ( 1*8)            ] = acc_frag_1_1.y;
    // smem_d[smem_d_off + 0 + ( 1*8) + (8*N_PAD)] = acc_frag_1_1.z;
    // smem_d[smem_d_off + 1 + ( 1*8) + (8*N_PAD)] = acc_frag_1_1.w;
    // smem_d[smem_d_off + 0 + ( 4*8)            ] = acc_frag_1_2.x;
    // smem_d[smem_d_off + 1 + ( 4*8)            ] = acc_frag_1_2.y;
    // smem_d[smem_d_off + 0 + ( 4*8) + (8*N_PAD)] = acc_frag_1_2.z;
    // smem_d[smem_d_off + 1 + ( 4*8) + (8*N_PAD)] = acc_frag_1_2.w;
    // smem_d[smem_d_off + 0 + ( 5*8)            ] = acc_frag_1_3.x;
    // smem_d[smem_d_off + 1 + ( 5*8)            ] = acc_frag_1_3.y;
    // smem_d[smem_d_off + 0 + ( 5*8) + (8*N_PAD)] = acc_frag_1_3.z;
    // smem_d[smem_d_off + 1 + ( 5*8) + (8*N_PAD)] = acc_frag_1_3.w;
    // smem_d[smem_d_off + 0 + ( 8*8)            ] = acc_frag_1_4.x;
    // smem_d[smem_d_off + 1 + ( 8*8)            ] = acc_frag_1_4.y;
    // smem_d[smem_d_off + 0 + ( 8*8) + (8*N_PAD)] = acc_frag_1_4.z;
    // smem_d[smem_d_off + 1 + ( 8*8) + (8*N_PAD)] = acc_frag_1_4.w;
    // smem_d[smem_d_off + 0 + ( 9*8)            ] = acc_frag_1_5.x;
    // smem_d[smem_d_off + 1 + ( 9*8)            ] = acc_frag_1_5.y;
    // smem_d[smem_d_off + 0 + ( 9*8) + (8*N_PAD)] = acc_frag_1_5.z;
    // smem_d[smem_d_off + 1 + ( 9*8) + (8*N_PAD)] = acc_frag_1_5.w;
    // smem_d[smem_d_off + 0 + (12*8)            ] = acc_frag_1_6.x;
    // smem_d[smem_d_off + 1 + (12*8)            ] = acc_frag_1_6.y;
    // smem_d[smem_d_off + 0 + (12*8) + (8*N_PAD)] = acc_frag_1_6.z;
    // smem_d[smem_d_off + 1 + (12*8) + (8*N_PAD)] = acc_frag_1_6.w;
    // smem_d[smem_d_off + 0 + (13*8)            ] = acc_frag_1_7.x;
    // smem_d[smem_d_off + 1 + (13*8)            ] = acc_frag_1_7.y;
    // smem_d[smem_d_off + 0 + (13*8) + (8*N_PAD)] = acc_frag_1_7.z;
    // smem_d[smem_d_off + 1 + (13*8) + (8*N_PAD)] = acc_frag_1_7.w;

    // __syncthreads();
    // float4 d_1_0 = *((float4 *)(smem_d + load_smem_d_off + ( 0 * N_PAD)));
    // float4 d_1_1 = *((float4 *)(smem_d + load_smem_d_off + ( 4 * N_PAD)));
    // float4 d_1_2 = *((float4 *)(smem_d + load_smem_d_off + ( 8 * N_PAD)));
    // float4 d_1_3 = *((float4 *)(smem_d + load_smem_d_off + (12 * N_PAD)));
    // float4 d_1_4 = *((float4 *)(smem_d + load_smem_d_off + (16 * N_PAD)));
    // float4 d_1_5 = *((float4 *)(smem_d + load_smem_d_off + (20 * N_PAD)));
    // float4 d_1_6 = *((float4 *)(smem_d + load_smem_d_off + (24 * N_PAD)));
    // float4 d_1_7 = *((float4 *)(smem_d + load_smem_d_off + (28 * N_PAD)));

    // __syncthreads();
    // float *global_d = &data0[((grid_m * 64) * N) + (grid_n * 128) + ((threads % 32) * 4) + ((threads / 32) * N)];
    // *((float4 *)(global_d +  0*N)) = d_0_0;
    // *((float4 *)(global_d +  4*N)) = d_0_1;
    // *((float4 *)(global_d +  8*N)) = d_0_2;
    // *((float4 *)(global_d + 12*N)) = d_0_3;
    // *((float4 *)(global_d + 16*N)) = d_0_4;
    // *((float4 *)(global_d + 20*N)) = d_0_5;
    // *((float4 *)(global_d + 24*N)) = d_0_6;
    // *((float4 *)(global_d + 28*N)) = d_0_7;
    // *((float4 *)(global_d + 32*N)) = d_1_0;
    // *((float4 *)(global_d + 36*N)) = d_1_1;
    // *((float4 *)(global_d + 40*N)) = d_1_2;
    // *((float4 *)(global_d + 44*N)) = d_1_3;
    // *((float4 *)(global_d + 48*N)) = d_1_4;
    // *((float4 *)(global_d + 52*N)) = d_1_5;
    // *((float4 *)(global_d + 56*N)) = d_1_6;
    // *((float4 *)(global_d + 60*N)) = d_1_7;

    // slower way: write floats one by one to data0
    size_t wg_c_off     = ((grid_m * 64) * N) + (grid_n * 128) + (wg_m * 16 * N) + (wg_n * 16);
    size_t thread_c_off = ((wg_threads % 4) * 2) + (((wg_threads / 4) % 8) * N);
    data0[wg_c_off + thread_c_off           + 0 + ( 0*8)] = acc_frag_0_0.x;
    data0[wg_c_off + thread_c_off           + 1 + ( 0*8)] = acc_frag_0_0.y;
    data0[wg_c_off + thread_c_off + (8 * N) + 0 + ( 0*8)] = acc_frag_0_0.z;
    data0[wg_c_off + thread_c_off + (8 * N) + 1 + ( 0*8)] = acc_frag_0_0.w;
    data0[wg_c_off + thread_c_off           + 0 + ( 1*8)] = acc_frag_0_1.x;
    data0[wg_c_off + thread_c_off           + 1 + ( 1*8)] = acc_frag_0_1.y;
    data0[wg_c_off + thread_c_off + (8 * N) + 0 + ( 1*8)] = acc_frag_0_1.z;
    data0[wg_c_off + thread_c_off + (8 * N) + 1 + ( 1*8)] = acc_frag_0_1.w;
    data0[wg_c_off + thread_c_off           + 0 + ( 4*8)] = acc_frag_0_2.x;
    data0[wg_c_off + thread_c_off           + 1 + ( 4*8)] = acc_frag_0_2.y;
    data0[wg_c_off + thread_c_off + (8 * N) + 0 + ( 4*8)] = acc_frag_0_2.z;
    data0[wg_c_off + thread_c_off + (8 * N) + 1 + ( 4*8)] = acc_frag_0_2.w;
    data0[wg_c_off + thread_c_off           + 0 + ( 5*8)] = acc_frag_0_3.x;
    data0[wg_c_off + thread_c_off           + 1 + ( 5*8)] = acc_frag_0_3.y;
    data0[wg_c_off + thread_c_off + (8 * N) + 0 + ( 5*8)] = acc_frag_0_3.z;
    data0[wg_c_off + thread_c_off + (8 * N) + 1 + ( 5*8)] = acc_frag_0_3.w;
    data0[wg_c_off + thread_c_off           + 0 + ( 8*8)] = acc_frag_0_4.x;
    data0[wg_c_off + thread_c_off           + 1 + ( 8*8)] = acc_frag_0_4.y;
    data0[wg_c_off + thread_c_off + (8 * N) + 0 + ( 8*8)] = acc_frag_0_4.z;
    data0[wg_c_off + thread_c_off + (8 * N) + 1 + ( 8*8)] = acc_frag_0_4.w;
    data0[wg_c_off + thread_c_off           + 0 + ( 9*8)] = acc_frag_0_5.x;
    data0[wg_c_off + thread_c_off           + 1 + ( 9*8)] = acc_frag_0_5.y;
    data0[wg_c_off + thread_c_off + (8 * N) + 0 + ( 9*8)] = acc_frag_0_5.z;
    data0[wg_c_off + thread_c_off + (8 * N) + 1 + ( 9*8)] = acc_frag_0_5.w;
    data0[wg_c_off + thread_c_off           + 0 + (12*8)] = acc_frag_0_6.x;
    data0[wg_c_off + thread_c_off           + 1 + (12*8)] = acc_frag_0_6.y;
    data0[wg_c_off + thread_c_off + (8 * N) + 0 + (12*8)] = acc_frag_0_6.z;
    data0[wg_c_off + thread_c_off + (8 * N) + 1 + (12*8)] = acc_frag_0_6.w;
    data0[wg_c_off + thread_c_off           + 0 + (13*8)] = acc_frag_0_7.x;
    data0[wg_c_off + thread_c_off           + 1 + (13*8)] = acc_frag_0_7.y;
    data0[wg_c_off + thread_c_off + (8 * N) + 0 + (13*8)] = acc_frag_0_7.z;
    data0[wg_c_off + thread_c_off + (8 * N) + 1 + (13*8)] = acc_frag_0_7.w;
    wg_c_off += 32*N;
    data0[wg_c_off + thread_c_off           + 0 + ( 0*8)] = acc_frag_1_0.x;
    data0[wg_c_off + thread_c_off           + 1 + ( 0*8)] = acc_frag_1_0.y;
    data0[wg_c_off + thread_c_off + (8 * N) + 0 + ( 0*8)] = acc_frag_1_0.z;
    data0[wg_c_off + thread_c_off + (8 * N) + 1 + ( 0*8)] = acc_frag_1_0.w;
    data0[wg_c_off + thread_c_off           + 0 + ( 1*8)] = acc_frag_1_1.x;
    data0[wg_c_off + thread_c_off           + 1 + ( 1*8)] = acc_frag_1_1.y;
    data0[wg_c_off + thread_c_off + (8 * N) + 0 + ( 1*8)] = acc_frag_1_1.z;
    data0[wg_c_off + thread_c_off + (8 * N) + 1 + ( 1*8)] = acc_frag_1_1.w;
    data0[wg_c_off + thread_c_off           + 0 + ( 4*8)] = acc_frag_1_2.x;
    data0[wg_c_off + thread_c_off           + 1 + ( 4*8)] = acc_frag_1_2.y;
    data0[wg_c_off + thread_c_off + (8 * N) + 0 + ( 4*8)] = acc_frag_1_2.z;
    data0[wg_c_off + thread_c_off + (8 * N) + 1 + ( 4*8)] = acc_frag_1_2.w;
    data0[wg_c_off + thread_c_off           + 0 + ( 5*8)] = acc_frag_1_3.x;
    data0[wg_c_off + thread_c_off           + 1 + ( 5*8)] = acc_frag_1_3.y;
    data0[wg_c_off + thread_c_off + (8 * N) + 0 + ( 5*8)] = acc_frag_1_3.z;
    data0[wg_c_off + thread_c_off + (8 * N) + 1 + ( 5*8)] = acc_frag_1_3.w;
    data0[wg_c_off + thread_c_off           + 0 + ( 8*8)] = acc_frag_1_4.x;
    data0[wg_c_off + thread_c_off           + 1 + ( 8*8)] = acc_frag_1_4.y;
    data0[wg_c_off + thread_c_off + (8 * N) + 0 + ( 8*8)] = acc_frag_1_4.z;
    data0[wg_c_off + thread_c_off + (8 * N) + 1 + ( 8*8)] = acc_frag_1_4.w;
    data0[wg_c_off + thread_c_off           + 0 + ( 9*8)] = acc_frag_1_5.x;
    data0[wg_c_off + thread_c_off           + 1 + ( 9*8)] = acc_frag_1_5.y;
    data0[wg_c_off + thread_c_off + (8 * N) + 0 + ( 9*8)] = acc_frag_1_5.z;
    data0[wg_c_off + thread_c_off + (8 * N) + 1 + ( 9*8)] = acc_frag_1_5.w;
    data0[wg_c_off + thread_c_off           + 0 + (12*8)] = acc_frag_1_6.x;
    data0[wg_c_off + thread_c_off           + 1 + (12*8)] = acc_frag_1_6.y;
    data0[wg_c_off + thread_c_off + (8 * N) + 0 + (12*8)] = acc_frag_1_6.z;
    data0[wg_c_off + thread_c_off + (8 * N) + 1 + (12*8)] = acc_frag_1_6.w;
    data0[wg_c_off + thread_c_off           + 0 + (13*8)] = acc_frag_1_7.x;
    data0[wg_c_off + thread_c_off           + 1 + (13*8)] = acc_frag_1_7.y;
    data0[wg_c_off + thread_c_off + (8 * N) + 0 + (13*8)] = acc_frag_1_7.z;
    data0[wg_c_off + thread_c_off + (8 * N) + 1 + (13*8)] = acc_frag_1_7.w;
}
