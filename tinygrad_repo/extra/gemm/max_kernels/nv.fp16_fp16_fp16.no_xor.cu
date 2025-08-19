#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#define SMEM_N_WIDTH 136

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

extern "C" __global__ void __launch_bounds__(256) wmma_example(half* data0, const half* data1, const half* data2, int N, int K) {
    extern __shared__ char smem[];
    half *smem_a_0 = (half *)(smem);
    half *smem_a_1 = (half *)(smem + 16384);
    half *smem_a_2 = (half *)(smem + 32768);
    half *smem_b_0 = (half *)(smem + 49152);
    half *smem_b_1 = (half *)(smem + 57344);
    half *smem_b_2 = (half *)(smem + 65536);

    int grid_m = blockIdx.x;        /* M//256 */
    int grid_n = blockIdx.y;        /* N//128 */
    int wg_threads = threadIdx.x;   // 32
    int wg_m = threadIdx.y;         // 4
    int wg_n = threadIdx.z;         // 2
    int threads = threadIdx.x + (threadIdx.y * 32) + (threadIdx.z * 128); /* 256 */
    int num_k_blocks = K / 32;

    // ldmatrix indices - 4x loads of 8x8 matrices by 32 threads
    // threads 0-7 are row starts for A, 8-15 for B, 16-23 for C, 24-31 for D
    // [ A | C ]
    // [ - + - ]
    // [ B | D ]

    // swizzled A - SMEM_A is 128 rows x 64 cols
    size_t global_a_off = ((grid_m * 256) * K) + ((threads %  4) * 8) + (((threads /  4) % 2) * 8 * 16 * K) + ((threads / 8) * K);
    size_t store_smem_a_off  = ((threads /  8) *  64) + (((threads * 8) ^ threads) & 56); // 32 rows / 64 cols per copy
    size_t load_smem_a_row   = ((wg_m * 16) + (threads % 16)) * 64;
    size_t load_smem_a_phase = (threads / 16) % 2;
    size_t load_smem_a_0_k_0 = load_smem_a_row + ( 0 * 64) + (((load_smem_a_phase + 0) ^ (threads % 8)) * 8);
    size_t load_smem_a_1_k_0 = load_smem_a_row + (64 * 64) + (((load_smem_a_phase + 0) ^ (threads % 8)) * 8);
    size_t load_smem_a_2_k_0 = load_smem_a_row + ( 0 * 64) + (((load_smem_a_phase + 4) ^ (threads % 8)) * 8);
    size_t load_smem_a_3_k_0 = load_smem_a_row + (64 * 64) + (((load_smem_a_phase + 4) ^ (threads % 8)) * 8);
    size_t load_smem_a_0_k_1 = load_smem_a_row + ( 0 * 64) + (((load_smem_a_phase + 2) ^ (threads % 8)) * 8);
    size_t load_smem_a_1_k_1 = load_smem_a_row + (64 * 64) + (((load_smem_a_phase + 2) ^ (threads % 8)) * 8);
    size_t load_smem_a_2_k_1 = load_smem_a_row + ( 0 * 64) + (((load_smem_a_phase + 6) ^ (threads % 8)) * 8);
    size_t load_smem_a_3_k_1 = load_smem_a_row + (64 * 64) + (((load_smem_a_phase + 6) ^ (threads % 8)) * 8);

    // swizzled B - SMEM_B is 64 rows x 64 cols
    size_t global_b_off = (grid_n * 128) + ((threads % 16) * 8) + ((threads / 16) * N);
    size_t store_smem_b_off = // 32 rows of 64 cols per copy
        ((threads / 128) * (64)) +                  // [A,C] vs [B,D] in ldmatrix
        ((threads % 2) * (2 * 64)) +                // [A vs C] or [B vs. D]
        (((threads / 2) % 2) * (4 * 64)) +          // WG_N in [0, 1]
        (((threads / 4) % 4) * (8 * 64)) +          // B in [0, 1, 2, 3]
        (((threads / 16) % 8) * (8));               // cols in SMEM_B i.e. rows of 8x8

    size_t load_smem_b_0_k_0 = (wg_n * 4 * 64) + ((wg_threads / 8) * 64) + ((wg_threads % 8) * 8);
    size_t load_smem_b_1_k_0 = load_smem_b_0_k_0 + ( 8 * 64);
    size_t load_smem_b_2_k_0 = load_smem_b_0_k_0 + (16 * 64);
    size_t load_smem_b_3_k_0 = load_smem_b_0_k_0 + (24 * 64);
    size_t load_smem_b_0_k_1 = load_smem_b_0_k_0 + (32 * 64);
    size_t load_smem_b_1_k_1 = load_smem_b_1_k_0 + (32 * 64);
    size_t load_smem_b_2_k_1 = load_smem_b_2_k_0 + (32 * 64);
    size_t load_smem_b_3_k_1 = load_smem_b_3_k_0 + (32 * 64);

    // create accs (M=4, N=8)
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
    half4 acc_frag_2_0 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_2_1 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_2_2 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_2_3 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_2_4 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_2_5 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_2_6 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_2_7 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_3_0 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_3_1 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_3_2 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_3_3 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_3_4 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_3_5 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_3_6 = make_half4(0.0f,0.0f,0.0f,0.0f);
    half4 acc_frag_3_7 = make_half4(0.0f,0.0f,0.0f,0.0f);

    // create registers for block A elements
    half8 a_frag_0_k_0;
    half8 a_frag_1_k_0;
    half8 a_frag_2_k_0;
    half8 a_frag_3_k_0;
    half8 a_frag_0_k_1;
    half8 a_frag_1_k_1;
    half8 a_frag_2_k_1;
    half8 a_frag_3_k_1;

    // create register for block B elements
    half4 b_frag_0_k_0;
    half4 b_frag_1_k_0;
    half4 b_frag_2_k_0;
    half4 b_frag_3_k_0;
    half4 b_frag_4_k_0;
    half4 b_frag_5_k_0;
    half4 b_frag_6_k_0;
    half4 b_frag_7_k_0;
    half4 b_frag_0_k_1;
    half4 b_frag_1_k_1;
    half4 b_frag_2_k_1;
    half4 b_frag_3_k_1;
    half4 b_frag_4_k_1;
    half4 b_frag_5_k_1;
    half4 b_frag_6_k_1;
    half4 b_frag_7_k_1;

    __syncthreads();

    // load first tile
    __pipeline_memcpy_async(&smem_a_0[store_smem_a_off + (     0)], &data1[global_a_off + (    0)], 16);
    __pipeline_memcpy_async(&smem_a_0[store_smem_a_off + ( 32*64)], &data1[global_a_off + ( 32*K)], 16);
    __pipeline_memcpy_async(&smem_a_0[store_smem_a_off + ( 64*64)], &data1[global_a_off + ( 64*K)], 16);
    __pipeline_memcpy_async(&smem_a_0[store_smem_a_off + ( 96*64)], &data1[global_a_off + ( 96*K)], 16);
    __pipeline_memcpy_async(&smem_b_0[store_smem_b_off + (     0)], &data2[global_b_off + (    0)], 16);
    __pipeline_memcpy_async(&smem_b_0[store_smem_b_off + ( 32*64)], &data2[global_b_off + ( 16*N)], 16);
    __pipeline_commit();
    global_a_off += 32;
    global_b_off += 32 * N;

    // load second tile
    __pipeline_memcpy_async(&smem_a_1[store_smem_a_off + (     0)], &data1[global_a_off + (    0)], 16);
    __pipeline_memcpy_async(&smem_a_1[store_smem_a_off + ( 32*64)], &data1[global_a_off + ( 32*K)], 16);
    __pipeline_memcpy_async(&smem_a_1[store_smem_a_off + ( 64*64)], &data1[global_a_off + ( 64*K)], 16);
    __pipeline_memcpy_async(&smem_a_1[store_smem_a_off + ( 96*64)], &data1[global_a_off + ( 96*K)], 16);
    __pipeline_memcpy_async(&smem_b_1[store_smem_b_off + (     0)], &data2[global_b_off + (    0)], 16);
    __pipeline_memcpy_async(&smem_b_1[store_smem_b_off + ( 32*64)], &data2[global_b_off + ( 16*N)], 16);
    __pipeline_commit();

    global_a_off += 32;
    global_b_off += 32 * N;

    // wait on first pre-fetch load
    __pipeline_wait_prior(1);
    __syncthreads();

    // load K=0 elements for the first tile
    __ldmatrix_a_elems(&a_frag_0_k_0,                &smem_a_0[load_smem_a_0_k_0]);
    __ldmatrix_a_elems(&a_frag_1_k_0,                &smem_a_0[load_smem_a_1_k_0]);
    __ldmatrix_a_elems(&a_frag_2_k_0,                &smem_a_0[load_smem_a_2_k_0]);
    __ldmatrix_a_elems(&a_frag_3_k_0,                &smem_a_0[load_smem_a_3_k_0]);
    __ldmatrix_b_elems(&b_frag_0_k_0, &b_frag_1_k_0, &smem_b_0[load_smem_b_0_k_0]);
    __ldmatrix_b_elems(&b_frag_2_k_0, &b_frag_3_k_0, &smem_b_0[load_smem_b_1_k_0]);
    __ldmatrix_b_elems(&b_frag_4_k_0, &b_frag_5_k_0, &smem_b_0[load_smem_b_2_k_0]);
    __ldmatrix_b_elems(&b_frag_6_k_0, &b_frag_7_k_0, &smem_b_0[load_smem_b_3_k_0]);

    for (int block_k = 0; block_k < num_k_blocks; block_k++) {
        int phase_k = block_k % 3;
        half *smem_a_curr = (phase_k == 0) ? smem_a_0 : ((phase_k == 1) ? smem_a_1 : smem_a_2);
        half *smem_b_curr = (phase_k == 0) ? smem_b_0 : ((phase_k == 1) ? smem_b_1 : smem_b_2);

        int next_phase_k = (block_k+1) % 3;
        half *smem_a_next = (next_phase_k == 0) ? smem_a_0 : ((next_phase_k == 1) ? smem_a_1 : smem_a_2);
        half *smem_b_next = (next_phase_k == 0) ? smem_b_0 : ((next_phase_k == 1) ? smem_b_1 : smem_b_2);

        int store_phase_k = (block_k+2) % 3;
        half *smem_a_store = (store_phase_k == 0) ? smem_a_0 : ((store_phase_k == 1) ? smem_a_1 : smem_a_2);
        half *smem_b_store = (store_phase_k == 0) ? smem_b_0 : ((store_phase_k == 1) ? smem_b_1 : smem_b_2);

        // load K=1 elements for the current tile
        __ldmatrix_a_elems(&a_frag_0_k_1,                &smem_a_curr[load_smem_a_0_k_1]);
        __ldmatrix_a_elems(&a_frag_1_k_1,                &smem_a_curr[load_smem_a_1_k_1]);
        __ldmatrix_a_elems(&a_frag_2_k_1,                &smem_a_curr[load_smem_a_2_k_1]);
        __ldmatrix_a_elems(&a_frag_3_k_1,                &smem_a_curr[load_smem_a_3_k_1]);
        __ldmatrix_b_elems(&b_frag_0_k_1, &b_frag_1_k_1, &smem_b_curr[load_smem_b_0_k_1]);
        __ldmatrix_b_elems(&b_frag_2_k_1, &b_frag_3_k_1, &smem_b_curr[load_smem_b_1_k_1]);
        __ldmatrix_b_elems(&b_frag_4_k_1, &b_frag_5_k_1, &smem_b_curr[load_smem_b_2_k_1]);
        __ldmatrix_b_elems(&b_frag_6_k_1, &b_frag_7_k_1, &smem_b_curr[load_smem_b_3_k_1]);

        // MMA K=0, (M=4 x N=8)
        acc_frag_0_0 = __WMMA_8_16_16_half_half(a_frag_0_k_0, b_frag_0_k_0, acc_frag_0_0);
        acc_frag_0_1 = __WMMA_8_16_16_half_half(a_frag_0_k_0, b_frag_1_k_0, acc_frag_0_1);
        acc_frag_0_2 = __WMMA_8_16_16_half_half(a_frag_0_k_0, b_frag_2_k_0, acc_frag_0_2);
        acc_frag_0_3 = __WMMA_8_16_16_half_half(a_frag_0_k_0, b_frag_3_k_0, acc_frag_0_3);
        acc_frag_0_4 = __WMMA_8_16_16_half_half(a_frag_0_k_0, b_frag_4_k_0, acc_frag_0_4);
        acc_frag_0_5 = __WMMA_8_16_16_half_half(a_frag_0_k_0, b_frag_5_k_0, acc_frag_0_5);
        acc_frag_0_6 = __WMMA_8_16_16_half_half(a_frag_0_k_0, b_frag_6_k_0, acc_frag_0_6);
        acc_frag_0_7 = __WMMA_8_16_16_half_half(a_frag_0_k_0, b_frag_7_k_0, acc_frag_0_7);
        acc_frag_1_0 = __WMMA_8_16_16_half_half(a_frag_1_k_0, b_frag_0_k_0, acc_frag_1_0);
        acc_frag_1_1 = __WMMA_8_16_16_half_half(a_frag_1_k_0, b_frag_1_k_0, acc_frag_1_1);
        acc_frag_1_2 = __WMMA_8_16_16_half_half(a_frag_1_k_0, b_frag_2_k_0, acc_frag_1_2);
        acc_frag_1_3 = __WMMA_8_16_16_half_half(a_frag_1_k_0, b_frag_3_k_0, acc_frag_1_3);
        acc_frag_1_4 = __WMMA_8_16_16_half_half(a_frag_1_k_0, b_frag_4_k_0, acc_frag_1_4);
        acc_frag_1_5 = __WMMA_8_16_16_half_half(a_frag_1_k_0, b_frag_5_k_0, acc_frag_1_5);
        acc_frag_1_6 = __WMMA_8_16_16_half_half(a_frag_1_k_0, b_frag_6_k_0, acc_frag_1_6);
        acc_frag_1_7 = __WMMA_8_16_16_half_half(a_frag_1_k_0, b_frag_7_k_0, acc_frag_1_7);
        acc_frag_2_0 = __WMMA_8_16_16_half_half(a_frag_2_k_0, b_frag_0_k_0, acc_frag_2_0);
        acc_frag_2_1 = __WMMA_8_16_16_half_half(a_frag_2_k_0, b_frag_1_k_0, acc_frag_2_1);
        acc_frag_2_2 = __WMMA_8_16_16_half_half(a_frag_2_k_0, b_frag_2_k_0, acc_frag_2_2);
        acc_frag_2_3 = __WMMA_8_16_16_half_half(a_frag_2_k_0, b_frag_3_k_0, acc_frag_2_3);
        acc_frag_2_4 = __WMMA_8_16_16_half_half(a_frag_2_k_0, b_frag_4_k_0, acc_frag_2_4);
        acc_frag_2_5 = __WMMA_8_16_16_half_half(a_frag_2_k_0, b_frag_5_k_0, acc_frag_2_5);
        acc_frag_2_6 = __WMMA_8_16_16_half_half(a_frag_2_k_0, b_frag_6_k_0, acc_frag_2_6);
        acc_frag_2_7 = __WMMA_8_16_16_half_half(a_frag_2_k_0, b_frag_7_k_0, acc_frag_2_7);
        acc_frag_3_0 = __WMMA_8_16_16_half_half(a_frag_3_k_0, b_frag_0_k_0, acc_frag_3_0);
        acc_frag_3_1 = __WMMA_8_16_16_half_half(a_frag_3_k_0, b_frag_1_k_0, acc_frag_3_1);
        acc_frag_3_2 = __WMMA_8_16_16_half_half(a_frag_3_k_0, b_frag_2_k_0, acc_frag_3_2);
        acc_frag_3_3 = __WMMA_8_16_16_half_half(a_frag_3_k_0, b_frag_3_k_0, acc_frag_3_3);
        acc_frag_3_4 = __WMMA_8_16_16_half_half(a_frag_3_k_0, b_frag_4_k_0, acc_frag_3_4);
        acc_frag_3_5 = __WMMA_8_16_16_half_half(a_frag_3_k_0, b_frag_5_k_0, acc_frag_3_5);
        acc_frag_3_6 = __WMMA_8_16_16_half_half(a_frag_3_k_0, b_frag_6_k_0, acc_frag_3_6);
        acc_frag_3_7 = __WMMA_8_16_16_half_half(a_frag_3_k_0, b_frag_7_k_0, acc_frag_3_7);

        // load next tile if needed
        if (block_k < (num_k_blocks-2)) {
            __pipeline_memcpy_async(&smem_a_store[store_smem_a_off + (     0)], &data1[global_a_off + (    0)], 16);
            __pipeline_memcpy_async(&smem_a_store[store_smem_a_off + ( 32*64)], &data1[global_a_off + ( 32*K)], 16);
            __pipeline_memcpy_async(&smem_a_store[store_smem_a_off + ( 64*64)], &data1[global_a_off + ( 64*K)], 16);
            __pipeline_memcpy_async(&smem_a_store[store_smem_a_off + ( 96*64)], &data1[global_a_off + ( 96*K)], 16);
            __pipeline_memcpy_async(&smem_b_store[store_smem_b_off + (     0)], &data2[global_b_off + (    0)], 16);
            __pipeline_memcpy_async(&smem_b_store[store_smem_b_off + ( 32*64)], &data2[global_b_off + ( 16*N)], 16);
            global_a_off += 32;
            global_b_off += 32 * N;
        }
        __pipeline_commit();

        // wait next tile
        __pipeline_wait_prior(1);
        __syncthreads();

        // load K=0 elements for the next tile
        __ldmatrix_a_elems(&a_frag_0_k_0,                &smem_a_next[load_smem_a_0_k_0]);
        __ldmatrix_a_elems(&a_frag_1_k_0,                &smem_a_next[load_smem_a_1_k_0]);
        __ldmatrix_a_elems(&a_frag_2_k_0,                &smem_a_next[load_smem_a_2_k_0]);
        __ldmatrix_a_elems(&a_frag_3_k_0,                &smem_a_next[load_smem_a_3_k_0]);
        __ldmatrix_b_elems(&b_frag_0_k_0, &b_frag_1_k_0, &smem_b_next[load_smem_b_0_k_0]);
        __ldmatrix_b_elems(&b_frag_2_k_0, &b_frag_3_k_0, &smem_b_next[load_smem_b_1_k_0]);
        __ldmatrix_b_elems(&b_frag_4_k_0, &b_frag_5_k_0, &smem_b_next[load_smem_b_2_k_0]);
        __ldmatrix_b_elems(&b_frag_6_k_0, &b_frag_7_k_0, &smem_b_next[load_smem_b_3_k_0]);

        // MMA K=1, (M=4 x N=8)
        acc_frag_0_0 = __WMMA_8_16_16_half_half(a_frag_0_k_1, b_frag_0_k_1, acc_frag_0_0);
        acc_frag_0_1 = __WMMA_8_16_16_half_half(a_frag_0_k_1, b_frag_1_k_1, acc_frag_0_1);
        acc_frag_0_2 = __WMMA_8_16_16_half_half(a_frag_0_k_1, b_frag_2_k_1, acc_frag_0_2);
        acc_frag_0_3 = __WMMA_8_16_16_half_half(a_frag_0_k_1, b_frag_3_k_1, acc_frag_0_3);
        acc_frag_0_4 = __WMMA_8_16_16_half_half(a_frag_0_k_1, b_frag_4_k_1, acc_frag_0_4);
        acc_frag_0_5 = __WMMA_8_16_16_half_half(a_frag_0_k_1, b_frag_5_k_1, acc_frag_0_5);
        acc_frag_0_6 = __WMMA_8_16_16_half_half(a_frag_0_k_1, b_frag_6_k_1, acc_frag_0_6);
        acc_frag_0_7 = __WMMA_8_16_16_half_half(a_frag_0_k_1, b_frag_7_k_1, acc_frag_0_7);
        acc_frag_1_0 = __WMMA_8_16_16_half_half(a_frag_1_k_1, b_frag_0_k_1, acc_frag_1_0);
        acc_frag_1_1 = __WMMA_8_16_16_half_half(a_frag_1_k_1, b_frag_1_k_1, acc_frag_1_1);
        acc_frag_1_2 = __WMMA_8_16_16_half_half(a_frag_1_k_1, b_frag_2_k_1, acc_frag_1_2);
        acc_frag_1_3 = __WMMA_8_16_16_half_half(a_frag_1_k_1, b_frag_3_k_1, acc_frag_1_3);
        acc_frag_1_4 = __WMMA_8_16_16_half_half(a_frag_1_k_1, b_frag_4_k_1, acc_frag_1_4);
        acc_frag_1_5 = __WMMA_8_16_16_half_half(a_frag_1_k_1, b_frag_5_k_1, acc_frag_1_5);
        acc_frag_1_6 = __WMMA_8_16_16_half_half(a_frag_1_k_1, b_frag_6_k_1, acc_frag_1_6);
        acc_frag_1_7 = __WMMA_8_16_16_half_half(a_frag_1_k_1, b_frag_7_k_1, acc_frag_1_7);
        acc_frag_2_0 = __WMMA_8_16_16_half_half(a_frag_2_k_1, b_frag_0_k_1, acc_frag_2_0);
        acc_frag_2_1 = __WMMA_8_16_16_half_half(a_frag_2_k_1, b_frag_1_k_1, acc_frag_2_1);
        acc_frag_2_2 = __WMMA_8_16_16_half_half(a_frag_2_k_1, b_frag_2_k_1, acc_frag_2_2);
        acc_frag_2_3 = __WMMA_8_16_16_half_half(a_frag_2_k_1, b_frag_3_k_1, acc_frag_2_3);
        acc_frag_2_4 = __WMMA_8_16_16_half_half(a_frag_2_k_1, b_frag_4_k_1, acc_frag_2_4);
        acc_frag_2_5 = __WMMA_8_16_16_half_half(a_frag_2_k_1, b_frag_5_k_1, acc_frag_2_5);
        acc_frag_2_6 = __WMMA_8_16_16_half_half(a_frag_2_k_1, b_frag_6_k_1, acc_frag_2_6);
        acc_frag_2_7 = __WMMA_8_16_16_half_half(a_frag_2_k_1, b_frag_7_k_1, acc_frag_2_7);
        acc_frag_3_0 = __WMMA_8_16_16_half_half(a_frag_3_k_1, b_frag_0_k_1, acc_frag_3_0);
        acc_frag_3_1 = __WMMA_8_16_16_half_half(a_frag_3_k_1, b_frag_1_k_1, acc_frag_3_1);
        acc_frag_3_2 = __WMMA_8_16_16_half_half(a_frag_3_k_1, b_frag_2_k_1, acc_frag_3_2);
        acc_frag_3_3 = __WMMA_8_16_16_half_half(a_frag_3_k_1, b_frag_3_k_1, acc_frag_3_3);
        acc_frag_3_4 = __WMMA_8_16_16_half_half(a_frag_3_k_1, b_frag_4_k_1, acc_frag_3_4);
        acc_frag_3_5 = __WMMA_8_16_16_half_half(a_frag_3_k_1, b_frag_5_k_1, acc_frag_3_5);
        acc_frag_3_6 = __WMMA_8_16_16_half_half(a_frag_3_k_1, b_frag_6_k_1, acc_frag_3_6);
        acc_frag_3_7 = __WMMA_8_16_16_half_half(a_frag_3_k_1, b_frag_7_k_1, acc_frag_3_7);
    }

    // write accumulators to output
    __pipeline_wait_prior(0);
    __syncthreads();

    // faster epilogue: write each 8x8 TC accs to SMEM first
    //   - SMEM_N_WIDTH 8 larger than 128 required to deconflict bank access
    //   - around 14 micros
    //   - check bank conflict with in sudo with: "PYTHONPATH=. CUDA=1 GEMM_VARIATION="max" DTYPE_IN=half DTYPE_OUT=half DTYPE_ACC=half CNT=8 INPUT=ONES /usr/local/cuda/bin/ncu --section MemoryWorkloadAnalysis --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum python3 ./extra/gemm/max_matmul.py"

    // epilogue chunk with 256 threads / WG_M=4 / WG_N=2: split into 8 chunks (hi/lo for each in TC M)
    //   1) write 32 rows of 128 cols (rows 0-7, 16-23, 32-39, 48-53 in acc_frag_0.lo, then acc_frag_0.hi, etc.)
    //   2) read/write 16 rows of 128 elements in 8 elem (16B) chunks

    half2 *smem32_d  = (half2 *)(smem);
    half8 *smem128_d = (half8 *)(smem);
    half8 *out128_d  = (half8 *)(data0);
    size_t smem32_d_write_off  = (wg_m * 8 * (SMEM_N_WIDTH / 2)) + (wg_n * (16 / 2));
    size_t smem32_d_thread_off = ((wg_threads / 4) * (SMEM_N_WIDTH / 2)) + (wg_threads % 4);
    size_t smem128_d_read_off = ((threads / 16) * (SMEM_N_WIDTH / 8)) + (threads % 16);
    size_t out128_d_off = ((grid_m * 256) * (N / 8)) + (grid_n * (128 / 8)) +
        ((threads / 128) * 16 * (N / 8)) + (((threads / 16) % 8) * (N / 8)) + (threads % 16);

    // write acc_frag_0_*

    // write 32 rows of 128 N elements to SMEM
    __syncthreads();
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 0*4)] = half2(acc_frag_0_0.x, acc_frag_0_0.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 1*4)] = half2(acc_frag_0_1.x, acc_frag_0_1.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 4*4)] = half2(acc_frag_0_2.x, acc_frag_0_2.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 5*4)] = half2(acc_frag_0_3.x, acc_frag_0_3.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 8*4)] = half2(acc_frag_0_4.x, acc_frag_0_4.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 9*4)] = half2(acc_frag_0_5.x, acc_frag_0_5.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + (12*4)] = half2(acc_frag_0_6.x, acc_frag_0_6.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + (13*4)] = half2(acc_frag_0_7.x, acc_frag_0_7.y);

    // each thread reads and writes two 8 element chunks
    __syncthreads();
    out128_d[out128_d_off + ( 0 * (N / 8))] = smem128_d[smem128_d_read_off];
    out128_d[out128_d_off + (32 * (N / 8))] = smem128_d[smem128_d_read_off + (16 * (SMEM_N_WIDTH / 8))];

    // write 32 rows of 128 N elements to SMEM
    __syncthreads();
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 0*4)] = half2(acc_frag_0_0.z, acc_frag_0_0.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 1*4)] = half2(acc_frag_0_1.z, acc_frag_0_1.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 4*4)] = half2(acc_frag_0_2.z, acc_frag_0_2.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 5*4)] = half2(acc_frag_0_3.z, acc_frag_0_3.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 8*4)] = half2(acc_frag_0_4.z, acc_frag_0_4.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 9*4)] = half2(acc_frag_0_5.z, acc_frag_0_5.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + (12*4)] = half2(acc_frag_0_6.z, acc_frag_0_6.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + (13*4)] = half2(acc_frag_0_7.z, acc_frag_0_7.w);

    // each thread reads and writes two 8 element chunks
    __syncthreads();
    out128_d[out128_d_off + ( 8 * (N / 8))] = smem128_d[smem128_d_read_off];
    out128_d[out128_d_off + (40 * (N / 8))] = smem128_d[smem128_d_read_off + (16 * (SMEM_N_WIDTH / 8))];

    // write acc_frag_1_*
    out128_d_off += (64 * (N / 8));

    // write 32 rows of 128 N elements to SMEM
    __syncthreads();
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 0*4)] = half2(acc_frag_1_0.x, acc_frag_1_0.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 1*4)] = half2(acc_frag_1_1.x, acc_frag_1_1.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 4*4)] = half2(acc_frag_1_2.x, acc_frag_1_2.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 5*4)] = half2(acc_frag_1_3.x, acc_frag_1_3.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 8*4)] = half2(acc_frag_1_4.x, acc_frag_1_4.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 9*4)] = half2(acc_frag_1_5.x, acc_frag_1_5.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + (12*4)] = half2(acc_frag_1_6.x, acc_frag_1_6.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + (13*4)] = half2(acc_frag_1_7.x, acc_frag_1_7.y);

    // each thread reads and writes two 8 element chunks
    __syncthreads();
    out128_d[out128_d_off + ( 0 * (N / 8))] = smem128_d[smem128_d_read_off];
    out128_d[out128_d_off + (32 * (N / 8))] = smem128_d[smem128_d_read_off + (16 * (SMEM_N_WIDTH / 8))];

    // write 32 rows of 128 N elements to SMEM
    __syncthreads();
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 0*4)] = half2(acc_frag_1_0.z, acc_frag_1_0.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 1*4)] = half2(acc_frag_1_1.z, acc_frag_1_1.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 4*4)] = half2(acc_frag_1_2.z, acc_frag_1_2.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 5*4)] = half2(acc_frag_1_3.z, acc_frag_1_3.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 8*4)] = half2(acc_frag_1_4.z, acc_frag_1_4.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 9*4)] = half2(acc_frag_1_5.z, acc_frag_1_5.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + (12*4)] = half2(acc_frag_1_6.z, acc_frag_1_6.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + (13*4)] = half2(acc_frag_1_7.z, acc_frag_1_7.w);

    // each thread reads and writes two 8 element chunks
    __syncthreads();
    out128_d[out128_d_off + ( 8 * (N / 8))] = smem128_d[smem128_d_read_off];
    out128_d[out128_d_off + (40 * (N / 8))] = smem128_d[smem128_d_read_off + (16 * (SMEM_N_WIDTH / 8))];

    // write acc_frag_2_*
    out128_d_off += (64 * (N / 8));

    // write 32 rows of 128 N elements to SMEM
    __syncthreads();
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 0*4)] = half2(acc_frag_2_0.x, acc_frag_2_0.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 1*4)] = half2(acc_frag_2_1.x, acc_frag_2_1.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 4*4)] = half2(acc_frag_2_2.x, acc_frag_2_2.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 5*4)] = half2(acc_frag_2_3.x, acc_frag_2_3.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 8*4)] = half2(acc_frag_2_4.x, acc_frag_2_4.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 9*4)] = half2(acc_frag_2_5.x, acc_frag_2_5.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + (12*4)] = half2(acc_frag_2_6.x, acc_frag_2_6.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + (13*4)] = half2(acc_frag_2_7.x, acc_frag_2_7.y);

    // each thread reads and writes two 8 element chunks
    __syncthreads();
    out128_d[out128_d_off + ( 0 * (N / 8))] = smem128_d[smem128_d_read_off];
    out128_d[out128_d_off + (32 * (N / 8))] = smem128_d[smem128_d_read_off + (16 * (SMEM_N_WIDTH / 8))];

    // write 32 rows of 128 N elements to SMEM
    __syncthreads();
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 0*4)] = half2(acc_frag_2_0.z, acc_frag_2_0.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 1*4)] = half2(acc_frag_2_1.z, acc_frag_2_1.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 4*4)] = half2(acc_frag_2_2.z, acc_frag_2_2.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 5*4)] = half2(acc_frag_2_3.z, acc_frag_2_3.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 8*4)] = half2(acc_frag_2_4.z, acc_frag_2_4.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 9*4)] = half2(acc_frag_2_5.z, acc_frag_2_5.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + (12*4)] = half2(acc_frag_2_6.z, acc_frag_2_6.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + (13*4)] = half2(acc_frag_2_7.z, acc_frag_2_7.w);

    // each thread reads and writes two 8 element chunks
    __syncthreads();
    out128_d[out128_d_off + ( 8 * (N / 8))] = smem128_d[smem128_d_read_off];
    out128_d[out128_d_off + (40 * (N / 8))] = smem128_d[smem128_d_read_off + (16 * (SMEM_N_WIDTH / 8))];

    // write acc_frag_3_*
    out128_d_off += (64 * (N / 8));

    // write 32 rows of 128 N elements to SMEM
    __syncthreads();
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 0*4)] = half2(acc_frag_3_0.x, acc_frag_3_0.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 1*4)] = half2(acc_frag_3_1.x, acc_frag_3_1.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 4*4)] = half2(acc_frag_3_2.x, acc_frag_3_2.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 5*4)] = half2(acc_frag_3_3.x, acc_frag_3_3.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 8*4)] = half2(acc_frag_3_4.x, acc_frag_3_4.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 9*4)] = half2(acc_frag_3_5.x, acc_frag_3_5.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + (12*4)] = half2(acc_frag_3_6.x, acc_frag_3_6.y);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + (13*4)] = half2(acc_frag_3_7.x, acc_frag_3_7.y);

    // each thread reads and writes two 8 element chunks
    __syncthreads();
    out128_d[out128_d_off + ( 0 * (N / 8))] = smem128_d[smem128_d_read_off];
    out128_d[out128_d_off + (32 * (N / 8))] = smem128_d[smem128_d_read_off + (16 * (SMEM_N_WIDTH / 8))];

    // write 32 rows of 128 N elements to SMEM
    __syncthreads();
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 0*4)] = half2(acc_frag_3_0.z, acc_frag_3_0.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 1*4)] = half2(acc_frag_3_1.z, acc_frag_3_1.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 4*4)] = half2(acc_frag_3_2.z, acc_frag_3_2.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 5*4)] = half2(acc_frag_3_3.z, acc_frag_3_3.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 8*4)] = half2(acc_frag_3_4.z, acc_frag_3_4.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + ( 9*4)] = half2(acc_frag_3_5.z, acc_frag_3_5.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + (12*4)] = half2(acc_frag_3_6.z, acc_frag_3_6.w);
    smem32_d[smem32_d_write_off + smem32_d_thread_off + (13*4)] = half2(acc_frag_3_7.z, acc_frag_3_7.w);

    // each thread reads and writes two 8 element chunks
    __syncthreads();
    out128_d[out128_d_off + ( 8 * (N / 8))] = smem128_d[smem128_d_read_off];
    out128_d[out128_d_off + (40 * (N / 8))] = smem128_d[smem128_d_read_off + (16 * (SMEM_N_WIDTH / 8))];

    __syncthreads();
}
