/**
 * @file
 * @brief Matrix multiply-accumulate operations for tiles stored in registers.
 */

/**
 * @brief Perform the HMMA.16816 operation.
 *
 * This function performs the half-precision matrix multiply-accumulate operation
 * using the `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` instruction.
 *
 * @param[out] d0 The first half of the output float2 accumulator.
 * @param[out] d1 The second half of the output float2 accumulator.
 * @param[in] a0 The first half of the first input bf16_2 matrix.
 * @param[in] a1 The second half of the first input bf16_2 matrix.
 * @param[in] a2 The first half of the second input bf16_2 matrix.
 * @param[in] a3 The second half of the second input bf16_2 matrix.
 * @param[in] b0 The first half of the bf16_2 matrix B.
 * @param[in] b1 The second half of the bf16_2 matrix B.
 * @param[in] c0 The first half of the float2 accumulator matrix C.
 * @param[in] c1 The second half of the float2 accumulator matrix C.
 */
__device__ static inline void hmma16816(      float2 &d0,       float2 &d1,
                                        const bf16_2 &a0, const bf16_2 &a1, const bf16_2 &a2, const bf16_2 &a3,
                                        const bf16_2 &b0, const bf16_2 &b1,
                                        const float2 &c0, const float2 &c1                                    ) {
    asm volatile(
        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
        "{%0, %1, %2, %3}, " \
        "{%4, %5, %6, %7}, " \
        "{%8, %9}, " \
        "{%10, %11, %12, %13};"

        // D matrix
    :   "+f"(d0.x), "+f"(d0.y),
        "+f"(d1.x), "+f"(d1.y)

        // A matrix
    :   "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
        "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),

        // B matrix
        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),

        // C matrix
        "f"(c0.x), "f"(c0.y),
        "f"(c1.x), "f"(c1.y)
    );
}
/**
 * @brief Perform the HMMA.16816 operation with inputs as fp16 and fp32 accumulators
 *
 * This function performs the half-precision matrix multiply-accumulate operation
 * using the `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` instruction.
 *
 * @param[out] d0 The first half of the output float2 accumulator.
 * @param[out] d1 The second half of the output float2 accumulator.
 * @param[in] a0 The first half of the first input half_2 matrix.
 * @param[in] a1 The second half of the first input half_2 matrix.
 * @param[in] a2 The first half of the second input half_2 matrix.
 * @param[in] a3 The second half of the second input half_2 matrix.
 * @param[in] b0 The first half of the half_2 matrix B.
 * @param[in] b1 The second half of the half_2 matrix B.
 * @param[in] c0 The first half of the float2 accumulator matrix C.
 * @param[in] c1 The second half of the float2 accumulator matrix C.
 */
__device__ static inline void hmma16816(      float2 &d0,       float2 &d1,
                                        const half_2 &a0, const half_2 &a1, const half_2 &a2, const half_2 &a3,
                                        const half_2 &b0, const half_2 &b1,
                                        const float2 &c0, const float2 &c1                                    ) {
    asm volatile(
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#multiply-and-accumulate-instruction-mma
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " \
        "{%0, %1, %2, %3}, " \
        "{%4, %5, %6, %7}, " \
        "{%8, %9}, " \
        "{%10, %11, %12, %13};"

        // D matrix
    :   "+f"(d0.x), "+f"(d0.y),
        "+f"(d1.x), "+f"(d1.y)

        // A matrix
    :   "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
        "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),

        // B matrix
        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),

        // C matrix
        "f"(c0.x), "f"(c0.y),
        "f"(c1.x), "f"(c1.y)
    );
}
/**
 * @brief Perform the HMMA.16816 operation.
 *
 * This function performs the half-precision matrix multiply-accumulate operation
 * using the `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` instruction.
 *
 * @param[out] d0 The first half of the output half_2 accumulator.
 * @param[out] d1 The second half of the output half_2 accumulator.
 * @param[in] a0 The first half of the first input half_2 matrix.
 * @param[in] a1 The second half of the first input half_2 matrix.
 * @param[in] a2 The first half of the second input half_2 matrix.
 * @param[in] a3 The second half of the second input half_2 matrix.
 * @param[in] b0 The first half of the half_2 matrix B.
 * @param[in] b1 The second half of the half_2 matrix B.
 * @param[in] c0 The first half of the half_2 accumulator matrix C.
 * @param[in] c1 The second half of the half_2 accumulator matrix C.
 */
__device__ static inline void hmma16816(      half_2 &d0,       half_2 &d1,
                                        const half_2 &a0, const half_2 &a1, const half_2 &a2, const half_2 &a3,
                                        const half_2 &b0, const half_2 &b1,
                                        const half_2 &c0, const half_2 &c1                                    ) {
    asm volatile(
        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 " \
        "{%0, %1}, " \
        "{%2, %3, %4, %5}, " \
        "{%6, %7}, " \
        "{%8, %9};"

        // D matrix
    :   "=r"(*(uint32_t*)(&d0)), "=r"(*(uint32_t*)(&d1))

        // A matrix
    :   "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
        "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),

        // B matrix
        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),

        // C matrix
        "r"(*(uint32_t*)(&c0)), "r"(*(uint32_t*)(&c1))
    );
}

#ifdef KITTENS_HOPPER
/**
* @brief Perform the HMMA.16816 operation for FP8 using fp8e4m3_2.
*
* Using mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 instruction
* but with fp8e4m3_2 (2 FP8 values) instead of fp8e4m3_4
*/
/**
 * @brief Perform the HMMA.16816 operation for FP8.
 *
 * This function performs the fp8-precision matrix multiply-accumulate operation
 * using the `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` instruction.
 *
 * @param[out] d0 The first half of the output float2 accumulator.
 * @param[out] d1 The second half of the output float2 accumulator.
 * @param[in] a0,a1,a2,a3 Input FP8 matrix A values
 * @param[in] b0,b1 Input FP8 matrix B values
 * @param[in] c0,c1 Input float2 accumulator matrix C values
 */
__device__ static inline void hmma16816(      float2 &d0,       float2 &d1,
                                       const fp8e4m3_4 &a0, const fp8e4m3_4 &a1, 
                                       const fp8e4m3_4 &a2, const fp8e4m3_4 &a3,
                                       const fp8e4m3_4 &b0, const fp8e4m3_4 &b1,
                                       const float2 &c0, const float2 &c1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        
        // D matrix (output)
        : "+f"(d0.x), "+f"(d0.y),
          "+f"(d1.x), "+f"(d1.y)
        
        // A matrix
        : "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
          "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),
        
        // B matrix
        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),
        
        // C matrix
        "f"(c0.x), "f"(c0.y),
        "f"(c1.x), "f"(c1.y)
    );
}
#endif

/**
 * @brief Base matrix multiply-accumulate operation for row layout.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::row> &d,
                                    const rt_base<bf16,  ducks::rt_layout::row> &a,
                                    const rt_base<bf16,  ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
/**
 * @brief Base matrix multiply-accumulate operation for row layout
 * with fp16 inputs and fp32 accumulators.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<half_2, row_layout> matrix.
 * @param[in] b The second input rt_base<half_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::row> &d,
                                    const rt_base<half,  ducks::rt_layout::row> &a,
                                    const rt_base<half,  ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#ifdef KITTENS_HOPPER
/**
 * @brief Base matrix multiply-accumulate operation for row layout.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<fp8e4m3, row_layout> matrix.
 * @param[in] b The second input rt_base<fp8e4m3, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::row> &d,
                                    const rt_base<fp8e4m3,  ducks::rt_layout::row> &a,
                                    const rt_base<fp8e4m3,  ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#endif
/**
 * @brief Base matrix multiply-accumulate operation for row layout.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<half_2, row_layout> accumulator.
 * @param[in] a The first input rt_base<half_2, row_layout> matrix.
 * @param[in] b The second input rt_base<half_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<half_2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AB_base(rt_base<half, ducks::rt_layout::row> &d,
                                    const rt_base<half, ducks::rt_layout::row> &a,
                                    const rt_base<half, ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<half, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
/**
 * @brief Base dot product operation for row layout.
 *
 * This function performs the base dot product operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<bf16,  ducks::rt_layout::row> &a,
                                     const rt_base<bf16,  ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2], // for some reason this one seems to need to be backwards
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3], // for some reason this one seems to need to be backwards
        c.data[2], c.data[3]
    );
}
/**
 * @brief Base dot product operation for row layout
 * with fp16 inputs and fp32 accumulators.
 *
 * This function performs the base dot product operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<half_2, row_layout> matrix.
 * @param[in] b The second input rt_base<half_2, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<half,  ducks::rt_layout::row> &a,
                                     const rt_base<half,  ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2], // for some reason this one seems to need to be backwards
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3], // for some reason this one seems to need to be backwards
        c.data[2], c.data[3]
    );
}
#ifdef KITTENS_HOPPER
/**
 * @brief Base dot product operation for row layout.
 *
 * This function performs the base dot product operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<fp8e4m3x4, row_layout> matrix.
 * @param[in] b The second input rt_base<fp8e4m3x4, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<fp8e4m3,  ducks::rt_layout::row> &a,
                                     const rt_base<fp8e4m3,  ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2], // for some reason this one seems to need to be backwards
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3], // for some reason this one seems to need to be backwards
        c.data[2], c.data[3]
    );
}
#endif


/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, col_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<bf16,  ducks::rt_layout::col> &a,
                                     const rt_base<bf16,  ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A
 * with fp16 inputs and fp32 accumulators.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<half_2, col_layout> matrix.
 * @param[in] b The second input rt_base<half_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<half,  ducks::rt_layout::col> &a,
                                     const rt_base<half,  ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#ifdef KITTENS_HOPPER
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<fp8e4m3x4, col_layout> matrix.
 * @param[in] b The second input rt_base<fp8e4m3x4, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<fp8e4m3,  ducks::rt_layout::col> &a,
                                     const rt_base<fp8e4m3,  ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#endif

/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A and B.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, col_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::row> &d,
                                      const rt_base<bf16,  ducks::rt_layout::col> &a,
                                      const rt_base<bf16,  ducks::rt_layout::row> &b, // in col-major mode
                                      const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A and B
 * with fp16 inputs and fp32 accumulators.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<half_2, col_layout> matrix.
 * @param[in] b The second input rt_base<half_2, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::row> &d,
                                      const rt_base<half,  ducks::rt_layout::col> &a,
                                      const rt_base<half,  ducks::rt_layout::row> &b, // in row-major mode
                                      const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#ifdef KITTENS_HOPPER
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A and B.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<fp8e4m3x4, col_layout> matrix.
 * @param[in] b The second input rt_base<fp8e4m3x4, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::row> &d,
                                      const rt_base<fp8e4m3,  ducks::rt_layout::col> &a,
                                      const rt_base<fp8e4m3,  ducks::rt_layout::row> &b, // in col-major mode
                                      const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#endif

/**
 * @brief Matrix multiply-accumulate operation.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` function.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_hf<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_hf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_hf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_hf<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::rt::col_layout B, ducks::rt::row_layout C>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b,
                               const C &c) {
    KITTENS_CHECK_WARP
    static_assert(D::rows == A::rows && D::cols == B::cols); // Check D matches A, B
    static_assert(A::cols == B::rows); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    #ifdef KITTENS_HOPPER
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)   ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)
    );
    #else
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif
    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AB_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_AB_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}
/**
 * @brief Dot product operation for row layout.
 *
 * This function performs the dot product operation
 * using the `hmma16816` function.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::rt::row_layout B, ducks::rt::row_layout C>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b, // notice row and (M, K) instead of col and (K, M)
                                const C &c) {
    KITTENS_CHECK_WARP
    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    #ifdef KITTENS_HOPPER
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)  ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)
    );
    #else
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif
    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_ABt_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_ABt_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}
/**
 * @brief Matrix multiply-accumulate operation with transposed A.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` instruction.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, row_layout> matrix.
 * @param[in] b The second input rt_bf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::row_layout D, ducks::rt::col_layout A, ducks::rt::col_layout B, ducks::rt::row_layout C>
__device__ static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b,
                                const C &c) {
    KITTENS_CHECK_WARP
    static_assert(D::rows == A::cols && D::cols == B::cols); // Check D matches A, B
    static_assert(A::rows == B::rows); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    #ifdef KITTENS_HOPPER
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)   ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)
    );
    #else
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif
    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AtB_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtB_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}
/**
 * @brief Matrix multiply-accumulate operation with transposed A and B.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` instruction.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, col_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::row_layout D, ducks::rt::col_layout A, ducks::rt::row_layout B, ducks::rt::row_layout C>
__device__ static inline void mma_AtBt(D &d,
                                 const A &a,
                                 const B &b,
                                 const C &c) {
    KITTENS_CHECK_WARP
    static_assert(D::rows == A::cols && D::cols == B::rows); // Check D matches A, B
    static_assert(A::rows == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    #ifdef KITTENS_HOPPER
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)   ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)
    );
    #else
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif
    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AtBt_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtBt_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}

template<int trans_A, int trans_B, ducks::rt::all D, ducks::rt::all A, ducks::rt::all B, ducks::rt::all C>
__device__ static inline void mma(D &d,
                                  const A &a,
                                  const B &b,
                                  const C &c) {
    KITTENS_CHECK_WARP
    if constexpr(trans_A == transpose::T) {
        if constexpr(trans_B == transpose::T) {
            mma_AtBt(d, a, b, c);
        } else {
            mma_AtB(d, a, b, c);
        }
    } else {
        if constexpr(trans_B == transpose::T) {
            mma_ABt(d, a, b, c);
        } else {
            mma_AB(d, a, b, c);
        }
    }
}
template<int trans_A, int trans_B, ducks::rt::all A, ducks::rt::all B, ducks::rt::all C>
__device__ static inline C mma(const A &a,
                               const B &b,
                               const C &c) {
    KITTENS_CHECK_WARP
    C d;
    if constexpr(trans_A == transpose::T) {
        if constexpr(trans_B == transpose::T) {
            mma_AtBt(d, a, b, c);
        } else {
            mma_AtB(d, a, b, c);
        }
    } else {
        if constexpr(trans_B == transpose::T) {
            mma_ABt(d, a, b, c);
        } else {
            mma_AB(d, a, b, c);
        }
    }
    return d;
}


//  --------------------------------------------------------------------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------
//  -------------------------------------------------- COMPLEX INPUTS --------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------



/**
 * @brief Matrix multiply-accumulate operation for complex tiles
 *
 * This function calls mma_AB with hf arguments
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_cmplx_hf<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_cmplx_hf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_cmplx_hf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_cmplx_hf<N, M, row_layout> accumulator matrix.
 */
template<int N, int K, int M>
__device__ static inline void mma_AB(crt_hf<N, M, ducks::rt_layout::row> &d,
                               const crt_hf<N, K, ducks::rt_layout::row> &a,
                               const crt_hf<K, M, ducks::rt_layout::col> &b,
                               const crt_hf<N, M, ducks::rt_layout::row> &c) {
    KITTENS_CHECK_WARP
    
    // Copy data from input accumulate register into output
    ::kittens::group<1>::copy(d.real, c.real);
    ::kittens::group<1>::copy(d.imag, c.imag);

    // Negative on B matrix so we can use single accum register
    rt_hf<N, K, ducks::rt_layout::row> tmp;
    // Hex value for -1 in float16
    constexpr half factor = std::bit_cast<__half>(uint16_t(0xFB80));
    ::kittens::group<1>::mul(tmp, a.imag, factor);
    mma_AB(d.real, a.real, b.real, d.real);
    mma_AB(d.real, tmp, b.imag, d.real);

    mma_AB(d.imag, a.real, b.imag, d.imag);
    mma_AB(d.imag, a.imag, b.real, d.imag);
}
/**
 * @brief Matrix multiply-accumulate operation for complex tiles
 *
 * This function calls mma_AB with bf16 arguments
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_cmplx_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_cmplx_bf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_cmplx_bf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_cmplx_fl<N, M, row_layout> accumulator matrix.
 */

template<int N, int K, int M>
__device__ static inline void mma_AB(crt_fl<N, M, ducks::rt_layout::row> &d,
                               const crt_bf<N, K, ducks::rt_layout::row> &a,
                               const crt_bf<K, M, ducks::rt_layout::col> &b,
                               const crt_fl<N, M, ducks::rt_layout::row> &c) {
    KITTENS_CHECK_WARP
    
    // Copy data from input accumulate register into output
    ::kittens::group<1>::copy(d.real, c.real);
    ::kittens::group<1>::copy(d.imag, c.imag);

    // Negative on B matrix so we can use single accum register
    kittens::rt_bf<N, K, ducks::rt_layout::row> tmp;
    // Hex value for -1 in bf16
    constexpr bf16 factor = std::bit_cast<__nv_bfloat16>(uint16_t(0xBF80));
    ::kittens::group<1>::mul(tmp, a.imag, factor);
    mma_AB(d.real, a.real, b.real, d.real);
    mma_AB(d.real, tmp, b.imag, d.real);

    mma_AB(d.imag, a.real, b.imag, d.imag);
    mma_AB(d.imag, a.imag, b.real, d.imag);
}