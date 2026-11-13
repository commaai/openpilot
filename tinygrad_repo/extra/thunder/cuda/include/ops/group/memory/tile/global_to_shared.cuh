/**
 * @file
 * @brief Group (collaborative warp) ops for loading shared tiles from and storing to global memory. 
 */
 

/**
 * @brief Loads data from global memory into a shared memory tile.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination shared memory tile.
 * @param[in] src The source global memory array.
 * @param[in] idx The coordinate of the tile in the global memory array.
 */
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    using T = typename ST::dtype;
    const int row_stride = src.template stride<axis>();
    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = dst.cols / elem_per_memcpy;
    constexpr int total_calls = (dst.height*dst.width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T> + GROUP_THREADS*elem_per_memcpy-1) / (GROUP_THREADS*elem_per_memcpy); // round up
    constexpr int total_rows = dst.height*dst.width;

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[unit_coord];
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    int laneid = threadIdx.x % GROUP_THREADS;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int load_idx = i * GROUP_THREADS + laneid;
        
        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % dst.cols;

        if constexpr (assume_aligned) {
            float4 tmp;
            move<float4>::ldg(tmp, (float4*)&src_ptr[row*row_stride + col]);
            move<float4>::sts(dst.idx(dst_ptr, {row, col}), tmp);
        }
        else {
            if (row + unit_coord.template dim<axis>() < src.template shape<axis>()) {
                float4 tmp;
                move<float4>::ldg(tmp, (float4*)&src_ptr[row*row_stride + col]);
                move<float4>::sts(dst.idx(dst_ptr, {row, col}), tmp);
            }
            else {
                float4 zeros = {0.f,0.f,0.f,0.f};
                move<float4>::sts(dst.idx(dst_ptr, {row, col}), zeros); // use the default value
            }
        }
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    load<2, false, ST, GL, COORD>(dst, src, idx);
}

/**
 * @brief Stores data from a shared memory tile into global memory.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param row_stride[in] The stride between rows in the destination array.
 */
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    using T = typename ST::dtype;
    const int row_stride = dst.template stride<axis>();
    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = src.cols / elem_per_memcpy;
    constexpr int total_calls = (src.height*src.width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T> + GROUP_THREADS*elem_per_memcpy-1) / (GROUP_THREADS*elem_per_memcpy); // round up

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    typename GL::dtype *dst_ptr = (typename GL::dtype*)&dst[unit_coord];
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    int laneid = threadIdx.x % GROUP_THREADS;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int load_idx = i * GROUP_THREADS + laneid;
        
        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % src.cols;

        if constexpr (assume_aligned) {
            float4 tmp;
            move<float4>::lds(tmp, src.idx(src_ptr, {row, col}));
            move<float4>::stg((float4*)&dst_ptr[row*row_stride + col], tmp);
        }
        else {
            if (row + unit_coord.template dim<axis>() < dst.template shape<axis>()) {
                float4 tmp;
                move<float4>::lds(tmp, src.idx(src_ptr, {row, col}));
                move<float4>::stg((float4*)&dst_ptr[row*row_stride + col], tmp);
            }
        }
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    store<2, false, ST, GL, COORD>(dst, src, idx);
}

/**
 * @brief Asynchronously loads data from global memory into a shared memory tile.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination shared memory tile.
 * @param[in] src The source global memory array.
 *
 * @note This function expects 16-byte alignments. Otherwise, behavior is undefined.
 */
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx) {
    using T = typename ST::dtype;
    const int row_stride = src.template stride<axis>();
    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = dst.cols / elem_per_memcpy;
    constexpr int total_calls = (dst.height*dst.width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T> + GROUP_THREADS*elem_per_memcpy-1) / (GROUP_THREADS*elem_per_memcpy); // round up

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[unit_coord];
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    int laneid = threadIdx.x % GROUP_THREADS;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int load_idx = i * GROUP_THREADS + laneid;
        
        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % dst.cols;

        if constexpr (assume_aligned) {
            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                :: "r"(dst.idx(dst_ptr, {row, col})), "l"(&src_ptr[row*row_stride + col])
                : "memory"
            );
        }
        else {
            if (row + unit_coord.template dim<axis>() < src.template shape<axis>()) {
                asm volatile(
                    "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                    :: "r"(dst.idx(dst_ptr, {row, col})), "l"(&src_ptr[row*row_stride + col])
                    : "memory"
                );
            }
            else {
                // printf("thread %d skipping async load on row %d, col %d\n", threadIdx.x, row + unit_coord.template dim<axis>(), col);
                float4 zeros = {0.f,0.f,0.f,0.f};
                move<float4>::sts(dst.idx(dst_ptr, {row, col}), zeros); // use the default value
            }
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx) {
    load_async<2, false, ST, GL, COORD>(dst, src, idx);
}