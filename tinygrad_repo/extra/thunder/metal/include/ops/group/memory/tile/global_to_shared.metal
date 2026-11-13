/**
 * @file
 * @brief Group (collaborative warp) ops for loading shared tiles from and storing to global memory.
 */


//template<typename ST, typename U>
//static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
//load(int i,
//     threadgroup ST *dst, device U* src,
//     thread const int& group_laneid,
//     thread const int& memcpy_per_row,
//     thread const int& elem_per_memcpy,
//     thread const int& row_stride)
//{
//    int idx = i * GROUP_THREADS + group_laneid;
//    int row = idx / memcpy_per_row;
//    int col = (idx*elem_per_memcpy) % ST::cols;
//    if (row < ST::rows) {
//        *(threadgroup float4*)(&(*dst)[{row, col}]) = *(device float4*)(&src[row*row_stride + col]);
//    }
//}


template<typename ST, typename GL>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_global_layout<GL>(), void>::type
load(threadgroup ST &dst, thread const GL &_src, thread const coord &idx, const int threadIdx) {
    int group_laneid = threadIdx % GROUP_THREADS;
    using T = typename ST::T;
    using U = typename GL::dtype;
    device U *src = (device U*)&_src.template get<ST>(idx);
    const int row_stride = _src.row_stride();
    using read_vector = ReadVector<1>;
    // we can handle this many rows each time we run a memcpy_async
    constexpr const int elem_per_memcpy = sizeof(read_vector)/sizeof(typename ST::dtype);
    constexpr const int memcpy_per_row = ST::cols / elem_per_memcpy;
    int total_calls = ((ST::height * ST::width + (N_WARPS-1))) * TILE_DIM*TILE_DIM / (N_WARPS*SIMD_THREADS*elem_per_memcpy); // round up
    #pragma clang loop unroll(full)
    for(int i = 0; i < total_calls; i++) {

        int idx = i * GROUP_THREADS + group_laneid;
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % dst.cols;
        if (row<dst.rows && col < dst.cols) {
            *(threadgroup read_vector*)(&dst[{row, col}]) = *(device read_vector*)(&src[row*row_stride + col]);
//            *(threadgroup float*)(&dst[{row, col}]) = 1.0f;
        }
    }
//    dst[{0, 0}] = base_types::convertor<T, float>::convert(1.f);
//    dst[{0, 0}] = total_calls;
//    meta::unroll_i_in_range<0, total_calls, 1>::run(load<ST, typename GL::dtype>, &dst, src, group_laneid, memcpy_per_row, elem_per_memcpy, row_stride);
}


//template<typename ST, typename GL>
//static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_global_layout<GL>(), void>::type
//load(threadgroup ST &dst, thread const GL &_src, thread const coord &idx, const int threadIdx) {
//    int group_laneid = threadIdx % GROUP_THREADS;
//    int groupid = threadIdx / GROUP_THREADS;
//    int laneid = threadIdx % SIMD_THREADS;
//    
//    using U = typename GL::dtype;
//    device U *src = (device U*)&_src.template get<ST>(idx);
//    const int row_stride = _src.row_stride();
//    
//    int elem_per_memcpy = sizeof(float)/sizeof(typename ST::dtype);
//    int memcpy_per_row = ST::cols / elem_per_memcpy;
//    int total_calls = ((ST::height * ST::width + (N_WARPS-1))) * TILE_DIM*TILE_DIM / (N_WARPS*SIMD_THREADS*elem_per_memcpy); // round up
//    /*
//     1x16 or 8 x 128
//     */
//    int offset = ST::num_elements / (GROUP_WARPS);
////    int offset = group_laneid
//    #pragma clang loop unroll(full)
//    for(int i = 0; i < total_calls; i++) {
//        int idx = i * SIMD_THREADS + laneid;
////        int idx = i * () + group_laneid;
//        int row = idx / memcpy_per_row;
//        int col = (idx*elem_per_memcpy) % dst.cols;
//        if (row<dst.rows) {
//            *(threadgroup float*)(&dst[{row, col}]) = *(device float*)(&src[row*row_stride + col]);
//        }
//    }
//}
//
//template<typename ST, typename GL>
//static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_global_layout<GL>(), void>::type
//load(threadgroup ST &dst, thread const GL &_src, thread const coord &idx, const int threadIdx) {
//    int warp_id = threadIdx / SIMD_THREADS;
//    int lane_id = threadIdx % SIMD_THREADS;
////    int N_WARPS = /* number of warps in your group */;
//    
//    using U = typename GL::dtype;
//    device U *src = (device U*)&_src.template get<ST>(idx);
//    const int row_stride = _src.row_stride();
//    
//    int elem_per_memcpy = sizeof(float)/sizeof(typename ST::dtype);
//    int memcpy_per_row = ST::cols / elem_per_memcpy;
//    int total_memcpy_elems = (ST::height * ST::cols) / elem_per_memcpy;
//    int elems_per_warp = (total_memcpy_elems + N_WARPS - 1) / N_WARPS;  // Ceiling division
//    
//    int start_idx = warp_id * elems_per_warp;
//    int end_idx = metal::min(start_idx + elems_per_warp, total_memcpy_elems);
//    
//    #pragma clang loop unroll(full)
//    for (int idx = start_idx + lane_id; idx < end_idx; idx += SIMD_THREADS) {
//        int row = idx / memcpy_per_row;
//        int col = (idx % memcpy_per_row) * elem_per_memcpy;
//        if (row < ST::height) {
//            *(threadgroup float*)(&dst[{row, col}]) = *(device float*)(&src[row * row_stride + col]);
//        }
//    }
//}

template<typename ST, typename GL>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_global_layout<GL>(), void>::type
store(thread const GL &_dst, threadgroup const ST &src, thread const coord &idx, const int threadIdx) {
    int group_laneid = threadIdx % GROUP_THREADS;
    using U = typename GL::dtype;
    device U *dst = (device U*)&_dst.template get<ST>(idx);
    const int row_stride = _dst.row_stride();
    using read_vector = ReadVector<1>;
    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(read_vector)/sizeof(typename ST::dtype); // float/float -> 1
    int memcpy_per_row = ST::cols / elem_per_memcpy; // 240 memcpy per row
    int total_calls = ((src.height * src.width + (N_WARPS-1))) * TILE_DIM*TILE_DIM / (N_WARPS*SIMD_THREADS*elem_per_memcpy); // round up
    
    #pragma clang loop unroll(full)
    for(int i = 0; i < total_calls; i++) {

        int idx = i * GROUP_THREADS + group_laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % src.cols;
        if (row<src.rows && col < src.cols) {
            *(device read_vector*)(&dst[row*row_stride + col]) = *(threadgroup read_vector*)(&src[{row, col}]);
//            *(device float*)(&dst[row*row_stride + col]) = 1.f;
        }
    }
//    dst[0] = src[{0,0}];
//    dst[0] = total_calls;
//    dst[0] = base_types::convertor<U, float>::convert(1);
}

