/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once // not done!
#include "../../../../types/types.metal"
#include "../../../../common/common.metal"
#include <metal_stdlib>
namespace mittens {

//    
namespace meta {
template<typename ST, int memcpy_per_row, int elem_per_memcpy, int READ_FLOATS>
METAL_FUNC static typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
load(int i, threadgroup ST *dst, device const typename ST::dtype *src, thread const int& row_stride, thread const short& laneid) {
    {
        unsigned idx = i + laneid;
        unsigned row = idx / memcpy_per_row;
        unsigned col = (idx*elem_per_memcpy) % ST::cols;
        *(threadgroup ReadVector<READ_FLOATS>*)(&(*dst)[int2(row, col)]) = *(device ReadVector<READ_FLOATS>*)(&src[row*row_stride + col]);
    }
}

template<typename ST, int memcpy_per_row, int elem_per_memcpy, int READ_FLOATS>
METAL_FUNC static typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
store(int i, device typename ST::dtype *dst, threadgroup const ST *src, thread const int& row_stride, thread const short& laneid) {
    {
        unsigned idx = i + laneid;
        unsigned row = idx / memcpy_per_row;
        unsigned col = (idx*elem_per_memcpy) % ST::cols;
        *(device ReadVector<READ_FLOATS>*)(&dst[row*row_stride + col]) = *(threadgroup ReadVector<READ_FLOATS>*)(&(*src)[int2(row, col)]);
    }
}
    
} // namespace meta

//
///**
// * @brief Loads data from global memory into a shared memory tile with a row layout.
// *
// * @tparam ST The type of the shared tile.
// * @param[out] dst The destination shared memory tile.
// * @param[in] src The source global memory array.
// * @param row_stride[in] The stride between rows in the source array.
// * @param laneid[in] Thread's index in SIMD group
// */
//template<typename ST>
//static METAL_FUNC void load(threadgroup ST &dst, device const typename ST::dtype *src, const int row_stride, short laneid) {
//    using read_type = float;
//    ducks::assert_shared_tile<ST>();
//    constexpr const unsigned elem_per_memcpy = sizeof(read_type)/sizeof(typename ST::dtype); // 2
//    constexpr const unsigned memcpy_per_row = ST::cols / elem_per_memcpy;                    // 32/2=16 not power of 2
//    constexpr const unsigned total_calls = ST::num_elements / (SIMD_THREADS*elem_per_memcpy); // 1024/(32*2)=16
////    #pragma clang loop unroll_count(1)
////    #pragma clang loop unroll(disable)
//    #pragma clang loop unroll(full)
//    for(unsigned i = 0; i < total_calls; i++) {
//        unsigned idx = i * 32 + laneid;
//        unsigned row = idx / memcpy_per_row;
//        unsigned col = (idx*elem_per_memcpy) % ST::cols;
//        *(threadgroup read_type*)(&dst[int2(row, col)]) = *(device read_type*)(&src[row*row_stride + col]);
//    }
//    
////    ducks::assert_shared_tile<ST>();
////    const constexpr int read_size = 1;
////    using read_type = ReadVector<read_size>;
////    constexpr const unsigned elem_per_memcpy = sizeof(read_type)/sizeof(typename ST::dtype); // 2
////    constexpr const unsigned memcpy_per_row = ST::cols / elem_per_memcpy;                    // 32/2=16 not power of 2
////    constexpr const unsigned total_calls = ST::num_elements / (SIMD_THREADS*elem_per_memcpy); // 1024/(32*2)=16
////    
////    
////    meta::unroll_i_in_range<0, total_calls * SIMD_THREADS, SIMD_THREADS>::run(meta::load<ST, memcpy_per_row, elem_per_memcpy, read_size>, &dst, src, row_stride, laneid);
//}
//    
//    
///**
// * @brief Stores data from a shared memory tile with a row layout into global memory.
// *
// * @tparam ST The type of the shared tile.
// * @param[out] dst The destination global memory array.
// * @param[in] src The source shared memory tile.
// * @param row_stride[in] The stride between rows in the destination array.
// * @param laneid[in] Thread's index in SIMD group
// */
//template<typename ST>
//static METAL_FUNC void store(device typename ST::dtype *dst, threadgroup const ST &src, const int row_stride, short laneid) {
//    using read_type = float4;
//    ducks::assert_shared_tile<ST>();
//    constexpr const unsigned elem_per_memcpy = sizeof(read_type)/sizeof(typename ST::dtype);
//    constexpr const unsigned memcpy_per_row = ST::cols / elem_per_memcpy;
//    constexpr const unsigned total_calls = ST::num_elements / (SIMD_THREADS*elem_per_memcpy);
////    #pragma clang loop unroll_count(READ_SIZE)
////#pragma clang loop unroll(disable)
//    #pragma clang loop unroll(full)
//    for(unsigned i = 0; i < total_calls; i++) {
//        unsigned idx = i * 32 + laneid;
//        unsigned row = idx / memcpy_per_row;
//        unsigned col = (idx*elem_per_memcpy) % src.cols;
//        *(device read_type*)(&dst[row*row_stride + col]) = *(threadgroup read_type*)(&src[int2(row, col)]);
//    }
//    
////    
////    ducks::assert_shared_tile<ST>();
////    const constexpr int read_size = 1;
////    using read_type = ReadVector<read_size>;
////
////    constexpr const unsigned elem_per_memcpy = sizeof(read_type)/sizeof(typename ST::dtype);
////    constexpr const unsigned memcpy_per_row = ST::cols / elem_per_memcpy;
////    constexpr const unsigned total_calls = ST::num_elements / (SIMD_THREADS*elem_per_memcpy);
////
////
////    meta::unroll_i_in_range<0, total_calls * SIMD_THREADS, SIMD_THREADS>::run(meta::store<ST, memcpy_per_row, elem_per_memcpy, read_size>, dst, &src, row_stride, laneid);
//}
    


/**
 * @brief Loads data from global memory into a shared memory tile with a row layout.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination shared memory tile.
 * @param[in] src The source global memory array.
 * @param row_stride[in] The stride between rows in the source array.
 * @param laneid[in] Thread's index in SIMD group
 */
template<typename ST, typename GL>
METAL_FUNC static typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_global_layout<GL>(), void>::type
load(threadgroup ST &dst, thread const GL &src, thread const coord &idx, short laneid) {
    using U = typename GL::dtype;
    constexpr const int read_size = 1;
    using read_type = ReadVector<read_size>;
    device U *src_ptr = (device U*)&src.template get<ST>(idx);
    const int row_stride = src.row_stride();
    constexpr const unsigned elem_per_memcpy = sizeof(read_type)/sizeof(typename ST::dtype); // 2
    constexpr const unsigned memcpy_per_row = ST::cols / elem_per_memcpy;                    // 32/2=16 not power of 2
    constexpr const unsigned total_calls = ST::num_elements / (SIMD_THREADS*elem_per_memcpy); // 1024/(32*2)=16
//    #pragma clang loop unroll_count(1)
//    #pragma clang loop unroll(disable)
//    #pragma clang loop unroll(full)
//    for(unsigned i = 0; i < total_calls; i++) {
//        unsigned idx = i * 32 + laneid;
//        unsigned row = idx / memcpy_per_row;
//        unsigned col = (idx*elem_per_memcpy) % ST::cols;
//        *(threadgroup read_type*)(&dst[int2(row, col)]) = *(device read_type*)(&src_ptr[row*row_stride + col]);
//    }
    meta::unroll_i_in_range<0, total_calls * SIMD_THREADS, SIMD_THREADS>::run(meta::load<ST, memcpy_per_row, elem_per_memcpy, read_size>, &dst, src_ptr, row_stride, laneid);
}
    /*
     
     */
    
    
/**
 * @brief Stores data from a shared memory tile with a row layout into global memory.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param row_stride[in] The stride between rows in the destination array.
 * @param laneid[in] Thread's index in SIMD group
 */
template<typename ST, typename GL>
METAL_FUNC static typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_global_layout<GL>(), void>::type
store(thread GL &dst, threadgroup const ST &src, thread const coord &idx, short laneid) {
    using U = typename GL::dtype;
    constexpr const int read_size = 1;
    using read_type = ReadVector<read_size>;
    device U *dst_ptr = (device U*)&dst.template get<ST>(idx);
    const int row_stride = dst.row_stride();
    
    constexpr const unsigned elem_per_memcpy = sizeof(read_type)/sizeof(typename ST::dtype);
    constexpr const unsigned memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr const unsigned total_calls = ST::num_elements / (SIMD_THREADS*elem_per_memcpy);
//    #pragma clang loop unroll_count(READ_SIZE)
//#pragma clang loop unroll(disable)
//    #pragma clang loop unroll(full)
//    for(unsigned i = 0; i < total_calls; i++) {
//        unsigned idx = i * 32 + laneid;
//        unsigned row = idx / memcpy_per_row;
//        unsigned col = (idx*elem_per_memcpy) % src.cols;
//        *(device read_type*)(&dst_ptr[row*row_stride + col]) = *(threadgroup read_type*)(&src[int2(row, col)]);
//    }
    
    meta::unroll_i_in_range<0, total_calls * SIMD_THREADS, SIMD_THREADS>::run(meta::store<ST, memcpy_per_row, elem_per_memcpy, read_size>, dst_ptr, &src, row_stride, laneid);
}
    

    
}


