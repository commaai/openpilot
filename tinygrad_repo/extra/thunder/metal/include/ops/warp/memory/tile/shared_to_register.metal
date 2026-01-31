/**
 * @file
 * @brief Functions for transferring data directly between shared memory and registers and back.
 */
#pragma once // done!

#include "../../../../types/types.metal"
#include "../../../../common/common.metal"
#include <metal_stdlib>
namespace mittens {

// These probably need to be redone to reduce bank conflicts.
// They currently work fine with xor layout but it should be
// possible to reduce their bank conflicts with other layouts too.
//
namespace meta {

template<typename RT, typename ST>
METAL_FUNC static typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_shared_tile<ST>(), void>::type
loadStR(int i, int j, thread RT *dst, threadgroup const ST *src, short laneid, int offsetY, int offsetX) {
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    int y = offsetY + i * mittens::TILE_DIM;
    int x = offsetX + j * mittens::TILE_DIM;
    T2 values = base_types::convertor<T2, U2>::convert(*((threadgroup U2*)(&(*src)[int2(y, x)])));
    dst->tiles[i][j].data.thread_elements()[0] = values[0];
    dst->tiles[i][j].data.thread_elements()[1] = values[1];
//    
//    simdgroup_load(dst->tiles[i][j].data,
//                   (threadgroup T*)(src->data),
//                   src->cols,
//                   {i * mittens::TILE_DIM, j * mittens::TILE_DIM},
//                   
}
    
template<typename RT, typename ST>
METAL_FUNC static typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_shared_tile<ST>(), void>::type
storeStR(int i, int j, threadgroup ST *dst, thread const RT *src, short laneid, int offsetY, int offsetX) {
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    int y = offsetY + i * mittens::TILE_DIM;
    int x = offsetX + j * mittens::TILE_DIM;
    U2 values = base_types::convertor<U2, T2>::convert({src->tiles[i][j].data.thread_elements()[0], src->tiles[i][j].data.thread_elements()[1]});
    *((threadgroup U2*)(&(*dst)[int2(y, x)])) = values;
}

template<typename RT, typename ST>
METAL_FUNC static typename metal::enable_if<ducks::is_col_register_tile<RT>() && ducks::is_shared_tile<ST>(), void>::type
loadStR(int i, int j, thread RT *dst, threadgroup const ST *src, short laneid, int offsetY, int offsetX) {
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    int y = offsetY + i * mittens::TILE_DIM;
    int x = offsetX + j * mittens::TILE_DIM;
//    dst->tiles[i][j].data.thread_elements()[0] = base_types::convertor<T, U>::convert((*src)[int2(y  , x)]);
//    dst->tiles[i][j].data.thread_elements()[1] = base_types::convertor<T, U>::convert((*src)[int2(y+1, x)]);
    T2 vals = base_types::convertor<T2, U2>::convert({(*src)[int2(y  , x)], (*src)[int2(y+1, x)]});
    dst->tiles[i][j].data.thread_elements()[0] = vals[0];
    dst->tiles[i][j].data.thread_elements()[1] = vals[1];
}
    
template<typename RT, typename ST>
METAL_FUNC static typename metal::enable_if<ducks::is_col_register_tile<RT>() && ducks::is_shared_tile<ST>(), void>::type
storeStR(int i, int j, threadgroup ST *dst, thread const RT *src, short laneid, int offsetY, int offsetX) {
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    int y = offsetY + i * mittens::TILE_DIM;
    int x = offsetX + j * mittens::TILE_DIM;
//    (*dst)[int2(y  , x)] = base_types::convertor<U, T>::convert(src->tiles[i][j].data.thread_elements()[0]);
//    (*dst)[int2(y+1, x)] = base_types::convertor<U, T>::convert(src->tiles[i][j].data.thread_elements()[1]);
    
    U2 vals = base_types::convertor<U2, T2>::convert({src->tiles[i][j].data.thread_elements()[0], src->tiles[i][j].data.thread_elements()[1]});
    (*dst)[int2(y  , x)] = vals[0];
    (*dst)[int2(y+1, x)] = vals[1];
}

}
    
/**
 * @brief Load data from a shared tile into a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source shared tile.
 * @param laneid[in] Thread's index in SIMD group
 */
template<typename RT, typename ST>
METAL_FUNC static typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_shared_tile<ST>(), void>::type
load(thread RT &dst, threadgroup const ST &src, short laneid) {
    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    const short qid = laneid / 4;
    int offsetY = (qid & 4) + (laneid / 2) % 4;
    int offsetX = (qid & 2) * 2 + (laneid % 2) * 2;
//    #pragma clang loop unroll(full)
//    for(int i = 0; i < dst.height; i++) {
//        #pragma clang loop unroll(full)
//        for(int j = 0; j < dst.width; j++) {
//            int y = offsetY + i * mittens::TILE_DIM;
//            int x = offsetX + j * mittens::TILE_DIM;
//            T2 values = base_types::convertor<T2, U2>::convert(*((threadgroup U2*)(&src[int2(y, x)])));
//            dst.tiles[i][j].data.thread_elements()[0] = values[0];
//            dst.tiles[i][j].data.thread_elements()[1] = values[1];
//        }
//    }
    meta::unroll_i_j_in_range<0, RT::height, 1, 0, RT::width, 1>::run(meta::loadStR<RT, ST>, &dst, &src, laneid, offsetY, offsetX);
}
    
/**
 * @brief Load data from a shared tile into a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source shared tile.
 * @param laneid[in] Thread's index in SIMD group
 */
template<typename RT, typename ST>
METAL_FUNC static typename metal::enable_if<ducks::is_col_register_tile<RT>() && ducks::is_shared_tile<ST>(), void>::type
load(thread RT &dst, threadgroup const ST &src, short laneid) {
    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    const short qid = laneid / 4;
//    int offsetY = (qid & 4) + (laneid / 2) % 4;
//    int offsetX = (qid & 2) * 2 + (laneid % 2) * 2;
    int offsetX = (qid & 4) + (laneid / 2) % 4;
    int offsetY = (qid & 2) * 2 + (laneid % 2) * 2;
//    #pragma clang loop unroll(full)
//    for(int i = 0; i < dst.height; i++) {
//        #pragma clang loop unroll(full)
//        for(int j = 0; j < dst.width; j++) {
//            int y = offsetY + i * mittens::TILE_DIM;
//            int x = offsetX + j * mittens::TILE_DIM;
//            dst.tiles[i][j].data.thread_elements()[0] = base_types::convertor<T, U>::convert(src[int2(y  , x)]);
//            dst.tiles[i][j].data.thread_elements()[1] = base_types::convertor<T, U>::convert(src[int2(y+1, x)]);
//        }
//    }
    meta::unroll_i_j_in_range<0, RT::height, 1, 0, RT::width, 1>::run(meta::loadStR<RT, ST>, &dst, &src, laneid, offsetY, offsetX);
}

/**
 * @brief Store data into a shared tile from a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination shared tile.
 * @param src[in]  The source register tile.
 * @param laneid[in] Thread's index in SIMD group
 */
template<typename RT, typename ST>
METAL_FUNC static typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_shared_tile<ST>(), void>::type
store(threadgroup ST &dst, thread const RT &src, short laneid) {
    ducks::assert_register_tile<RT>();
    ducks::assert_shared_tile<ST>();
    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;

    const short qid = laneid / 4;
    int offsetY = (qid & 4) + (laneid / 2) % 4;
    int offsetX = (qid & 2) * 2 + (laneid % 2) * 2;
    
//    #pragma clang loop unroll(full)
//    for(int i = 0; i < src.height; i++) {
//        #pragma clang loop unroll(full)
//        for(int j = 0; j < src.width; j++) {
//            int y = offsetY + i * mittens::TILE_DIM;
//            int x = offsetX + j * mittens::TILE_DIM;
//            U2 values = base_types::convertor<U2, T2>::convert({src.tiles[i][j].data.thread_elements()[0], src.tiles[i][j].data.thread_elements()[1]});
//            *((threadgroup U2*)(&dst[int2(y, x)])) = values;
//        }
//    }
    meta::unroll_i_j_in_range<0, RT::height, 1, 0, RT::width, 1>::run(meta::storeStR<RT, ST>, &dst, &src, laneid, offsetY, offsetX);
}
    
/**
 * @brief Store data into a shared tile from a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination shared tile.
 * @param src[in]  The source register tile.
 * @param laneid[in] Thread's index in SIMD group
 */
template<typename RT, typename ST>
METAL_FUNC static typename metal::enable_if<ducks::is_col_register_tile<RT>() && ducks::is_shared_tile<ST>(), void>::type
store(threadgroup ST &dst, thread const RT &src, short laneid) {
    ducks::assert_register_tile<RT>();
    ducks::assert_shared_tile<ST>();
    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;

    const short qid = laneid / 4;
//    int offsetY = (qid & 4) + (laneid / 2) % 4;
//    int offsetX = (qid & 2) * 2 + (laneid % 2) * 2;
    int offsetX = (qid & 4) + (laneid / 2) % 4;
    int offsetY = (qid & 2) * 2 + (laneid % 2) * 2;
    
//    #pragma clang loop unroll(full)
//    for(int i = 0; i < src.height; i++) {
//        #pragma clang loop unroll(full)
//        for(int j = 0; j < src.width; j++) {
//            int y = offsetY + i * mittens::TILE_DIM;
//            int x = offsetX + j * mittens::TILE_DIM;
//            dst[int2(y  , x)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data.thread_elements()[0]);
//            dst[int2(y+1, x)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data.thread_elements()[1]);
//        }
//    }
    meta::unroll_i_j_in_range<0, RT::height, 1, 0, RT::width, 1>::run(meta::storeStR<RT, ST>, &dst, &src, laneid, offsetY, offsetX);
}
    
/*---------------------------------------------------------------------------------*/
// These probably need to be redone to reduce bank conflicts.
// They currently work fine with xor layout but it should be
// possible to reduce their bank conflicts with other layouts too.
//
namespace meta {

template<typename RT, typename ST>
METAL_FUNC static typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_shared_tile<ST>(), void>::type
loadStR_r(int i, int j, thread RT *dst, thread const ST *src, short laneid, int offsetY, int offsetX) {
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    int y = offsetY + i * mittens::TILE_DIM;
    int x = offsetX + j * mittens::TILE_DIM;
    T2 values = base_types::convertor<T2, U2>::convert(*((threadgroup U2*)(&(*src)[int2(y, x)])));
    dst->tiles[i][j].data.thread_elements()[0] = values[0];
    dst->tiles[i][j].data.thread_elements()[1] = values[1];
//
//    simdgroup_load(dst->tiles[i][j].data,
//                   (threadgroup T*)(src->data),
//                   src->cols,
//                   {i * mittens::TILE_DIM, j * mittens::TILE_DIM},
//
}
    
template<typename RT, typename ST>
METAL_FUNC static typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_shared_tile<ST>(), void>::type
storeStR_r(int i, int j, thread ST *dst, thread const RT *src, short laneid, int offsetY, int offsetX) {
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    int y = offsetY + i * mittens::TILE_DIM;
    int x = offsetX + j * mittens::TILE_DIM;
    U2 values = base_types::convertor<U2, T2>::convert({src->tiles[i][j].data.thread_elements()[0], src->tiles[i][j].data.thread_elements()[1]});
    *((threadgroup U2*)(&(*dst)[int2(y, x)])) = values;
}

template<typename RT, typename ST>
METAL_FUNC static typename metal::enable_if<ducks::is_col_register_tile<RT>() && ducks::is_shared_tile<ST>(), void>::type
loadStR_c(int i, int j, thread RT *dst, thread const ST *src, short laneid, int offsetY, int offsetX) {
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    int y = offsetY + i * mittens::TILE_DIM;
    int x = offsetX + j * mittens::TILE_DIM;
//    dst->tiles[i][j].data.thread_elements()[0] = base_types::convertor<T, U>::convert((*src)[int2(y  , x)]);
//    dst->tiles[i][j].data.thread_elements()[1] = base_types::convertor<T, U>::convert((*src)[int2(y+1, x)]);
    T2 vals = base_types::convertor<T2, U2>::convert({(*src)[int2(y  , x)], (*src)[int2(y+1, x)]});
    dst->tiles[i][j].data.thread_elements()[0] = vals[0];
    dst->tiles[i][j].data.thread_elements()[1] = vals[1];
}
    
template<typename RT, typename ST>
METAL_FUNC static typename metal::enable_if<ducks::is_col_register_tile<RT>() && ducks::is_shared_tile<ST>(), void>::type
storeStR_c(int i, int j, thread ST *dst, thread const RT *src, short laneid, int offsetY, int offsetX) {
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    int y = offsetY + i * mittens::TILE_DIM;
    int x = offsetX + j * mittens::TILE_DIM;
//    (*dst)[int2(y  , x)] = base_types::convertor<U, T>::convert(src->tiles[i][j].data.thread_elements()[0]);
//    (*dst)[int2(y+1, x)] = base_types::convertor<U, T>::convert(src->tiles[i][j].data.thread_elements()[1]);
    
    U2 vals = base_types::convertor<U2, T2>::convert({src->tiles[i][j].data.thread_elements()[0], src->tiles[i][j].data.thread_elements()[1]});
    (*dst)[int2(y  , x)] = vals[0];
    (*dst)[int2(y+1, x)] = vals[1];
}

}
    
/**
 * @brief Load data from a shared tile into a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source shared tile.
 * @param laneid[in] Thread's index in SIMD group
 */
template<typename RT, typename ST>
METAL_FUNC static typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_shared_tile<ST>(), void>::type
load(thread RT &dst, thread const ST &src, short laneid) {
    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    const short qid = laneid / 4;
    int offsetY = (qid & 4) + (laneid / 2) % 4;
    int offsetX = (qid & 2) * 2 + (laneid % 2) * 2;
//    #pragma clang loop unroll(full)
//    for(int i = 0; i < dst.height; i++) {
//        #pragma clang loop unroll(full)
//        for(int j = 0; j < dst.width; j++) {
//            int y = offsetY + i * mittens::TILE_DIM;
//            int x = offsetX + j * mittens::TILE_DIM;
//            T2 values = base_types::convertor<T2, U2>::convert(*((threadgroup U2*)(&src[int2(y, x)])));
//            dst.tiles[i][j].data.thread_elements()[0] = values[0];
//            dst.tiles[i][j].data.thread_elements()[1] = values[1];
//        }
//    }
    meta::unroll_i_j_in_range<0, RT::height, 1, 0, RT::width, 1>::run(meta::loadStR_r<RT, ST>, &dst, &src, laneid, offsetY, offsetX);
}
    
/**
 * @brief Load data from a shared tile into a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source shared tile.
 * @param laneid[in] Thread's index in SIMD group
 */
template<typename RT, typename ST>
METAL_FUNC static typename metal::enable_if<ducks::is_col_register_tile<RT>() && ducks::is_shared_tile<ST>(), void>::type
load(thread RT &dst, thread const ST &src, short laneid) {
    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    const short qid = laneid / 4;
//    int offsetY = (qid & 4) + (laneid / 2) % 4;
//    int offsetX = (qid & 2) * 2 + (laneid % 2) * 2;
    int offsetX = (qid & 4) + (laneid / 2) % 4;
    int offsetY = (qid & 2) * 2 + (laneid % 2) * 2;
//    #pragma clang loop unroll(full)
//    for(int i = 0; i < dst.height; i++) {
//        #pragma clang loop unroll(full)
//        for(int j = 0; j < dst.width; j++) {
//            int y = offsetY + i * mittens::TILE_DIM;
//            int x = offsetX + j * mittens::TILE_DIM;
//            dst.tiles[i][j].data.thread_elements()[0] = base_types::convertor<T, U>::convert(src[int2(y  , x)]);
//            dst.tiles[i][j].data.thread_elements()[1] = base_types::convertor<T, U>::convert(src[int2(y+1, x)]);
//        }
//    }
    meta::unroll_i_j_in_range<0, RT::height, 1, 0, RT::width, 1>::run(meta::loadStR_c<RT, ST>, &dst, &src, laneid, offsetY, offsetX);
}

/**
 * @brief Store data into a shared tile from a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination shared tile.
 * @param src[in]  The source register tile.
 * @param laneid[in] Thread's index in SIMD group
 */
template<typename RT, typename ST>
METAL_FUNC static typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_shared_tile<ST>(), void>::type
store(thread ST &dst, thread const RT &src, short laneid) {
    ducks::assert_register_tile<RT>();
    ducks::assert_shared_tile<ST>();
    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;

    const short qid = laneid / 4;
    int offsetY = (qid & 4) + (laneid / 2) % 4;
    int offsetX = (qid & 2) * 2 + (laneid % 2) * 2;
    
//    #pragma clang loop unroll(full)
//    for(int i = 0; i < src.height; i++) {
//        #pragma clang loop unroll(full)
//        for(int j = 0; j < src.width; j++) {
//            int y = offsetY + i * mittens::TILE_DIM;
//            int x = offsetX + j * mittens::TILE_DIM;
//            U2 values = base_types::convertor<U2, T2>::convert({src.tiles[i][j].data.thread_elements()[0], src.tiles[i][j].data.thread_elements()[1]});
//            *((threadgroup U2*)(&dst[int2(y, x)])) = values;
//        }
//    }
    meta::unroll_i_j_in_range<0, RT::height, 1, 0, RT::width, 1>::run(meta::storeStR_r<RT, ST>, &dst, &src, laneid, offsetY, offsetX);
}
    
/**
 * @brief Store data into a shared tile from a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination shared tile.
 * @param src[in]  The source register tile.
 * @param laneid[in] Thread's index in SIMD group
 */
template<typename RT, typename ST>
METAL_FUNC static typename metal::enable_if<ducks::is_col_register_tile<RT>() && ducks::is_shared_tile<ST>(), void>::type
store(thread ST &dst, thread const RT &src, short laneid) {
    ducks::assert_register_tile<RT>();
    ducks::assert_shared_tile<ST>();
    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;

    const short qid = laneid / 4;
//    int offsetY = (qid & 4) + (laneid / 2) % 4;
//    int offsetX = (qid & 2) * 2 + (laneid % 2) * 2;
    int offsetX = (qid & 4) + (laneid / 2) % 4;
    int offsetY = (qid & 2) * 2 + (laneid % 2) * 2;
    
//    #pragma clang loop unroll(full)
//    for(int i = 0; i < src.height; i++) {
//        #pragma clang loop unroll(full)
//        for(int j = 0; j < src.width; j++) {
//            int y = offsetY + i * mittens::TILE_DIM;
//            int x = offsetX + j * mittens::TILE_DIM;
//            dst[int2(y  , x)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data.thread_elements()[0]);
//            dst[int2(y+1, x)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data.thread_elements()[1]);
//        }
//    }
    meta::unroll_i_j_in_range<0, RT::height, 1, 0, RT::width, 1>::run(meta::storeStR_c<RT, ST>, &dst, &src, laneid, offsetY, offsetX);
}
    
}


