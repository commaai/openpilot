/**
 * @file
 * @brief Conversions on vectors stored in registers.
 */

struct vec_conversion_detail {

// i am not smart enough to figure out these indices without these helpers :/
// again, blame nvidia for these stupid, stupid layouts
__device__ static inline int row_from_indices_dim2(int laneid, int inner_dim, int x_or_y) {
    return 8*inner_dim + (laneid%4)*2 + x_or_y;
}
__device__ static inline int row_from_indices_dim1(int laneid, int x_or_y) {
    return 8*x_or_y + (laneid/4);
}
__device__ static inline int canonical_src_lane_dim2(int row) {
    return (row/2)%4 + 4*(row%2); // draw even rows from 0...3 and odds from 4...7
}
__device__ static inline int canonical_src_lane_dim1(int row) {
    return (row*4)%32;
}

};

/**
 * @brief Copies data from one register vector to another.
 *
 * @tparam RV1 The type of the destination register vector.
 * @tparam RV2 The type of the source register vector.
 * @param dst[out] The destination register vector.
 * @param src[in] The source register vector to copy from.
 */
template<ducks::rv::all RV1, ducks::rv::all RV2>
__device__ static inline void copy(RV1 &dst, const RV2 &src) {
    KITTENS_CHECK_WARP
    static_assert(RV1::length == RV2::length, "Register vectors must be the same length.");
    using D1 = RV1::dtype;
    using D2 = RV2::dtype;
    if constexpr (std::is_same_v<typename RV1::layout, typename RV2::layout>) { // just a simple copy / typecast
        #pragma unroll
        for(int i = 0; i < RV1::outer_dim; i++) {
            #pragma unroll
            for(int j = 0; j < RV1::inner_dim; j++) {
                dst[i][j] = base_types::convertor<D1, D2>::convert(src[i][j]);
            }
        }
    }
    else { // Inner dimensions are not the same, this is really a layout conversion.
        int laneid = ::kittens::laneid();
        if constexpr (std::is_same_v<typename RV1::layout, ortho_l> && std::is_same_v<typename RV2::layout, align_l>) { // align -> ortho layout
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                dst[i][0].x = packed_shfl_sync(
                    kittens::MASK_ALL,
                    laneid < 4 ? src[i][0].x : src[i][0].y, // mirrors canonical_src_lane_dim2
                    vec_conversion_detail::canonical_src_lane_dim2(vec_conversion_detail::row_from_indices_dim1(laneid, 0))
                );
                dst[i][0].y = packed_shfl_sync(
                    kittens::MASK_ALL,
                    laneid < 4 ? src[i][1].x : src[i][1].y, // mirrors canonical_src_lane_dim2
                    vec_conversion_detail::canonical_src_lane_dim2(vec_conversion_detail::row_from_indices_dim1(laneid, 1))
                );
            }
        }
        else if constexpr (std::is_same_v<typename RV1::layout, align_l> && std::is_same_v<typename RV2::layout, ortho_l>) { // ortho -> align layout
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                dst[i][0].x = packed_shfl_sync(
                    kittens::MASK_ALL,
                    src[i][0].x, // first 8 rows
                    vec_conversion_detail::canonical_src_lane_dim1(vec_conversion_detail::row_from_indices_dim2(laneid, 0, 0))
                );
                dst[i][0].y = packed_shfl_sync(
                    kittens::MASK_ALL,
                    src[i][0].x, // first 8 rows
                    vec_conversion_detail::canonical_src_lane_dim1(vec_conversion_detail::row_from_indices_dim2(laneid, 0, 1))
                );
                dst[i][1].x = packed_shfl_sync(
                    kittens::MASK_ALL,
                    src[i][0].y, // last 8 rows
                    vec_conversion_detail::canonical_src_lane_dim1(vec_conversion_detail::row_from_indices_dim2(laneid, 1, 0))
                );
                dst[i][1].y = packed_shfl_sync(
                    kittens::MASK_ALL,
                    src[i][0].y, // last 8 rows
                    vec_conversion_detail::canonical_src_lane_dim1(vec_conversion_detail::row_from_indices_dim2(laneid, 1, 1))
                );
            }
        }
        else if constexpr (std::is_same_v<typename RV1::layout, ortho_l> && std::is_same_v<typename RV2::layout, naive_l>) { // naive -> ortho layout
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                dst[i][0].x = packed_shfl_sync(
                    kittens::MASK_ALL, src[i/2][0],
                    16*(i%2) + 0 + (laneid/4)
                );
                dst[i][0].y = packed_shfl_sync(
                    kittens::MASK_ALL, src[i/2][0],
                    16*(i%2) + 8 + (laneid/4)
                );
            }
        }
        else if constexpr (std::is_same_v<typename RV1::layout, naive_l> && std::is_same_v<typename RV2::layout, ortho_l>) { // ortho -> naive layout
            int lane_replication = laneid%4; // 0...3
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                D1 tmp = 0;
                if(RV1::length%32==0 || i < RV1::outer_dim-1 || lane_replication<2) {
                    tmp = lane_replication%2 ? src[2*i + (lane_replication>=2)][0].y : src[2*i + (lane_replication>=2)][0].x;
                }
                dst[i][0] = packed_shfl_sync(
                    kittens::MASK_ALL, tmp,
                    (laneid%8)*4 + (laneid/8)
                );
            }
        }
        else if constexpr (std::is_same_v<typename RV1::layout, align_l> && std::is_same_v<typename RV2::layout, naive_l>) { // naive -> align layout
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                dst[i][0].x = packed_shfl_sync(
                    kittens::MASK_ALL, src[i/2][0],
                    16*(i%2) + 0 + 2*(laneid%4) + 0
                );
                dst[i][0].y = packed_shfl_sync(
                    kittens::MASK_ALL, src[i/2][0],
                    16*(i%2) + 0 + 2*(laneid%4) + 1
                );
                dst[i][1].x = packed_shfl_sync(
                    kittens::MASK_ALL, src[i/2][0],
                    16*(i%2) + 8 + 2*(laneid%4) + 0
                );
                dst[i][1].y = packed_shfl_sync(
                    kittens::MASK_ALL, src[i/2][0],
                    16*(i%2) + 8 + 2*(laneid%4) + 1
                );
            }
        }
        else if constexpr (std::is_same_v<typename RV1::layout, naive_l> && std::is_same_v<typename RV2::layout, align_l>) { // align -> naive layout
            int lane_replication = laneid/8; // 0...3
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                D1 tmp = 0;
                if(RV1::length%32==0 || i < RV1::outer_dim-1 || laneid<16) {
                    tmp = (laneid%8)<4 ? src[2*i + (lane_replication>=2)][lane_replication%2].x : src[2*i + (lane_replication>=2)][lane_replication%2].y;
                }
                dst[i][0] = packed_shfl_sync(
                    kittens::MASK_ALL, tmp,
                    4*(laneid%2) + (laneid%8)/2 + (laneid&0b11000)
                );
            }
        }
    }
}