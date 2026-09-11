/**
 * @file
 * @brief Functions for a group to collaboratively transfer data directly between shared memory and registers and back.
 */

/**
 * @brief Collaboratively load data from a shared vector into register vectors split across a warpgroup.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination register vector.
 * @param src[in]  The source shared vector.
 */
template<ducks::rv::all RV, ducks::sv::all SV>
__device__ inline static void load(RV &dst, const SV &src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;
    if constexpr (GROUP_WARPS == 1) {
        static_assert(SV::length == RV::length);
        
        int laneid = ::kittens::laneid();
        uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
        
        __syncwarp();
        if constexpr (std::is_same_v<typename RV::layout, align_l>) {
            #pragma unroll
            for(auto w = 0; w < (dst.outer_dim+3)/4; w++) {
                int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
                int o_dim = w*4 + (laneid/4) / 2;
                int i_dim = (laneid/4) % 2;
                // this should be a maximally coalesced load.
                if(idx < dst.outer_dim*16) {
                    U2 tmp;
                    move<U2>::lds(tmp, src_ptr + sizeof(typename SV::dtype)*idx);
                    dst[o_dim][i_dim] = base_types::convertor<T2, U2>::convert(tmp);
                }
            }
            __syncwarp();
            // now we need to do a bunch of shuffle_sync's to make sure everyone has everything they need.
            #pragma unroll
            for(auto w = 0; w < dst.outer_dim; w++) {
                int leader = 8*(w%4) + (laneid%4); // repeats every 64 columns
                dst[w][0] = packed_shfl_sync(MASK_ALL, dst[w][0], leader);
                dst[w][1] = packed_shfl_sync(MASK_ALL, dst[w][1], leader+4);
            }
        }
        else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
            // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
            // otherwise there will be some pain :/
            #pragma unroll
            for(auto w = 0; w < (dst.outer_dim+1)/2; w++) {
                int idx = w*32 + (laneid%4)*8 + (laneid/4);
                int o_dim = w*2 + (laneid%4) / 2;
                // this should be a maximally coalesced load.
                if(idx < dst.outer_dim*16) {
                    U tmp;
                    move<U>::lds(tmp, src_ptr + sizeof(typename SV::dtype)*idx);
                    if(laneid%2==0) dst[o_dim][0].x =  base_types::convertor<T, U>::convert(tmp);
                    else dst[o_dim][0].y = base_types::convertor<T, U>::convert(tmp);
                }
            }
            __syncwarp();
            // now we need to do a bunch of shuffle_sync's to make sure everyone has everything they need.
            #pragma unroll
            for(auto w = 0; w < dst.outer_dim; w++) {
                int leader = (laneid/4)*4 + 2*(w%2); // repeats every 64 columns
                dst[w][0].x = __shfl_sync(MASK_ALL, dst[w][0].x, leader);
                dst[w][0].y = __shfl_sync(MASK_ALL, dst[w][0].y, leader+1);
            }
        }
        else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
            #pragma unroll
            for(auto w = 0; w < dst.outer_dim; w++) {
                if(w < dst.outer_dim-1 || RV::length%32 == 0 || laneid<16) {
                    U tmp;
                    move<U>::lds(tmp, src_ptr + sizeof(typename SV::dtype)*(w*32 + laneid));
                    dst[w][0] = base_types::convertor<T, U>::convert(tmp);
                }
            }
        }
    }
    else {
        static_assert(SV::length == RV::length*GROUP_WARPS);// confirm size correct
        auto &_src = src.template subvec<RV::length>(warpid()); // pretend it's smaller and do warp-level load

        ::kittens::group<1>::load(dst, _src); // warp-level
    }
}

/**
 * @brief Collaboratively store data into a shared vector from register vectors split across a warpgroup.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination shared vector.
 * @param src[in]  The source register vector.
 */
template<ducks::sv::all SV, ducks::rv::all RV>
__device__ inline static void store(SV &dst, const RV &src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    if constexpr (GROUP_WARPS == 1) {
        static_assert(SV::length == RV::length);
        
        int laneid = ::kittens::laneid();
        uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));

        __syncwarp();
        if constexpr (std::is_same_v<typename RV::layout, align_l>) {
            #pragma unroll
            for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
                int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
                int o_dim = w*4 + (laneid/4) / 2;
                int i_dim = (laneid/4) % 2;
                // this should be a maximally coalesced store. I hope!
                if(idx < src.outer_dim*16) {
                    U2 tmp = base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]);
                    move<U2>::sts(dst_ptr + sizeof(typename SV::dtype)*idx, tmp);
                }
            }
        }
        else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
            // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
            // otherwise there will be some pain :/
            #pragma unroll
            for(auto w = 0; w < (src.outer_dim+1)/2; w++) {
                int idx = w*32 + (laneid%4)*8 + (laneid/4);
                int o_dim = w*2 + (laneid%4) / 2;
                // this should be a maximally coalesced load.
                if(idx < src.outer_dim*16) {
                    U tmp;
                    if(laneid%2==0) tmp = base_types::convertor<U, T>::convert(src[o_dim][0].x);
                    else tmp = base_types::convertor<U, T>::convert(src[o_dim][0].y);
                    move<U>::sts(dst_ptr + sizeof(typename SV::dtype)*idx, tmp);
                }
            }
        }
        else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
            #pragma unroll
            for(auto w = 0; w < src.outer_dim; w++) {
                if(w < src.outer_dim-1 || RV::length%32 == 0 || laneid<16) {
                    U tmp = base_types::convertor<U, T>::convert(src[w][0]);
                    move<U>::sts(dst_ptr + sizeof(typename SV::dtype)*(w*32 + laneid), tmp);
                }
            }
        }
    }
    else {
        static_assert(SV::length == RV::length*GROUP_WARPS);// confirm size correct
        auto &_dst = dst.template subvec<RV::length>(warpid()); // pretend it's smaller and do warp-level load

        ::kittens::group<1>::store(_dst, src); // warp-level
    }
}