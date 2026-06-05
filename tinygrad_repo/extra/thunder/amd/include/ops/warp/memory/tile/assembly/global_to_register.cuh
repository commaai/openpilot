/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

 #pragma once

 #include "../../../../../common/common.cuh"
 #include "../../../../../types/types.cuh"
 #include "../../util/util.cuh"
 
 namespace kittens {
 
 /**
  * @brief Load data from a source array into a row-major layout tile.
  *
  * @tparam RT The row-major layout tile type.
  * @tparam U The data type of the source array.
  * @param dst[out] The destination tile to load data into.
  * @param src[in] The source array to load data from.
  * @param idx[in] The index of the tile to load data from.
  */
 
 template<int axis, ducks::art::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
 __device__ inline static void load(RT &dst, const GL &src, const COORD &idx, const COORD &warp_idx) {
     using T2 = RT::dtype;
     constexpr int packing = base_types::packing<typename RT::dtype>::num();
     using U = typename GL::dtype;
     using U2 = base_types::packing<U>::packed_type;
     static_assert(std::is_same_v<typename RT::layout, ducks::rt_layout::row>, "RT must be a row-major layout tile");
 
     U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
     const int row_stride = src.template stride<axis>();
     int laneid = kittens::laneid();
 
     int row_offset = laneid%(dst.base_tile_rows);
     int col_offset = dst.base_tile_stride*(laneid/dst.base_tile_rows);
 
     uint32_t buffer_size = src.batch() * src.depth() * src.rows() * src.cols() * sizeof(U);
     std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(src_ptr);
     std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);    // widen if host is 32-bit
     buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
 
     int warp_offset = src.idx(warp_idx.template unit_coord<axis, 3>());
 
     // Compile-time loop to load data into the tile
     auto perform_load_at = [&]<int N, int M, int K>() {
         using tile_range = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
         const int register_offset = K * RT::registers_per_stride;
 
         const int col = RT::base_tile_cols*M + col_offset + K * RT::base_tile_elements_per_stride_group;
         const int row = RT::base_tile_rows*N + row_offset;
         const int offset = (row*row_stride + col + warp_offset) * sizeof(U);
 
         if constexpr (std::is_same_v<U2, bf16_2>) {
             if constexpr (RT::base_tile_stride == 8) {
                 macros::buffer_load_dwordx4<tile_range::lo + register_offset>(br, offset);
             } else if constexpr (RT::base_tile_stride == 4) {
                 macros::buffer_load_dwordx2<tile_range::lo + register_offset>(br, offset);
             }
         }
     };
 
     [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
         ([&]<std::size_t N>() {
             [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                 ([&]<std::size_t M>() {
                     [&]<std::size_t... Ks>(std::index_sequence<Ks...>) {
                         ([&]<std::size_t K>() {
                             perform_load_at.template operator()<N, M, K>();
                         }.template operator()<Ks>(), ...);
                     }(std::make_index_sequence<RT::base_tile_num_strides>{});
                 }.template operator()<Ms>(), ...);
             }(std::make_index_sequence<RT::width>{});
         }.template operator()<Ns>(), ...);
     }(std::make_index_sequence<RT::height>{});
     
 }
 
 template<ducks::art::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
 __device__ inline static void load(RT &dst, const GL &src, const COORD &idx, const COORD &warp_idx) {
     load<2, RT, GL>(dst, src, idx, warp_idx);
 }
 
 /**
  * @brief Store data from a register tile to a destination array in global memory with a row-major layout.
  *
  * @tparam RT The register tile type with a row-major layout.
  * @tparam U The data type of the destination array.
  * @param[out] dst The destination array in global memory to store data into.
  * @param[in] src The source register tile to store data from.
  * @param row_stride[in] The stride in elements between rows in the destination array.
  */
 template<int axis, ducks::art::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
 __device__ inline static void store(const GL &dst, const RT &src, const COORD &idx, const COORD &warp_idx) {
     using T2 = RT::dtype;
     using T = base_types::packing<typename RT::dtype>::unpacked_type;
     using U = typename GL::dtype;
     using U2 = base_types::packing<U>::packed_type;
 
     U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
     const int row_stride = dst.template stride<axis>();
     int laneid = kittens::laneid();
 
     std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(dst_ptr);
     std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);    // widen if host is 32-bit
     buffer_resource br = make_buffer_resource(as_u64, 0xFFFFFFFF, 0x00020000);
 
     int warp_offset = dst.idx(warp_idx.template unit_coord<axis, 3>());

     if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
        const int row_offset = laneid%(src.base_tile_rows);
        const int col_offset = src.base_tile_stride*(laneid/src.base_tile_rows);
    
        if constexpr ((std::is_same_v<typename RT::shape, rt_32x16_s> || std::is_same_v<typename RT::shape, rt_16x32_s>) && std::is_same_v<typename RT::T, bf16>) {
            auto perform_store_at = [&]<int N, int M>() {
                using tile_range = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
                static_assert(tile_range::lo + 3 == tile_range::hi, "tile_range must be 4 contiguous registers");
    
                const int col = RT::base_tile_cols*M + col_offset;
                const int row = RT::base_tile_rows*N + row_offset;
    
                constexpr int GPR_START = tile_range::lo;
    
                macros::buffer_store_dwordx4<GPR_START>(br, (row*row_stride + col + warp_offset) * sizeof(U));
            };
    
            [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
                ([&]<std::size_t N>() {
                    [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                        ([&]<std::size_t M>() {
                            perform_store_at.template operator()<N, M>();
                        }.template operator()<Ms>(), ...);
                    }(std::make_index_sequence<RT::width>{});
                }.template operator()<Ns>(), ...);
            }(std::make_index_sequence<RT::height>{});
        } else if constexpr (std::is_same_v<typename RT::shape, rt_32x32_s> || std::is_same_v<typename RT::shape, rt_16x16_s>) {
            auto perform_store_at = [&]<int N, int M>() {
                using tile_range = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
        
                const int col = RT::base_tile_cols*M + col_offset;
                const int row = RT::base_tile_rows*N + row_offset;
        
        
                if constexpr (std::is_same_v<typename RT::shape, rt_32x32_s>) {
        
                    if constexpr (std::is_same_v<U, bf16> && std::is_same_v<T, float>) {
        
                        constexpr int GPR_START_0 = tile_range::lo;
                        constexpr int GPR_START_1 = tile_range::lo + 4;
                        constexpr int GPR_START_2 = tile_range::lo + 8;
                        constexpr int GPR_START_3 = tile_range::lo + 12;
        
                        constexpr int GPR_START_0_BF16 = tile_range::lo;
                        constexpr int GPR_START_1_BF16 = tile_range::lo + 2;
                        constexpr int GPR_START_2_BF16 = tile_range::lo + 4;
                        constexpr int GPR_START_3_BF16 = tile_range::lo + 6;
        
                        macros::v_cvt_pk_bf16_f32<GPR_START_0_BF16, GPR_START_0, GPR_START_0 + 1>();
                        macros::v_cvt_pk_bf16_f32<GPR_START_0_BF16 + 1, GPR_START_0 + 2, GPR_START_0 + 3>();
                        macros::v_cvt_pk_bf16_f32<GPR_START_1_BF16, GPR_START_1, GPR_START_1 + 1>();
                        macros::v_cvt_pk_bf16_f32<GPR_START_1_BF16 + 1, GPR_START_1 + 2, GPR_START_1 + 3>();
                        macros::v_cvt_pk_bf16_f32<GPR_START_2_BF16, GPR_START_2, GPR_START_2 + 1>();
                        macros::v_cvt_pk_bf16_f32<GPR_START_2_BF16 + 1, GPR_START_2 + 2, GPR_START_2 + 3>();
                        macros::v_cvt_pk_bf16_f32<GPR_START_3_BF16, GPR_START_3, GPR_START_3 + 1>();
                        macros::v_cvt_pk_bf16_f32<GPR_START_3_BF16 + 1, GPR_START_3 + 2, GPR_START_3 + 3>();
        
                        macros::buffer_store_dwordx2<GPR_START_0_BF16>(br, (row*row_stride + col + warp_offset) * sizeof(U));
                        macros::buffer_store_dwordx2<GPR_START_1_BF16>(br, (row*row_stride + col + 8 + warp_offset) * sizeof(U));
                        macros::buffer_store_dwordx2<GPR_START_2_BF16>(br, (row*row_stride + col + 16 + warp_offset) * sizeof(U));
                        macros::buffer_store_dwordx2<GPR_START_3_BF16>(br, (row*row_stride + col + 24 + warp_offset) * sizeof(U));
                    } else if constexpr (std::is_same_v<U, bf16> && std::is_same_v<T, bf16>) {
                        constexpr int GPR_START_0_BF16 = tile_range::lo;
                        constexpr int GPR_START_1_BF16 = tile_range::lo + 2;
                        constexpr int GPR_START_2_BF16 = tile_range::lo + 4;
                        constexpr int GPR_START_3_BF16 = tile_range::lo + 6;
        
                        macros::buffer_store_dwordx2<GPR_START_0_BF16>(br, (row*row_stride + col + warp_offset) * sizeof(U));
                        macros::buffer_store_dwordx2<GPR_START_1_BF16>(br, (row*row_stride + col + 8 + warp_offset) * sizeof(U));
                        macros::buffer_store_dwordx2<GPR_START_2_BF16>(br, (row*row_stride + col + 16 + warp_offset) * sizeof(U));
                        macros::buffer_store_dwordx2<GPR_START_3_BF16>(br, (row*row_stride + col + 24 + warp_offset) * sizeof(U));
                    } else {
                        static_assert(false, "Unsupported dtype");
                    }
                } else {
                    if constexpr (std::is_same_v<U, bf16> && std::is_same_v<T, bf16>) {
                        constexpr int GPR_START_0_BF16 = tile_range::lo;
                        macros::buffer_store_dwordx2<GPR_START_0_BF16>(br, (row*row_stride + col + warp_offset) * sizeof(U));
                    }
                }
                
            };
        
            [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
                ([&]<std::size_t N>() {
                    [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                        ([&]<std::size_t M>() {
                            perform_store_at.template operator()<N, M>();
                        }.template operator()<Ms>(), ...);
                    }(std::make_index_sequence<RT::width>{});
                }.template operator()<Ns>(), ...);
            }(std::make_index_sequence<RT::height>{});
        } else {
            static_assert(false, "Unsupported tile shape");
        }
    } else {
        int col_offset = laneid%(src.base_tile_cols);
        int row_offset = laneid/(src.base_tile_cols)*RT::base_tile_stride;
    
        if constexpr (std::is_same_v<typename RT::shape, rt_16x16_s> && std::is_same_v<T, float> && std::is_same_v<U, float>) {
            // Compile-tile nested loops over RT::height and RT::width
            auto perform_store_at = [&]<int N, int M>() {
                using tile_range = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
    
                const int col = RT::base_tile_cols*M + col_offset;
                const int row = RT::base_tile_rows*N + row_offset;
    
                const uint32_t byte_offset_0 = static_cast<uint32_t>((row * row_stride + col + warp_offset) * sizeof(U));
                const uint32_t byte_offset_1 = static_cast<uint32_t>(((row + 1)*row_stride + col + warp_offset) * sizeof(U));
                const uint32_t byte_offset_2 = static_cast<uint32_t>(((row + 2)*row_stride + col + warp_offset) * sizeof(U));
                const uint32_t byte_offset_3 = static_cast<uint32_t>(((row + 3)*row_stride + col + warp_offset) * sizeof(U));
    
                static_assert(tile_range::lo + 3 == tile_range::hi, "tile_range must be contiguous");
                constexpr int R0 = tile_range::lo;
                constexpr int R1 = tile_range::lo + 1;
                constexpr int R2 = tile_range::lo + 2;
                constexpr int R3 = tile_range::lo + 3;
    
                macros::buffer_store_dword<R0>(br, byte_offset_0);
                macros::buffer_store_dword<R1>(br, byte_offset_1);
                macros::buffer_store_dword<R2>(br, byte_offset_2);
                macros::buffer_store_dword<R3>(br, byte_offset_3);
            };
    
            [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
                ([&]<std::size_t N>() {
                    [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                        ([&]<std::size_t M>() {
                            perform_store_at.template operator()<N, M>();
                        }.template operator()<Ms>(), ...);
                    }(std::make_index_sequence<RT::width>{});
                }.template operator()<Ns>(), ...);
            }(std::make_index_sequence<RT::height>{});
        } else {
            static_assert(false, "Unsupported tile shape and dtype");
        }
    }
 
 }
 
 template<ducks::art::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
 __device__ inline static void store(const GL &dst, const RT &src, const COORD &idx, const COORD &warp_idx) {
     store<2, RT, GL, COORD>(dst, src, idx, warp_idx);
 }
 
 }