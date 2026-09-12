/**
 * @file
 * @brief Functions for transferring data directly between shared memory and registers and back.
 */

 #pragma once

 #include <type_traits>
 
 #include "../../../../../common/common.cuh"
 #include "../../../../../types/types.cuh"
 #include "../../util/util.cuh"
 
 namespace kittens {
 // These probably need to be redone to reduce bank conflicts.
 // They currently work fine with xor layout but it should be
 // possible to reduce their bank conflicts with other layouts too.
 
 /**
  * @brief Load data from a shared tile into a register tile.
  *
  * @tparam RT The register tile type
  * @tparam ST The shared tile type
  * @param dst[out] The destination register tile.
  * @param src[in]  The source shared tile.
  */
 
 template<ducks::art::all RT, ducks::st::all ST>
 __device__ inline static uint32_t get_address(RT &dst, const ST &src) {

     if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
 
        static_assert(RT::rows == ST::rows, "register tile and shared tile must match rows");
        static_assert(RT::cols == ST::cols,  "register tile and shared tile must match cols");
    
        using T2 = RT::dtype;
        using T  = base_types::packing<T2>::unpacked_type;
        using U  = ST::dtype;
        using U2 = base_types::packing<U >::packed_type;
        constexpr int packing = base_types::packing<typename RT::dtype>::num();
    
        const int laneid = kittens::laneid();
    
        const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
    
        if constexpr (std::is_same_v<typename ST::shape, st_16x16_s> && std::is_same_v<typename RT::shape, rt_16x32_s>) {
            const int shared_tile_stride = 16 * 16 * sizeof(U);
    
            const int shared_tile_offset = (laneid / 32) * shared_tile_stride;
            const int lane_col_offset = ((laneid % 32) / 16) * 8;
            const int lane_row_offset = (laneid % 16);
    
            const int lane_byte_offset = (lane_row_offset * 16 + lane_col_offset) * sizeof(U) + shared_tile_offset;
            const uint32_t addr = src_ptr + lane_byte_offset;
    
            return addr;
        } else if constexpr (std::is_same_v<typename ST::shape, st_16x32_s> && std::is_same_v<typename RT::shape, rt_16x32_s>) {
            const int lane_col_offset = (laneid / 16) * 8;
            const int lane_row_offset = (laneid % 16);
    
            const int lane_byte_offset = (lane_row_offset * 32 + lane_col_offset) * sizeof(U);
            const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
            const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
    
            return addr;
        } else {
            static_assert(false, "Unsupported shape");
        }
    } else {

        static_assert(RT::rows == ST::rows, "register tile and shared tile must match rows");
        static_assert(RT::cols == ST::cols,  "register tile and shared tile must match cols");
    
        using T2 = RT::dtype;
        using T  = base_types::packing<T2>::unpacked_type;
        using U  = ST::dtype;
        using U2 = base_types::packing<U >::packed_type;
        constexpr int packing = base_types::packing<typename RT::dtype>::num();
    
        const int laneid = kittens::laneid();
    
        const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
        
        if constexpr (std::is_same_v<typename ST::shape, st_16x16_swizzled_s> && std::is_same_v<typename RT::shape, rt_32x16_4_s>) {
            const int row_offset = (laneid % 16) / 4 + (laneid / 16) * 4;
            const int col_offset = ((laneid % 4) * 4);
            const int lane_byte_offset = (row_offset * 16 + col_offset) * sizeof(U);
            const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 7) << 3);
    
            const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
            return addr;
        } else if constexpr (std::is_same_v<typename ST::shape, st_16x16_s> && std::is_same_v<typename RT::shape, rt_32x16_4_s>) {
            const int row_offset = (laneid % 16) / 4 + (laneid / 16) * 4;
            const int col_offset = ((laneid % 4) * 4);
            const int lane_byte_offset = (row_offset * 16 + col_offset) * sizeof(U);
    
            const uint32_t addr = src_ptr + lane_byte_offset;
            return addr;
        } else if constexpr (std::is_same_v<typename ST::shape, st_16x32_s> && std::is_same_v<typename RT::shape, rt_16x32_s>) {
            const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
            const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
            const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(U);
            const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
    
            const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
            return addr;
        } else {
            static_assert(false, "Unsupported shape");
        }
    }
 }
 
 template<int N, int M, ducks::art::all RT, ducks::st::all ST>
 __device__ inline static void load(RT &dst, const ST &src, uint32_t addr) {
 
    if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
        static_assert(RT::rows == ST::rows, "register tile and shared tile must match rows");
        static_assert(RT::cols == ST::cols,  "register tile and shared tile must match cols");
    
        using T2 = RT::dtype;
        using T  = base_types::packing<T2>::unpacked_type;
        using U  = ST::dtype;
        using U2 = base_types::packing<U >::packed_type;
        constexpr int packing = base_types::packing<typename RT::dtype>::num();
    
        if constexpr (std::is_same_v<typename ST::shape, st_16x16_s> && std::is_same_v<typename RT::shape, rt_16x32_s>) {
            constexpr int shared_tile_stride = 16 * 16 * sizeof(U);
            constexpr int row_stride = shared_tile_stride * ST::underlying_subtiles_per_row;
            constexpr int register_tile_stride = shared_tile_stride * 2;
    
            using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
            static_assert(range_type::lo + 3 == range_type::hi, "ds_read_b128 requires 4 consecutive registers");
            
            constexpr int offset = N * row_stride + M * register_tile_stride;
            macros::ds_read_b128<range_type::lo>(addr, offset);
        } else if constexpr (std::is_same_v<typename ST::shape, st_16x32_s> && std::is_same_v<typename RT::shape, rt_16x32_s>) {
            constexpr int tile_stride = 16 * 32 * sizeof(U);
            constexpr int row_stride = tile_stride * ST::underlying_subtiles_per_row;
    
            using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
            static_assert(range_type::lo + 3 == range_type::hi, "ds_read_b128 requires 4 consecutive registers");
    
            constexpr int offset = N * row_stride + M * tile_stride;
            macros::ds_read_b128<range_type::lo>(addr, offset); 
        } else {
            static_assert(false, "Unsupported shape");
        }
    } else {
        static_assert(RT::rows == ST::rows, "register tile and shared tile must match rows");
        static_assert(RT::cols == ST::cols,  "register tile and shared tile must match cols");
    
        using T2 = RT::dtype;
        using T  = base_types::packing<T2>::unpacked_type;
        using U  = ST::dtype;
        using U2 = base_types::packing<U >::packed_type;
        
        if constexpr (std::is_same_v<typename ST::shape, st_16x16_swizzled_s> && std::is_same_v<typename RT::shape, rt_32x16_4_s>) {
            constexpr int tile_stride = 16 * 16 * sizeof(U);
            constexpr int shared_tile_offset = 16 * 16 * ST::underlying_subtiles_per_row * sizeof(U);
            constexpr int row_stride = shared_tile_offset * 2;
    
            using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
            static_assert(range_type::lo + 3 == range_type::hi, "ds_read_b64_tr_b16 requires 4 consecutive registers");
            
            constexpr int offset_0 = N * row_stride + M * tile_stride;
            constexpr int offset_1 = N * row_stride + M * tile_stride + shared_tile_offset;
            macros::ds_read_b64_tr_b16<range_type::lo>(addr, offset_0);
            macros::ds_read_b64_tr_b16<range_type::lo + 2>(addr, offset_1);
        } else if constexpr (std::is_same_v<typename ST::shape, st_16x16_s> && std::is_same_v<typename RT::shape, rt_32x16_4_s>) {
            constexpr int tile_stride = 16 * 16 * sizeof(U);
            constexpr int row_stride = tile_stride * ST::underlying_subtiles_per_row;
    
            using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
            static_assert(range_type::lo + 3 == range_type::hi, "ds_read_b64_tr_b16 requires 4 consecutive registers");
            
            macros::ds_read_b64_tr_b16<range_type::lo>(addr, N * (2 * row_stride) + M * tile_stride);
            macros::ds_read_b64_tr_b16<range_type::lo + 2>(addr, N * (2 * row_stride) + M * tile_stride + row_stride);
    
        } else if constexpr (std::is_same_v<typename ST::shape, st_16x32_s> && std::is_same_v<typename RT::shape, rt_16x32_s>) {
            constexpr int tile_stride = 16 * 32 * sizeof(U);
            constexpr int row_stride = tile_stride * ST::underlying_subtiles_per_row;
    
            using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
            static_assert(range_type::lo + 3 == range_type::hi, "ds_read_b64_tr_b16 requires 4 consecutive registers");
            
            constexpr int offset_0 = N * row_stride + M * tile_stride;
            constexpr int offset_1 = N * row_stride + M * tile_stride + 4 * 32 * sizeof(U);
            macros::ds_read_b64_tr_b16<range_type::lo>(addr, offset_0);
            macros::ds_read_b64_tr_b16<range_type::lo + 2>(addr, offset_1);
        } else {
            static_assert(false, "Unsupported shape");
        }
    }
 }
 
 
 template<ducks::art::all RT, ducks::st::all ST>
 __device__ inline static void load(RT &dst, const ST &src) {
 
    if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
        static_assert(RT::rows == ST::rows, "register tile and shared tile must match rows");
        static_assert(RT::cols == ST::cols,  "register tile and shared tile must match cols");
    
        using T2 = RT::dtype;
        using T  = base_types::packing<T2>::unpacked_type;
        using U  = ST::dtype;
        using U2 = base_types::packing<U >::packed_type;
        constexpr int packing = base_types::packing<typename RT::dtype>::num();
    
        const int laneid = kittens::laneid();
    
        const int row_offset = laneid % dst.base_tile_rows;
        const int col_offset = dst.base_tile_stride * (laneid / dst.base_tile_rows);
    
        const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
    
        if constexpr (std::is_same_v<typename ST::shape, st_16x16_s> && std::is_same_v<typename RT::shape, rt_16x32_s>) {
            const int shared_tile_stride = 16 * 16 * sizeof(U);
            const int row_stride = shared_tile_stride * ST::underlying_subtiles_per_row;
            const int register_tile_stride = shared_tile_stride * 2;
    
            const int shared_tile_offset = (laneid / 32) * shared_tile_stride;
            const int lane_col_offset = ((laneid % 32) / 16) * 8;
            const int lane_row_offset = (laneid % 16);
    
            const int lane_byte_offset = (lane_row_offset * 16 + lane_col_offset) * sizeof(U) + shared_tile_offset;
            const uint32_t addr = src_ptr + lane_byte_offset;
    
            auto perform_load_at = [&]<int N, int M>() {
                using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
                static_assert(range_type::lo + 3 == range_type::hi, "ds_read_b64_tr_b16 requires 4 consecutive registers");
                
                const int offset = N * row_stride + M * register_tile_stride;
    
                macros::ds_read_b128<range_type::lo>(addr, offset);
            };
    
            // Compile-time nested loops over N and M
            [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
                ([&]<std::size_t N>() {
                    [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                        ([&]<std::size_t M>() {
                            perform_load_at.template operator()<N, M>();
                        }.template operator()<Ms>(), ...);
                    }(std::make_index_sequence<RT::width>{});
                }.template operator()<Ns>(), ...);
            }(std::make_index_sequence<RT::height>{});
        } else if constexpr (std::is_same_v<typename ST::shape, st_16x32_s> && std::is_same_v<typename RT::shape, rt_16x32_s>) {
            const int tile_stride = 16 * 32 * sizeof(U);
            const int row_stride = tile_stride * ST::underlying_subtiles_per_row;
            const int lane_col_offset = (laneid / 16) * 8;
            const int lane_row_offset = (laneid % 16);
    
            const int lane_byte_offset = (lane_row_offset * 32 + lane_col_offset) * sizeof(U);
            const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
            const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
    
            auto perform_load_at = [&]<int N, int M>() {
                using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
                static_assert(range_type::lo + 3 == range_type::hi, "ds_read_b128 requires 4 consecutive registers");
                const int offset = N * row_stride + M * tile_stride;
                macros::ds_read_b128<range_type::lo>(addr, offset); 
            };
    
            // Compile-time nested loops over N and M
            [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
                ([&]<std::size_t N>() {
                    [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                        ([&]<std::size_t M>() {
                            perform_load_at.template operator()<N, M>();
                        }.template operator()<Ms>(), ...);
                    }(std::make_index_sequence<RT::width>{});
                }.template operator()<Ns>(), ...);
            }(std::make_index_sequence<RT::height>{});
        } else {
            static_assert(false, "Unsupported shape");
        }
    } else {
        static_assert(RT::rows == ST::rows, "register tile and shared tile must match rows");
        static_assert(RT::cols == ST::cols,  "register tile and shared tile must match cols");
    
        using T2 = RT::dtype;
        using T  = base_types::packing<T2>::unpacked_type;
        using U  = ST::dtype;
        using U2 = base_types::packing<U >::packed_type;
        constexpr int packing = base_types::packing<typename RT::dtype>::num();
    
        const int laneid = kittens::laneid();
    
        const int row_offset = ((laneid % 16) / 4) + ((laneid / dst.base_tile_cols) * dst.base_tile_stride);
        const int col_offset = ((laneid % 4) * 4) + (16 * ((laneid % dst.base_tile_cols) / 16));
    
        const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
        
        if constexpr (std::is_same_v<typename ST::shape, st_16x16_swizzled_s> && std::is_same_v<typename RT::shape, rt_32x16_4_s>) {
            const int tile_stride = 16 * 16 * sizeof(U);
            const int shared_tile_offset = 16 * 16 * ST::underlying_subtiles_per_row * sizeof(U);
            const int row_stride = shared_tile_offset * 2;
            const int row_offset = (laneid % 16) / 4 + (laneid / 16) * 4;
            const int col_offset = ((laneid % 4) * 4);
            const int lane_byte_offset = (row_offset * 16 + col_offset) * sizeof(U);
            const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 7) << 3);
    
            const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
    
            auto perform_load_at = [&]<int N, int M>() {
                using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
                static_assert(range_type::lo + 3 == range_type::hi, "ds_read_b64_tr_b16 requires 4 consecutive registers");
                
                const int offset_0 = N * row_stride + M * tile_stride;
                const int offset_1 = N * row_stride + M * tile_stride + shared_tile_offset;
                macros::ds_read_b64_tr_b16<range_type::lo>(addr, offset_0);
                macros::ds_read_b64_tr_b16<range_type::lo + 2>(addr, offset_1);
            };
    
            // Compile-time nested loops over N and M
            [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
                ([&]<std::size_t N>() {
                    [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                        ([&]<std::size_t M>() {
                            perform_load_at.template operator()<N, M>();
                        }.template operator()<Ms>(), ...);
                    }(std::make_index_sequence<RT::width>{});
                }.template operator()<Ns>(), ...);
            }(std::make_index_sequence<RT::height>{});
        } else if constexpr (std::is_same_v<typename ST::shape, st_16x16_s> && std::is_same_v<typename RT::shape, rt_32x16_4_s>) {
            const int tile_stride = 16 * 16 * sizeof(U);
            const int row_stride = tile_stride * ST::underlying_subtiles_per_row;
            const int row_offset = (laneid % 16) / 4 + (laneid / 16) * 4;
            const int col_offset = ((laneid % 4) * 4);
            const int lane_byte_offset = (row_offset * 16 + col_offset) * sizeof(U);
    
            const uint32_t addr = src_ptr + lane_byte_offset;
    
            auto perform_load_at = [&]<int N, int M>() {
                using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
                static_assert(range_type::lo + 3 == range_type::hi, "ds_read_b64_tr_b16 requires 4 consecutive registers");
                
                macros::ds_read_b64_tr_b16<range_type::lo>(addr, N * (2 * row_stride) + M * tile_stride);
                macros::ds_read_b64_tr_b16<range_type::lo + 2>(addr, N * (2 * row_stride) + M * tile_stride + row_stride);
            };
            // Compile-time nested loops over N and M
            [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
                ([&]<std::size_t N>() {
                    [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                        ([&]<std::size_t M>() {
                            perform_load_at.template operator()<N, M>();
                        }.template operator()<Ms>(), ...);
                    }(std::make_index_sequence<RT::width>{});
                }.template operator()<Ns>(), ...);
            }(std::make_index_sequence<RT::height>{});
        } else if constexpr (std::is_same_v<typename ST::shape, st_16x32_s> && std::is_same_v<typename RT::shape, rt_16x32_s>) {
            const int tile_stride = 16 * 32 * sizeof(U);
            const int row_stride = tile_stride * ST::underlying_subtiles_per_row;
            const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
            const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
            const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(U);
            const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
    
            const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
    
            auto perform_load_at = [&]<int N, int M>() {
                using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
                static_assert(range_type::lo + 3 == range_type::hi, "ds_read_b64_tr_b16 requires 4 consecutive registers");
                
                const int offset_0 = N * row_stride + M * tile_stride;
                const int offset_1 = N * row_stride + M * tile_stride + 4 * 32 * sizeof(U);
                macros::ds_read_b64_tr_b16<range_type::lo>(addr, offset_0);
                macros::ds_read_b64_tr_b16<range_type::lo + 2>(addr, offset_1);
            };
    
            // Compile-time nested loops over N and M
            [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
                ([&]<std::size_t N>() {
                    [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                        ([&]<std::size_t M>() {
                            perform_load_at.template operator()<N, M>();
                        }.template operator()<Ms>(), ...);
                    }(std::make_index_sequence<RT::width>{});
                }.template operator()<Ns>(), ...);
            }(std::make_index_sequence<RT::height>{});
        } else {
            static_assert(false, "Unsupported shape");
        }
    }
 }
 
 /**
  * @brief Store data into a shared tile from a register tile.
  *
  * @tparam RT The register tile type
  * @tparam ST The shared tile type
  * @param dst[out] The destination shared tile.
  * @param src[in]  The source register tile.
  */
 
 template<ducks::art::all RT, ducks::st::all ST>
 __device__ inline static uint32_t get_address(ST &dst, const RT &src) {
 
     static_assert(std::is_same_v<typename RT::layout, ducks::rt_layout::row>, "RT must be a row-major layout tile");
     static_assert(std::is_same_v<typename ST::shape, st_16x16_swizzled_s>, "Only st_16x16_swizzled_s is supported");
     static_assert(std::is_same_v<typename RT::shape, rt_16x16_s>, "Only rt_16x16_s is supported");
 
     using T2 = RT::dtype;
     using T  = base_types::packing<T2>::unpacked_type;
     using U  = ST::dtype;
     using U2 = base_types::packing<U >::packed_type;
 
     const int laneid = kittens::laneid();
     const int col_offset = (laneid / 16) * 4;
     const int row_offset = (laneid % 16);
 
     const int lane_byte_offset = (row_offset * ST::underlying_subtile_cols + col_offset) * sizeof(U);
     const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 7) << 3);
     const uint32_t addr = reinterpret_cast<uintptr_t>(&dst.data[0]) + swizzled_lane_byte_offset;
 
     return addr;
 }
 
 template<int N, int M, ducks::art::all RT, ducks::st::all ST>
 __device__ inline static void store(ST &dst, const RT &src, uint32_t addr) {
     static_assert(std::is_same_v<typename RT::layout, ducks::rt_layout::row>, "RT must be a row-major layout tile");
     static_assert(std::is_same_v<typename ST::shape, st_16x16_swizzled_s>, "Only st_16x16_swizzled_s is supported");
     static_assert(std::is_same_v<typename RT::shape, rt_16x16_s>, "Only rt_16x16_s is supported");
 
     using T2 = RT::dtype;
     using T  = base_types::packing<T2>::unpacked_type;
     using U  = ST::dtype;
     using U2 = base_types::packing<U >::packed_type;
 
     constexpr int tile_stride = ST::underlying_subtile_rows * ST::underlying_subtile_cols;
     constexpr int row_stride = tile_stride * ST::underlying_subtiles_per_row;
 
     using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
     static_assert(range_type::lo + 1 == range_type::hi, "ds_write_b64 requires 2 consecutive registers");
 
     constexpr int offset = (N * row_stride + M * tile_stride) * sizeof(U);
     macros::ds_write_b64<range_type::lo>(addr, offset);
 }
 
 template<ducks::art::all RT, ducks::st::all ST>
 __device__ inline static void store(ST &dst, const RT &src) {
     static_assert(std::is_same_v<typename RT::layout, ducks::rt_layout::row>, "RT must be a row-major layout tile");
     static_assert(std::is_same_v<typename ST::shape, st_16x16_swizzled_s>, "Only st_16x16_swizzled_s is supported");
     static_assert(std::is_same_v<typename RT::shape, rt_16x16_s>, "Only rt_16x16_s is supported");
 
     using T2 = RT::dtype;
     using T  = base_types::packing<T2>::unpacked_type;
     using U  = ST::dtype;
     using U2 = base_types::packing<U >::packed_type;
 
     const int laneid = kittens::laneid();
     const int col_offset = (laneid / 16) * 4;
     const int row_offset = (laneid % 16);
 
     const int lane_byte_offset = (row_offset * ST::underlying_subtile_cols + col_offset) * sizeof(U);
     const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 7) << 3);
     const uint32_t addr = reinterpret_cast<uintptr_t>(&dst.data[0]) + swizzled_lane_byte_offset;
 
     const int tile_stride = ST::underlying_subtile_rows * ST::underlying_subtile_cols;
     const int row_stride = tile_stride * ST::underlying_subtiles_per_row;
 
     auto perform_store_at = [&]<int N, int M>() {
         using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
 
         static_assert(range_type::lo + 1 == range_type::hi, "ds_write_b64 requires 2 consecutive registers");
 
         const int offset = (N * row_stride + M * tile_stride) * sizeof(U);
 
         macros::ds_write_b64<range_type::lo>(addr, offset);
     };
 
     // Compile-time nested loops over N and M
     [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
         ([&]<std::size_t N>() {
             [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                 ([&]<std::size_t M>() {
                     perform_store_at.template operator()<N, M>();
                 }.template operator()<Ms>(), ...);
             }(std::make_index_sequence<ST::subtiles_per_row>{});
         }.template operator()<Ns>(), ...);
     }(std::make_index_sequence<ST::subtiles_per_col>{});
 }
 
 }