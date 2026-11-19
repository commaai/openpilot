/**
 * @file
 * @brief Conversions between data layouts and types for register tiles.
 */

#pragma once

#include "../../../../../common/common.cuh"
#include "../../../../../types/types.cuh"

namespace kittens {

/* ----------  LAYOUT SWAPS  ---------- */

/**
 * @brief Swaps the layout of a register tile in place.
 *
 * This function swaps the layout of a register tile in place by iterating over its height and width
 * and performing in-place layout swaps on each of its base elements.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the register tile.
 * @tparam _width The width of the register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be swapped in place.
 * @return A reference to the swapped register tile.
 */

template<ducks::art::all T0, ducks::art::all T1>
__device__ static inline void swap_layout_inplace(T0 &dst, const T1 &src) {
    if constexpr (std::is_same_v<typename T0::shape, typename ducks::rt_shape::rt_16x32> && std::is_same_v<typename T1::shape, typename ducks::rt_shape::rt_16x16>) {

        if constexpr (std::is_same_v<typename T1::layout, typename ducks::rt_layout::col> && std::is_same_v<typename T1::layout, typename ducks::rt_layout::col>) {
            // src consists of 16x16 tiles while dst consists of 16x32 tiles.
            // the reduction dimension (rows) stays the same, while the column dimension (cols) is doubled.
            // For every two 16x16 tiles in src along the (width) axis, we fill one 16x32 tile in dst along the (width) axis.
            // To do this for bf16, we issue 4 v_permlane16_swap instructions.
            static_assert(std::is_same_v<typename T0::T, bf16> && std::is_same_v<typename T1::T, bf16>, "only supports bf16");
            static_assert(T0::width == T1::width / 2);
            static_assert(T0::height == T1::height);

            auto perform_swap_layout_at = [&]<int N, int M>() {
                using range_type_T1_0 = ducks::art::get_nth_range_t<typename T1::register_ranges, N * T1::width + M * 2>;
                using range_type_T1_1 = ducks::art::get_nth_range_t<typename T1::register_ranges, N * T1::width + M * 2 + 1>;
                using range_type_T0 = ducks::art::get_nth_range_t<typename T0::register_ranges, N * T0::width + M>;

                static_assert(range_type_T1_0::size == 2 && range_type_T1_1::size == 2 && range_type_T0::size == 4);
                static_assert(range_type_T1_0::lo == range_type_T0::lo && range_type_T1_1::lo == range_type_T0::lo + 2);

                macros::v_permlane16_swap_b32_e32<range_type_T1_0::lo, range_type_T1_1::lo>();
                macros::v_permlane16_swap_b32_e32<range_type_T1_0::lo + 1, range_type_T1_1::lo + 1>();
            };

            // Compile-time nested loops over N and M
            [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
                ([&]<std::size_t N>() {
                    [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                        ([&]<std::size_t M>() {
                            perform_swap_layout_at.template operator()<N, M>();
                        }.template operator()<Ms>(), ...);
                    }(std::make_index_sequence<T0::width>{});
                }.template operator()<Ns>(), ...);
            }(std::make_index_sequence<T0::height>{});
        } else {
            static_assert(false, "Unsupported layout swap");
        }
    } else {
        static_assert(false, "Unsupported matrix layout swap");
    }
}

/* ----------  TYPE SWAPS  ---------- */
template<int N, int M, ducks::art::all T0, ducks::art::all T1>
__device__ static inline void copy(T0 &dst, const T1 &src) {

    static_assert(std::is_same_v<typename T0::T, bf16> && std::is_same_v<typename T1::T, float>, "Only float to bf16 copy is supported");

    static_assert(T0::width == T1::width);
    static_assert(T0::height == T1::height);

    using range_type_T0 = ducks::art::get_nth_range_t<typename T0::register_ranges, N * T0::width + M>;
    using registers_T0 = ducks::art::split_many_t<ducks::art::type_list<range_type_T0>, 1>;

    using range_type_T1 = ducks::art::get_nth_range_t<typename T1::register_ranges, N * T1::width + M>;
    using registers_T1 = ducks::art::split_many_t<ducks::art::type_list<range_type_T1>, 1>;

    static_assert(registers_T0::size == 2 && registers_T1::size == 4);

    macros::v_cvt_pk_bf16_f32<range_type_T0::lo, range_type_T1::lo, range_type_T1::lo + 1>();
    macros::v_cvt_pk_bf16_f32<range_type_T0::lo + 1, range_type_T1::lo + 2, range_type_T1::lo + 3>();
}

template<ducks::art::all T0, ducks::art::all T1>
__device__ static inline void copy(T0 &dst, const T1 &src) {

    static_assert(std::is_same_v<typename T0::T, bf16> && std::is_same_v<typename T1::T, float>, "Only float to bf16 copy is supported");

    static_assert(T0::width == T1::width);
    static_assert(T0::height == T1::height);

    auto perform_copy_at = [&]<int N, int M>() {
        using range_type_T0 = ducks::art::get_nth_range_t<typename T0::register_ranges, N * T0::width + M>;
        using registers_T0 = ducks::art::split_many_t<ducks::art::type_list<range_type_T0>, 1>;

        using range_type_T1 = ducks::art::get_nth_range_t<typename T1::register_ranges, N * T1::width + M>;
        using registers_T1 = ducks::art::split_many_t<ducks::art::type_list<range_type_T1>, 1>;

        static_assert(registers_T0::size == 2 && registers_T1::size == 4);

        macros::v_cvt_pk_bf16_f32<range_type_T0::lo, range_type_T1::lo, range_type_T1::lo + 1>();
        macros::v_cvt_pk_bf16_f32<range_type_T0::lo + 1, range_type_T1::lo + 2, range_type_T1::lo + 3>();
    };

    // Compile-time nested loops over N and M
    [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
        ([&]<std::size_t N>() {
            [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                ([&]<std::size_t M>() {
                    perform_copy_at.template operator()<N, M>();
                }.template operator()<Ms>(), ...);
            }(std::make_index_sequence<T0::width>{});
        }.template operator()<Ns>(), ...);
    }(std::make_index_sequence<T0::height>{});
}

template<ducks::art::all T0, ducks::art::all T1>
__device__ static inline void accvgpr_read(T0 &dst, const T1 &src) {

    static_assert(T0::width == T1::width);
    static_assert(T0::height == T1::height);

    auto perform_accvgpr_read_at = [&]<int N, int M>() {
        using range_type_T0 = ducks::art::get_nth_range_t<typename T0::register_ranges, N * T0::width + M>;
        using registers_T0 = ducks::art::split_many_t<ducks::art::type_list<range_type_T0>, 1>;

        using range_type_T1 = ducks::art::get_nth_range_t<typename T1::register_ranges, N * T1::width + M>;
        using registers_T1 = ducks::art::split_many_t<ducks::art::type_list<range_type_T1>, 1>;

        static_assert(registers_T0::size == registers_T1::size);
        static_assert(range_type_T0::hi < 256 && range_type_T1::lo >= 256);

        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            ([&]<std::size_t I>() {
                constexpr int register_0 = ducks::art::get_nth_range_t<registers_T0, I>::lo;
                constexpr int register_1 = ducks::art::get_nth_range_t<registers_T1, I>::lo;
                macros::v_accvgpr_read_b32<register_0, register_1>();
            }.template operator()<Is>(), ...);
        }(std::make_index_sequence<registers_T0::size>{});
    };

    // Compile-time nested loops over N and M
    [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
        ([&]<std::size_t N>() {
            [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                ([&]<std::size_t M>() {
                    perform_accvgpr_read_at.template operator()<N, M>();
                }.template operator()<Ms>(), ...);
            }(std::make_index_sequence<T0::width>{});
        }.template operator()<Ns>(), ...);
    }(std::make_index_sequence<T0::height>{});
}

template<int N, int M, int GPR, ducks::art::all T0, ducks::art::all T1>
__device__ static inline void make_causal(T0 &dst, const T1 &src) {
    static_assert(std::is_same_v<typename T0::T, float> && std::is_same_v<typename T1::T, float>, "Only float to float causal mask is supported");
    static_assert(std::is_same_v<typename T0::layout, typename T1::layout>, "Only same layout is supported");
    static_assert(std::is_same_v<typename T0::shape, typename T1::shape>, "Only same shape is supported");

    if constexpr (std::is_same_v<typename T0::layout, typename ducks::rt_layout::col> && std::is_same_v<typename T0::shape, typename ducks::rt_shape::rt_16x16>) {
        // Move the causal mask to our SGPRs
        using range_type_T0 = ducks::art::get_nth_range_t<typename T0::register_ranges, N * T0::width + M>;
        using registers_T0 = ducks::art::split_many_t<ducks::art::type_list<range_type_T0>, 1>;

        using range_type_T1 = ducks::art::get_nth_range_t<typename T1::register_ranges, N * T1::width + M>;
        using registers_T1 = ducks::art::split_many_t<ducks::art::type_list<range_type_T1>, 1>;

        static_assert(registers_T0::size == registers_T1::size);

        uint64_t causal_mask = 0xE000FE00FFE0FFFE;
        macros::v_cndmask_b32_e64<ducks::art::get_nth_range_t<registers_T0, 0>::lo, ducks::art::get_nth_range_t<registers_T1, 0>::lo, GPR>(causal_mask);

        causal_mask = 0xC000FC00FFC0FFFC;
        macros::v_cndmask_b32_e64<ducks::art::get_nth_range_t<registers_T0, 1>::lo, ducks::art::get_nth_range_t<registers_T1, 1>::lo, GPR>(causal_mask);

        causal_mask = 0x8000F800FF80FFF8;
        macros::v_cndmask_b32_e64<ducks::art::get_nth_range_t<registers_T0, 2>::lo, ducks::art::get_nth_range_t<registers_T1, 2>::lo, GPR>(causal_mask);

        causal_mask = 0x0000F000FF00FFF0;
        macros::v_cndmask_b32_e64<ducks::art::get_nth_range_t<registers_T0, 3>::lo, ducks::art::get_nth_range_t<registers_T1, 3>::lo, GPR>(causal_mask);
    } else {
        static_assert(false, "Unsupported causal mask");
    }
}

}
