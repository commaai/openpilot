/**
 * @file
 * @brief Map operations: between tiles, and those which apply vectors to tiles.
 */

#pragma once

#include "../../../../../common/common.cuh"
#include "../../../../../types/types.cuh"

namespace kittens {

/* ----------  Uniform tile maps (independent of layout)  ---------- */

/**
 * @brief Applies a unary operation to each element of a tile.
 *
 * @tparam op Unary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 */
 template<int N, int M, int R, typename op, ducks::art::all T0, ducks::art::all T1>
 __device__ static inline void unary_map() {
     static_assert(T0::width == T1::width);
     static_assert(T0::height == T1::height);
     static_assert(std::is_same_v<typename T0::dtype, typename T1::dtype>);
 
     using range_type_T0 = ducks::art::get_nth_range_t<typename T0::register_ranges, N * T0::width + M>;
     using registers_T0 = ducks::art::split_many_t<ducks::art::type_list<range_type_T0>, 1>;
 
     using range_type_T1 = ducks::art::get_nth_range_t<typename T1::register_ranges, N * T1::width + M>;
     using registers_T1 = ducks::art::split_many_t<ducks::art::type_list<range_type_T1>, 1>;
 
     static_assert(registers_T0::size == registers_T1::size);
 
     op::template op<ducks::art::get_nth_range_t<registers_T0, R>::lo, ducks::art::get_nth_range_t<registers_T1, R>::lo>();
 }

template<int N, int M, typename op, ducks::art::all T0, ducks::art::all T1>
__device__ static inline void unary_map() {
    static_assert(T0::width == T1::width);
    static_assert(T0::height == T1::height);
    static_assert(std::is_same_v<typename T0::dtype, typename T1::dtype>);

    using range_type_T0 = ducks::art::get_nth_range_t<typename T0::register_ranges, N * T0::width + M>;
    using registers_T0 = ducks::art::split_many_t<ducks::art::type_list<range_type_T0>, 1>;

    using range_type_T1 = ducks::art::get_nth_range_t<typename T1::register_ranges, N * T1::width + M>;
    using registers_T1 = ducks::art::split_many_t<ducks::art::type_list<range_type_T1>, 1>;

    static_assert(registers_T0::size == registers_T1::size);

    [&]<std::size_t... Rs>(std::index_sequence<Rs...>) {
        ([&]<std::size_t R>() {
        op::template op<ducks::art::get_nth_range_t<registers_T0, R>::lo, ducks::art::get_nth_range_t<registers_T1, R>::lo>();
        }.template operator()<Rs>(), ...);
    }(std::make_index_sequence<registers_T0::size>{});
}

template<typename op, ducks::art::all T0, ducks::art::all T1>
__device__ static inline void unary_map() {
    static_assert(T0::width == T1::width);
    static_assert(T0::height == T1::height);
    static_assert(std::is_same_v<typename T0::dtype, typename T1::dtype>);
    
    auto perform_unary_map_at = [&]<int N, int M>() {
        using range_type_T0 = ducks::art::get_nth_range_t<typename T0::register_ranges, N * T0::width + M>;
        using registers_T0 = ducks::art::split_many_t<ducks::art::type_list<range_type_T0>, 1>;

        using range_type_T1 = ducks::art::get_nth_range_t<typename T1::register_ranges, N * T1::width + M>;
        using registers_T1 = ducks::art::split_many_t<ducks::art::type_list<range_type_T1>, 1>;

        static_assert(registers_T0::size == registers_T1::size);

        [&]<std::size_t... Rs>(std::index_sequence<Rs...>) {
            ([&]<std::size_t R>() {
              op::template op<ducks::art::get_nth_range_t<registers_T0, R>::lo, ducks::art::get_nth_range_t<registers_T1, R>::lo>();
            }.template operator()<Rs>(), ...);
        }(std::make_index_sequence<registers_T0::size>{});
    };

    // Compile-time nested loops over N and M
    [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
        ([&]<std::size_t N>() {
            [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                ([&]<std::size_t M>() {
                    perform_unary_map_at.template operator()<N, M>();
                }.template operator()<Ms>(), ...);
            }(std::make_index_sequence<T0::width>{});
        }.template operator()<Ns>(), ...);
    }(std::make_index_sequence<T0::height>{});
}

/**
 * @brief Applies a binary operation to each element of a tile with a scalar parameter.
 *
 * @tparam op Binary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param param[in] Scalar parameter for the binary operation.
 */
 template<int N, int M, int R, typename op, ducks::art::all T0, ducks::art::all T1>
 __device__ static inline void bin_map(T0 &dst, const T1 &src, const typename base_types::packing<typename T0::dtype>::unpacked_type &param) {
     static_assert(T0::width == T1::width);
     static_assert(T0::height == T1::height);
     static_assert(std::is_same_v<typename T0::dtype, typename T1::dtype>);
 
     using range_type_T0 = ducks::art::get_nth_range_t<typename T0::register_ranges, N * T0::width + M>;
     using registers_T0 = ducks::art::split_many_t<ducks::art::type_list<range_type_T0>, 1>;
 
     using range_type_T1 = ducks::art::get_nth_range_t<typename T1::register_ranges, N * T1::width + M>;
     using registers_T1 = ducks::art::split_many_t<ducks::art::type_list<range_type_T1>, 1>;
 
     static_assert(registers_T0::size == registers_T1::size);
 
     op::template op<ducks::art::get_nth_range_t<registers_T0, R>::lo, ducks::art::get_nth_range_t<registers_T1, R>::lo>(param);
 }

template<int N, int M, typename op, ducks::art::all T0, ducks::art::all T1>
__device__ static inline void bin_map(T0 &dst, const T1 &src, const typename base_types::packing<typename T0::dtype>::unpacked_type &param) {
    static_assert(T0::width == T1::width);
    static_assert(T0::height == T1::height);
    static_assert(std::is_same_v<typename T0::dtype, typename T1::dtype>);

    using range_type_T0 = ducks::art::get_nth_range_t<typename T0::register_ranges, N * T0::width + M>;
    using registers_T0 = ducks::art::split_many_t<ducks::art::type_list<range_type_T0>, 1>;

    using range_type_T1 = ducks::art::get_nth_range_t<typename T1::register_ranges, N * T1::width + M>;
    using registers_T1 = ducks::art::split_many_t<ducks::art::type_list<range_type_T1>, 1>;

    static_assert(registers_T0::size == registers_T1::size);

    [&]<std::size_t... Rs>(std::index_sequence<Rs...>) {
        ([&]<std::size_t R>() {
            op::template op<ducks::art::get_nth_range_t<registers_T0, R>::lo, ducks::art::get_nth_range_t<registers_T1, R>::lo>(param);
        }.template operator()<Rs>(), ...);
    }(std::make_index_sequence<registers_T0::size>{});
}

template<typename op, ducks::art::all T0, ducks::art::all T1>
__device__ static inline void bin_map(T0 &dst, const T1 &src, const typename base_types::packing<typename T0::dtype>::unpacked_type &param) {
    static_assert(T0::width == T1::width);
    static_assert(T0::height == T1::height);
    static_assert(std::is_same_v<typename T0::dtype, typename T1::dtype>);

    auto perform_bin_map_at = [&]<int N, int M>() {
        using range_type_T0 = ducks::art::get_nth_range_t<typename T0::register_ranges, N * T0::width + M>;
        using registers_T0 = ducks::art::split_many_t<ducks::art::type_list<range_type_T0>, 1>;

        using range_type_T1 = ducks::art::get_nth_range_t<typename T1::register_ranges, N * T1::width + M>;
        using registers_T1 = ducks::art::split_many_t<ducks::art::type_list<range_type_T1>, 1>;

        static_assert(registers_T0::size == registers_T1::size);

        [&]<std::size_t... Rs>(std::index_sequence<Rs...>) {
            ([&]<std::size_t R>() {
              op::template op<ducks::art::get_nth_range_t<registers_T0, R>::lo, ducks::art::get_nth_range_t<registers_T1, R>::lo>(param);
            }.template operator()<Rs>(), ...);
        }(std::make_index_sequence<registers_T0::size>{});
    };

    // Compile-time nested loops over N and M
    [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
        ([&]<std::size_t N>() {
            [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                ([&]<std::size_t M>() {
                    perform_bin_map_at.template operator()<N, M>();
                }.template operator()<Ms>(), ...);
            }(std::make_index_sequence<T0::width>{});
        }.template operator()<Ns>(), ...);
    }(std::make_index_sequence<T0::height>{});
}

/**
 * @brief Applies a binary operation element-wise between two tiles.
 *
 * @tparam op Binary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the operation.
 * @param rhs[in] Right-hand side source tile for the operation.
 */
template<int N, int M, typename op, ducks::art::all T0, ducks::art::all T1, ducks::art::all T2>
__device__ static inline void bin_map(T0 &dst, const T1 &lhs, const T2 &rhs) {
    static_assert(T0::width == T1::width);
    static_assert(T0::height == T1::height);
    static_assert(T0::width == T2::width);
    static_assert(T0::height == T2::height);
    static_assert(std::is_same_v<typename T0::dtype, typename T1::dtype>);
    static_assert(std::is_same_v<typename T0::dtype, typename T2::dtype>);

    using range_type_T0 = ducks::art::get_nth_range_t<typename T0::register_ranges, N * T0::width + M>;
    using registers_T0 = ducks::art::split_many_t<ducks::art::type_list<range_type_T0>, 1>;

    using range_type_T1 = ducks::art::get_nth_range_t<typename T1::register_ranges, N * T1::width + M>;
    using registers_T1 = ducks::art::split_many_t<ducks::art::type_list<range_type_T1>, 1>;

    using range_type_T2 = ducks::art::get_nth_range_t<typename T2::register_ranges, N * T2::width + M>;
    using registers_T2 = ducks::art::split_many_t<ducks::art::type_list<range_type_T2>, 1>;

    static_assert(registers_T0::size == registers_T1::size);
    static_assert(registers_T0::size == registers_T2::size);

    [&]<std::size_t... Rs>(std::index_sequence<Rs...>) {
        ([&]<std::size_t R>() {
        op::template op<ducks::art::get_nth_range_t<registers_T0, R>::lo, ducks::art::get_nth_range_t<registers_T1, R>::lo, ducks::art::get_nth_range_t<registers_T2, R>::lo>();
        }.template operator()<Rs>(), ...);
    }(std::make_index_sequence<registers_T0::size>{});
}

template<typename op, ducks::art::all T0, ducks::art::all T1, ducks::art::all T2>
__device__ static inline void bin_map(T0 &dst, const T1 &lhs, const T2 &rhs) {
    static_assert(T0::width == T1::width);
    static_assert(T0::height == T1::height);
    static_assert(T0::width == T2::width);
    static_assert(T0::height == T2::height);
    static_assert(std::is_same_v<typename T0::dtype, typename T1::dtype>);
    static_assert(std::is_same_v<typename T0::dtype, typename T2::dtype>);

    auto perform_bin_map_at = [&]<int N, int M>() {
        using range_type_T0 = ducks::art::get_nth_range_t<typename T0::register_ranges, N * T0::width + M>;
        using registers_T0 = ducks::art::split_many_t<ducks::art::type_list<range_type_T0>, 1>;

        using range_type_T1 = ducks::art::get_nth_range_t<typename T1::register_ranges, N * T1::width + M>;
        using registers_T1 = ducks::art::split_many_t<ducks::art::type_list<range_type_T1>, 1>;

        using range_type_T2 = ducks::art::get_nth_range_t<typename T2::register_ranges, N * T2::width + M>;
        using registers_T2 = ducks::art::split_many_t<ducks::art::type_list<range_type_T2>, 1>;

        static_assert(registers_T0::size == registers_T1::size);
        static_assert(registers_T0::size == registers_T2::size);

        [&]<std::size_t... Rs>(std::index_sequence<Rs...>) {
            ([&]<std::size_t R>() {
              op::template op<ducks::art::get_nth_range_t<registers_T0, R>::lo, ducks::art::get_nth_range_t<registers_T1, R>::lo, ducks::art::get_nth_range_t<registers_T2, R>::lo>();
            }.template operator()<Rs>(), ...);
        }(std::make_index_sequence<registers_T0::size>{});
    };

    // Compile-time nested loops over N and M
    [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
        ([&]<std::size_t N>() {
            [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                ([&]<std::size_t M>() {
                    perform_bin_map_at.template operator()<N, M>();
                }.template operator()<Ms>(), ...);
            }(std::make_index_sequence<T0::width>{});
        }.template operator()<Ns>(), ...);
    }(std::make_index_sequence<T0::height>{});
}


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// All of the annoying qualifiers *should* be automatically inferred during compile-time.
// So, syntax should just be kittens::add_row(tile, colvec);

/**
 * @brief Applies the exponential function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the exponential function on.
 */
template<int N, int M, int R, ducks::art::all T0, ducks::art::all T1>
__device__ static inline void exp2(T0 &dst, const T1 &src) {
    unary_map<N, M, R, macros::exp2, T0, T1>();
}

template<int N, int M, ducks::art::all T0, ducks::art::all T1>
__device__ static inline void exp2(T0 &dst, const T1 &src) {
    unary_map<N, M, macros::exp2, T0, T1>();
}

template<ducks::art::all T0, ducks::art::all T1>
__device__ static inline void exp2(T0 &dst, const T1 &src) {
    unary_map<macros::exp2, T0, T1>();
}

/**
 * @brief Sets all elements of a tile to zero.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template<int N, int M, ducks::art::all T0>
__device__ static inline void zero(T0 &dst) {
    unary_map<N, M, macros::zero, T0, T0>();
}
template<ducks::art::all T0>
__device__ static inline void zero(T0 &dst) {
    unary_map<macros::zero, T0, T0>();
}

template<int N, int M, int R, int GPR, ducks::art::all T0>
__device__ static inline void mov(T0 &dst) {
    using range_type_T0 = ducks::art::get_nth_range_t<typename T0::register_ranges, N * T0::width + M>;
    using registers_T0 = ducks::art::split_many_t<ducks::art::type_list<range_type_T0>, 1>;

    macros::v_mov_b32_e32<ducks::art::get_nth_range_t<registers_T0, R>::lo, GPR>();
}

template<int N, int M, int GPR, ducks::art::all T0>
__device__ static inline void mov(T0 &dst) {
    using range_type_T0 = ducks::art::get_nth_range_t<typename T0::register_ranges, N * T0::width + M>;
    using registers_T0 = ducks::art::split_many_t<ducks::art::type_list<range_type_T0>, 1>;

    [&]<std::size_t... Rs>(std::index_sequence<Rs...>) {
        ([&]<std::size_t R>() {
            macros::v_mov_b32_e32<ducks::art::get_nth_range_t<registers_T0, R>::lo, GPR>();
        }.template operator()<Rs>(), ...);
    }(std::make_index_sequence<registers_T0::size>{});
}

template<int GPR, ducks::art::all T0>
__device__ static inline void mov(T0 &dst) {
    
    auto perform_mov_at = [&]<int N, int M>() {
        using range_type_T0 = ducks::art::get_nth_range_t<typename T0::register_ranges, N * T0::width + M>;
        using registers_T0 = ducks::art::split_many_t<ducks::art::type_list<range_type_T0>, 1>;

        [&]<std::size_t... Rs>(std::index_sequence<Rs...>) {
            ([&]<std::size_t R>() {
                macros::v_mov_b32_e32<ducks::art::get_nth_range_t<registers_T0, R>::lo, GPR>();
            }.template operator()<Rs>(), ...);
        }(std::make_index_sequence<registers_T0::size>{});
    };

    // Compile-time nested loops over N and M
    [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
        ([&]<std::size_t N>() {
            [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                ([&]<std::size_t M>() {
                    perform_mov_at.template operator()<N, M>();
                }.template operator()<Ms>(), ...);
            }(std::make_index_sequence<T0::width>{});
        }.template operator()<Ns>(), ...);
    }(std::make_index_sequence<T0::height>{});
}

/**
 * @brief Multiplies two tiles element-wise or multiplies each element of a tile by a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the multiplication.
 * @param rhs[in] Right-hand side source tile or scalar for the multiplication.
 */
 template<int N, int M, int R, ducks::art::all T0, ducks::art::all T1, typename U>
 __device__ static inline void mul(T0 &dst, const T1 &lhs, const U &rhs) {
     bin_map<N, M, R, macros::mul, T0, T1>(dst, lhs, rhs);
 }

template<int N, int M, ducks::art::all T0, ducks::art::all T1, typename U>
__device__ static inline void mul(T0 &dst, const T1 &lhs, const U &rhs) {
    bin_map<N, M, macros::mul, T0, T1>(dst, lhs, rhs);
}

template<ducks::art::all T0, ducks::art::all T1, typename U>
__device__ static inline void mul(T0 &dst, const T1 &lhs, const U &rhs) {
    bin_map<macros::mul, T0, T1>(dst, lhs, rhs);
}

/**
 * @brief Subtracts row values from each row of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the subtraction on.
 * @param row_values[in] Column vector containing values to subtract from each row.
 */
template<int N, int M, int GPR, ducks::art::all T0, ducks::art::all T1>
__device__ static inline void sub_row(T0 &dst, const T1 &src) {
 
    static_assert(T0::width == T1::width);
    static_assert(T0::height == T1::height);
    static_assert(std::is_same_v<typename T0::dtype, typename T1::dtype>);
 
    using range_type_T0 = ducks::art::get_nth_range_t<typename T0::register_ranges, N * T0::width + M>;
    using registers_T0 = ducks::art::split_many_t<ducks::art::type_list<range_type_T0>, 1>;

    using range_type_T1 = ducks::art::get_nth_range_t<typename T1::register_ranges, N * T1::width + M>;
    using registers_T1 = ducks::art::split_many_t<ducks::art::type_list<range_type_T1>, 1>;

    static_assert(registers_T0::size == 4 && registers_T1::size == 4);

    macros::v_subrev_f32_dpp<range_type_T0::lo, range_type_T1::lo, GPR>();
}

template<int GPR, ducks::art::all T0, ducks::art::all T1>
__device__ static inline void sub_row(T0 &dst, const T1 &src) {

    static_assert(T0::width == T1::width);
    static_assert(T0::height == T1::height);
    static_assert(std::is_same_v<typename T0::dtype, typename T1::dtype>);

    auto perform_sub_row_at = [&]<int N, int M>() {
        using range_type_T0 = ducks::art::get_nth_range_t<typename T0::register_ranges, N * T0::width + M>;
        using registers_T0 = ducks::art::split_many_t<ducks::art::type_list<range_type_T0>, 1>;

        using range_type_T1 = ducks::art::get_nth_range_t<typename T1::register_ranges, N * T1::width + M>;
        using registers_T1 = ducks::art::split_many_t<ducks::art::type_list<range_type_T1>, 1>;

        static_assert(registers_T0::size == 4 && registers_T1::size == 4);

        macros::v_subrev_f32_dpp<range_type_T0::lo, range_type_T1::lo, GPR>();
    };

    // Compile-time nested loops over N and M
    [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
        ([&]<std::size_t N>() {
            [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                ([&]<std::size_t M>() {
                    perform_sub_row_at.template operator()<N, M>();
                }.template operator()<Ms>(), ...);
            }(std::make_index_sequence<T0::width>{});
        }.template operator()<Ns>(), ...);
    }(std::make_index_sequence<T0::height>{});
}

}