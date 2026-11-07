/**
 * @file
 * @brief Templated layouts for global memory.
 */
 
#pragma once

#include "../../common/common.cuh"
#include "../shared/shared.cuh"
#include "util.cuh"
#ifdef KITTENS_HOPPER
#include <utility>
#include "tma.cuh"
#endif

namespace kittens {

/* ----------   Global layout axes  ---------- */

struct dim {
    static constexpr int BATCH = 0;
    static constexpr int DEPTH = 1;
    static constexpr int ROW   = 2;
    static constexpr int COL   = 3;
};

/* ----------   Associative dictionary for global layouts  ---------- */

#ifdef KITTENS_HOPPER
namespace ducks {
namespace tma {
namespace descriptor {
struct identifier {};
template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;
} // namespace descriptor
} // namespace tma
} // namespace ducks
namespace detail {
namespace tma {
template<typename T> struct descriptor_copy_helper {};
template<kittens::ducks::tma::descriptor::all _T> struct descriptor_copy_helper<_T> { static constexpr int value = _T::axis; using T = _T::T; static constexpr bool swizzle_flag = _T::swizzle_flag; };
template<kittens::ducks::st::all _T> struct descriptor_copy_helper<_T> { static constexpr int value = 2; using T = _T; static constexpr bool swizzle_flag = true; };
template<kittens::ducks::sv::all _T> struct descriptor_copy_helper<_T> { static constexpr int value = -1; using T = _T; static constexpr bool swizzle_flag = true; };
template<typename T> using descriptor_copy_helper_t = descriptor_copy_helper<T>::T;
template<typename T> static constexpr int descriptor_copy_helper_v = descriptor_copy_helper<T>::value;
template<typename T> static constexpr bool descriptor_copy_helper_swizzle_flag = descriptor_copy_helper<T>::swizzle_flag;
} // namespace tma
} // namespace detail
namespace tma {
template<typename _T, int _axis=-9999, bool _swizzle_flag=true> struct descriptor {
    using identifier = ducks::tma::descriptor::identifier;
    using T = detail::tma::descriptor_copy_helper_t<_T>;
    static_assert(ducks::st::all<T> || ducks::sv::all<T> || ducks::tma::descriptor::all<T>, "Must be a shared TK type to generate a TMA descriptor.");
    static constexpr int axis = (
        ducks::tma::descriptor::all<_T> ? detail::tma::descriptor_copy_helper_v<_T> : // if a copy, inherit the axis from the original descriptor. 
        (_axis != -9999) ? _axis : detail::tma::descriptor_copy_helper_v<_T>); // if a default value was provided, use it.
    static_assert((kittens::ducks::st::all<T> && axis >= 0 && axis <= 2) || (kittens::ducks::sv::all<T> && axis == -1), "Internal template error detected.");
    static constexpr bool swizzle_flag = ducks::tma::descriptor::all<_T> ? detail::tma::descriptor_copy_helper_swizzle_flag<_T> : _swizzle_flag;
};
} // namespace tma
#endif

namespace detail {
template<typename... Args>
struct descriptor_dict {
    __host__ __device__ descriptor_dict() {}
    template<typename T> __host__ __device__ descriptor_dict(T _, int b, int d, int r, int c) {}
    __host__ __device__ descriptor_dict(const descriptor_dict &other) {}
#ifdef KITTENS_HOPPER
    template<typename T, int U> __device__ const CUtensorMap* get() const {
        static_assert(
            std::is_same_v<T, std::true_type> && std::is_same_v<T, std::false_type>,
            "SKILL ISSUE: Requested a TMA descriptor for a type not initialized in the global layout."
        );
    }
#endif
};

#ifdef KITTENS_HOPPER
template<typename _T, typename... Args>
struct descriptor_dict<_T, Args...> {
    static_assert(ducks::sv::all<_T> || ducks::st::all<_T> || ducks::tma::descriptor::all<_T>, "Must be a shared TK type to generate a TMA descriptor.");
    using DESC = kittens::tma::descriptor<_T>; // copy or initialize with a default value
    CUtensorMap tma_desc;
    descriptor_dict<Args...> other_descs;
    __host__ __device__ descriptor_dict() {}
    __host__ __device__ descriptor_dict(typename DESC::T::dtype *data, int b, int d, int r, int c): other_descs(data, b, d, r, c) {
        kittens::detail::tma::create_tensor_map<typename DESC::T, DESC::axis, DESC::swizzle_flag>(&tma_desc, data, b, d, r, c);
    }
    __host__ __device__ inline descriptor_dict(const descriptor_dict &other) :
        tma_desc(other.tma_desc), other_descs(other.other_descs) {}
    template<typename U, int axis> __device__ inline const CUtensorMap* get() const {
        if constexpr (std::is_same_v<typename DESC::T, U> && DESC::axis == axis) { return &tma_desc; }
        else                                                                     { return other_descs.template get<U, axis>(); }
    }
};
#endif
}

/* ----------  Global layout descriptor  ---------- */

namespace ducks {
namespace gl {
struct identifier {};
}
}

template<typename _T, int b, int d, int r, int c, typename... TMA_Types>
struct gl {
    using identifier = ducks::gl::identifier;

    using T     = base_types::packing<_T>::unpacked_type;
    using T2    = base_types::packing<_T>::packed_type;
    using dtype = T;

    T* raw_ptr;

    static constexpr int __b__ = b, __d__ = d, __r__ = r, __c__ = c; // Not to be touched by the user.

    ducks::gl::make_dim_t<b> batch_internal;
    ducks::gl::make_dim_t<d> depth_internal;
    ducks::gl::make_dim_t<r> rows_internal;
    ducks::gl::make_dim_t<c> cols_internal;

    template <int B=__b__> __device__ __host__ static constexpr std::enable_if_t<(B > 0), int> batch() { return B; }
    template <int B=__b__> __device__ __host__ std::enable_if_t<(B == -1), int> batch() const { return batch_internal; }
    template <int D=__d__> __device__ __host__ static constexpr std::enable_if_t<(D > 0), int> depth() { return D; }
    template <int D=__d__> __device__ __host__ std::enable_if_t<(D == -1), int> depth() const { return depth_internal; }
    template <int R=__r__> __device__ __host__ static constexpr std::enable_if_t<(R > 0), int> rows() { return R; }
    template <int R=__r__> __device__ __host__ std::enable_if_t<(R == -1), int> rows() const { return rows_internal; }
    template <int C=__c__> __device__ __host__ static constexpr std::enable_if_t<(C > 0), int> cols() { return C; }
    template <int C=__c__> __device__ __host__ std::enable_if_t<(C == -1), int> cols() const { return cols_internal; }

    detail::descriptor_dict<TMA_Types...> tma_descs;

    __host__ __device__ inline gl(T *_data,
                        ducks::gl::make_arg_t<b> _batch,
                        ducks::gl::make_arg_t<d> _depth,
                        ducks::gl::make_arg_t<r> _rows,
                        ducks::gl::make_arg_t<c> _cols) :
            raw_ptr(_data), batch_internal(_batch), depth_internal(_depth), rows_internal(_rows), cols_internal(_cols) {
        tma_descs = detail::descriptor_dict<TMA_Types...>(raw_ptr, batch_internal, depth_internal, rows_internal, cols_internal);
    }
    __host__ __device__ inline gl(const gl &other) :
            raw_ptr(other.raw_ptr), batch_internal(other.batch_internal), depth_internal(other.depth_internal), rows_internal(other.rows_internal), cols_internal(other.cols_internal), tma_descs(other.tma_descs) {}
#ifdef KITTENS_HOPPER
    template<typename U, int axis> __device__ inline const CUtensorMap* get_tma() const {
        return tma_descs.template get<U, axis>();
    }
#endif
    __device__ inline T& operator[](const coord<ducks::default_type> &idx) const { // yes I am abusing the const qualifier here a bit.
        return raw_ptr[((idx.b*depth() + idx.d)*rows() + idx.r)*cols() + idx.c];
    }
    template<int axis> __device__ inline size_t shape() const {
        static_assert(axis==0 || axis==1 || axis==2 || axis==3, "Axis must be 0, 1, 2, or 3.");
        if constexpr (axis==0) { return size_t(batch()); }
        else if constexpr (axis==1) { return size_t(depth()); }
        else if constexpr (axis==2) { return size_t(rows()); }
        else if constexpr (axis==3) { return size_t(cols()); }
    }
    template<int axis> __device__ inline size_t stride() const { 
        static_assert(axis==0 || axis==1 || axis==2 || axis==3, "Axis must be 0, 1, 2, or 3.");
        if      constexpr (axis==0) { return depth()*rows()*cols(); }
        else if constexpr (axis==1) { return rows()*cols(); }
        else if constexpr (axis==2) { return cols(); }
        else if constexpr (axis==3) { return 1; }
    }
};

template<typename _T, int d, int r, int c, typename... TMA_Types> using gl3 = gl<_T, 1, d, r, c, TMA_Types...>;
template<typename _T, int r, int c, typename... TMA_Types>        using gl2 = gl<_T, 1, 1, r, c, TMA_Types...>;
template<typename _T, int c, typename... TMA_Types>               using gl1 = gl<_T, 1, 1, 1, c, TMA_Types...>;

namespace ducks {
namespace gl {
/**
* @brief Concept for all global layouts.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as ducks::gl::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::gl::identifier
}
}

// Structs for initializing global layouts automatically.
// struct unsafe_gl {
//     uint64_t data;
//     int b, d, r, c;
//     unsafe_gl(uint64_t data, int b, int d, int r, int c) : data(data), b(b), d(d), r(r), c(c) {}
// };
template<int N> auto make_unsafe_gl_arg(int param) { // typename std::conditional_t<(N < 0), std::nullptr_t, int>
    if constexpr (N > 0) { return nullptr; }
    else                 { return param;   }
}
template<ducks::gl::all GL, bool safe=true> __host__ inline GL make_gl(uint64_t data, int b, int d, int r, int c) {
    if constexpr (safe) {
        if(GL::__b__ > 0 && b != GL::__b__) {
            throw std::runtime_error("Batch dimension mismatch. Expected: " + std::to_string(GL::__b__) + ", Got: " + std::to_string(b));
        }
        if(GL::__d__ > 0 && d != GL::__d__) {
            throw std::runtime_error("Depth dimension mismatch. Expected: " + std::to_string(GL::__d__) + ", Got: " + std::to_string(d));
        }
        if(GL::__r__ > 0 && r != GL::__r__) {
            throw std::runtime_error("Row dimension mismatch. Expected: " + std::to_string(GL::__r__) + ", Got: " + std::to_string(r));
        }
        if(GL::__c__ > 0 && c != GL::__c__) {
            throw std::runtime_error("Column dimension mismatch. Expected: " + std::to_string(GL::__c__) + ", Got: " + std::to_string(c));
        }
    }
    return GL(
        reinterpret_cast<typename GL::dtype*>(data),
        make_unsafe_gl_arg<GL::__b__>(b),
        make_unsafe_gl_arg<GL::__d__>(d),
        make_unsafe_gl_arg<GL::__r__>(r),
        make_unsafe_gl_arg<GL::__c__>(c)
    );
}

} // namespace kittens
