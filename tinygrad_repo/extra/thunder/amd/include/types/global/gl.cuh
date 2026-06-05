/**
 * @file
 * @brief Templated layouts for global memory.
 */
 
#pragma once

#include "../../common/common.cuh"
#include "../shared/shared.cuh"
#include "util.cuh"

namespace kittens {

/* ----------   Associative dictionary for global layouts  ---------- */

namespace detail {
template<typename... Args>
struct descriptor_dict {
    __host__ __device__ descriptor_dict() {}
    template<typename T> __host__ __device__ descriptor_dict(T _, int b, int d, int r, int c) {}
    __host__ __device__ descriptor_dict(const descriptor_dict &other) {}
};
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
    __device__ inline T& operator[](const coord<ducks::default_type> &idx) const { // yes I am abusing the const qualifier here a bit.
        return raw_ptr[((idx.b*depth() + idx.d)*rows() + idx.r)*cols() + idx.c];
    }
    __device__ inline int idx(const coord<ducks::default_type> &idx) const {
        return ((idx.b*depth() + idx.d)*rows() + idx.r)*cols() + idx.c;
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
            throw std::runtime_error("Batch dimension mismatch.");
        }
        if(GL::__d__ > 0 && d != GL::__d__) {
            throw std::runtime_error("Depth dimension mismatch.");
        }
        if(GL::__r__ > 0 && r != GL::__r__) {
            throw std::runtime_error("Row dimension mismatch.");
        }
        if(GL::__c__ > 0 && c != GL::__c__) {
            throw std::runtime_error("Column dimension mismatch.");
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
