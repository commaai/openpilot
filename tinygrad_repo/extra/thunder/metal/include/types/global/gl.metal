/**
 * @file
 * @brief Templated layouts for global memory.
 */
 
#pragma once

#include "../../common/common.metal"
#include "../shared/shared.metal"
#include "../register/register.metal"
#include "util.metal"


namespace mittens {
/* ----------   Associative dictionary for global layouts  ---------- */

namespace detail {
template<typename... Args>
struct descriptor_dict {
    METAL_FUNC descriptor_dict() {}
    template<typename T> METAL_FUNC descriptor_dict(T _, int b, int d, int r, int c) {}
    METAL_FUNC descriptor_dict(thread const descriptor_dict &other) {}
};
}

/* ----------  Global layout descriptor  ---------- */

namespace ducks {
namespace gl {
struct identifier {};
}

template <typename T>
static constexpr bool is_tile() {
    return mittens::ducks::is_shared_tile<T>() || mittens::ducks::is_register_tile<T>();
}
    
template <typename T>
static constexpr bool is_vec() {
    return mittens::ducks::is_shared_vector<T>() || mittens::ducks::is_register_vector<T>();
}
}


template<typename _T, int b, int d, int r, int c>
struct gl {
    using identifier = ducks::gl::identifier;
    
    using T     = typename base_types::packing<_T>::unpacked_type;
    using T2    = typename base_types::packing<_T>::packed_type;
    using dtype = T;
    
    device T* raw_ptr;
    
    ducks::g::make_dim_t<b> batch;
    ducks::g::make_dim_t<d> depth;
    ducks::g::make_dim_t<r> rows;
    ducks::g::make_dim_t<c> cols;
//    int batch;
//    int depth;
//    int rows;
//    int cols;
        
    METAL_FUNC gl(device T *_data,
                  ducks::g::make_arg_t<b> _batch,
                  ducks::g::make_arg_t<d> _depth,
                  ducks::g::make_arg_t<r> _rows,
                  ducks::g::make_arg_t<c> _cols) :
    raw_ptr(_data), batch(_batch), depth(_depth), rows(_rows), cols(_cols) {
    }
//    METAL_FUNC gl(device T *_data,
//                  int _batch,
//                  int _depth,
//                  int _rows,
//                  int _cols) :
//    raw_ptr(_data), batch(_batch), depth(_depth), rows(_rows), cols(_cols) {
//    }
//    
    METAL_FUNC gl(thread const gl &other) :
    raw_ptr(other.raw_ptr), batch(other.batch), depth(other.depth), rows(other.rows), cols(other.cols) {}
    
    METAL_FUNC gl(constant const gl &other) :
    raw_ptr(other.raw_ptr), batch(other.batch), depth(other.depth), rows(other.rows), cols(other.cols) {}

    METAL_FUNC device T& operator[](const thread coord &idx) {
        return raw_ptr[((idx.b*depth + idx.d)*rows + idx.r)*cols + idx.c];
    }
    METAL_FUNC device const T& operator[](const thread coord &idx) const {
        return raw_ptr[((idx.b*depth + idx.d)*rows + idx.r)*cols + idx.c];
    }
    template<typename TILE>
    METAL_FUNC typename metal::enable_if<ducks::is_tile<TILE>(), device T&>::type
    get(const thread coord &idx) {
        return raw_ptr[((idx.b*depth + idx.d)*rows + idx.r*TILE::rows)*cols + idx.c*TILE::cols];
    }
    template<typename TILE>
    METAL_FUNC typename metal::enable_if<ducks::is_tile<TILE>(), device const T&>::type
    get(const thread coord &idx) const {
        return raw_ptr[((idx.b*depth + idx.d)*rows + idx.r*TILE::rows)*cols + idx.c*TILE::cols];
    }
    template<typename VEC>
    METAL_FUNC typename metal::enable_if<ducks::is_vec<VEC>(), device T&>::type
    get(const thread coord &idx) {
        return raw_ptr[((idx.b*depth + idx.d)*rows + idx.r)*cols + idx.c*VEC::length];
    }
    template<typename VEC>
    METAL_FUNC typename metal::enable_if<ducks::is_vec<VEC>(), device const T&>::type
    get(const thread coord &idx) const {
        return raw_ptr[((idx.b*depth + idx.d)*rows + idx.r)*cols + idx.c*VEC::length];
    }
    METAL_FUNC size_t row_stride() const { return cols; }
};

namespace ducks {
template <typename T>
struct has_gl_identifier {
    static constant constexpr bool value = false; // Default case
};

template <typename _T, int b, int d, int r, int c>
struct has_gl_identifier<mittens::gl<_T, b, d, r, c>> {
    static constant constexpr bool value = true;
};

template <typename GL>
static constexpr bool is_global_layout() {
    return has_gl_identifier<GL>::value;
}
template <typename GL>
static constexpr void assert_gl() {
    static_assert(is_global_layout<GL>(), "T must be a gl");
}
}

    
    
    
    
    
    
template<typename _T, int b, int d, int r, int c>
struct gl2 {
    using identifier = ducks::gl::identifier;
    
    using T     = typename base_types::packing<_T>::unpacked_type;
    using T2    = typename base_types::packing<_T>::packed_type;
    using dtype = T;
    
    device T* raw_ptr;
    
//    ducks::g::make_dim_t<b> batch;
//    ducks::g::make_dim_t<d> depth;
//    ducks::g::make_dim_t<r> rows;
//    ducks::g::make_dim_t<c> cols;
//        
//    METAL_FUNC gl2(device T *_data,
//                  ducks::g::make_arg_t<b> _batch,
//                  ducks::g::make_arg_t<d> _depth,
//                  ducks::g::make_arg_t<r> _rows,
//                  ducks::g::make_arg_t<c> _cols) :
//    raw_ptr(_data), batch(_batch), depth(_depth), rows(_rows), cols(_cols) {
//    }
    
    int batch;
    int depth;
    int rows;
    int cols;
        
    METAL_FUNC gl2(device T *_data,
                  int _batch,
                  int _depth,
                  int _rows,
                  int _cols) :
    raw_ptr(_data), batch(_batch), depth(_depth), rows(_rows), cols(_cols) {
    }
    
    
//    METAL_FUNC gl2(thread const gl2 &other) :
//    raw_ptr(other.raw_ptr), batch(other.batch), depth(other.depth), rows(other.rows), cols(other.cols) {}
//    
//    METAL_FUNC gl2(constant const gl2 &other) :
//    raw_ptr(other.raw_ptr), batch(other.batch), depth(other.depth), rows(other.rows), cols(other.cols) {}

    METAL_FUNC device T& operator[](const thread coord &idx) {
        return raw_ptr[((idx.b*depth + idx.d)*rows + idx.r)*cols + idx.c];
    }
    METAL_FUNC device const T& operator[](const thread coord &idx) const {
        return raw_ptr[((idx.b*depth + idx.d)*rows + idx.r)*cols + idx.c];
    }
    template<typename TILE>
    METAL_FUNC typename metal::enable_if<ducks::is_tile<TILE>(), device T&>::type
    get(const thread coord &idx) {
        return raw_ptr[((idx.b*depth + idx.d)*rows + idx.r*TILE::rows)*cols + idx.c*TILE::cols];
    }
    template<typename TILE>
    METAL_FUNC typename metal::enable_if<ducks::is_tile<TILE>(), device const T&>::type
    get(const thread coord &idx) const {
        return raw_ptr[((idx.b*depth + idx.d)*rows + idx.r*TILE::rows)*cols + idx.c*TILE::cols];
    }
    template<typename VEC>
    METAL_FUNC typename metal::enable_if<ducks::is_vec<VEC>(), device T&>::type
    get(const thread coord &idx) {
        return raw_ptr[((idx.b*depth + idx.d)*rows + idx.r)*cols + idx.c*VEC::length];
    }
    template<typename VEC>
    METAL_FUNC typename metal::enable_if<ducks::is_vec<VEC>(), device const T&>::type
    get(const thread coord &idx) const {
        return raw_ptr[((idx.b*depth + idx.d)*rows + idx.r)*cols + idx.c*VEC::length];
    }
    METAL_FUNC size_t row_stride() const { return cols; }
};

}
