/**
 * @file
 * @brief The basic 8x8 register tile on which larger register tiles are built.
 */
#pragma once // todo: col/row layout if needed
#include <metal_stdlib>

#include "../../common/common.metal"
#include "rt_layout.metal"
#include "rv_layout.metal"
namespace mittens {
/* ----------  BASE 8x8 SUBTILE STRUCT  ---------- */
namespace ducks {
/**
 * @namespace rt_base
 *
 * @brief The namespace where concepts and abstract types for register base (16x16) tiles live.
 */
namespace rt_base {
/**
 * @brief A dummy type used to identify register base tiles.
 *
 * For a type to quack like an rt_base, it should define its identifier as ducks::rt_base::identifier.
 * If a type quacks like ducks::rt_base::identifier, it will be treated as an rt_base by compiler checks.
 */
struct identifier {};
}
template <typename T>
static constexpr bool is_register_tile_base() {
    return metal::is_same<typename T::identifier, ducks::rt_base::identifier>::value;
}
template <typename RT>
static constexpr void assert_register_tile_base() {
    static_assert(is_register_tile_base<RT>(), "T must be a rt_base");
}
} // namespace ducks
    
/**
 * @brief Basic tile structure for computation in registers.
 *
 * @tparam T2 The packed data type used for the matrix elements.
 * @tparam _layout The layout of the base tile, either row-major or column-major.
 *
 * This type is a primarily utility for building larger inline templates
 * out of PTX primitives and managing layouts.
 *
 * In general, you probably want a row-major tile, unless you specifically want to call mma
 */
template <typename _T, typename _layout>
struct rt_base {
    using identifier = ducks::rt_base::identifier; ///< Type identifier for the rt_base structure.
    using layout = _layout; ///< Layout of the matrix tile.
    static_assert(ducks::base_types::isT1Type<_T>(), "rt_base was provided an unsupported type");
    static_assert(ducks::is_rt_layout<layout>(), "rt_base was provided an unsupported layout");
    using T  = typename base_types::packing<_T>::unpacked_type;
    using T2 = typename base_types::packing<_T>::packed_type;
    using dtype = T;
    
    
    
    static constant constexpr const int tile_size            = mittens::TILE_DIM;
    static constant constexpr const int rows                 = tile_size;
    static constant constexpr const int cols                 = tile_size;
    static constant constexpr const int num_elements         = rows*cols;
    static constant constexpr const int elements_per_thread  = num_elements / mittens::SIMD_THREADS;
    
    static constant constexpr const int registers_per_thread = elements_per_thread;
    static constant constexpr const int packed_per_thread    = elements_per_thread / base_types::packing<T2>::num();
    metal::simdgroup_matrix<dtype, mittens::TILE_DIM, mittens::TILE_DIM> data;
    
    using row_vec_layout = metal::conditional_t<metal::is_same_v<layout, ducks::rt_layout::row>, ducks::rv_layout::align, ducks::rv_layout::ortho>; // for holding column reductions
    
    using col_vec_layout = metal::conditional_t<metal::is_same_v<layout, ducks::rt_layout::row>, ducks::rv_layout::ortho, ducks::rv_layout::align>; // for holding row reductions
};
    
/* ----------  WRAPPERS FOR PRETTINESS  ---------- */
    
template<typename L=ducks::rt_layout::row> using rt_base_fl = rt_base<float, L>;
template<typename L=ducks::rt_layout::row> using rt_base_bf = rt_base<bf16, L>;
template<typename L=ducks::rt_layout::row> using rt_base_hf = rt_base<half, L>;

     
}
 
