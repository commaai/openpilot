/**
 * @file
 * @brief The basic 16x16 register tile with assembly mode on which larger register tiles are built.
 */

 #pragma once

 #include <type_traits>
 
 #include "../../common/common.cuh"
 #include "rt_layout.cuh"
 #include "rt_shape.cuh"
 #include "rv_layout.cuh"
 
 namespace kittens {
 
 /* ----------  BASE 16x16 SUBTILE STRUCT WITH ASSEMBLY MODE  ---------- */
 
 namespace ducks {
 /**
  * @namespace art_base
  *
  * @brief The namespace where concepts and abstract types for register base (16x16) tiles with assembly mode live.
  */
 namespace art_base {
 /**
  * @brief A dummy type used to identify register base tiles with assembly mode.
  *
  * For a type to quack like an art_base, it should define its identifier as ducks::art_base::identifier.
  * If a type quacks like ducks::art_base::identifier, it will be treated as an art_base by compiler checks.
  */
 struct identifier {};
 }
 } // namespace ducks
 
 /**
  * @brief Basic tile structure for computation in registers with assembly mode.
  *
  * @tparam _T The data type used for the matrix elements.
  * @tparam _layout The layout of the base tile, either row-major or column-major.
  * @tparam _matrix_layout The matrix layout (mfma dimensions).
  * @tparam _register_range The register range for this tile.
  *
  * This type is a mirror of art_base but uses register ranges instead of data arrays
  * for assembly-level register management.
  */
 template<typename _T, ducks::rt_layout::all _layout, ducks::rt_shape::all _shape, typename _register_range>
 struct art_base {
     using identifier = ducks::art_base::identifier; ///< Type identifier for the art_base structure.
     using layout = _layout; ///< Layout of the matrix tile.
     using shape = _shape; ///< Shape of the matrix tile.
     static_assert(kittens::ducks::base_types::T1<_T>); // confirm it's a supported type
     using T = kittens::base_types::packing<_T>::unpacked_type;
     using T2 = kittens::base_types::packing<_T>::packed_type;
     using dtype = T2; ///< Data type of the matrix elements
     using register_range = _register_range; ///< Register range for this tile.
 
     static_assert(
         std::is_same_v<dtype, bf16_2> || std::is_same_v<dtype, float2> || std::is_same_v<dtype, half_2> ||
         std::is_same_v<dtype, __hip_fp8x4_e4m3> || std::is_same_v<dtype, __hip_fp8x4_e4m3_fnuz> ||
         std::is_same_v<dtype, __hip_fp8x4_e5m2> || std::is_same_v<dtype, __hip_fp8x4_e5m2_fnuz>,
         "art_base was provided an unsupported type."
     );
 
     static constexpr int rows                 = shape::rows; ///< Number of rows.
     static constexpr int cols                 = shape::cols; ///< Number of cols.
     static constexpr int stride               = shape::stride; ///< Stride of the matrix tile.
     static constexpr int num_elements         = rows*cols;
     static constexpr int elements_per_thread  = num_elements / kittens::WARP_THREADS;
     static constexpr int num_strides          = shape::num_strides;
 
     static constexpr int reductions = std::is_same_v<layout, ducks::rt_layout::row> ? cols : rows;
     static constexpr int threads_per_reduction = reductions / elements_per_thread;
     static constexpr int elements_per_stride_group = threads_per_reduction * stride;
 
     static_assert(num_elements % stride == 0, "num_elements must be divisible by stride");
 
     static constexpr int packed_per_thread    = (elements_per_thread / base_types::packing<dtype>::num()) ; // 2
     static constexpr int registers_per_thread = packed_per_thread * sizeof(dtype) / 4; // 2 or 4, registers are 32-bit words
     static constexpr int registers_per_stride = registers_per_thread / num_strides;
 
     // Type check: ensure register range size matches the required number of registers per thread
     static_assert(register_range::size == registers_per_thread,
                   "Register range size must match registers_per_thread for art_base");
 
     using row_vec_layout = std::conditional_t<std::is_same_v<layout, ducks::rt_layout::row>, ducks::rv_layout::align, ducks::rv_layout::ortho>; // for holding column reductions
     using col_vec_layout = std::conditional_t<std::is_same_v<layout, ducks::rt_layout::row>, ducks::rv_layout::ortho, ducks::rv_layout::align>; // for holding row reductions
 
     register_range registers; ///< The register range for the base tile instead of data array
 };
 
 /* ----------  CONCEPTS  ---------- */
 
 namespace ducks {
 namespace art_base {
 /**
 * @brief Concept for all register base tiles with assembly mode.
 * @tparam T The type to check against the concept requirements.
 *
 * Requires:
 * - T has a nested type identifier that is the same as art_base::identifier.
 */
 template<typename T> concept all = requires {
     typename T::identifier; // Checks if T::identifier exists
 } && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::art_base::identifier
 } // namespace art_base
 } // namespace ducks
 
 /* ----------  WRAPPERS FOR PRETTINESS  ---------- */
 
 // Forward declare range for default template parameter
 namespace ducks { namespace art { template<int, int> struct range; } }
 
 template<ducks::rt_layout::all L=ducks::rt_layout::row, ducks::rt_shape::all S=ducks::rt_shape::rt_16x16, typename R=ducks::art::range<0, 1>> using art_base_fl = art_base<float, L, S, R>;
 template<ducks::rt_layout::all L=ducks::rt_layout::row, ducks::rt_shape::all S=ducks::rt_shape::rt_16x16, typename R=ducks::art::range<0, 1>> using art_base_bf = art_base<bf16, L, S, R>;
 template<ducks::rt_layout::all L=ducks::rt_layout::row, ducks::rt_shape::all S=ducks::rt_shape::rt_16x16, typename R=ducks::art::range<0, 1>> using art_base_hf = art_base<half, L, S, R>;
 
 } // namespace kittens