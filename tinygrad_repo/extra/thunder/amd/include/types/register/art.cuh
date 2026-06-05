/**
 * @file
 * @brief The primitives for register tiles with assembly mode.
 */

 #pragma once

 #include <type_traits>
 
 #include "../../common/common.cuh"
 #include "art_base.cuh"
 #include "rv.cuh"
 
 namespace kittens {
 
 /* ----------  MAIN TILE STRUCT WITH ASSEMBLY MODE  ---------- */
 
 // helper struct for type inference
 namespace ducks {
 /**
  * @namespace art
  *
  * @brief The namespace where concepts and abstract types for register tiles with assembly mode live.
  */
 namespace art {
 
 // Primitives to define register ranges
 // ---------- type-list ----------
 template <typename... Ts> struct type_list {
     static constexpr int size = sizeof...(Ts);
 };
 
 template <typename L1, typename L2> struct concat;
 template <typename... A, typename... B>
 struct concat<type_list<A...>, type_list<B...>> { using type = type_list<A..., B...>; };
 
 // Helper to get size of type_list
 template <typename TList> struct type_list_size;
 template <typename... Ts>
 struct type_list_size<type_list<Ts...>> {
     static constexpr int value = sizeof...(Ts);
 };
 template <typename TList>
 static constexpr int type_list_size_v = type_list_size<TList>::value;
 
 // ---------- range ----------
 template <int L, int R>
 struct range {
     static_assert(L <= R, "range requires L <= R");
     static constexpr int lo = L, hi = R;
     static constexpr int size = R - L + 1; ///< Number of registers in this range
 };
 
 // ---------- split one range with alignment to multiples of N ----------
 template <int L, int R, int N, bool Done = (L > R)>
 struct split_one;
 
 // base
 template <int L, int R, int N>
 struct split_one<L, R, N, true> { using type = type_list<>; };
 
 // step
 template <int L, int R, int N>
 struct split_one<L, R, N, false> {
     static_assert(N > 0, "N must be > 0");
     static_assert(L + N - 1 <= R, "L + N - 1 must be <= R");
     // Highest index within L's alignment block: floor(L/N)*N + (N-1)
     static constexpr int end = L + N - 1;
 
     using head = range<L, end>;
     using tail = typename split_one<end + 1, R, N>::type;
     using type = typename concat<type_list<head>, tail>::type;
 };
 
 // ---------- split many ranges ----------
 template <typename RList, int N> struct split_many;
 template <int N>
 struct split_many<type_list<>, N> { using type = type_list<>; };
 
 template <typename R1, typename... Rs, int N>
 struct split_many<type_list<R1, Rs...>, N> {
     using first = typename split_one<R1::lo, R1::hi, N>::type;
     using rest  = typename split_many<type_list<Rs...>, N>::type;
     using type  = typename concat<first, rest>::type;
 };
 
 template <typename RList, int N>
 using split_many_t = typename split_many<RList, N>::type;
 
 // Helper to get the Nth range from a type_list
 template <typename RangeList, int N>
 struct get_nth_range;
 
 template <typename R1, typename... Rs, int N>
 struct get_nth_range<type_list<R1, Rs...>, N> {
     using type = typename std::conditional_t<N == 0, R1, typename get_nth_range<type_list<Rs...>, N-1>::type>;
 };
 
 template <typename R1, typename... Rs>
 struct get_nth_range<type_list<R1, Rs...>, 0> {
     using type = R1;
 };
 
 template <typename RangeList, int N>
 using get_nth_range_t = typename get_nth_range<RangeList, N>::type;
 
 // ---------- transpose 2D layout ----------
 // Transposes a type_list representing an H×W grid into W×H
 // Original: ranges are in row-major order [r0c0, r0c1, ..., r1c0, r1c1, ...]
 // Result: ranges are in column-major order [r0c0, r1c0, ..., r0c1, r1c1, ...]
 template <typename TList, int H, int W, int... Indices>
 struct transpose_2d_impl;
 
 // Base case: no more indices to process
 template <typename TList, int H, int W>
 struct transpose_2d_impl<TList, H, W> {
     using type = type_list<>;
 };
 
 // Recursive case: process one index at a time
 template <typename TList, int H, int W, int I, int... Rest>
 struct transpose_2d_impl<TList, H, W, I, Rest...> {
     // Convert linear index I (in column-major order) to row-major index
     // In col-major: element at column c, row r has index r + c*H
     // We want to map this to row-major: element at row r, column c has index r*W + c
     static constexpr int r = I % H;  // row index
     static constexpr int c = I / H;  // column index
     static constexpr int src_idx = r * W + c;  // source index in row-major
 
     using current = type_list<get_nth_range_t<TList, src_idx>>;
     using rest = typename transpose_2d_impl<TList, H, W, Rest...>::type;
     using type = typename concat<current, rest>::type;
 };
 
 // Helper to generate index sequence and call impl
 template <typename TList, int H, int W>
 struct transpose_2d_helper {
     static_assert(type_list_size_v<TList> == H * W, "List size must equal H * W");
 
     template <int... Is>
     static auto make_impl(std::integer_sequence<int, Is...>)
         -> typename transpose_2d_impl<TList, H, W, Is...>::type;
 
     using type = decltype(make_impl(std::make_integer_sequence<int, H * W>{}));
 };
 
 template <typename TList, int H, int W>
 using transpose_2d = typename transpose_2d_helper<TList, H, W>::type;
 
 // Type alias for register range types - any range type works
 template<typename T>
 concept register_range_t = requires {
     T::lo;
     T::hi;
     T::size;
 };
 
 template<typename RList>
 __device__ inline static void clobber() {
 
   using registers = ducks::art::split_many_t<RList, 1>;
   [&]<std::size_t... Rs>(std::index_sequence<Rs...>) {
     ([&]<std::size_t R>() {
       macros::clobber_gpr<ducks::art::get_nth_range_t<registers, R>::lo>();
     }.template operator()<Rs>(), ...);
   }(std::make_index_sequence<registers::size>{});
 
 }
 
 /**
  * @brief A dummy type used to identify register tiles with assembly mode.
  *
  * For a type to quack like an art, it should define its identifier as ducks::art::asm_identifier.
  * If a type quacks like ducks::art::asm_identifier, it will be treated as an art by compiler checks.
  */
 struct asm_identifier {}; ///< Unique identifier for assembly-mode tiles only
 } // namespace art
 } // namespace ducks
 
 /**
  * @brief Main tile structure for manipulating data in registers with assembly mode.
  *
  * @tparam T The data type used for the matrix elements.
  * @tparam _rows The number of rows in the tile.
  * @tparam _cols The number of columns in the tile.
  * @tparam _layout The layout of the internal base tiles, either row-major or column-major.
  * @tparam _matrix_layout The matrix layout (mfma dimensions).
  * @tparam _register_ranges A type_list of register ranges to distribute among base tiles.
  *
  * This structure is designed to handle matrix tiles with explicit register management,
  * automatically distributing register ranges among the constituent base tiles.
  */
 template<typename _T, int _rows, int _cols, ducks::rt_layout::all _layout=ducks::rt_layout::row, ducks::rt_shape::all _shape=ducks::rt_shape::rt_16x16, typename _register_ranges=ducks::art::type_list<ducks::art::range<0, 1>>>
 struct art {
     using identifier = ducks::art::asm_identifier; ///< Type identifier for the art structure - distinct from art.
     using layout = _layout; ///< Layout of the matrix tile.
     using shape = _shape; ///< Shape of the matrix tile.
     static_assert(kittens::ducks::base_types::T1<_T>); // confirm it's a supported type
     using T = kittens::base_types::packing<_T>::unpacked_type;
     using T2 = kittens::base_types::packing<_T>::packed_type;
     using dtype = T2; ///< Data type of the matrix elements
     using register_ranges = _register_ranges; ///< The list of register ranges for distribution
 
     static constexpr int rows                = _rows; ///< Total number of rows.
     static_assert(rows % art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::rows == 0, "Rows must be divisible by the tile size");
     static constexpr int cols                = _cols; ///< Total number of columns.
     static_assert(cols % art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::cols == 0, "Columns must be divisible by the tile size");
     static constexpr int height              = rows / art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::rows; ///< Height in subtiles.
     static constexpr int width               = cols / art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::cols; ///< Width in subtiles.
     
     // Base tile attributes
     static constexpr int base_tile_rows        = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::rows;        ///< Size of the base tile.
     static constexpr int base_tile_cols        = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::cols;        ///< Size of the base tile.
     static constexpr int base_tile_stride      = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::stride;        ///< Stride of the base tile.
     static constexpr int base_tile_num_strides = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::num_strides;        ///< Number of strides of the base tile.
     static constexpr int base_tile_reductions  = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::reductions;        ///< Number of reductions of the base tile.
     static constexpr int base_tile_threads_per_reduction = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::threads_per_reduction;        ///< Number of threads per reduction of the base tile.
     static constexpr int base_tile_elements_per_stride_group = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::elements_per_stride_group;        ///< Number of elements per stride group of the base tile.
     
     static constexpr int num_elements        = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::num_elements        * width * height; ///< Total number of elements.
     static constexpr int elements_per_thread = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::elements_per_thread * width * height; ///< Elements handled per thread.
     static constexpr int packed_per_thread   = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::packed_per_thread   * width * height; ///< Packed elements per thread.
     static constexpr int packed_per_base_tile    = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::packed_per_thread; ///< Packed elements per tile.
     static constexpr int elements_per_base_tile  = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::elements_per_thread; ///< Elements per thread per base tile.
 
     static constexpr int registers_per_stride = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::registers_per_stride;        ///< Number of registers per stride of the base tile.
 
     // Static assertion to ensure we have enough register ranges for all base tiles
     static_assert(ducks::art::type_list_size_v<register_ranges> == height * width,
         "Not enough register ranges provided for all base tiles in art");
     // Helper template to create base tiles with specific register ranges
     template<int Row, int Col>
     using base_tile_type = art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, Row * width + Col>>;
    // Note: actual tiles are created via base_tile_type template, not stored as array
    using row_vec = rv<T, cols, base_tile_cols, shape, typename art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::row_vec_layout>; ///< A type representing a row vector for this tile.
    using col_vec = rv<T, rows, base_tile_rows, shape, typename art_base<T, layout, shape, ducks::art::get_nth_range_t<register_ranges, 0>>::col_vec_layout>; ///< A type representing a column vector for this tile.
 };
 
 /* ----------  CONCEPTS  ---------- */
 
 namespace ducks {
     namespace art {
     /**
     * @brief Concept for all assembly register tiles.
     * @tparam T The type to check against the concept requirements.
     *
     * Requires:
     * - T has a nested type identifier that is the same as art::asm_identifier.
     */
     template<typename T> concept all = requires {
         typename T::identifier; // Checks if T::identifier exists
     } && std::is_same_v<typename T::identifier, asm_identifier>; // Checks if T::identifier is ducks::art::asm_identifier
     /**
     * @brief Concept for register tiles with row layout.
     * @tparam T The type to check against the concept requirements.
     *
     * Requires:
     * - T is a register tile.
     * - T has an internal type layout that is ducks::rt_layout::row.
     */
     template<typename T>
     concept row_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::row>;
     /**
     * @brief Concept for register tiles with col layout.
     * @tparam T The type to check against the concept requirements.
     *
     * Requires:
     * - T is a register tile.
     * - T has an internal type layout that is ducks::rt_layout::col.
     */
     template<typename T>
     concept col_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::col>;
     
     
     } // namespace art
     } // namespace ducks
 
 /* ----------  WRAPPERS FOR PRETTINESS  ---------- */
 
 template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row, ducks::rt_shape::all shape=ducks::rt_shape::rt_16x16, typename ranges=ducks::art::type_list<ducks::art::range<0, 1>>> using art_fl = art<float, _r, _c, layout, shape, ranges>;
 template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row, ducks::rt_shape::all shape=ducks::rt_shape::rt_16x16, typename ranges=ducks::art::type_list<ducks::art::range<0, 1>>> using art_bf = art<bf16,  _r, _c, layout, shape, ranges>;
 template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row, ducks::rt_shape::all shape=ducks::rt_shape::rt_16x16, typename ranges=ducks::art::type_list<ducks::art::range<0, 1>>> using art_hf = art<half,  _r, _c, layout, shape, ranges>;
 
 } // namespace kittens