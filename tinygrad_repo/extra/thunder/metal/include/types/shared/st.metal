/**
 * @file
 * @brief The Thundermittens shared tile struct.
 */

#pragma once // kinda done

/*
 add subtile, make it work
 */
#include <metal_stdlib>
#include "../../common/common.metal"
#include "sv.metal"
/* ----------  MAIN TILE STRUCT  ---------- */

// these are helper structs for type inference
namespace mittens {
    
namespace ducks {
/**
 * @namespace st
 *
 * @brief The namespace where concepts and abstract types for shared tiles live.
 */
namespace st {
/**
 * @brief A dummy type used to identify shared tiles.
 *
 * For a type to quack like an st, it should define its identifier as ducks::st::identifier.
 * If a type quacks like ducks::st::identifier, it will be treated as an st by compiler checks.
 * This is particularly useful for subtiles.
 */
struct identifier {};
} // namespace st
    
}// namespace ducks
 
// Forward declaration of subtile
template<
    typename ST,
    int _subtile_height,
    int _subtile_width
>
struct st_subtile;

/**
 * @brief Shared memory tile structure for various data types and layouts.
 *
 * @tparam T The data type of the elements in the tile. Not packed!
 * @tparam _height The height of the tile in units of 8-element subtiles.
 * @tparam _width The width of the tile in units of 8-element subtiles.
 */
template<typename _T, int _rows, int _cols>
struct mittens_DEFAULT_ALIGN st {
    using identifier = ducks::st::identifier; ///< Type identifier for the rt structure.
    using T  = typename base_types::packing<_T>::unpacked_type;
    using T2 = typename base_types::packing<_T>::packed_type;
    using dtype = T; ///< Data type of the elements in the tile.
    static_assert(base_types::packing<dtype>::num() == 1, "st type must be 1-packed (float, bf16, etc)"); // must be a 1-packed type (e.g. float, bf16, etc)
    // define underlying data as same as that projected, to make clear that this is *not* a subtile.
    static constant constexpr const int underlying_rows          = _rows;
    static constant constexpr const int underlying_cols          = _cols;
    static constant constexpr const int underlying_height        = _rows / TILE_DIM;
    static constant constexpr const int underlying_width         = _cols / TILE_DIM;
    static constant constexpr const int underlying_num_elements  = underlying_rows * underlying_cols;

    static constant constexpr const int rows                = _rows; ///< Total number of rows in the tile.
    static_assert(rows % TILE_DIM == 0, "Rows must be divisible by the tile dimension");
    static constant constexpr const int cols                = _cols; ///< Total number of cols in the tile.
    static_assert(cols % TILE_DIM == 0, "Rows must be divisible by the tile dimension");
    static constant constexpr const int height              = _rows / TILE_DIM; ///< Height of the tile in terms of 16-element subtiles.
    static constant constexpr const int width               = _cols / TILE_DIM; ///< Width of the tile in terms of 16-element subtiles.
    
    static constant constexpr const int num_elements        = rows * cols; ///< Total number of elements in the tile.
//    static constant constexpr const int row_incr            = 32 / memcpy_per_row;
    

    
    dtype data[rows*cols]; ///< Raw data storage for the tile.
    
    
     
    /* ---------- static vars ---------- */
//    /* static METAL_FUNC threadgroup float* idx(threadgroup float *ptr, int r, int c)*/
    static constant constexpr const int swizzle_bytes = underlying_width % 4 == 0 ? 128 : underlying_width%2==0 ? 64 : 32;
    static constant constexpr const int swizzle_repeat = swizzle_bytes * 8;
    static constant constexpr const int subtile_cols   = swizzle_bytes / sizeof(T);
    
    static constant constexpr const int subtile_cols_log2 = (swizzle_bytes == 128) ? 5 : (swizzle_bytes == 64) ? 4 : 3;
    static constant constexpr const int subtile_cols_mask = subtile_cols - 1;
    static constant constexpr int swizzle_mask = swizzle_repeat - 1;
    static constant constexpr int swizzle_offset_shift = 7;
    static constant constexpr int swizzle_adjust_shift = 4;
    static constant constexpr int mask = (swizzle_repeat - 1) >> swizzle_offset_shift;
    
//    static constant constexpr const int load_block_bytes = 8;
    static constant constexpr const int laod_block_words = 4;
//    static constant constexpr const int load_block_words = 2;
    static constant constexpr const int col_load_block_words = cols / laod_block_words;
    static constant constexpr const int load_block_words_mask = laod_block_words - 1;
    
 
    static METAL_FUNC threadgroup T* idx(threadgroup T * __restrict ptr, int2 coord) { // naive row-major index default
        int r = coord.x, c = coord.y;
        return ptr + r * underlying_cols + c;
//
//        c = (c + ((r / 2) * 8)) % cols;
//        return ptr + r * underlying_cols + c;
//// CORRECT 0.124 | 0.168
//        const int outer_idx = c/subtile_cols;
//        const uint64_t addr = (uint64_t)(&ptr[outer_idx*rows*subtile_cols + r*subtile_cols + c%subtile_cols]);
//        const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
//        return (threadgroup T*)(addr ^ swizzle);
            
//        const int outer_idx = c/subtile_cols;
//        ptr = &ptr[outer_idx*rows*subtile_cols + r*subtile_cols + c%subtile_cols];
//        const int swizzle = (((uintptr_t)ptr % swizzle_repeat) >> 7) << 4;
//        return (threadgroup T*)((uintptr_t)ptr ^ swizzle);
////
//// CORRECT  0.097 | 0.120
//        int idx = (((c >> subtile_cols_log2) * rows + r) << subtile_cols_log2) + (c & subtile_cols_mask);
//        // Compute address in bytes (since ptr is float*, multiply idx by sizeof(float) = 4)
//        int addr_bytes = idx << 2; // Equivalent to idx * 4
//        // Compute swizzle without modulo operation
//        int swizzle = (((addr_bytes & swizzle_mask) >> 7) << 4);
//        // Compute final swizzled address
//        return (threadgroup T*)((threadgroup char*)ptr + (addr_bytes ^ swizzle));
//    
//// CORRECT ____ | 0.169
//    int idx = (((c >> subtile_cols_log2) * rows + r) << subtile_cols_log2) + (c & subtile_cols_mask);
//
//    // Compute address in bytes (since ptr is float*, multiply idx by sizeof(float) = 4)
//    uint64_t addr_bytes = ((uint64_t)ptr) + ((uint64_t)idx << 2); // Full address in bytes
//
//    // Compute swizzle including the base address
//    int swizzle = ((addr_bytes % swizzle_repeat) >> 7) << 4;
//
//    // Compute final swizzled address
//    addr_bytes ^= swizzle;
//
//    // Return the swizzled address
//    return (threadgroup float*)(addr_bytes);
//
    }
    static METAL_FUNC uint32_t idx(uint32_t ptr, int2 coord) { // naive row-major index
        int r = coord.x, c = coord.y; // alias
        return ptr + sizeof(T) * (r * underlying_cols + c);
        
//        c = (c + ((r / 2) * 8)) % cols;
//        return ptr + r * underlying_cols + c;
//        return ptr + sizeof(T) * (r * underlying_cols + c);
    }
    /**
     * @brief Access a shared tile element using a row and column, as if the tile were row-major.
     *
     * This is the preferred way to access memory within a shared tile, which abstracts
     * indexing calculations for swizzled layouts.
     */
    METAL_FUNC threadgroup T& operator[](thread const int2& rowcol) threadgroup {
        return *idx(data, rowcol);
    }
    METAL_FUNC const threadgroup T& operator[](thread const int2 &rowcol) const threadgroup {
        return *(const threadgroup T*)idx((threadgroup T*)data, rowcol);
    }
        
    METAL_FUNC threadgroup T& operator[](int idx) threadgroup {
        return data[idx];
    }
    METAL_FUNC const threadgroup T& operator[](int idx) const threadgroup {
        return data[idx];
    }
    
    using col_vec = sv<dtype, rows>; ///< Column vector type for this tile
    using row_vec = sv<dtype, cols>; ///< Row vector type for this tile
    template<int subtile_rows, int subtile_cols> using subtile = st_subtile<
            st<T, rows, cols>, subtile_rows, subtile_cols
        >; ///< A templated subtile type wrapper for this tile.
};


/**
 * @brief A reference into a chunk of shared tile memory.
 *
 * The st_subtile is a drop-in replacement for an st which internally
 * references the appropriate memory while performing minimal address
 * calculations. You should never create this directly, but instead
 * have subtile_inplace return it for you instead. (`auto` is nice.)
 *
 * You can generally just pretend this is an st. But not for wgmma's.
 */
template<
    typename _ST,
    int _subtile_rows,
    int _subtile_cols
>
struct st_subtile {
    using identifier = ducks::st::identifier; // i quack like an st, gcc will never know the difference
    using ST = _ST;
    using T = typename ST::T;
    using T2 = typename ST::T2;
    using dtype = T; ///< Data type of the elements in the tile.
    
    
    constant static constexpr int underlying_rows          = ST::underlying_rows;
    static_assert(underlying_rows % TILE_DIM == 0, "Underlying rows must be divisible by the tile dimension");
    constant static constexpr int underlying_cols          = ST::underlying_cols;
    static_assert(underlying_cols % TILE_DIM == 0, "Underlying cols must be divisible by the tile dimension");
    constant static constexpr int underlying_height        = ST::underlying_height;
    constant static constexpr int underlying_width         = ST::underlying_width;
    constant static constexpr int underlying_num_elements  = ST::underlying_num_elements;

    constant static constexpr int rows                = _subtile_rows;
    static_assert(rows % TILE_DIM == 0, "Rows must be divisible by the tile dimension");
    constant static constexpr int cols                = _subtile_cols;
    static_assert(cols % TILE_DIM == 0, "Cols must be divisible by the tile dimension");
    constant static constexpr int height              = rows / TILE_DIM;
    constant static constexpr int width               = cols / TILE_DIM;
    constant static constexpr int num_elements        = rows * cols;

//    constant static constexpr int swizzle_bytes = ST::swizzle_bytes;

//    device dtype *data;
    threadgroup T* data;
    int row_offset, col_offset;

//    METAL_FUNC st_subtile(threadgroup ST &src, int2 rowcol) {
//        data = reinterpret_cast<uint64_t>(&src.data[0]);
//        row_offset = rowcol.x * rows;
//        col_offset = rowcol.y * cols;
//    }
//    void METAL_FUNC init_subtile(threadgroup ST &src, int2 rowcol) {
////        data = &(src.data[0]);
//        row_offset = rowcol.x * rows;
//        col_offset = rowcol.y * cols;
//    }
    template<typename SUBTILE, typename ST>
    static void METAL_FUNC init_subtile(threadgroup SUBTILE& sub_st, threadgroup ST& src, int2 rowcol) {
        sub_st.data = (threadgroup T*)src.data;
        sub_st.row_offset = rowcol.x * rows;
        sub_st.col_offset = rowcol.y * cols;
    }
    
    template<typename SUBTILE, typename ST>
    static void METAL_FUNC init_subtile(thread SUBTILE& sub_st, threadgroup ST& src, int2 rowcol) {
        sub_st.data = (threadgroup T*)src.data;
        sub_st.row_offset = rowcol.x * rows;
        sub_st.col_offset = rowcol.y * cols;
    }

//    METAL_FUNC threadgroup T* idx(threadgroup T *ptr, const int2 coord) { // naive row-major index default
//        int r = coord.x+row_offset, c = coord.y+col_offset; // alias
//        return ptr + r * underlying_cols + c;
//    }
//    // Add this const overload of idx
//    METAL_FUNC const threadgroup T* idx(const threadgroup T *ptr, const int2 coord) const {
//        int r = coord.x + row_offset, c = coord.y + col_offset;
//        return ptr + r * underlying_cols + c;
//    }
//
//    METAL_FUNC uint32_t idx(uint32_t ptr, const int2 coord) const { // naive row-major index default
//        int r = coord.x+row_offset, c = coord.y+col_offset; // alias
//        return ptr + sizeof(T) * (r * underlying_cols + c);
//    }
//    METAL_FUNC threadgroup T& operator[](thread const int2 &rowcol) threadgroup {
//        return *idx(data, rowcol);
//    }
//    METAL_FUNC const threadgroup T& operator[](thread const int2 &rowcol) const threadgroup  {
//        return *idx(data, rowcol);
//    }
    // Declare idx as a const member function
//    METAL_FUNC threadgroup T* idx(threadgroup T * __restrict ptr, const int2 coord) const {
//        int r = coord.x + row_offset, c = coord.y + col_offset;
//        return ptr + r * underlying_cols + c;
//    }
//
//    // New idx function (const overload)
//    METAL_FUNC uint32_t idx(uint32_t ptr, int2 coord) {
//        int r = coord.x + row_offset, c = coord.y + col_offset;
//        return ptr + r * underlying_cols + c;
//    }
//
//    // Non-const operator[]
//    METAL_FUNC threadgroup T& operator[](thread const int2& rowcol) threadgroup {
//        return *idx(data, rowcol);
//    }
//
//    // Const operator[]
//    METAL_FUNC const threadgroup T& operator[](thread const int2 &rowcol) threadgroup const {
//        return *idx(data, rowcol);
//    }
    // idx function returning threadgroup T*
    METAL_FUNC threadgroup T* idx(threadgroup T * __restrict ptr, const int2 coord) threadgroup const {
        int r = coord.x + row_offset, c = coord.y + col_offset;
        return ptr + r * underlying_cols + c;
    }

    // idx function returning uint32_t
    METAL_FUNC uint32_t idx(uint32_t ptr, int2 coord) threadgroup const {
        int r = coord.x + row_offset, c = coord.y + col_offset;
        return ptr + r * underlying_cols + c;
    }

    // Non-const operator[]
    METAL_FUNC threadgroup T& operator[](thread const int2& rowcol) threadgroup {
        return *idx(data, rowcol);
    }

    // Const operator[]
    METAL_FUNC const threadgroup T& operator[](thread const int2 &rowcol) threadgroup const {
        return *idx(data, rowcol);
    }
    
    
    METAL_FUNC threadgroup T* idx(threadgroup T * __restrict ptr, const int2 coord) thread const {
        int r = coord.x + row_offset, c = coord.y + col_offset;
        return ptr + r * underlying_cols + c;
    }

    // idx function returning uint32_t
    METAL_FUNC uint32_t idx(uint32_t ptr, int2 coord) thread const {
        int r = coord.x + row_offset, c = coord.y + col_offset;
        return ptr + r * underlying_cols + c;
    }

    // Non-const operator[]
    METAL_FUNC threadgroup T& operator[](thread const int2& rowcol) thread {
        return *idx(data, rowcol);
    }

    // Const operator[]
    METAL_FUNC const threadgroup T& operator[](thread const int2 &rowcol) thread const {
        return *idx(data, rowcol);
    }

    


    // single-index operator[] is left undefined as it would likely be an improper use of st_subtile type.
    // can of course be end-run by just accessing .data directly.

};

namespace ducks{
template <typename T>
struct has_st_identifier {
    static constant constexpr bool value = false; // Default case
};
 
// Specialize for specific template instantiations of st
template <typename _T, int _height, int _width>
struct has_st_identifier<mittens::st<_T, _height, _width>> {
    static constant constexpr bool value = true;
};

template<typename _T, int _subtile_rows, int _subtile_cols>
struct has_st_identifier<mittens::st_subtile<_T, _subtile_rows, _subtile_cols>> {
    static constant constexpr bool value = true;
};
    
template <typename ST>
static constexpr bool is_shared_tile() {
    return has_st_identifier<ST>::value;
}
template <typename ST>
static constexpr void assert_shared_tile() {
    static_assert(is_shared_tile<ST>(), "T must be a st");
}
}
    
    

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// layout and type wrappers
template<int _height, int _width> using st_bf = st<bf16, _height, _width>;
template<int _height, int _width> using st_hf = st<half, _height, _width>;
template<int _height, int _width> using st_fl = st<float, _height, _width>;
} // namespace mittens

