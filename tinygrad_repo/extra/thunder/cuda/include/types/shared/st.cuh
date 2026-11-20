/**
 * @file
 * @brief The ThunderKittens shared tile struct.
 */

#pragma once

#include "../../common/common.cuh"
#include "sv.cuh"

/* ----------  MAIN TILE STRUCT  ---------- */

// these are helper structs for type inference
namespace kittens {
namespace ducks {
/**
 * @namespace rt
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
/**
* @brief Concept for all shared tiles.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as st::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::st::identifier
}
} // namespace ducks

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
 * @tparam _rows The height of the tile.
 * @tparam _cols The width of the tile.
 */
template<typename _T, int _rows, int _cols>
struct KITTENS_DEFAULT_ALIGN st {
    using identifier = ducks::st::identifier; ///< Type identifier for shared memory tile.
    using T = base_types::packing<_T>::unpacked_type;
    using T2 = base_types::packing<_T>::packed_type;
    using dtype = T; ///< Data type of the elements in the tile.

    // define underlying data as same as that projected, to make clear that this is *not* a subtile.
    static constexpr int underlying_rows          = _rows;
    static constexpr int underlying_cols          = _cols;
    static constexpr int underlying_height        = _rows / kittens::TILE_ROW_DIM<T>;
    static constexpr int underlying_width         = _cols / kittens::TILE_COL_DIM<T>;
    static constexpr int underlying_num_elements  = underlying_rows * underlying_cols;

    static constexpr int rows                = _rows; ///< Total number of rows in the tile.
    static_assert(rows % kittens::TILE_ROW_DIM<T> == 0, "Rows must be divisible by the tile dimension");
    static constexpr int cols                = _cols; ///< Total number of cols in the tile.
    static_assert(cols % kittens::TILE_COL_DIM<T> == 0, "Cols must be divisible by the tile dimension");
    static constexpr int height              = _rows / kittens::TILE_ROW_DIM<T>; ///< Height of the tile in terms of 16-element subtiles.
    static constexpr int width               = _cols / kittens::TILE_COL_DIM<T>; ///< Width of the tile in terms of 16-element subtiles.
    static constexpr int num_elements        = rows * cols; ///< Total number of elements in the tile.

    static_assert(base_types::packing<dtype>::num() == 1); // must be a 1-packed type (e.g. float, bf16, etc)

    static constexpr int swizzle_bytes = (
        sizeof(dtype) == 1 ? (  // Add FP8 case
            underlying_width%4 == 0 ? 128 :
            underlying_width%2 == 0 ?  64 : 32
        ) :
        sizeof(dtype) == 2 ? (
            underlying_width%4 == 0 ? 128 :
            underlying_width%2 == 0 ?  64 : 32
        ) :
        sizeof(dtype) == 4 ? (
            underlying_width%2 == 0 ? 128 : 64
        ) : -1
    );

    // wgmma layout with swizzling
    dtype data[rows*cols]; ///< Raw data storage for the tile.

    __device__ static inline T* idx(T *ptr, int2 coord) { // naive row-major coord default
        int r = coord.x, c = coord.y; // alias
        static constexpr int swizzle_repeat = swizzle_bytes * 8;
        static constexpr int subtile_cols   = swizzle_bytes / sizeof(T);
        const int outer_idx = c/subtile_cols;
        const uint64_t addr = (uint64_t)(&ptr[outer_idx*rows*subtile_cols + r*subtile_cols + c%subtile_cols]);
        const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
        return (T*)(addr ^ swizzle);
    }
    __device__ static inline uint32_t idx(uint32_t ptr, int2 coord) {
        int r = coord.x, c = coord.y; // alias
        static constexpr int swizzle_repeat = swizzle_bytes * 8;
        static constexpr int subtile_cols   = swizzle_bytes / sizeof(T);
        const int outer_idx = c/subtile_cols;
        const uint32_t addr = ptr + sizeof(T)*(outer_idx*rows*subtile_cols + r*subtile_cols + c%subtile_cols);
        const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
        return (addr ^ swizzle);
    }
    /**
     * @brief Access a shared tile element using a row and column, as if the tile were row-major.
     *
     * This is the preferred way to access memory within a shared tile, which abstracts
     * indexing calculations for swizzled layouts.
     */
    __device__ inline       dtype& operator[](const int2 &rowcol)       {
        return *idx(data, rowcol);
    }
    __device__ inline const dtype& operator[](const int2 &rowcol) const {
        return *(const dtype*)idx((dtype*)data, rowcol);
    }
    __device__ inline       dtype& operator[](int idx)       {
        return data[idx];
    }
    __device__ inline const dtype& operator[](int idx) const {
        return data[idx];
    }

    template<int subtile_rows, int subtile_cols>
    __device__ inline st_subtile<st<_T, _rows, _cols>, subtile_rows, subtile_cols> subtile(int2 rowcol);

    // vector types
    using col_vec = sv<dtype, rows>; ///< Column vector type for this tile
    using row_vec = sv<dtype, cols>; ///< Row vector type for this tile
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
    using T = ST::T;
    using T2 = ST::T2;
    using dtype = T; ///< Data type of the elements in the tile.

    static constexpr int underlying_rows          = ST::underlying_rows;
    static_assert(underlying_rows % kittens::TILE_ROW_DIM<T> == 0, "Underlying rows must be divisible by the tile dimension");
    static constexpr int underlying_cols          = ST::underlying_cols;
    static_assert(underlying_cols % kittens::TILE_COL_DIM<T> == 0, "Underlying cols must be divisible by the tile dimension");
    static constexpr int underlying_height        = ST::underlying_height;
    static constexpr int underlying_width         = ST::underlying_width;
    static constexpr int underlying_num_elements  = ST::underlying_num_elements;

    static constexpr int rows                = _subtile_rows;
    static_assert(rows % kittens::TILE_ROW_DIM<T> == 0, "Rows must be divisible by the tile dimension");
    static constexpr int cols                = _subtile_cols;
    static_assert(cols % kittens::TILE_COL_DIM<T> == 0, "Cols must be divisible by the tile dimension");
    static constexpr int height              = rows / kittens::TILE_ROW_DIM<T>;
    static constexpr int width               = cols / kittens::TILE_COL_DIM<T>;
    static constexpr int num_elements        = rows * cols;

    static constexpr int swizzle_bytes = ST::swizzle_bytes;

    dtype *data;
    int row_offset, col_offset;

    __device__ st_subtile(ST &src, int2 rowcol) {
        data = &src.data[0];
        row_offset = rowcol.x * rows;
        col_offset = rowcol.y * cols;
    }

    __device__ inline T* idx(T *ptr, const int2 coord) { // naive row-major coord default
        int r = coord.x+row_offset, c = coord.y+col_offset; // alias
        static constexpr int swizzle_repeat = swizzle_bytes * 8;
        static constexpr int subtile_cols   = swizzle_bytes / sizeof(T);
        const int outer_idx = c/subtile_cols;
        const uint64_t addr = (uint64_t)(&ptr[outer_idx*underlying_rows*subtile_cols + r*subtile_cols + c%subtile_cols]);
        const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
        return (T*)(addr ^ swizzle);
    }
    __device__ inline uint32_t idx(uint32_t ptr, const int2 coord) const { // naive row-major coord default
        int r = coord.x+row_offset, c = coord.y+col_offset; // alias
        static constexpr int swizzle_repeat = swizzle_bytes * 8;
        static constexpr int subtile_cols   = swizzle_bytes / sizeof(T);
        const int outer_idx = c/subtile_cols;
        const uint32_t addr = ptr + sizeof(T)*(outer_idx*underlying_rows*subtile_cols + r*subtile_cols + c%subtile_cols);
        const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
        return (addr ^ swizzle);
    }
    /**
     * @brief Access a shared tile element using a row and column, as if the tile were row-major.
     *
     * This is the preferred way to access memory within a shared tile, which abstracts
     * indexing calculations for swizzled layouts.
     */
    __device__ inline       dtype& operator[](const int2 &rowcol)       {
        return *idx(data, rowcol);
    }
    __device__ inline const dtype& operator[](const int2 &rowcol) const {
        return *(const dtype*)idx((dtype*)data, rowcol);
    }

    // single-coord operator[] is left undefined as it would likely be an improper use of st_subtile type.
    // can of course be end-run by just accessing .data directly.

    // vector types
    using col_vec = sv<dtype, rows>;
    using row_vec = sv<dtype, cols>;

    __device__ inline void operator=(const dtype &value) { // runs at warp scope by default
        #pragma unroll
        for(int i = kittens::laneid(); i < num_elements; i += WARP_THREADS) {
            data[i] = value;
        }
    }
};

template <typename _T, int _rows, int _cols> // Class template parameters
template <int subtile_rows, int subtile_cols> // Function template parameters
__device__ inline st_subtile<st<_T, _rows, _cols>, subtile_rows, subtile_cols> // Return type
st<_T, _rows, _cols>::subtile(int2 rowcol) // Qualified function name and parameters
{
    // Type aliases for convenience within the function body
    using ST_t = st<_T, _rows, _cols>; // Alias for the parent tile type
    using dtype = typename ST_t::dtype;  // Alias for the data type

    // Static assertions (as provided in the initial request)
    static_assert(subtile_rows > 0 && subtile_cols > 0, "Subtile dimensions must be positive.");
    static_assert(subtile_rows % kittens::TILE_ROW_DIM<dtype> == 0,
        "Subtile rows must be divisible by the base tile row dimension.");
    static_assert(subtile_cols % kittens::TILE_COL_DIM<dtype> == 0,
        "Subtile cols must be divisible by the base tile col dimension.");

    // Calculate height/width in terms of base tiles for further checks
    constexpr int subtile_height = subtile_rows / kittens::TILE_ROW_DIM<dtype>;
    constexpr int subtile_width = subtile_cols / kittens::TILE_COL_DIM<dtype>;
    static_assert(subtile_height > 0 && subtile_width > 0, "Subtile height/width in base tiles must be positive.");

    // Check divisibility of parent height/width by subtile height/width
    static_assert(ST_t::height % subtile_height == 0,
        "Parent tile height (in base tiles) must be divisible by subtile height (in base tiles).");
    static_assert(ST_t::width % subtile_width == 0,
        "Parent tile width (in base tiles) must be divisible by subtile width (in base tiles).");

    // Ensure the parent st object is not itself a subtile view by comparing its
    // dimensions to its underlying dimensions.
    static_assert(ST_t::height == ST_t::underlying_height && ST_t::width == ST_t::underlying_width,
        "Cannot create a subtile from an object that appears to be a subtile view (height/width mismatch underlying).");
    // Also check rows/cols directly for robustness, though height/width check might suffice.
    static_assert(ST_t::rows == ST_t::underlying_rows && ST_t::cols == ST_t::underlying_cols,
        "Cannot create a subtile from an object that appears to be a subtile view (rows/cols mismatch underlying).");


    // Construct and return the st_subtile object using its constructor:
    // st_subtile(ST &src, int2 rowcol)
    // Here, 'src' is the current 'st' object (*this)
    return st_subtile<ST_t, subtile_rows, subtile_cols>(*this, rowcol);
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

template<int _height, int _width> using st_bf = st<bf16,  _height, _width>;
template<int _height, int _width> using st_hf = st<half,  _height, _width>;
template<int _height, int _width> using st_fl = st<float, _height, _width>;
#ifdef KITTENS_HOPPER
template<int _height, int _width> using st_fp8e4m3 = st<fp8e4m3, _height, _width>;
template<int _height, int _width> using st_fp8e5m2 = st<fp8e5m2, _height, _width>;
#ifdef KITTENS_BLACKWELL
template<int _height, int _width> using st_fp8e8m0 = st<fp8e8m0, _height, _width>;
#endif
#endif

/* ----------  PRINTOUTS  ---------- */

/**
 * @brief Print the contents of a shared tile as a formatted table.
 * 
 * This function should be called by a single thread in the warp.
 * It will print the entire tile atomically to avoid interleaved output.
 * 
 * @param tile The shared tile to print
 */
template<ducks::st::all ST>
__device__ inline void print(const ST& tile) {
    printf("Shared Tile %dx%d:\n", ST::rows, ST::cols);
    
    // Print column headers
    printf("     "); // Padding for row indices
    for (int c = 0; c < ST::cols; c++) {
        printf("%8d ", c);
    }
    printf("\n");
    
    // Print separator line
    printf("     ");
    for (int c = 0; c < ST::cols; c++) {
        printf("--------+");
    }
    printf("\n");
    
    // Print data rows
    for (int r = 0; r < ST::rows; r++) {
        printf("%3d |", r); // Row index
        for (int c = 0; c < ST::cols; c++) {
            if constexpr (std::is_same_v<typename ST::dtype, fp8e4m3>) {
                printf("%8.3f ", static_cast<float>(tile[{r,c}]));
#ifdef KITTENS_BLACKWELL
            } else if constexpr (std::is_same_v<typename ST::dtype, fp8e8m0>) {
                printf("%8.3f ", static_cast<float>(tile[{r,c}]));
#endif
            } else if constexpr (std::is_same_v<typename ST::dtype, float>) {
                printf("%8.3f ", tile[{r,c}]);
            } else if constexpr (std::is_same_v<typename ST::dtype, __nv_bfloat16>) {
                printf("%8.3f ", __bfloat162float(tile[{r,c}]));
            } else if constexpr (std::is_integral_v<typename ST::dtype>) {
                printf("%8d ", (int)tile[{r,c}]);
            } else {
                printf("%8.3f ", (float)tile[{r,c}]);
            }
        }
        printf("\n");
    }
    printf("\n");
}

}
