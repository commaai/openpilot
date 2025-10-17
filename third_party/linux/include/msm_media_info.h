#ifndef __MEDIA_INFO_H__
#define __MEDIA_INFO_H__

#ifndef MSM_MEDIA_ALIGN
#define MSM_MEDIA_ALIGN(__sz, __align) (((__sz) + (__align-1)) & (~(__align-1)))
#endif

#ifndef MSM_MEDIA_ROUNDUP
#define MSM_MEDIA_ROUNDUP(__sz, __r) (((__sz) + ((__r) - 1)) / (__r))
#endif

#ifndef MSM_MEDIA_MAX
#define MSM_MEDIA_MAX(__a, __b) ((__a) > (__b)?(__a):(__b))
#endif

enum color_fmts {
	/* Venus NV12:
	 * YUV 4:2:0 image with a plane of 8 bit Y samples followed
	 * by an interleaved U/V plane containing 8 bit 2x2 subsampled
	 * colour difference samples.
	 *
	 * <-------- Y/UV_Stride -------->
	 * <------- Width ------->
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  ^           ^
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  |           |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  Height      |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  |          Y_Scanlines
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  |           |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  |           |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  |           |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  V           |
	 * . . . . . . . . . . . . . . . .              |
	 * . . . . . . . . . . . . . . . .              |
	 * . . . . . . . . . . . . . . . .              |
	 * . . . . . . . . . . . . . . . .              V
	 * U V U V U V U V U V U V . . . .  ^
	 * U V U V U V U V U V U V . . . .  |
	 * U V U V U V U V U V U V . . . .  |
	 * U V U V U V U V U V U V . . . .  UV_Scanlines
	 * . . . . . . . . . . . . . . . .  |
	 * . . . . . . . . . . . . . . . .  V
	 * . . . . . . . . . . . . . . . .  --> Buffer size alignment
	 *
	 * Y_Stride : Width aligned to 128
	 * UV_Stride : Width aligned to 128
	 * Y_Scanlines: Height aligned to 32
	 * UV_Scanlines: Height/2 aligned to 16
	 * Extradata: Arbitrary (software-imposed) padding
	 * Total size = align((Y_Stride * Y_Scanlines
	 *          + UV_Stride * UV_Scanlines
	 *          + max(Extradata, Y_Stride * 8), 4096)
	 */
	COLOR_FMT_NV12,

	/* Venus NV21:
	 * YUV 4:2:0 image with a plane of 8 bit Y samples followed
	 * by an interleaved V/U plane containing 8 bit 2x2 subsampled
	 * colour difference samples.
	 *
	 * <-------- Y/UV_Stride -------->
	 * <------- Width ------->
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  ^           ^
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  |           |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  Height      |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  |          Y_Scanlines
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  |           |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  |           |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  |           |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  V           |
	 * . . . . . . . . . . . . . . . .              |
	 * . . . . . . . . . . . . . . . .              |
	 * . . . . . . . . . . . . . . . .              |
	 * . . . . . . . . . . . . . . . .              V
	 * V U V U V U V U V U V U . . . .  ^
	 * V U V U V U V U V U V U . . . .  |
	 * V U V U V U V U V U V U . . . .  |
	 * V U V U V U V U V U V U . . . .  UV_Scanlines
	 * . . . . . . . . . . . . . . . .  |
	 * . . . . . . . . . . . . . . . .  V
	 * . . . . . . . . . . . . . . . .  --> Padding & Buffer size alignment
	 *
	 * Y_Stride : Width aligned to 128
	 * UV_Stride : Width aligned to 128
	 * Y_Scanlines: Height aligned to 32
	 * UV_Scanlines: Height/2 aligned to 16
	 * Extradata: Arbitrary (software-imposed) padding
	 * Total size = align((Y_Stride * Y_Scanlines
	 *          + UV_Stride * UV_Scanlines
	 *          + max(Extradata, Y_Stride * 8), 4096)
	 */
	COLOR_FMT_NV21,
	/* Venus NV12_MVTB:
	 * Two YUV 4:2:0 images/views one after the other
	 * in a top-bottom layout, same as NV12
	 * with a plane of 8 bit Y samples followed
	 * by an interleaved U/V plane containing 8 bit 2x2 subsampled
	 * colour difference samples.
	 *
	 *
	 * <-------- Y/UV_Stride -------->
	 * <------- Width ------->
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  ^           ^               ^
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  |           |               |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  Height      |               |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  |          Y_Scanlines      |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  |           |               |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  |           |               |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  |           |               |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  V           |               |
	 * . . . . . . . . . . . . . . . .              |             View_1
	 * . . . . . . . . . . . . . . . .              |               |
	 * . . . . . . . . . . . . . . . .              |               |
	 * . . . . . . . . . . . . . . . .              V               |
	 * U V U V U V U V U V U V . . . .  ^                           |
	 * U V U V U V U V U V U V . . . .  |                           |
	 * U V U V U V U V U V U V . . . .  |                           |
	 * U V U V U V U V U V U V . . . .  UV_Scanlines                |
	 * . . . . . . . . . . . . . . . .  |                           |
	 * . . . . . . . . . . . . . . . .  V                           V
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  ^           ^               ^
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  |           |               |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  Height      |               |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  |          Y_Scanlines      |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  |           |               |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  |           |               |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  |           |               |
	 * Y Y Y Y Y Y Y Y Y Y Y Y . . . .  V           |               |
	 * . . . . . . . . . . . . . . . .              |             View_2
	 * . . . . . . . . . . . . . . . .              |               |
	 * . . . . . . . . . . . . . . . .              |               |
	 * . . . . . . . . . . . . . . . .              V               |
	 * U V U V U V U V U V U V . . . .  ^                           |
	 * U V U V U V U V U V U V . . . .  |                           |
	 * U V U V U V U V U V U V . . . .  |                           |
	 * U V U V U V U V U V U V . . . .  UV_Scanlines                |
	 * . . . . . . . . . . . . . . . .  |                           |
	 * . . . . . . . . . . . . . . . .  V                           V
	 * . . . . . . . . . . . . . . . .  --> Buffer size alignment
	 *
	 * Y_Stride : Width aligned to 128
	 * UV_Stride : Width aligned to 128
	 * Y_Scanlines: Height aligned to 32
	 * UV_Scanlines: Height/2 aligned to 16
	 * View_1 begin at: 0 (zero)
	 * View_2 begin at: Y_Stride * Y_Scanlines + UV_Stride * UV_Scanlines
	 * Extradata: Arbitrary (software-imposed) padding
	 * Total size = align((2*(Y_Stride * Y_Scanlines)
	 *          + 2*(UV_Stride * UV_Scanlines) + Extradata), 4096)
	 */
	COLOR_FMT_NV12_MVTB,
	/* Venus NV12 UBWC:
	 * Compressed Macro-tile format for NV12.
	 * Contains 4 planes in the following order -
	 * (A) Y_Meta_Plane
	 * (B) Y_UBWC_Plane
	 * (C) UV_Meta_Plane
	 * (D) UV_UBWC_Plane
	 *
	 * Y_Meta_Plane consists of meta information to decode compressed
	 * tile data in Y_UBWC_Plane.
	 * Y_UBWC_Plane consists of Y data in compressed macro-tile format.
	 * UBWC decoder block will use the Y_Meta_Plane data together with
	 * Y_UBWC_Plane data to produce loss-less uncompressed 8 bit Y samples.
	 *
	 * UV_Meta_Plane consists of meta information to decode compressed
	 * tile data in UV_UBWC_Plane.
	 * UV_UBWC_Plane consists of UV data in compressed macro-tile format.
	 * UBWC decoder block will use UV_Meta_Plane data together with
	 * UV_UBWC_Plane data to produce loss-less uncompressed 8 bit 2x2
	 * subsampled color difference samples.
	 *
	 * Each tile in Y_UBWC_Plane/UV_UBWC_Plane is independently decodable
	 * and randomly accessible. There is no dependency between tiles.
	 *
	 * <----- Y_Meta_Stride ---->
	 * <-------- Width ------>
	 * M M M M M M M M M M M M . .      ^           ^
	 * M M M M M M M M M M M M . .      |           |
	 * M M M M M M M M M M M M . .      Height      |
	 * M M M M M M M M M M M M . .      |         Meta_Y_Scanlines
	 * M M M M M M M M M M M M . .      |           |
	 * M M M M M M M M M M M M . .      |           |
	 * M M M M M M M M M M M M . .      |           |
	 * M M M M M M M M M M M M . .      V           |
	 * . . . . . . . . . . . . . .                  |
	 * . . . . . . . . . . . . . .                  |
	 * . . . . . . . . . . . . . .      -------> Buffer size aligned to 4k
	 * . . . . . . . . . . . . . .                  V
	 * <--Compressed tile Y Stride--->
	 * <------- Width ------->
	 * Y* Y* Y* Y* Y* Y* Y* Y* . . . .  ^           ^
	 * Y* Y* Y* Y* Y* Y* Y* Y* . . . .  |           |
	 * Y* Y* Y* Y* Y* Y* Y* Y* . . . .  Height      |
	 * Y* Y* Y* Y* Y* Y* Y* Y* . . . .  |        Macro_tile_Y_Scanlines
	 * Y* Y* Y* Y* Y* Y* Y* Y* . . . .  |           |
	 * Y* Y* Y* Y* Y* Y* Y* Y* . . . .  |           |
	 * Y* Y* Y* Y* Y* Y* Y* Y* . . . .  |           |
	 * Y* Y* Y* Y* Y* Y* Y* Y* . . . .  V           |
	 * . . . . . . . . . . . . . . . .              |
	 * . . . . . . . . . . . . . . . .              |
	 * . . . . . . . . . . . . . . . .  -------> Buffer size aligned to 4k
	 * . . . . . . . . . . . . . . . .              V
	 * <----- UV_Meta_Stride ---->
	 * M M M M M M M M M M M M . .      ^
	 * M M M M M M M M M M M M . .      |
	 * M M M M M M M M M M M M . .      |
	 * M M M M M M M M M M M M . .      M_UV_Scanlines
	 * . . . . . . . . . . . . . .      |
	 * . . . . . . . . . . . . . .      V
	 * . . . . . . . . . . . . . .      -------> Buffer size aligned to 4k
	 * <--Compressed tile UV Stride--->
	 * U* V* U* V* U* V* U* V* . . . .  ^
	 * U* V* U* V* U* V* U* V* . . . .  |
	 * U* V* U* V* U* V* U* V* . . . .  |
	 * U* V* U* V* U* V* U* V* . . . .  UV_Scanlines
	 * . . . . . . . . . . . . . . . .  |
	 * . . . . . . . . . . . . . . . .  V
	 * . . . . . . . . . . . . . . . .  -------> Buffer size aligned to 4k
	 *
	 * Y_Stride = align(Width, 128)
	 * UV_Stride = align(Width, 128)
	 * Y_Scanlines = align(Height, 32)
	 * UV_Scanlines = align(Height/2, 16)
	 * Y_UBWC_Plane_size = align(Y_Stride * Y_Scanlines, 4096)
	 * UV_UBWC_Plane_size = align(UV_Stride * UV_Scanlines, 4096)
	 * Y_Meta_Stride = align(roundup(Width, Y_TileWidth), 64)
	 * Y_Meta_Scanlines = align(roundup(Height, Y_TileHeight), 16)
	 * Y_Meta_Plane_size = align(Y_Meta_Stride * Y_Meta_Scanlines, 4096)
	 * UV_Meta_Stride = align(roundup(Width, UV_TileWidth), 64)
	 * UV_Meta_Scanlines = align(roundup(Height, UV_TileHeight), 16)
	 * UV_Meta_Plane_size = align(UV_Meta_Stride * UV_Meta_Scanlines, 4096)
	 * Extradata = 8k
	 *
	 * Total size = align( Y_UBWC_Plane_size + UV_UBWC_Plane_size +
	 *           Y_Meta_Plane_size + UV_Meta_Plane_size
	 *           + max(Extradata, Y_Stride * 48), 4096)
	 */
	COLOR_FMT_NV12_UBWC,
	/* Venus NV12 10-bit UBWC:
	 * Compressed Macro-tile format for NV12.
	 * Contains 4 planes in the following order -
	 * (A) Y_Meta_Plane
	 * (B) Y_UBWC_Plane
	 * (C) UV_Meta_Plane
	 * (D) UV_UBWC_Plane
	 *
	 * Y_Meta_Plane consists of meta information to decode compressed
	 * tile data in Y_UBWC_Plane.
	 * Y_UBWC_Plane consists of Y data in compressed macro-tile format.
	 * UBWC decoder block will use the Y_Meta_Plane data together with
	 * Y_UBWC_Plane data to produce loss-less uncompressed 10 bit Y samples.
	 *
	 * UV_Meta_Plane consists of meta information to decode compressed
	 * tile data in UV_UBWC_Plane.
	 * UV_UBWC_Plane consists of UV data in compressed macro-tile format.
	 * UBWC decoder block will use UV_Meta_Plane data together with
	 * UV_UBWC_Plane data to produce loss-less uncompressed 10 bit 2x2
	 * subsampled color difference samples.
	 *
	 * Each tile in Y_UBWC_Plane/UV_UBWC_Plane is independently decodable
	 * and randomly accessible. There is no dependency between tiles.
	 *
	 * <----- Y_Meta_Stride ----->
	 * <-------- Width ------>
	 * M M M M M M M M M M M M . .      ^           ^
	 * M M M M M M M M M M M M . .      |           |
	 * M M M M M M M M M M M M . .      Height      |
	 * M M M M M M M M M M M M . .      |         Meta_Y_Scanlines
	 * M M M M M M M M M M M M . .      |           |
	 * M M M M M M M M M M M M . .      |           |
	 * M M M M M M M M M M M M . .      |           |
	 * M M M M M M M M M M M M . .      V           |
	 * . . . . . . . . . . . . . .                  |
	 * . . . . . . . . . . . . . .                  |
	 * . . . . . . . . . . . . . .      -------> Buffer size aligned to 4k
	 * . . . . . . . . . . . . . .                  V
	 * <--Compressed tile Y Stride--->
	 * <------- Width ------->
	 * Y* Y* Y* Y* Y* Y* Y* Y* . . . .  ^           ^
	 * Y* Y* Y* Y* Y* Y* Y* Y* . . . .  |           |
	 * Y* Y* Y* Y* Y* Y* Y* Y* . . . .  Height      |
	 * Y* Y* Y* Y* Y* Y* Y* Y* . . . .  |        Macro_tile_Y_Scanlines
	 * Y* Y* Y* Y* Y* Y* Y* Y* . . . .  |           |
	 * Y* Y* Y* Y* Y* Y* Y* Y* . . . .  |           |
	 * Y* Y* Y* Y* Y* Y* Y* Y* . . . .  |           |
	 * Y* Y* Y* Y* Y* Y* Y* Y* . . . .  V           |
	 * . . . . . . . . . . . . . . . .              |
	 * . . . . . . . . . . . . . . . .              |
	 * . . . . . . . . . . . . . . . .  -------> Buffer size aligned to 4k
	 * . . . . . . . . . . . . . . . .              V
	 * <----- UV_Meta_Stride ---->
	 * M M M M M M M M M M M M . .      ^
	 * M M M M M M M M M M M M . .      |
	 * M M M M M M M M M M M M . .      |
	 * M M M M M M M M M M M M . .      M_UV_Scanlines
	 * . . . . . . . . . . . . . .      |
	 * . . . . . . . . . . . . . .      V
	 * . . . . . . . . . . . . . .      -------> Buffer size aligned to 4k
	 * <--Compressed tile UV Stride--->
	 * U* V* U* V* U* V* U* V* . . . .  ^
	 * U* V* U* V* U* V* U* V* . . . .  |
	 * U* V* U* V* U* V* U* V* . . . .  |
	 * U* V* U* V* U* V* U* V* . . . .  UV_Scanlines
	 * . . . . . . . . . . . . . . . .  |
	 * . . . . . . . . . . . . . . . .  V
	 * . . . . . . . . . . . . . . . .  -------> Buffer size aligned to 4k
	 *
	 *
	 * Y_Stride = align(Width * 4/3, 128)
	 * UV_Stride = align(Width * 4/3, 128)
	 * Y_Scanlines = align(Height, 32)
	 * UV_Scanlines = align(Height/2, 16)
	 * Y_UBWC_Plane_Size = align(Y_Stride * Y_Scanlines, 4096)
	 * UV_UBWC_Plane_Size = align(UV_Stride * UV_Scanlines, 4096)
	 * Y_Meta_Stride = align(roundup(Width, Y_TileWidth), 64)
	 * Y_Meta_Scanlines = align(roundup(Height, Y_TileHeight), 16)
	 * Y_Meta_Plane_size = align(Y_Meta_Stride * Y_Meta_Scanlines, 4096)
	 * UV_Meta_Stride = align(roundup(Width, UV_TileWidth), 64)
	 * UV_Meta_Scanlines = align(roundup(Height, UV_TileHeight), 16)
	 * UV_Meta_Plane_size = align(UV_Meta_Stride * UV_Meta_Scanlines, 4096)
	 * Extradata = 8k
	 *
	 * Total size = align(Y_UBWC_Plane_size + UV_UBWC_Plane_size +
	 *           Y_Meta_Plane_size + UV_Meta_Plane_size
	 *           + max(Extradata, Y_Stride * 48), 4096)
	 */
	COLOR_FMT_NV12_BPP10_UBWC,
	/* Venus RGBA8888 format:
	 * Contains 1 plane in the following order -
	 * (A) RGBA plane
	 *
	 * <-------- RGB_Stride -------->
	 * <------- Width ------->
	 * R R R R R R R R R R R R . . . .  ^           ^
	 * R R R R R R R R R R R R . . . .  |           |
	 * R R R R R R R R R R R R . . . .  Height      |
	 * R R R R R R R R R R R R . . . .  |       RGB_Scanlines
	 * R R R R R R R R R R R R . . . .  |           |
	 * R R R R R R R R R R R R . . . .  |           |
	 * R R R R R R R R R R R R . . . .  |           |
	 * R R R R R R R R R R R R . . . .  V           |
	 * . . . . . . . . . . . . . . . .              |
	 * . . . . . . . . . . . . . . . .              |
	 * . . . . . . . . . . . . . . . .              |
	 * . . . . . . . . . . . . . . . .              V
	 *
	 * RGB_Stride = align(Width * 4, 128)
	 * RGB_Scanlines = align(Height, 32)
	 * RGB_Plane_size = align(RGB_Stride * RGB_Scanlines, 4096)
	 * Extradata = 8k
	 *
	 * Total size = align(RGB_Plane_size + Extradata, 4096)
	 */
	COLOR_FMT_RGBA8888,
	/* Venus RGBA8888 UBWC format:
	 * Contains 2 planes in the following order -
	 * (A) Meta plane
	 * (B) RGBA plane
	 *
	 * <--- RGB_Meta_Stride ---->
	 * <-------- Width ------>
	 * M M M M M M M M M M M M . .      ^           ^
	 * M M M M M M M M M M M M . .      |           |
	 * M M M M M M M M M M M M . .      Height      |
	 * M M M M M M M M M M M M . .      |       Meta_RGB_Scanlines
	 * M M M M M M M M M M M M . .      |           |
	 * M M M M M M M M M M M M . .      |           |
	 * M M M M M M M M M M M M . .      |           |
	 * M M M M M M M M M M M M . .      V           |
	 * . . . . . . . . . . . . . .                  |
	 * . . . . . . . . . . . . . .                  |
	 * . . . . . . . . . . . . . .      -------> Buffer size aligned to 4k
	 * . . . . . . . . . . . . . .                  V
	 * <-------- RGB_Stride -------->
	 * <------- Width ------->
	 * R R R R R R R R R R R R . . . .  ^           ^
	 * R R R R R R R R R R R R . . . .  |           |
	 * R R R R R R R R R R R R . . . .  Height      |
	 * R R R R R R R R R R R R . . . .  |       RGB_Scanlines
	 * R R R R R R R R R R R R . . . .  |           |
	 * R R R R R R R R R R R R . . . .  |           |
	 * R R R R R R R R R R R R . . . .  |           |
	 * R R R R R R R R R R R R . . . .  V           |
	 * . . . . . . . . . . . . . . . .              |
	 * . . . . . . . . . . . . . . . .              |
	 * . . . . . . . . . . . . . . . .    -------> Buffer size aligned to 4k
	 * . . . . . . . . . . . . . . . .              V
	 *
	 * RGB_Stride = align(Width * 4, 128)
	 * RGB_Scanlines = align(Height, 32)
	 * RGB_Plane_size = align(RGB_Stride * RGB_Scanlines, 4096)
	 * RGB_Meta_Stride = align(roundup(Width, RGB_TileWidth), 64)
	 * RGB_Meta_Scanline = align(roundup(Height, RGB_TileHeight), 16)
	 * RGB_Meta_Plane_size = align(RGB_Meta_Stride *
	 *		RGB_Meta_Scanlines, 4096)
	 * Extradata = 8k
	 *
	 * Total size = align(RGB_Meta_Plane_size + RGB_Plane_size +
	 *		Extradata, 4096)
	 */
	COLOR_FMT_RGBA8888_UBWC,
};

static inline unsigned int VENUS_EXTRADATA_SIZE(int width, int height)
{
	(void)height;
	(void)width;

	/*
	 * In the future, calculate the size based on the w/h but just
	 * hardcode it for now since 16K satisfies all current usecases.
	 */
	return 16 * 1024;
}

static inline unsigned int VENUS_Y_STRIDE(int color_fmt, int width)
{
	unsigned int alignment, stride = 0;
	if (!width)
		goto invalid_input;

	switch (color_fmt) {
	case COLOR_FMT_NV21:
	case COLOR_FMT_NV12:
	case COLOR_FMT_NV12_MVTB:
	case COLOR_FMT_NV12_UBWC:
		alignment = 128;
		stride = MSM_MEDIA_ALIGN(width, alignment);
		break;
	case COLOR_FMT_NV12_BPP10_UBWC:
		alignment = 256;
		stride = MSM_MEDIA_ALIGN(width, 192);
		stride = MSM_MEDIA_ALIGN(stride * 4/3, alignment);
		break;
	default:
		break;
	}
invalid_input:
	return stride;
}

static inline unsigned int VENUS_UV_STRIDE(int color_fmt, int width)
{
	unsigned int alignment, stride = 0;
	if (!width)
		goto invalid_input;

	switch (color_fmt) {
	case COLOR_FMT_NV21:
	case COLOR_FMT_NV12:
	case COLOR_FMT_NV12_MVTB:
	case COLOR_FMT_NV12_UBWC:
		alignment = 128;
		stride = MSM_MEDIA_ALIGN(width, alignment);
		break;
	case COLOR_FMT_NV12_BPP10_UBWC:
		alignment = 256;
		stride = MSM_MEDIA_ALIGN(width, 192);
		stride = MSM_MEDIA_ALIGN(stride * 4/3, alignment);
		break;
	default:
		break;
	}
invalid_input:
	return stride;
}

static inline unsigned int VENUS_Y_SCANLINES(int color_fmt, int height)
{
	unsigned int alignment, sclines = 0;
	if (!height)
		goto invalid_input;

	switch (color_fmt) {
	case COLOR_FMT_NV21:
	case COLOR_FMT_NV12:
	case COLOR_FMT_NV12_MVTB:
	case COLOR_FMT_NV12_UBWC:
		alignment = 32;
		break;
	case COLOR_FMT_NV12_BPP10_UBWC:
		alignment = 16;
		break;
	default:
		return 0;
	}
	sclines = MSM_MEDIA_ALIGN(height, alignment);
invalid_input:
	return sclines;
}

static inline unsigned int VENUS_UV_SCANLINES(int color_fmt, int height)
{
	unsigned int alignment, sclines = 0;
	if (!height)
		goto invalid_input;

	switch (color_fmt) {
	case COLOR_FMT_NV21:
	case COLOR_FMT_NV12:
	case COLOR_FMT_NV12_MVTB:
	case COLOR_FMT_NV12_BPP10_UBWC:
		alignment = 16;
		break;
	case COLOR_FMT_NV12_UBWC:
		alignment = 32;
		break;
	default:
		goto invalid_input;
	}

	sclines = MSM_MEDIA_ALIGN(height / 2, alignment);

invalid_input:
	return sclines;
}

static inline unsigned int VENUS_Y_META_STRIDE(int color_fmt, int width)
{
	int y_tile_width = 0, y_meta_stride = 0;

	if (!width)
		goto invalid_input;

	switch (color_fmt) {
	case COLOR_FMT_NV12_UBWC:
		y_tile_width = 32;
		break;
	case COLOR_FMT_NV12_BPP10_UBWC:
		y_tile_width = 48;
		break;
	default:
		goto invalid_input;
	}

	y_meta_stride = MSM_MEDIA_ROUNDUP(width, y_tile_width);
	y_meta_stride = MSM_MEDIA_ALIGN(y_meta_stride, 64);

invalid_input:
	return y_meta_stride;
}

static inline unsigned int VENUS_Y_META_SCANLINES(int color_fmt, int height)
{
	int y_tile_height = 0, y_meta_scanlines = 0;

	if (!height)
		goto invalid_input;

	switch (color_fmt) {
	case COLOR_FMT_NV12_UBWC:
		y_tile_height = 8;
		break;
	case COLOR_FMT_NV12_BPP10_UBWC:
		y_tile_height = 4;
		break;
	default:
		goto invalid_input;
	}

	y_meta_scanlines = MSM_MEDIA_ROUNDUP(height, y_tile_height);
	y_meta_scanlines = MSM_MEDIA_ALIGN(y_meta_scanlines, 16);

invalid_input:
	return y_meta_scanlines;
}

static inline unsigned int VENUS_UV_META_STRIDE(int color_fmt, int width)
{
	int uv_tile_width = 0, uv_meta_stride = 0;

	if (!width)
		goto invalid_input;

	switch (color_fmt) {
	case COLOR_FMT_NV12_UBWC:
		uv_tile_width = 16;
		break;
	case COLOR_FMT_NV12_BPP10_UBWC:
		uv_tile_width = 24;
		break;
	default:
		goto invalid_input;
	}

	uv_meta_stride = MSM_MEDIA_ROUNDUP(width / 2, uv_tile_width);
	uv_meta_stride = MSM_MEDIA_ALIGN(uv_meta_stride, 64);

invalid_input:
	return uv_meta_stride;
}

static inline unsigned int VENUS_UV_META_SCANLINES(int color_fmt, int height)
{
	int uv_tile_height = 0, uv_meta_scanlines = 0;

	if (!height)
		goto invalid_input;

	switch (color_fmt) {
	case COLOR_FMT_NV12_UBWC:
		uv_tile_height = 8;
		break;
	case COLOR_FMT_NV12_BPP10_UBWC:
		uv_tile_height = 4;
		break;
	default:
		goto invalid_input;
	}

	uv_meta_scanlines = MSM_MEDIA_ROUNDUP(height / 2, uv_tile_height);
	uv_meta_scanlines = MSM_MEDIA_ALIGN(uv_meta_scanlines, 16);

invalid_input:
	return uv_meta_scanlines;
}

static inline unsigned int VENUS_RGB_STRIDE(int color_fmt, int width)
{
	unsigned int alignment = 0, stride = 0;
	if (!width)
		goto invalid_input;

	switch (color_fmt) {
	case COLOR_FMT_RGBA8888:
		alignment = 128;
		break;
	case COLOR_FMT_RGBA8888_UBWC:
		alignment = 256;
		break;
	default:
		goto invalid_input;
	}

	stride = MSM_MEDIA_ALIGN(width * 4, alignment);

invalid_input:
	return stride;
}

static inline unsigned int VENUS_RGB_SCANLINES(int color_fmt, int height)
{
	unsigned int alignment = 0, scanlines = 0;

	if (!height)
		goto invalid_input;

	switch (color_fmt) {
	case COLOR_FMT_RGBA8888:
		alignment = 32;
		break;
	case COLOR_FMT_RGBA8888_UBWC:
		alignment = 16;
		break;
	default:
		goto invalid_input;
	}

	scanlines = MSM_MEDIA_ALIGN(height, alignment);

invalid_input:
	return scanlines;
}

static inline unsigned int VENUS_RGB_META_STRIDE(int color_fmt, int width)
{
	int rgb_tile_width = 0, rgb_meta_stride = 0;

	if (!width)
		goto invalid_input;

	switch (color_fmt) {
	case COLOR_FMT_RGBA8888_UBWC:
		rgb_tile_width = 16;
		break;
	default:
		goto invalid_input;
	}

	rgb_meta_stride = MSM_MEDIA_ROUNDUP(width, rgb_tile_width);
	rgb_meta_stride = MSM_MEDIA_ALIGN(rgb_meta_stride, 64);

invalid_input:
	return rgb_meta_stride;
}

static inline unsigned int VENUS_RGB_META_SCANLINES(int color_fmt, int height)
{
	int rgb_tile_height = 0, rgb_meta_scanlines = 0;

	if (!height)
		goto invalid_input;

	switch (color_fmt) {
	case COLOR_FMT_RGBA8888_UBWC:
		rgb_tile_height = 4;
		break;
	default:
		goto invalid_input;
	}

	rgb_meta_scanlines = MSM_MEDIA_ROUNDUP(height, rgb_tile_height);
	rgb_meta_scanlines = MSM_MEDIA_ALIGN(rgb_meta_scanlines, 16);

invalid_input:
	return rgb_meta_scanlines;
}

static inline unsigned int VENUS_BUFFER_SIZE(
	int color_fmt, int width, int height)
{
	const unsigned int extra_size = VENUS_EXTRADATA_SIZE(width, height);
	unsigned int uv_alignment = 0, size = 0;
	unsigned int y_plane, uv_plane, y_stride,
		uv_stride, y_sclines, uv_sclines;
	unsigned int y_ubwc_plane = 0, uv_ubwc_plane = 0;
	unsigned int y_meta_stride = 0, y_meta_scanlines = 0;
	unsigned int uv_meta_stride = 0, uv_meta_scanlines = 0;
	unsigned int y_meta_plane = 0, uv_meta_plane = 0;
	unsigned int rgb_stride = 0, rgb_scanlines = 0;
	unsigned int rgb_plane = 0, rgb_ubwc_plane = 0, rgb_meta_plane = 0;
	unsigned int rgb_meta_stride = 0, rgb_meta_scanlines = 0;

	if (!width || !height)
		goto invalid_input;

	y_stride = VENUS_Y_STRIDE(color_fmt, width);
	uv_stride = VENUS_UV_STRIDE(color_fmt, width);
	y_sclines = VENUS_Y_SCANLINES(color_fmt, height);
	uv_sclines = VENUS_UV_SCANLINES(color_fmt, height);
	rgb_stride = VENUS_RGB_STRIDE(color_fmt, width);
	rgb_scanlines = VENUS_RGB_SCANLINES(color_fmt, height);

	switch (color_fmt) {
	case COLOR_FMT_NV21:
	case COLOR_FMT_NV12:
		uv_alignment = 4096;
		y_plane = y_stride * y_sclines;
		uv_plane = uv_stride * uv_sclines + uv_alignment;
		size = y_plane + uv_plane +
				MSM_MEDIA_MAX(extra_size, 8 * y_stride);
		size = MSM_MEDIA_ALIGN(size, 4096);
		break;
	case COLOR_FMT_NV12_MVTB:
		uv_alignment = 4096;
		y_plane = y_stride * y_sclines;
		uv_plane = uv_stride * uv_sclines + uv_alignment;
		size = y_plane + uv_plane;
		size = 2 * size + extra_size;
		size = MSM_MEDIA_ALIGN(size, 4096);
		break;
	case COLOR_FMT_NV12_UBWC:
	case COLOR_FMT_NV12_BPP10_UBWC:
		y_ubwc_plane = MSM_MEDIA_ALIGN(y_stride * y_sclines, 4096);
		uv_ubwc_plane = MSM_MEDIA_ALIGN(uv_stride * uv_sclines, 4096);
		y_meta_stride = VENUS_Y_META_STRIDE(color_fmt, width);
		y_meta_scanlines = VENUS_Y_META_SCANLINES(color_fmt, height);
		y_meta_plane = MSM_MEDIA_ALIGN(
				y_meta_stride * y_meta_scanlines, 4096);
		uv_meta_stride = VENUS_UV_META_STRIDE(color_fmt, width);
		uv_meta_scanlines = VENUS_UV_META_SCANLINES(color_fmt, height);
		uv_meta_plane = MSM_MEDIA_ALIGN(uv_meta_stride *
					uv_meta_scanlines, 4096);

		size = y_ubwc_plane + uv_ubwc_plane + y_meta_plane +
			uv_meta_plane +
			MSM_MEDIA_MAX(extra_size + 8192, 48 * y_stride);
		size = MSM_MEDIA_ALIGN(size, 4096);
		break;
	case COLOR_FMT_RGBA8888:
		rgb_plane = MSM_MEDIA_ALIGN(rgb_stride  * rgb_scanlines, 4096);
		size = rgb_plane;
		size =  MSM_MEDIA_ALIGN(size, 4096);
		break;
	case COLOR_FMT_RGBA8888_UBWC:
		rgb_ubwc_plane = MSM_MEDIA_ALIGN(rgb_stride * rgb_scanlines,
							4096);
		rgb_meta_stride = VENUS_RGB_META_STRIDE(color_fmt, width);
		rgb_meta_scanlines = VENUS_RGB_META_SCANLINES(color_fmt,
					height);
		rgb_meta_plane = MSM_MEDIA_ALIGN(rgb_meta_stride *
					rgb_meta_scanlines, 4096);
		size = rgb_ubwc_plane + rgb_meta_plane;
		size = MSM_MEDIA_ALIGN(size, 4096);
		break;
	default:
		break;
	}
invalid_input:
	return size;
}

static inline unsigned int VENUS_VIEW2_OFFSET(
	int color_fmt, int width, int height)
{
	unsigned int offset = 0;
	unsigned int y_plane, uv_plane, y_stride,
		uv_stride, y_sclines, uv_sclines;
	if (!width || !height)
		goto invalid_input;

	y_stride = VENUS_Y_STRIDE(color_fmt, width);
	uv_stride = VENUS_UV_STRIDE(color_fmt, width);
	y_sclines = VENUS_Y_SCANLINES(color_fmt, height);
	uv_sclines = VENUS_UV_SCANLINES(color_fmt, height);
	switch (color_fmt) {
	case COLOR_FMT_NV12_MVTB:
		y_plane = y_stride * y_sclines;
		uv_plane = uv_stride * uv_sclines;
		offset = y_plane + uv_plane;
		break;
	default:
		break;
	}
invalid_input:
	return offset;
}

#endif
