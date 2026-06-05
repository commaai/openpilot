#pragma once

#include <cassert>
#include <cstdint>
#include <tuple>

// NV12 subset copied from media/msm_media_info.h.
#ifndef MSM_MEDIA_ALIGN
#define MSM_MEDIA_ALIGN(__sz, __align) (((__align) & ((__align) - 1)) ? \
  ((((__sz) + (__align) - 1) / (__align)) * (__align)) : \
  (((__sz) + (__align) - 1) & (~((__align) - 1))))
#endif

#ifndef MSM_MEDIA_MAX
#define MSM_MEDIA_MAX(__a, __b) ((__a) > (__b) ? (__a) : (__b))
#endif

enum color_fmts {
  COLOR_FMT_NV12,
};

static inline unsigned int VENUS_EXTRADATA_SIZE(int width, int height) {
  (void)height;
  (void)width;
  return 16 * 1024;
}

static inline unsigned int VENUS_Y_STRIDE(int color_fmt, int width) {
  unsigned int stride = 0;
  if (!width) goto invalid_input;

  switch (color_fmt) {
    case COLOR_FMT_NV12:
      stride = MSM_MEDIA_ALIGN(width, 128);
      break;
    default:
      break;
  }
invalid_input:
  return stride;
}

static inline unsigned int VENUS_UV_STRIDE(int color_fmt, int width) {
  unsigned int stride = 0;
  if (!width) goto invalid_input;

  switch (color_fmt) {
    case COLOR_FMT_NV12:
      stride = MSM_MEDIA_ALIGN(width, 128);
      break;
    default:
      break;
  }
invalid_input:
  return stride;
}

static inline unsigned int VENUS_Y_SCANLINES(int color_fmt, int height) {
  unsigned int sclines = 0;
  if (!height) goto invalid_input;

  switch (color_fmt) {
    case COLOR_FMT_NV12:
      sclines = MSM_MEDIA_ALIGN(height, 32);
      break;
    default:
      break;
  }
invalid_input:
  return sclines;
}

static inline unsigned int VENUS_UV_SCANLINES(int color_fmt, int height) {
  unsigned int sclines = 0;
  if (!height) goto invalid_input;

  switch (color_fmt) {
    case COLOR_FMT_NV12:
      sclines = MSM_MEDIA_ALIGN((height + 1) >> 1, 16);
      break;
    default:
      break;
  }
invalid_input:
  return sclines;
}

static inline unsigned int VENUS_BUFFER_SIZE(int color_fmt, int width, int height) {
  const unsigned int extra_size = VENUS_EXTRADATA_SIZE(width, height);
  unsigned int size = 0;
  unsigned int y_stride = 0, uv_stride = 0, y_sclines = 0, uv_sclines = 0;

  if (!width || !height) goto invalid_input;

  y_stride = VENUS_Y_STRIDE(color_fmt, width);
  uv_stride = VENUS_UV_STRIDE(color_fmt, width);
  y_sclines = VENUS_Y_SCANLINES(color_fmt, height);
  uv_sclines = VENUS_UV_SCANLINES(color_fmt, height);

  switch (color_fmt) {
    case COLOR_FMT_NV12: {
      const unsigned int y_plane = y_stride * y_sclines;
      const unsigned int uv_plane = uv_stride * uv_sclines + 4096;
      size = y_plane + uv_plane + MSM_MEDIA_MAX(extra_size, 8 * y_stride);
      size = MSM_MEDIA_ALIGN(size, 4096);
      size += MSM_MEDIA_ALIGN(width, 512) * 512;
      size = MSM_MEDIA_ALIGN(size, 4096);
      break;
    }
    default:
      break;
  }
invalid_input:
  return size;
}

// Returns NV12 aligned (stride, y_height, uv_height, buffer_size) for the given frame dimensions.
inline std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_nv12_info(int width, int height) {
  const uint32_t stride = VENUS_Y_STRIDE(COLOR_FMT_NV12, width);
  const uint32_t y_height = VENUS_Y_SCANLINES(COLOR_FMT_NV12, height);
  const uint32_t uv_height = VENUS_UV_SCANLINES(COLOR_FMT_NV12, height);
  const uint32_t size = VENUS_BUFFER_SIZE(COLOR_FMT_NV12, width, height);

  // Sanity checks for NV12 format assumptions
  assert(stride == VENUS_UV_STRIDE(COLOR_FMT_NV12, width));
  assert(y_height / 2 == uv_height);
  assert((stride * y_height) % 0x1000 == 0);  // uv_offset must be page-aligned

  return {stride, y_height, uv_height, size};
}
