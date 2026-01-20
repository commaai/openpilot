#pragma once

#include <cassert>
#include <cstdint>
#include <tuple>

#include "third_party/linux/include/msm_media_info.h"

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
