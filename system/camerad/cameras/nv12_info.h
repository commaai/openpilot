#pragma once

#include <cassert>
#include <cstdint>
#include <tuple>

#include "third_party/linux/include/msm_media_info.h"

// Returns NV12 aligned width, height, and buffer size for the given frame.
inline std::tuple<uint32_t, uint32_t, uint32_t> get_nv12_info(int width, int height) {
  const uint32_t nv12_width = VENUS_Y_STRIDE(COLOR_FMT_NV12, width);
  const uint32_t nv12_height = VENUS_Y_SCANLINES(COLOR_FMT_NV12, height);
  assert(nv12_width == VENUS_UV_STRIDE(COLOR_FMT_NV12, width));
  assert(nv12_height / 2 == VENUS_UV_SCANLINES(COLOR_FMT_NV12, height));
  const uint32_t nv12_buffer_size = 2346 * nv12_width;  // Matches camera driver sizeimage
  return {nv12_width, nv12_height, nv12_buffer_size};
}
