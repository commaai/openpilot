#pragma once

#include <cstdint>
#include <tuple>

#include "third_party/linux/include/msm_media_info.h"

// Returns NV12 aligned stride, height, and buffer size for the given frame dimensions.
inline std::tuple<uint32_t, uint32_t, uint32_t> get_nv12_info(int width, int height) {
  const uint32_t nv12_stride = VENUS_Y_STRIDE(COLOR_FMT_NV12, width);
  const uint32_t nv12_height = VENUS_Y_SCANLINES(COLOR_FMT_NV12, height);
  const uint32_t nv12_size = VENUS_BUFFER_SIZE(COLOR_FMT_NV12, width, height);
  return {nv12_stride, nv12_height, nv12_size};
}
