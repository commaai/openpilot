#pragma once

#include <cassert>
#include <cstdint>
#include <tuple>

constexpr uint32_t nv12_align(uint32_t val, uint32_t alignment) {
  return ((val + alignment - 1) / alignment) * alignment;
}

// Returns NV12 aligned (stride, y_height, uv_height, buffer_size) for the given frame dimensions.
inline std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_nv12_info(int width, int height) {
  if (width <= 0 || height <= 0) return {0, 0, 0, 0};

  const uint32_t frame_width = static_cast<uint32_t>(width);
  const uint32_t frame_height = static_cast<uint32_t>(height);
  const uint32_t stride = nv12_align(frame_width, 128);
  const uint32_t y_height = nv12_align(frame_height, 32);
  const uint32_t uv_height = nv12_align((frame_height + 1) / 2, 16);

  // NV12 case from VENUS_BUFFER_SIZE in media/msm_media_info.h.
  const uint32_t y_plane = stride * y_height;
  const uint32_t uv_plane = stride * uv_height + 4096;
  const uint32_t extra_size = 16 * 1024;
  const uint32_t padding = extra_size > 8 * stride ? extra_size : 8 * stride;
  uint32_t size = y_plane + uv_plane + padding;
  size = nv12_align(size, 4096);
  size += nv12_align(frame_width, 512) * 512;
  size = nv12_align(size, 4096);

  // Sanity checks for NV12 format assumptions
  assert(y_height / 2 == uv_height);
  assert((stride * y_height) % 0x1000 == 0);  // uv_offset must be page-aligned

  return {stride, y_height, uv_height, size};
}
