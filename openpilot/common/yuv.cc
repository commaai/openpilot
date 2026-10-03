#include "common/yuv.h"

#include <algorithm>
#include <cstring>

namespace yuv {

namespace {

inline uint8_t clamp_u8(int v) {
  return static_cast<uint8_t>(std::clamp(v, 0, 255));
}

void copy_plane(const uint8_t *src, int src_stride,
                uint8_t *dst, int dst_stride,
                int width, int height) {
  if (src_stride == width && dst_stride == width) {
    std::memcpy(dst, src, static_cast<size_t>(width) * height);
    return;
  }
  for (int y = 0; y < height; ++y) {
    std::memcpy(dst + y * dst_stride, src + y * src_stride, width);
  }
}

void scale_plane_point(const uint8_t *src, int src_stride, int src_width, int src_height,
                       uint8_t *dst, int dst_stride, int dst_width, int dst_height) {
  if (src_width == dst_width && src_height == dst_height) {
    copy_plane(src, src_stride, dst, dst_stride, dst_width, dst_height);
    return;
  }
  for (int y = 0; y < dst_height; ++y) {
    const int sy = y * src_height / dst_height;
    const uint8_t *src_row = src + sy * src_stride;
    uint8_t *dst_row = dst + y * dst_stride;
    for (int x = 0; x < dst_width; ++x) {
      dst_row[x] = src_row[x * src_width / dst_width];
    }
  }
}

// BT.601 limited range → RGB (integer form used widely, incl. similar to libyuv).
inline void yuv_to_rgb(int y, int u, int v, uint8_t *r, uint8_t *g, uint8_t *b) {
  const int c = (y - 16) * 298;
  const int d = u - 128;
  const int e = v - 128;
  *r = clamp_u8((c + 409 * e + 128) >> 8);
  *g = clamp_u8((c - 100 * d - 208 * e + 128) >> 8);
  *b = clamp_u8((c + 516 * d + 128) >> 8);
}

}  // namespace

void nv12_to_i420(const uint8_t *src_y, int src_stride_y,
                  const uint8_t *src_uv, int src_stride_uv,
                  uint8_t *dst_y, int dst_stride_y,
                  uint8_t *dst_u, int dst_stride_u,
                  uint8_t *dst_v, int dst_stride_v,
                  int width, int height) {
  copy_plane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);

  const int uv_width = width / 2;
  const int uv_height = height / 2;
  for (int y = 0; y < uv_height; ++y) {
    const uint8_t *uv = src_uv + y * src_stride_uv;
    uint8_t *u = dst_u + y * dst_stride_u;
    uint8_t *v = dst_v + y * dst_stride_v;
    for (int x = 0; x < uv_width; ++x) {
      u[x] = uv[2 * x];
      v[x] = uv[2 * x + 1];
    }
  }
}

void i420_to_nv12(const uint8_t *src_y, int src_stride_y,
                  const uint8_t *src_u, int src_stride_u,
                  const uint8_t *src_v, int src_stride_v,
                  uint8_t *dst_y, int dst_stride_y,
                  uint8_t *dst_uv, int dst_stride_uv,
                  int width, int height) {
  copy_plane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);

  const int uv_width = width / 2;
  const int uv_height = height / 2;
  for (int y = 0; y < uv_height; ++y) {
    const uint8_t *u = src_u + y * src_stride_u;
    const uint8_t *v = src_v + y * src_stride_v;
    uint8_t *uv = dst_uv + y * dst_stride_uv;
    for (int x = 0; x < uv_width; ++x) {
      uv[2 * x] = u[x];
      uv[2 * x + 1] = v[x];
    }
  }
}

void i420_scale(const uint8_t *src_y, int src_stride_y,
                const uint8_t *src_u, int src_stride_u,
                const uint8_t *src_v, int src_stride_v,
                int src_width, int src_height,
                uint8_t *dst_y, int dst_stride_y,
                uint8_t *dst_u, int dst_stride_u,
                uint8_t *dst_v, int dst_stride_v,
                int dst_width, int dst_height) {
  scale_plane_point(src_y, src_stride_y, src_width, src_height,
                    dst_y, dst_stride_y, dst_width, dst_height);
  scale_plane_point(src_u, src_stride_u, src_width / 2, src_height / 2,
                    dst_u, dst_stride_u, dst_width / 2, dst_height / 2);
  scale_plane_point(src_v, src_stride_v, src_width / 2, src_height / 2,
                    dst_v, dst_stride_v, dst_width / 2, dst_height / 2);
}

void nv12_to_rgba(const uint8_t *src_y, int src_stride_y,
                  const uint8_t *src_uv, int src_stride_uv,
                  uint8_t *dst_rgba, int dst_stride_rgba,
                  int width, int height) {
  for (int y = 0; y < height; ++y) {
    const uint8_t *y_row = src_y + y * src_stride_y;
    const uint8_t *uv_row = src_uv + (y / 2) * src_stride_uv;
    uint8_t *dst = dst_rgba + y * dst_stride_rgba;
    for (int x = 0; x < width; ++x) {
      const int uv_x = (x & ~1);
      uint8_t r, g, b;
      yuv_to_rgb(y_row[x], uv_row[uv_x], uv_row[uv_x + 1], &r, &g, &b);
      dst[4 * x + 0] = r;
      dst[4 * x + 1] = g;
      dst[4 * x + 2] = b;
      dst[4 * x + 3] = 255;
    }
  }
}

}  // namespace yuv
