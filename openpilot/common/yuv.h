#pragma once

#include <cstdint>

// NV12: Y plane + interleaved UV. I420: planar Y, U, V.

namespace yuv {

// Deinterleave NV12 UV into planar I420.
void nv12_to_i420(const uint8_t *src_y, int src_stride_y,
                  const uint8_t *src_uv, int src_stride_uv,
                  uint8_t *dst_y, int dst_stride_y,
                  uint8_t *dst_u, int dst_stride_u,
                  uint8_t *dst_v, int dst_stride_v,
                  int width, int height);

// Interleave planar I420 UV into NV12.
void i420_to_nv12(const uint8_t *src_y, int src_stride_y,
                  const uint8_t *src_u, int src_stride_u,
                  const uint8_t *src_v, int src_stride_v,
                  uint8_t *dst_y, int dst_stride_y,
                  uint8_t *dst_uv, int dst_stride_uv,
                  int width, int height);

// Point-sample scale I420 (equivalent to libyuv::I420Scale + kFilterNone).
void i420_scale(const uint8_t *src_y, int src_stride_y,
                const uint8_t *src_u, int src_stride_u,
                const uint8_t *src_v, int src_stride_v,
                int src_width, int src_height,
                uint8_t *dst_y, int dst_stride_y,
                uint8_t *dst_u, int dst_stride_u,
                uint8_t *dst_v, int dst_stride_v,
                int dst_width, int dst_height);

// Convert NV12 to packed RGBA (R,G,B,A bytes — suitable for GL_RGBA).
// BT.601 limited-range, matching common libyuv defaults.
void nv12_to_rgba(const uint8_t *src_y, int src_stride_y,
                  const uint8_t *src_uv, int src_stride_uv,
                  uint8_t *dst_rgba, int dst_stride_rgba,
                  int width, int height);

}
