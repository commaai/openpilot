#pragma once

#include <array>
#include "selfdrive/common/mat.h"
#include "selfdrive/hardware/hw.h"

const int  TRAJECTORY_SIZE = 33;
const int LAT_MPC_N = 16;
const int LON_MPC_N = 32;
const float MIN_DRAW_DISTANCE = 10.0;
const float MAX_DRAW_DISTANCE = 100.0;

template <typename T, size_t size>
constexpr std::array<T, size> build_idxs(float max_val) {
  std::array<T, size> result{};
  for (int i = 0; i < size; ++i) {
    result[i] = max_val * ((i / (double)(size - 1)) * (i / (double)(size - 1)));
  }
  return result;
}

constexpr auto T_IDXS = build_idxs<double, TRAJECTORY_SIZE>(10.0);
constexpr auto T_IDXS_FLOAT = build_idxs<float, TRAJECTORY_SIZE>(10.0);
constexpr auto X_IDXS = build_idxs<double, TRAJECTORY_SIZE>(192.0);
constexpr auto X_IDXS_FLOAT = build_idxs<float, TRAJECTORY_SIZE>(192.0);

#define ALIGN(x, align) (((x) + (align)-1) & ~((align)-1))

const int CAM_RGB_ALIGN = 4;
const int TICI_CAM_WIDTH = 1928;
const int TICI_QCAM_WIDTH = ALIGN(526, CAM_RGB_ALIGN); // same alignment as replay/FrameReader

namespace tici_dm_crop {
  const int x_offset = -72;
  const int y_offset = -144;
  const int width = 954;
};

inline mat3 get_intrinsic_matrix(int w, bool wide, float *y_offset = nullptr, float *zoom = nullptr) {
  constexpr mat3 eon_cam_intrinsics = {{910., 0., 1164.0 / 2,
                                        0., 910., 874.0 / 2,
                                        0., 0., 1.}};
  constexpr mat3 tici_cam_intrinsics = {{2648.0, 0.0, 1928.0 / 2,
                                         0.0, 2648.0, 1208.0 / 2,
                                         0.0, 0.0, 1.0}};
  // without unwarp, focal length is for center portion only
  constexpr mat3 tici_wide_cam_intrinsics = {{620.0, 0.0, 1928.0 / 2,
                                              0.0, 620.0, 1208.0 / 2,
                                              0.0, 0.0, 1.0}};
  mat3 mat = {};
  bool is_tici = (w == TICI_CAM_WIDTH || w == TICI_QCAM_WIDTH);
  if (is_tici) {
    mat = wide ? tici_wide_cam_intrinsics : tici_cam_intrinsics;
  } else {
    mat = eon_cam_intrinsics;
  }

  if (y_offset) {
    *y_offset = is_tici ? 150.0 : 0.0;
  }

  if (zoom) {
    *zoom = (is_tici ? 2912.8 : 2138.5) / mat.v[0];
    if (wide) *zoom *= 0.5;
  }

  return mat;
}

static inline mat3 get_model_yuv_transform(int w, bool bayer = true) {
  const bool is_tici = (w == TICI_CAM_WIDTH || w == TICI_QCAM_WIDTH);
  float db_s = is_tici ? 1.0 : 0.5; // debayering does a 2x downscale on EON
  const mat3 transform = (mat3){{
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0
  }};
  return bayer ? transform_scale_buffer(transform, db_s) : transform;
}
