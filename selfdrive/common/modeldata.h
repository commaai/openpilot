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

const int TICI_CAM_WIDTH = 1928;
const int TICI_QCAM_WIDTH = (526 + 3) & ~3; // same as replay/framereader

namespace tici_dm_crop {
  const int x_offset = -72;
  const int y_offset = -144;
  const int width = 954;
};

inline bool is_tici_frame(int frame_width) {
  return frame_width == TICI_CAM_WIDTH || frame_width == TICI_QCAM_WIDTH;
}

inline const mat3 get_camera_intrinsics(int frame_width, bool wide_camera) {
  static constexpr mat3 eon_intrinsics =
      (mat3){{910., 0., 1164.0 / 2,
              0., 910., 874.0 / 2,
              0., 0., 1.}};
  static constexpr mat3 tici_intrinsics =
      (mat3){{2648.0, 0.0, 1928.0 / 2,
              0.0, 2648.0, 1208.0 / 2,
              0.0, 0.0, 1.0}};
  // without unwarp, focal length is for center portion only
  static constexpr mat3 tici_wide_intrinsics =
      (mat3){{620.0, 0.0, 1928.0 / 2,
              0.0, 620.0, 1208.0 / 2,
              0.0, 0.0, 1.0}};

  if (is_tici_frame(frame_width)) {
    return wide_camera ? tici_wide_intrinsics : tici_intrinsics;
  } else {
    return eon_intrinsics;
  }
}

inline mat3 get_model_yuv_transform(int frame_width, bool bayer = true) {
  float db_s = is_tici_frame(frame_width) ? 1.0 : 0.5;  // debayering does a 2x downscale on EON
  const mat3 transform = (mat3){{
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0
  }};
  return bayer ? transform_scale_buffer(transform, db_s) : transform;
}
