#pragma once

#include <array>
#include "common/mat.h"
#include "system/hardware/hw.h"

const int TRAJECTORY_SIZE = 33;
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

const mat3 fcam_intrinsic_matrix = (mat3){{2648.0, 0.0, 1928.0 / 2,
                                           0.0, 2648.0, 1208.0 / 2,
                                           0.0, 0.0, 1.0}};

// tici ecam focal probably wrong? magnification is not consistent across frame
// Need to retrain model before this can be changed
const mat3 ecam_intrinsic_matrix = (mat3){{567.0, 0.0, 1928.0 / 2,
                                           0.0, 567.0, 1208.0 / 2,
                                           0.0, 0.0, 1.0}};

static inline mat3 get_model_yuv_transform() {
  float db_s = 1.0;
  const mat3 transform = (mat3){{
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0
  }};
  // Can this be removed since scale is 1?
  return transform_scale_buffer(transform, db_s);
}


const int FACE_KPTS_SIZE = 31;
const vec3 default_face_kpts_3d[FACE_KPTS_SIZE] = {
  {-5.76, -40.20, 8.00}, {-13.88, -38.03, 8.00}, {-19.84, -34.79, 8.00}, {-24.71, -29.91, 8.00}, {-26.88, -24.50, 8.00},
  {-27.42, -21.25, 8.00}, {-27.42, -9.34, 8.00}, {-26.88, -5.01, 8.00}, {-25.25, 0.94, 8.00}, {-25.25, 13.39, 8.00},
  {-24.71, 15.02, 8.00}, {-23.63, 17.18, 8.00}, {-16.59, 25.30, 8.00}, {-9.55, 32.88, 8.00}, {-5.76, 34.51, 8.00},
  {6.15, 34.51, 8.00}, {9.94, 32.88, 8.00}, {16.97, 25.30, 8.00}, {24.01, 17.18, 8.00}, {25.09, 15.02, 8.00},
  {25.64, 13.39, 8.00}, {25.64, 0.94, 8.00}, {27.26, -5.01, 8.00}, {27.80, -9.34, 8.00}, {27.80, -21.25, 8.00},
  {27.26, -24.50, 8.00}, {25.09, -29.91, 8.00}, {20.22, -34.79, 8.00}, {14.27, -38.03, 8.00}, {6.15, -40.20, 8.00},
  {-5.76, -40.20, 8.00},
};

const int face_end_idxs[1]= {FACE_KPTS_SIZE-1};
