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
  {-7.01, -48.88, 8.00}, {-16.88, -46.25, 8.00}, {-24.12, -42.30, 8.00}, {-30.05, -36.38, 8.00}, {-32.68, -29.79, 8.00},
  {-33.34, -25.84, 8.00}, {-33.34, -11.36, 8.00}, {-32.68, -6.09, 8.00}, {-30.71, 1.15, 8.00}, {-30.71, 16.29, 8.00},
  {-30.05, 18.26, 8.00}, {-28.73, 20.90, 8.00}, {-20.17, 30.77, 8.00}, {-11.62, 39.99, 8.00}, {-7.01, 41.96, 8.00},
  {7.47, 41.96, 8.00}, {12.08, 39.99, 8.00}, {20.64, 30.77, 8.00}, {29.20, 20.90, 8.00}, {30.51, 18.26, 8.00},
  {31.17, 16.29, 8.00}, {31.17, 1.15, 8.00}, {33.15, -6.09, 8.00}, {33.81, -11.36, 8.00}, {33.81, -25.84, 8.00},
  {33.15, -29.79, 8.00}, {30.51, -36.38, 8.00}, {24.59, -42.30, 8.00}, {17.35, -46.25, 8.00}, {7.47, -48.88, 8.00},
  {-7.01, -48.88, 8.00},
};

const int face_end_idxs[1]= {FACE_KPTS_SIZE-1};
