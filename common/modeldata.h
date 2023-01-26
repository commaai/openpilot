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


const int FACE_KPTS_SIZE = 63;
const vec3 default_face_kpts_3d[FACE_KPTS_SIZE] = {
  {-67.81, -20.85, -15.47}, {-65.98, -3.04, -13.35}, {-62.47, 13.09, -11.37}, {-59.14, 27.59, -7.22}, {-54.01, 42.94, 1.73}, {-44.48, 54.71, 17.22}, {-32.82, 61.18, 35.89}, {-18.42, 65.58, 53.77}, {2.02, 68.02, 60.66}, {22.11, 65.14, 53.20}, {35.84, 60.73, 35.01}, {46.52, 54.48, 16.40}, {54.94, 42.68, 1.18}, {59.20, 27.47, -7.62}, {61.81, 13.02, -11.88}, {64.48, -2.98, -13.95}, {65.88, -20.80, -15.94},
  {-53.36, -39.44, 40.38}, {-45.79, -45.50, 51.24}, {-36.05, -47.39, 58.98}, {-26.72, -46.71, 63.77}, {-18.26, -44.43, 66.03},
  {16.04, -44.53, 65.81}, {24.53, -46.78, 63.43}, {33.82, -47.62, 58.57}, {43.68, -45.86, 50.79}, {51.43, -39.42, 39.93},
  {-0.63, -27.00, 68.58}, {-0.45, -16.70, 76.98}, {-0.15, -6.71, 85.56}, {0.03, 2.04, 87.81}, {0.10, 12.20, 73.78},
  {-40.59, -26.39, 46.76}, {-34.78, -29.75, 54.40}, {-26.35, -29.84, 54.70}, {-18.49, -26.35, 52.64}, {-25.58, -24.33, 54.47}, {-34.32, -23.89, 52.21}, {-40.59, -26.39, 46.76},
  {16.54, -26.32, 52.22}, {24.34, -29.87, 54.18}, {32.96, -29.63, 53.46}, {38.94, -26.21, 46.49}, {32.60, -23.91, 52.00}, {23.79, -24.23, 54.16}, {16.54, -26.32, 52.22},
  {-24.99, 29.63, 57.84}, {-16.17, 25.40, 67.58}, {-5.21, 22.32, 73.25}, {0.48, 23.42, 73.75}, {5.92, 22.32, 73.18}, {16.83, 25.28, 67.31}, {25.53, 29.17, 57.34}, {16.52, 30.67, 67.94}, {8.99, 32.17, 72.79}, {1.41, 32.34, 73.91}, {-6.34, 32.25, 72.85}, {-14.31, 30.94, 68.17}, {-22.61, 28.83, 58.48}, {-7.00, 28.11, 68.73}, {0.61, 28.13, 70.32}, {7.98, 28.13, 68.68}, {24.00, 28.60, 57.80},
};

const int face_end_idxs[7]= {16, 21, 26, 31, 38, 45, FACE_KPTS_SIZE-1};
