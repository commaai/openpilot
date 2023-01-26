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


const int FACE_KPTS_SIZE = 57;
const vec3 default_face_kpts_3d[FACE_KPTS_SIZE] = {
  {-64.80, -19.71, -22.56}, {-63.04, -2.68, -20.54}, {-59.69, 12.73, -18.64}, {-56.51, 26.59, -14.68}, {-51.61, 41.26, -6.12}, {-42.50, 52.51, 8.68}, {-31.36, 58.69, 26.52}, {-17.60, 62.89, 43.60}, {1.93, 65.22, 50.18}, {21.13, 62.47, 43.06}, {34.25, 58.25, 25.68}, {44.45, 52.28, 7.90}, {52.50, 41.01, -6.65}, {56.57, 26.47, -15.06}, {59.06, 12.66, -19.13}, {61.61, -2.63, -21.11}, {62.95, -19.65, -23.01},
  {-50.99, -37.46, 30.80}, {-43.76, -43.25, 41.18}, {-34.45, -45.06, 48.58}, {-25.53, -44.41, 53.16}, {-17.44, -42.23, 55.32},
  {15.33, -42.33, 55.10}, {23.44, -44.48, 52.84}, {32.32, -45.28, 48.19}, {41.74, -43.60, 40.76}, {49.14, -37.44, 30.38},
  {-0.61, -25.58, 57.75}, {-0.43, -15.73, 65.78}, {-0.14, -6.19, 73.98}, {0.03, 2.17, 76.13}, {0.09, 11.88, 62.73},
  {-38.79, -25.00, 36.90}, {-33.23, -28.21, 44.21}, {-25.18, -28.29, 44.49}, {-17.67, -24.96, 42.53}, {-24.45, -23.02, 44.27}, {-32.79, -22.60, 42.11}, {-38.79, -25.00, 36.90},
  {15.81, -24.93, 42.12}, {23.25, -28.32, 44.00}, {31.50, -28.09, 43.31}, {37.21, -24.83, 36.65}, {31.15, -22.62, 41.91}, {22.73, -22.93, 43.98}, {15.81, -24.93, 42.12},
  {15.78, 29.53, 57.14}, {8.59, 30.96, 61.77}, {1.35, 31.13, 62.85}, {-6.06, 31.04, 61.83}, {-13.67, 29.79, 57.36}, {-21.60, 27.77, 48.10}, {-6.69, 27.08, 57.90}, {0.58, 27.10, 59.42}, {7.63, 27.10, 57.85}, {22.93, 27.55, 47.46}, {15.78, 29.53, 57.14},
};

const int face_end_idxs[7]= {16, 21, 26, 31, 38, 45, FACE_KPTS_SIZE-1};
