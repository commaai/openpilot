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
  {-52.74, -15.11, -20.92}, {-51.32, -1.25, -19.27}, {-48.59, 11.29, -17.73}, {-45.99, 22.57, -14.50}, {-42.01, 34.51, -7.54}, {-34.60, 43.67, 4.51}, {-25.53, 48.70, 19.03}, {-14.32, 52.12, 32.93}, {1.57, 54.01, 38.29}, {17.20, 51.77, 32.49}, {27.88, 48.34, 18.34}, {36.18, 43.48, 3.87}, {42.73, 34.31, -7.97}, {46.04, 22.47, -14.82}, {48.07, 11.24, -18.13}, {50.15, -1.21, -19.74}, {51.24, -15.07, -21.28},
  {-41.50, -29.56, 22.52}, {-35.61, -34.28, 30.96}, {-28.04, -35.75, 36.99}, {-20.78, -35.22, 40.71}, {-14.20, -33.45, 42.47},
  {12.48, -33.52, 42.29}, {19.08, -35.27, 40.45}, {26.30, -35.93, 36.67}, {33.97, -34.56, 30.62}, {40.00, -29.55, 22.17},
  {-0.49, -19.89, 44.45}, {-0.35, -11.88, 50.99}, {-0.11, -4.11, 57.66}, {0.03, 2.69, 59.41}, {0.08, 10.60, 48.50},
  {-31.57, -19.42, 27.48}, {-27.05, -22.03, 33.43}, {-20.49, -22.10, 33.66}, {-14.38, -19.38, 32.06}, {-19.90, -17.81, 33.48}, {-26.69, -17.47, 31.72}, {-31.57, -19.42, 27.48},
  {12.87, -19.36, 31.72}, {18.93, -22.12, 33.25}, {25.64, -21.94, 32.69}, {30.28, -19.28, 27.27}, {25.36, -17.48, 31.55}, {18.50, -17.73, 33.24}, {12.87, -19.36, 31.72},
  {12.85, 24.96, 43.95}, {7.00, 26.13, 47.72}, {1.10, 26.27, 48.60}, {-4.93, 26.20, 47.77}, {-11.13, 25.18, 44.13}, {-17.58, 23.53, 36.60}, {-5.45, 22.97, 44.57}, {0.47, 22.99, 45.81}, {6.21, 22.99, 44.53}, {18.67, 23.36, 36.07}, {12.85, 24.96, 43.95},
};

const int face_end_idxs[7]= {16, 21, 26, 31, 38, 45, FACE_KPTS_SIZE-1};
