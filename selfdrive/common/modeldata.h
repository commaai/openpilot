#pragma once
const int  TRAJECTORY_SIZE = 33;
const int LAT_MPC_N = 16;
const int LON_MPC_N = 32;
const float MIN_DRAW_DISTANCE = 10.0;
const float MAX_DRAW_DISTANCE = 100.0;

const double T_IDXS[TRAJECTORY_SIZE] = {
        0.        ,  0.00976562,  0.0390625 ,  0.08789062,  0.15625   ,
        0.24414062,  0.3515625 ,  0.47851562,  0.625     ,  0.79101562,
        0.9765625 ,  1.18164062,  1.40625   ,  1.65039062,  1.9140625 ,
        2.19726562,  2.5       ,  2.82226562,  3.1640625 ,  3.52539062,
        3.90625   ,  4.30664062,  4.7265625 ,  5.16601562,  5.625     ,
        6.10351562,  6.6015625 ,  7.11914062,  7.65625   ,  8.21289062,
        8.7890625 ,  9.38476562, 10.};
const double X_IDXS[TRAJECTORY_SIZE] = {
         0.    ,   0.1875,   0.75  ,   1.6875,   3.    ,   4.6875,
         6.75  ,   9.1875,  12.    ,  15.1875,  18.75  ,  22.6875,
        27.    ,  31.6875,  36.75  ,  42.1875,  48.    ,  54.1875,
        60.75  ,  67.6875,  75.    ,  82.6875,  90.75  ,  99.1875,
       108.    , 117.1875, 126.75  , 136.6875, 147.    , 157.6875,
       168.75  , 180.1875, 192.};

#ifdef __cplusplus

#include "selfdrive/common/mat.h"
#include "selfdrive/hardware/hw.h"
const mat3 fcam_intrinsic_matrix =
    Hardware::TICI() ? (mat3){{2648.0, 0.0, 1928.0 / 2,
                               0.0, 2648.0, 1208.0 / 2,
                               0.0, 0.0, 1.0}}
                     : (mat3){{910., 0., 1164.0 / 2,
                               0., 910., 874.0 / 2,
                               0., 0., 1.}};

// without unwarp, focal length is for center portion only
const mat3 ecam_intrinsic_matrix =
    Hardware::TICI() ? (mat3){{620.0, 0.0, 1928.0 / 2,
                               0.0, 620.0, 1208.0 / 2,
                               0.0, 0.0, 1.0}}
                     : (mat3){{0., 0., 0.,
                               0., 0., 0.,
                               0., 0., 0.}};

static inline mat3 get_model_yuv_transform(bool bayer = true) {
  float db_s = Hardware::TICI() ? 1.0 : 0.5; // debayering does a 2x downscale on EON
  const mat3 transform = (mat3){{
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0
  }};
  return bayer ? transform_scale_buffer(transform, db_s) : transform;
}

#endif
