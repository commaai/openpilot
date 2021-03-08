#pragma once

#include <stdbool.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "camera_common.h"

#define FRAME_BUF_COUNT 16

typedef struct CameraState {
  int camera_id;
  int camera_num;
  CameraInfo ci;

  int fps;
  float digital_gain;
  float cur_gain_frac;

  CameraBuf buf;
} CameraState;

typedef struct MultiCameraState {
  int ispif_fd;

  CameraState road_cam;
  CameraState driver_cam;

  SubMaster *sm;
  PubMaster *pm;
} MultiCameraState;
