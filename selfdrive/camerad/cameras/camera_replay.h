#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/common/framereader.h"

#define FRAME_BUF_COUNT 16

typedef struct CameraState {
  int camera_num;
  CameraInfo ci;

  int fps;
  float digital_gain = 0;

  CameraBuf buf;
  FrameReader *frame_reader = nullptr;
} CameraState;

typedef struct MultiCameraState {
  CameraState road_cam;
  CameraState driver_cam;

  SubMaster *sm;
  PubMaster *pm;
} MultiCameraState;
