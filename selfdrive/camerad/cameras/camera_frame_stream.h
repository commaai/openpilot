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

class CameraState : public CameraStateBase{
public:
  float digital_gain;
};
typedef struct MultiCameraState {
  CameraState road_cam;
  CameraState driver_cam;

  SubMaster *sm;
  PubMaster *pm;
} MultiCameraState;
