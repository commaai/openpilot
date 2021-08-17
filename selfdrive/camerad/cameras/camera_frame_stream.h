#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "camera_common.h"

#define FRAME_BUF_COUNT 16

typedef struct CameraState {
  CameraType cam_type;
  int camera_num;
  CameraInfo ci;

  int fps;
  float digital_gain;

  CameraBuf buf;
} CameraState;

class MultiCameraState : public CameraServerBase {
public:
  CameraState road_cam{.cam_type = RoadCam};
  CameraState driver_cam{.cam_type = DriverCam};
};
