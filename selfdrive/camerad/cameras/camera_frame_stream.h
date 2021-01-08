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
  CameraInfo ci;

  int fps;
  float digital_gain;
  float cur_gain_frac;

  CameraBuf buf;
} CameraState;

typedef struct MultiCameraState {
  int ispif_fd;

  CameraState rear;
  CameraState front;

  SubMaster *sm;
  PubMaster *pm;
} MultiCameraState;

void cameras_init(VisionIpcServer * v, MultiCameraState *s, cl_device_id device_id, cl_context ctx);
void cameras_open(MultiCameraState *s);
void cameras_run(MultiCameraState *s);
void cameras_close(MultiCameraState *s);
void camera_autoexposure(CameraState *s, float grey_frac);
