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
  int frame_size;

  VisionBuf *camera_bufs;
  FrameMetadata camera_bufs_metadata[FRAME_BUF_COUNT];
  TBuffer camera_tb;

  int fps;
  float digital_gain;

  float cur_gain_frac;

  mat3 transform;
} CameraState;

typedef struct MultiCameraState {
  int ispif_fd;

  CameraState rear;
  CameraState front;
} MultiCameraState;

void cameras_init(MultiCameraState *s);
void cameras_open(cl_device_id device_id, cl_context ctx, MultiCameraState *s, VisionBuf *camera_bufs_rear, VisionBuf *camera_bufs_front);
void cameras_run(MultiCameraState *s);
void cameras_close(MultiCameraState *s);
void camera_autoexposure(CameraState *s, float grey_frac);
void camera_process_frame(MultiCameraState *s, CameraBuf *b, int cnt);
