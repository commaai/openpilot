#ifndef CAMERA_FRAME_STREAM_H
#define CAMERA_FRAME_STREAM_H

#include <stdbool.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "common/mat.h"

#include "buffering.h"
#include "common/visionbuf.h"
#include "camera_common.h"

#define FRAME_BUF_COUNT 16

#ifdef __cplusplus
extern "C" {
#endif

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


typedef struct DualCameraState {
  int ispif_fd;

  CameraState rear;
  CameraState front;
} DualCameraState;

void cameras_init(DualCameraState *s);
void cameras_open(DualCameraState *s, VisionBuf *camera_bufs_rear, VisionBuf *camera_bufs_focus, VisionBuf *camera_bufs_stats, VisionBuf *camera_bufs_front);
void cameras_run(DualCameraState *s);
void cameras_close(DualCameraState *s);
void camera_autoexposure(CameraState *s, float grey_frac);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif
