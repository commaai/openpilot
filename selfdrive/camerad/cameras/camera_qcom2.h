#ifndef CAMERA_H
#define CAMERA_H

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

#include "common/mat.h"
#include "common/visionbuf.h"
#include "common/buffering.h"

#include "camera_common.h"

#define FRAME_BUF_COUNT 4
#define METADATA_BUF_COUNT 4

#ifdef __cplusplus
extern "C" {
#endif


typedef struct CameraState {
  CameraInfo ci;
  FrameMetadata camera_bufs_metadata[FRAME_BUF_COUNT];
  TBuffer camera_tb;

  int frame_size;
  float digital_gain;
  mat3 transform;

  int camera_num;
} CameraState;


typedef struct DualCameraState {
  int device;

  int ispif_fd;

  CameraState rear;
  CameraState front;
  CameraState wide;
} DualCameraState;

void cameras_init(DualCameraState *s);
void cameras_open(DualCameraState *s, VisionBuf *camera_bufs_rear, VisionBuf *camera_bufs_focus, VisionBuf *camera_bufs_stats, VisionBuf *camera_bufs_front);
void cameras_run(DualCameraState *s);
void camera_autoexposure(CameraState *s, float grey_frac);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif

