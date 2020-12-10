#pragma once

#include <stdbool.h>

#include "clutil.h"
#include "camera_common.h"

#define FRAME_BUF_COUNT 16

typedef struct CameraState {
  int camera_id;
  CameraInfo ci;

  int fps;
  float digital_gain;
  float cur_gain_frac;

  mat3 transform;

  CameraBuf buf;
} CameraState;

typedef struct MultiCameraState {
  int ispif_fd;

  CameraState rear;
  CameraState front;

  SubMaster *sm;
  PubMaster *pm;
} MultiCameraState;

void cameras_init(MultiCameraState *s, CLContext *ctx);
void cameras_open(MultiCameraState *s);
void cameras_run(MultiCameraState *s);
void cameras_close(MultiCameraState *s);
void camera_autoexposure(CameraState *s, float grey_frac);
