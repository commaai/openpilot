#pragma once

#include "selfdrive/camerad/cameras/camera_common.h"

#define FRAME_BUF_COUNT 16

typedef struct CameraState {
  CameraInfo ci;
  int camera_num;
  int fps;
  float digital_gain;
  CameraBuf buf;
} CameraState;


struct MultiCameraState : public CameraServerBase {
  CameraState road_cam;
  CameraState driver_cam;
};
