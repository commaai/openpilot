#pragma once

#include "camera_common.h"

#define FRAME_BUF_COUNT 16

typedef struct CameraState {
  int camera_num;
  CameraInfo ci;

  int fps;
  float digital_gain;

  CameraBuf buf;
} CameraState;

class CameraServer : public CameraServerBase {
public:
  CameraServer() : CameraServerBase() {}
  CameraState road_cam;
  CameraState driver_cam;
};
