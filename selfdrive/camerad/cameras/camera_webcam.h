#pragma once

#include <stdbool.h>

#include "camera_common.h"

#define FRAME_BUF_COUNT 16

typedef struct CameraState {
  CameraInfo ci;
  int camera_num;
  int fps;
  float digital_gain;
  CameraBuf buf;
} CameraState;


class MultiCameraState : public MultiCameraStateBase {
public:
  MultiCameraState() : MultiCameraStateBase() {}
  CameraState road_cam;
  CameraState driver_cam;
};;
