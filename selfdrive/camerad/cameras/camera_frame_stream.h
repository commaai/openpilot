#pragma once

#include "selfdrive/camerad/cameras/camera_common.h"

#define FRAME_BUF_COUNT 16

typedef struct CameraState {
  CameraType cam_type;
  int camera_num;
  CameraInfo ci;

  int fps;
  float digital_gain;

  CameraBuf buf;
} CameraState;

class CameraServer : public CameraServerBase {
public:
  CameraServer();
  void run() override;

  CameraState road_cam = {.cam_type = RoadCam};
  CameraState driver_cam = {.cam_type = DriverCam};
};
