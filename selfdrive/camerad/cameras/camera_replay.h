#pragma once

#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/ui/replay/framereader.h"

#define FRAME_BUF_COUNT 16

typedef struct CameraState {
  int camera_num;
  CameraInfo ci;

  int fps;
  float digital_gain = 0;

  CameraBuf buf;
  FrameReader *frame = nullptr;
} CameraState;

typedef struct MultiCameraState {
  CameraState road_cam;
  CameraState driver_cam;

  SubMaster *sm = nullptr;
  PubMaster *pm = nullptr;
} MultiCameraState;
