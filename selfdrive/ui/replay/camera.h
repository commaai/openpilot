#pragma once

#include "cereal/visionipc/visionipc_server.h"
#include "selfdrive/common/queue.h"
#include "selfdrive/ui/replay/framereader.h"

// TODO: move this to camerad
enum CameraType {
  RoadCam = 0,
  DriverCam,
  WideRoadCam
};

const CameraType ALL_CAMERAS[] = {RoadCam, DriverCam, WideRoadCam};
const int MAX_CAMERAS = std::size(ALL_CAMERAS);

class CameraServer {
public:
  CameraServer();
  ~CameraServer();
  void ensure(FrameReader *frs[MAX_CAMERAS]);
  void pushFrame(CameraType type, FrameReader *fr, uint32_t encodeFrameId);
  void stop();

 private:
  cl_device_id device_id_ = nullptr;
  cl_context context_ = nullptr;
  VisionIpcServer *vipc_server_ = nullptr;

  class CameraState;
  CameraState* camera_states_[MAX_CAMERAS] = {};
};
