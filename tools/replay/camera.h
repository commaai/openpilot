#pragma once

#include <unistd.h>

#include <memory>
#include <tuple>
#include <utility>

#include "cereal/visionipc/visionipc_server.h"
#include "common/queue.h"
#include "tools/replay/framereader.h"
#include "tools/replay/logreader.h"

std::tuple<size_t, size_t, size_t> get_nv12_info(int width, int height);

class CameraServer {
public:
  CameraServer(std::pair<int, int> camera_size[MAX_CAMERAS] = nullptr);
  ~CameraServer();
  void pushFrame(CameraType type, FrameReader* fr, const cereal::EncodeIndex::Reader& eidx);
  void waitForSent();

protected:
  struct Camera {
    CameraType type;
    VisionStreamType stream_type;
    int width;
    int height;
    std::thread thread;
    SafeQueue<std::pair<FrameReader*, const cereal::EncodeIndex::Reader>> queue;
    int cached_id = -1;
    int cached_seg = -1;
    VisionBuf * cached_buf;
  };
  void startVipcServer();
  void cameraThread(Camera &cam);

  Camera cameras_[MAX_CAMERAS] = {
      {.type = RoadCam, .stream_type = VISION_STREAM_ROAD},
      {.type = DriverCam, .stream_type = VISION_STREAM_DRIVER},
      {.type = WideRoadCam, .stream_type = VISION_STREAM_WIDE_ROAD},
  };
  std::atomic<int> publishing_ = 0;
  std::unique_ptr<VisionIpcServer> vipc_server_;
};
