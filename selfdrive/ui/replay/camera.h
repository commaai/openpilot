#pragma once

#include "cereal/visionipc/visionipc_server.h"
#include "selfdrive/common/queue.h"
#include "selfdrive/ui/replay/filereader.h"
#include "selfdrive/ui/replay/framereader.h"

class CameraServer {
public:
  CameraServer();
  ~CameraServer();
  inline void pushFrame(CameraType type, FrameReader* fr, uint32_t encodeFrameId) {
    queue_.push({type, fr, encodeFrameId});
  }
  inline void waitFramesSent() {
    while (!queue_.empty()) {
      std::this_thread::yield();
    }
  }

protected:
  void start();
  void thread();

  struct Camera {
    int width;
    int height;
  };

  std::array<Camera, MAX_CAMERAS> cameras_ = {};
  cl_device_id device_id_ = nullptr;
  cl_context context_ = nullptr;
  VisionIpcServer* vipc_server_ = nullptr;
  SafeQueue<std::tuple<CameraType, FrameReader*, uint32_t>> queue_;
  std::thread camera_thread_;
};
