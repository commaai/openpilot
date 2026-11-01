#pragma once

#include <memory>
#include <set>
#include <tuple>
#include <utility>

#include "tools/replay/framereader.h"
#include "tools/replay/logreader.h"

#ifdef __EMSCRIPTEN__
class CameraServer {
public:
  CameraServer(std::pair<int, int> camera_size[MAX_CAMERAS] = nullptr) {}
  void pushFrame(CameraType type, FrameReader *fr, const Event *event);
  void waitForSent() {}

private:
  // the decoder's position per camera, to send only what it still needs
  FrameReader *last_reader_[MAX_CAMERAS] = {};
  int last_idx_[MAX_CAMERAS] = {-1, -1, -1};
};
#else
#include "openpilot/cereal/visionstream.h"
#include "msgq/visionipc/visionipc_server.h"
#include "common/queue.h"

class CameraServer {
public:
  CameraServer(std::pair<int, int> camera_size[MAX_CAMERAS] = nullptr);
  ~CameraServer();
  void pushFrame(CameraType type, FrameReader* fr, const Event *event);
  void waitForSent();

protected:
  struct Camera {
    CameraType type;
    VisionStreamType stream_type;
    int width;
    int height;
    std::thread thread;
    SafeQueue<std::pair<FrameReader*, const Event *>> queue;
    std::set<VisionBuf *> cached_buf;
  };
  void startVipcServer();
  void cameraThread(Camera &cam);
  VisionBuf *getFrame(Camera &cam, FrameReader *fr, int32_t segment_id, uint32_t frame_id);

  Camera cameras_[MAX_CAMERAS] = {
      {.type = NarrowRoadCam, .stream_type = VISION_STREAM_NARROW_ROAD},
      {.type = CabinCam, .stream_type = VISION_STREAM_CABIN},
      {.type = WideRoadCam, .stream_type = VISION_STREAM_WIDE_ROAD},
  };
  std::atomic<int> publishing_ = 0;
  std::unique_ptr<VisionIpcServer> vipc_server_;
};
#endif  // __EMSCRIPTEN__
