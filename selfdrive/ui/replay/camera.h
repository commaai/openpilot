#pragma once

#include <unistd.h>
#include "cereal/visionipc/visionipc_server.h"
#include "selfdrive/common/queue.h"
#include "selfdrive/ui/replay/framereader.h"
#include "selfdrive/ui/replay/logreader.h"

class CameraServer {
public:
  CameraServer(std::pair<int, int> camera_size[MAX_CAMERAS] = nullptr, bool send_yuv = false);
  ~CameraServer();
  void pushFrame(CameraType type, FrameReader* fr, const cereal::EncodeIndex::Reader& eidx);
  inline void waitFinish() {
    while (publishing_ > 0) usleep(0);
  }

protected:
  struct Camera {
    CameraType type;
    VisionStreamType rgb_type; 
    VisionStreamType yuv_type;
    int width;
    int height;
    std::thread thread;
    SafeQueue<std::pair<FrameReader*, const cereal::EncodeIndex::Reader>> queue;
    int cached_id = -1;
    int cached_seg = -1;
    std::pair<VisionBuf *, VisionBuf*> cached_buf;
  };
  void startVipcServer();
  void cameraThread(Camera &cam);

  Camera cameras_[MAX_CAMERAS] = {
      {.type = RoadCam, .rgb_type = VISION_STREAM_RGB_BACK, .yuv_type = VISION_STREAM_YUV_BACK},
      {.type = DriverCam, .rgb_type = VISION_STREAM_RGB_FRONT, .yuv_type = VISION_STREAM_YUV_FRONT},
      {.type = WideRoadCam, .rgb_type = VISION_STREAM_RGB_WIDE, .yuv_type = VISION_STREAM_YUV_WIDE},
  };
  std::atomic<int> publishing_ = 0;
  std::unique_ptr<VisionIpcServer> vipc_server_;
  bool send_yuv;
};
