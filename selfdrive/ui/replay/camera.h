#pragma once

#include "cereal/visionipc/visionipc_server.h"
#include "selfdrive/common/queue.h"
#include "selfdrive/ui/replay/framereader.h"
#include "selfdrive/ui/replay/logreader.h"

class CameraServer {
public:
  CameraServer();
  ~CameraServer();
  void pushFrame(CameraType type, FrameReader* fr, uint32_t encodeFrameId, const cereal::FrameData::Reader &frame_data);
  void waitFramesSent();

protected:
  void start();
  void stop();
  void thread();

  struct Camera {
    VisionStreamType rgb_type;
    VisionStreamType yuv_type;
    int width;
    int height;
  };

  Camera cameras_[MAX_CAMERAS] = {
      {.rgb_type = VISION_STREAM_RGB_BACK, .yuv_type = VISION_STREAM_YUV_BACK},
      {.rgb_type = VISION_STREAM_RGB_FRONT, .yuv_type = VISION_STREAM_YUV_FRONT},
      {.rgb_type = VISION_STREAM_RGB_WIDE, .yuv_type = VISION_STREAM_YUV_WIDE},
  };
  cl_device_id device_id_ = nullptr;
  cl_context context_ = nullptr;
  VisionIpcServer* vipc_server_ = nullptr;
  std::thread camera_thread_;
  SafeQueue<std::tuple<CameraType, FrameReader*, uint32_t, const cereal::FrameData::Reader>> queue_;
};
