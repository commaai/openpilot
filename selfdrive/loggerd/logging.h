#pragma once

#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_client.h"

#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/loggerd/logger.h"

const int MAX_CAMERAS = WideRoadCam + 1;
struct LoggerdState {
  Context *ctx;
  LoggerState logger = {};
  char segment_path[4096];
  std::mutex rotate_lock;
  std::condition_variable rotate_cv;
  std::atomic<int> rotate_segment;
  std::atomic<double> last_camera_seen_tms;
  std::atomic<int> waiting_rotate;
  int max_waiting = 0;
  double last_rotate_tms = 0.;

  // Sync logic for startup
  std::atomic<int> encoders_ready = 0;
  std::atomic<uint32_t> latest_frame_id = 0;
  bool camera_ready[MAX_CAMERAS] = {};
};

bool sync_encoders(LoggerdState *s, CameraType cam_type, uint32_t frame_id);
