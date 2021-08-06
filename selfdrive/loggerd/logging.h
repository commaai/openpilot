#pragma once

#include <condition_variable>
#include <mutex>

#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionbuf.h"
#include "cereal/visionipc/visionipc.h"
#include "selfdrive/common/util.h"
#include "selfdrive/hardware/hw.h"
#include "selfdrive/loggerd/logger.h"

#define LOG_CAMERA_ID_FCAMERA 0
#define LOG_CAMERA_ID_DCAMERA 1
#define LOG_CAMERA_ID_ECAMERA 2
#define LOG_CAMERA_ID_QCAMERA 3
#define LOG_CAMERA_ID_MAX 4

#define NO_CAMERA_PATIENCE 500  // fall back to time-based rotation if all cameras are dead

const int SEGMENT_LENGTH = getenv("LOGGERD_TEST") ? atoi(getenv("LOGGERD_SEGMENT_LENGTH")) : 60;

typedef struct LogCameraInfo {
  const char* filename;
  const char* frame_packet_name;
  const char* encode_idx_name;
  VisionStreamType stream_type;
  int frame_width, frame_height;
  int fps;
  int bitrate;
  bool is_h265;
  bool downscale;
  bool has_qcamera;
  bool trigger_rotate;
} LogCameraInfo;

class LoggerdState {
public:
  LoggerdState() = default;
  LoggerdState(int segment_length_ms, int no_camera_patience, bool testing);
  void rotate();
  void rotate_if_needed();
  void triggerAndWait(int cur_seg, ExitHandler* do_exit);

  Context* ctx;
  LoggerState logger = {};
  char segment_path[4096];
  std::mutex rotate_lock;
  std::condition_variable rotate_cv;
  std::atomic<int> rotate_segment = -1;
  std::atomic<double> last_camera_seen_tms = 0;
  std::atomic<int> waiting_rotate = 0;
  int max_waiting = 0;
  double last_rotate_tms = 0.;

private:
  int segment_length_ms = SEGMENT_LENGTH * 1000;
  int no_camera_patience = NO_CAMERA_PATIENCE;
  bool testing = false;
};
