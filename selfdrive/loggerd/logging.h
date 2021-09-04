#include <atomic>
#include <condition_variable>
#include <mutex>

#include "cereal/messaging/messaging.h"

#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/loggerd/logger.h"

#define NO_CAMERA_PATIENCE 500 // fall back to time-based rotation if all cameras are dead

const bool LOGGERD_TEST = getenv("LOGGERD_TEST");
const int SEGMENT_LENGTH = LOGGERD_TEST ? atoi(getenv("LOGGERD_SEGMENT_LENGTH")) : 60;

typedef struct LogCameraInfo {
  CameraType type;
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
  bool enable;
} LogCameraInfo;

class LoggerdState {
public:
  LoggerdState() = default;
  LoggerdState(int segment_length_ms, int no_camera_patience, bool testing);
  std::optional<std::pair<int, std::string>> get_segment(int cur_seg, bool trigger_rotate, ExitHandler *do_exit);
  bool rotate_if_needed();
  void rotate();

  LoggerState logger = {};
  
  std::condition_variable rotate_cv;
  std::atomic<double> last_camera_seen_tms = 0.;
  int max_waiting = 0;

  // Sync logic for startup
  std::atomic<bool> encoders_synced;
  std::atomic<int> encoders_ready;
  std::atomic<uint32_t> start_frame_id;
  std::atomic<uint32_t> latest_frame_id;

protected:
  std::mutex rotate_lock;
  char segment_path[4096] = {};
  int waiting_rotate = 0;
  int rotate_segment = -1;
  double last_rotate_tms = 0.;

  int segment_length_ms = SEGMENT_LENGTH * 1000;
  int no_camera_patience = NO_CAMERA_PATIENCE;
  bool testing = false;
};
