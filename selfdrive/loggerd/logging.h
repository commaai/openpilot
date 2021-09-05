#include <atomic>
#include <condition_variable>
#include <mutex>

#include "cereal/messaging/messaging.h"

#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/loggerd/logger.h"

const int MAX_CAMERAS = WideRoadCam + 1;
const bool LOGGERD_TEST = getenv("LOGGERD_TEST");

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
  LoggerdState(int segment_length_ms, int no_camera_patience, bool testing = false);
  void init();
  void close(ExitHandler *do_exit);
  std::optional<std::pair<int, std::string>> get_segment(int cur_seg, bool trigger_rotate, ExitHandler *do_exit);
  bool rotate_if_needed();
  void rotate();
  bool sync_encoders(CameraType cam_type, uint32_t frame_id);

  LoggerState logger = {};
  
  std::atomic<double> last_camera_seen_tms = 0.;
  int max_waiting = 0;

protected:
  std::mutex rotate_lock;
  std::condition_variable rotate_cv;
  char segment_path[4096] = {};
  int waiting_rotate = 0;
  int rotate_segment = -1;
  double last_rotate_tms = 0.;

  // Sync logic for startup
  std::mutex sync_lock;
  int encoders_ready = 0;
  uint32_t latest_frame_id = 0;
  bool camera_ready[MAX_CAMERAS] = {};

  int segment_length_ms = 0;
  int no_camera_patience = 0;
  bool testing = false;
};
