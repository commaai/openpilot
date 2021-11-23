#pragma once

#include <ftw.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cerrno>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "cereal/messaging/messaging.h"
#include "cereal/services.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_client.h"
#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/common/params.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"
#include "selfdrive/hardware/hw.h"

#include "selfdrive/loggerd/encoder.h"
#include "selfdrive/loggerd/logger.h"
#if defined(QCOM) || defined(QCOM2)
#include "selfdrive/loggerd/omx_encoder.h"
#define Encoder OmxEncoder
#else
#include "selfdrive/loggerd/raw_logger.h"
#define Encoder RawLogger
#endif

constexpr int MAIN_FPS = 20;
const int MAIN_BITRATE = Hardware::TICI() ? 10000000 : 5000000;
const int DCAM_BITRATE = Hardware::TICI() ? MAIN_BITRATE : 2500000;

#define NO_CAMERA_PATIENCE 500 // fall back to time-based rotation if all cameras are dead

const bool LOGGERD_TEST = getenv("LOGGERD_TEST");
const int SEGMENT_LENGTH = LOGGERD_TEST ? atoi(getenv("LOGGERD_SEGMENT_LENGTH")) : 60;

const LogCameraInfo cameras_logged[] = {
  {
    .type = RoadCam,
    .stream_type = VISION_STREAM_YUV_BACK,
    .filename = "fcamera.hevc",
    .frame_packet_name = "roadCameraState",
    .fps = MAIN_FPS,
    .bitrate = MAIN_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = true,
    .trigger_rotate = true,
    .enable = true,
    .record = true,
  },
  {
    .type = DriverCam,
    .stream_type = VISION_STREAM_YUV_FRONT,
    .filename = "dcamera.hevc",
    .frame_packet_name = "driverCameraState",
    .fps = MAIN_FPS, // on EONs, more compressed this way
    .bitrate = DCAM_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = false,
    .trigger_rotate = Hardware::TICI(),
    .enable = !Hardware::PC(),
    .record = Params().getBool("RecordFront"),
  },
  {
    .type = WideRoadCam,
    .stream_type = VISION_STREAM_YUV_WIDE,
    .filename = "ecamera.hevc",
    .frame_packet_name = "wideRoadCameraState",
    .fps = MAIN_FPS,
    .bitrate = MAIN_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = false,
    .trigger_rotate = true,
    .enable = Hardware::TICI(),
    .record = Hardware::TICI(),
  },
};
const LogCameraInfo qcam_info = {
  .filename = "qcamera.ts",
  .fps = MAIN_FPS,
  .bitrate = 256000,
  .is_h265 = false,
  .downscale = true,
  .frame_width = Hardware::TICI() ? 526 : 480,
  .frame_height = Hardware::TICI() ? 330 : 360 // keep pixel count the same?
};

struct LoggerdState {
  Context *ctx;
  LoggerState logger = {};
  char segment_path[4096];
  std::mutex rotate_lock;
  std::condition_variable rotate_cv;
  std::atomic<int> rotate_segment;
  std::atomic<double> last_camera_seen_tms;
  std::atomic<int> ready_to_rotate;  // count of encoders ready to rotate
  int max_waiting = 0;
  double last_rotate_tms = 0.;      // last rotate time in ms

  // Sync logic for startup
  std::atomic<int> encoders_ready = 0;
  std::atomic<uint32_t> start_frame_id = 0;
  bool camera_ready[WideRoadCam + 1] = {};
  bool camera_synced[WideRoadCam + 1] = {};
};

void start_logging();
