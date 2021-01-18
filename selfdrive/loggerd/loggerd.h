#pragma once

#include "common/util.h"
#include "camerad/cameras/camera_common.h"
#include "logger.h"
constexpr int MAIN_BITRATE = 5000000;
constexpr int MAIN_FPS = 20;
#ifndef QCOM2
constexpr int MAX_CAM_IDX = LOG_CAMERA_ID_DCAMERA;
constexpr int DCAM_BITRATE = 2500000;
#else
constexpr int MAX_CAM_IDX = LOG_CAMERA_ID_ECAMERA;
constexpr int DCAM_BITRATE = MAIN_BITRATE;
#endif

#define NO_CAMERA_PATIENCE 500 // fall back to time-based rotation if all cameras are dead

constexpr int SEGMENT_LENGTH = 60;

ExitHandler do_exit;

static LogCameraInfo cameras_logged[LOG_CAMERA_ID_MAX] = {
  [LOG_CAMERA_ID_FCAMERA] = {
    .stream_type = VISION_STREAM_YUV_BACK,
    .filename = "fcamera.hevc",
    .frame_packet_name = "frame",
    .fps = MAIN_FPS,
    .bitrate = MAIN_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = true
  },
  [LOG_CAMERA_ID_DCAMERA] = {
    .stream_type = VISION_STREAM_YUV_FRONT,
    .filename = "dcamera.hevc",
    .frame_packet_name = "frontFrame",
    .fps = MAIN_FPS, // on EONs, more compressed this way
    .bitrate = DCAM_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = false
  },
  [LOG_CAMERA_ID_ECAMERA] = {
    .stream_type = VISION_STREAM_YUV_WIDE,
    .filename = "ecamera.hevc",
    .frame_packet_name = "wideFrame",
    .fps = MAIN_FPS,
    .bitrate = MAIN_BITRATE,
    .is_h265 = true,
    .downscale = false,
    .has_qcamera = false
  },
  [LOG_CAMERA_ID_QCAMERA] = {
    .filename = "qcamera.ts",
    .fps = MAIN_FPS,
    .bitrate = 128000,
    .is_h265 = false,
    .downscale = true,
#ifndef QCOM2
    .frame_width = 480, .frame_height = 360
#else
    .frame_width = 526, .frame_height = 330 // keep pixel count the same?
#endif
  },
};

class RotateState {
public:
  SubSocket* fpkt_sock;
  uint32_t stream_frame_id, log_frame_id, last_rotate_frame_id;
  bool enabled, should_rotate, initialized;

  RotateState() : fpkt_sock(nullptr), stream_frame_id(0), log_frame_id(0),
                  last_rotate_frame_id(UINT32_MAX), enabled(false), should_rotate(false), initialized(false) {};

  void waitLogThread() {
    std::unique_lock<std::mutex> lk(fid_lock);
    while (stream_frame_id > log_frame_id           // if the log camera is older, wait for it to catch up.
           && (stream_frame_id - log_frame_id) < 8  // but if its too old then there probably was a discontinuity (visiond restarted)
           && !do_exit) {
      cv.wait(lk);
    }
  }

  void cancelWait() {
    cv.notify_one();
  }

  void setStreamFrameId(uint32_t frame_id) {
    fid_lock.lock();
    stream_frame_id = frame_id;
    fid_lock.unlock();
    cv.notify_one();
  }

  void setLogFrameId(uint32_t frame_id) {
    fid_lock.lock();
    log_frame_id = frame_id;
    fid_lock.unlock();
    cv.notify_one();
  }

  void rotate() {
    if (enabled) {
      std::unique_lock<std::mutex> lk(fid_lock);
      should_rotate = true;
      last_rotate_frame_id = stream_frame_id;
    }
  }

  void finish_rotate() {
    std::unique_lock<std::mutex> lk(fid_lock);
    should_rotate = false;
  }

private:
  std::mutex fid_lock;
  std::condition_variable cv;
};

typedef struct SocketState {
  int counter, freq, fpkt_id;
} SocketState;

class LoggerdState {
public:
  Context *ctx;
  Poller *poller;
  int segment_length = 0;
  std::map<SubSocket*, SocketState> socket_states;
  std::vector<std::thread> encoder_threads;
  LoggerState logger;
  char segment_path[4096] = {};
  int rotate_segment = -1;
  std::mutex rotate_lock;
  RotateState rotate_state[LOG_CAMERA_ID_MAX-1];

  LoggerdState();
  ~LoggerdState();
  void run();

private:
  double last_rotate_tms = 0, last_camera_seen_tms = 0;
  void rotate();
  std::unique_ptr<Message> log(SubSocket *socket, SocketState& ss);
};
