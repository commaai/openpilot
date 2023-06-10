#pragma once

#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cerrno>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "cereal/messaging/messaging.h"
#include "cereal/services.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_client.h"
#include "system/camerad/cameras/camera_common.h"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/timing.h"
#include "common/util.h"
#include "system/hardware/hw.h"

#include "system/loggerd/encoder/encoder.h"
#include "system/loggerd/logger.h"
#ifdef QCOM2
#include "system/loggerd/encoder/v4l_encoder.h"
#define Encoder V4LEncoder
#else
#include "system/loggerd/encoder/ffmpeg_encoder.h"
#define Encoder FfmpegEncoder
#endif

constexpr int MAIN_FPS = 20;
const int MAIN_BITRATE = 10000000;
const int DCAM_BITRATE = MAIN_BITRATE;

#define NO_CAMERA_PATIENCE 500 // fall back to time-based rotation if all cameras are dead

const bool LOGGERD_TEST = getenv("LOGGERD_TEST");
const int SEGMENT_LENGTH = LOGGERD_TEST ? atoi(getenv("LOGGERD_SEGMENT_LENGTH")) : 60;

struct LogCameraInfo {
  CameraType type;
  const char *filename;
  VisionStreamType stream_type;
  int frame_width, frame_height;
  int fps;
  int bitrate;
  bool is_h265;
  bool has_qcamera;
  bool record;
};

const LogCameraInfo cameras_logged[] = {
  {
    .type = RoadCam,
    .stream_type = VISION_STREAM_ROAD,
    .filename = "fcamera.hevc",
    .fps = MAIN_FPS,
    .bitrate = MAIN_BITRATE,
    .is_h265 = true,
    .has_qcamera = true,
    .record = true,
    .frame_width = 1928,
    .frame_height = 1208,
  },
  {
    .type = DriverCam,
    .stream_type = VISION_STREAM_DRIVER,
    .filename = "dcamera.hevc",
    .fps = MAIN_FPS,
    .bitrate = DCAM_BITRATE,
    .is_h265 = true,
    .has_qcamera = false,
    .record = Params().getBool("RecordFront"),
    .frame_width = 1928,
    .frame_height = 1208,
  },
  {
    .type = WideRoadCam,
    .stream_type = VISION_STREAM_WIDE_ROAD,
    .filename = "ecamera.hevc",
    .fps = MAIN_FPS,
    .bitrate = MAIN_BITRATE,
    .is_h265 = true,
    .has_qcamera = false,
    .record = true,
    .frame_width = 1928,
    .frame_height = 1208,
  },
};
const LogCameraInfo qcam_info = {
  .filename = "qcamera.ts",
  .fps = MAIN_FPS,
  .bitrate = 256000,
  .is_h265 = false,
  .record = true,
  .frame_width = 526,
  .frame_height = 330,
};
