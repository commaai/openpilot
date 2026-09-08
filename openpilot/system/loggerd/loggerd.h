#pragma once

#include <cstdlib>
#include <vector>

#include "openpilot/cereal/messaging/messaging.h"
#include "openpilot/cereal/services.h"
#include "openpilot/cereal/visionstream.h"
#include "msgq/visionipc/visionipc_client.h"
#include "common/hardware/hw.h"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/util.h"

#include "system/loggerd/logger.h"

constexpr int MAIN_FPS = 20;
const auto MAIN_ENCODE_TYPE = Hardware::PC() ? cereal::EncodeIndex::Type::BIG_BOX_LOSSLESS : cereal::EncodeIndex::Type::FULL_H_E_V_C;
#define NO_CAMERA_PATIENCE 500  // fall back to time-based rotation if all cameras are dead

#define INIT_ENCODE_FUNCTIONS(encode_type)                                \
  .get_encode_data_func = &cereal::Event::Reader::get##encode_type##Data, \
  .set_encode_idx_func = &cereal::Event::Builder::set##encode_type##Idx,  \
  .init_encode_data_func = &cereal::Event::Builder::init##encode_type##Data

const bool LOGGERD_TEST = getenv("LOGGERD_TEST");
const int SEGMENT_LENGTH = LOGGERD_TEST ? atoi(getenv("LOGGERD_SEGMENT_LENGTH")) : 60;

inline int livestream_width() {
  switch (Hardware::get_device_type()) {
    case cereal::InitData::DeviceType::TIZI: return 1152;
    case cereal::InitData::DeviceType::MICI: return 1280;
    default: return -1;
  }
}

inline int livestream_height() {
  switch (Hardware::get_device_type()) {
    case cereal::InitData::DeviceType::TIZI:
    case cereal::InitData::DeviceType::MICI: return 720;
    default: return -1;
  }
}

constexpr char PRESERVE_ATTR_NAME[] = "user.preserve";
constexpr char PRESERVE_ATTR_VALUE = '1';

struct EncoderSettings {
  cereal::EncodeIndex::Type encode_type;
  int bitrate;
  int gop_size;
  int b_frames = 0; // we don't use b frames

  static EncoderSettings MainEncoderSettings(int in_width) {
    if (in_width <= 1344) {
      return EncoderSettings{.encode_type = MAIN_ENCODE_TYPE, .bitrate = 5'000'000, .gop_size = 20};
    } else {
      return EncoderSettings{.encode_type = MAIN_ENCODE_TYPE, .bitrate = 10'000'000, .gop_size = 30};
    }
  }

  static EncoderSettings QcamEncoderSettings() {
    return EncoderSettings{.encode_type = cereal::EncodeIndex::Type::QCAMERA_H264, .bitrate = 256'000, .gop_size = 15};
  }

  static EncoderSettings StreamEncoderSettings() {
    int _stream_bitrate = getenv("STREAM_BITRATE") ? atoi(getenv("STREAM_BITRATE")) : 5'000'000;
    return EncoderSettings{.encode_type = cereal::EncodeIndex::Type::QCAMERA_H264, .bitrate = _stream_bitrate , .gop_size = 5};
  }
};

class EncoderInfo {
public:
  const char *publish_name;
  const char *thumbnail_name = NULL;
  const char *filename = NULL;
  bool record = true;
  bool include_audio = false;
  bool is_live = false;
  int frame_width = -1;
  int frame_height = -1;
  int fps = MAIN_FPS;
  std::function<EncoderSettings(int)> get_settings;

  ::cereal::EncodeData::Reader (cereal::Event::Reader::*get_encode_data_func)() const;
  void (cereal::Event::Builder::*set_encode_idx_func)(::cereal::EncodeIndex::Reader);
  cereal::EncodeData::Builder (cereal::Event::Builder::*init_encode_data_func)();
};

class LogCameraInfo {
public:
  const char *thread_name;
  int fps = MAIN_FPS;
  VisionStreamType stream_type;
  std::vector<EncoderInfo> encoder_infos;
};

const EncoderInfo main_road_encoder_info = {
  .publish_name = "narrowRoadEncodeData",
  .thumbnail_name = "thumbnail",
  .filename = "fcamera.hevc",
  .get_settings = [](int in_width){return EncoderSettings::MainEncoderSettings(in_width);},
  INIT_ENCODE_FUNCTIONS(NarrowRoadEncode),
};

const EncoderInfo main_wide_road_encoder_info = {
  .publish_name = "wideRoadEncodeData",
  .filename = "ecamera.hevc",
  .get_settings = [](int in_width){return EncoderSettings::MainEncoderSettings(in_width);},
  INIT_ENCODE_FUNCTIONS(WideRoadEncode),
};

const EncoderInfo main_cabin_encoder_info = {
  .publish_name = "cabinEncodeData",
  .filename = "dcamera.hevc",
  .record = Params().getBool("RecordFront"),
  .get_settings = [](int in_width){return EncoderSettings::MainEncoderSettings(in_width);},
  INIT_ENCODE_FUNCTIONS(CabinEncode),
};

const EncoderInfo stream_road_encoder_info = {
  .publish_name = "livestreamNarrowRoadEncodeData",
  //.thumbnail_name = "thumbnail",
  .record = false,
  .is_live = true,
  .frame_width = livestream_width(),
  .frame_height = livestream_height(),
  .get_settings = [](int){return EncoderSettings::StreamEncoderSettings();},
  INIT_ENCODE_FUNCTIONS(LivestreamNarrowRoadEncode),
};

const EncoderInfo stream_wide_road_encoder_info = {
  .publish_name = "livestreamWideRoadEncodeData",
  .record = false,
  .is_live = true,
  .frame_width = livestream_width(),
  .frame_height = livestream_height(),
  .get_settings = [](int){return EncoderSettings::StreamEncoderSettings();},
  INIT_ENCODE_FUNCTIONS(LivestreamWideRoadEncode),
};

const EncoderInfo stream_cabin_encoder_info = {
  .publish_name = "livestreamCabinEncodeData",
  .record = false,
  .is_live = true,
  .frame_width = livestream_width(),
  .frame_height = livestream_height(),
  .get_settings = [](int){return EncoderSettings::StreamEncoderSettings();},
  INIT_ENCODE_FUNCTIONS(LivestreamCabinEncode),
};

const EncoderInfo qcam_encoder_info = {
  .publish_name = "qNarrowRoadEncodeData",
  .filename = "qcamera.ts",
  .include_audio = Params().getBool("RecordAudio"),
  .frame_width = 526,
  .frame_height = 330,
  .get_settings = [](int){return EncoderSettings::QcamEncoderSettings();},
  INIT_ENCODE_FUNCTIONS(QNarrowRoadEncode),
};

const LogCameraInfo narrow_road_camera_info{
  .thread_name = "narrow_road_cam_encoder",
  .stream_type = VISION_STREAM_NARROW_ROAD,
  .encoder_infos = {main_road_encoder_info, qcam_encoder_info}
};

const LogCameraInfo wide_road_camera_info{
  .thread_name = "wide_road_cam_encoder",
  .stream_type = VISION_STREAM_WIDE_ROAD,
  .encoder_infos = {main_wide_road_encoder_info}
};

const LogCameraInfo cabin_camera_info{
  .thread_name = "cabin_cam_encoder",
  .stream_type = VISION_STREAM_CABIN,
  .encoder_infos = {main_cabin_encoder_info}
};

const LogCameraInfo stream_road_camera_info{
  .thread_name = "narrow_road_cam_encoder",
  .stream_type = VISION_STREAM_NARROW_ROAD,
  .encoder_infos = {stream_road_encoder_info},
};

const LogCameraInfo stream_wide_road_camera_info{
  .thread_name = "wide_road_cam_encoder",
  .stream_type = VISION_STREAM_WIDE_ROAD,
  .encoder_infos = {stream_wide_road_encoder_info},
};

const LogCameraInfo stream_cabin_camera_info{
  .thread_name = "cabin_cam_encoder",
  .stream_type = VISION_STREAM_CABIN,
  .encoder_infos = {stream_cabin_encoder_info},
};

const LogCameraInfo cameras_logged[] = {narrow_road_camera_info, wide_road_camera_info, cabin_camera_info};
const LogCameraInfo stream_cameras_logged[] = {stream_road_camera_info, stream_wide_road_camera_info, stream_cabin_camera_info};
