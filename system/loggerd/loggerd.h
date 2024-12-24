#pragma once

#include <vector>

#include "cereal/messaging/messaging.h"
#include "cereal/services.h"
#include "msgq/visionipc/visionipc_client.h"
#include "system/hardware/hw.h"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/util.h"

#include "system/loggerd/logger.h"

constexpr int MAIN_FPS = 20;
const int MAIN_BITRATE = 1e7;
const int LIVESTREAM_BITRATE = 1e6;
const int QCAM_BITRATE = 256000;

#define NO_CAMERA_PATIENCE 500  // fall back to time-based rotation if all cameras are dead

#define INIT_ENCODE_FUNCTIONS(encode_type)                                \
  .get_encode_data_func = &cereal::Event::Reader::get##encode_type##Data, \
  .set_encode_idx_func = &cereal::Event::Builder::set##encode_type##Idx,  \
  .init_encode_data_func = &cereal::Event::Builder::init##encode_type##Data

const bool LOGGERD_TEST = getenv("LOGGERD_TEST");
const int SEGMENT_LENGTH = LOGGERD_TEST ? atoi(getenv("LOGGERD_SEGMENT_LENGTH")) : 60;

constexpr char PRESERVE_ATTR_NAME[] = "user.preserve";
constexpr char PRESERVE_ATTR_VALUE = '1';
class EncoderInfo {
public:
  const char *publish_name;
  const char *thumbnail_name = NULL;
  const char *filename = NULL;
  bool record = true;
  int frame_width = -1;
  int frame_height = -1;
  int fps = MAIN_FPS;
  int bitrate = MAIN_BITRATE;
  cereal::EncodeIndex::Type encode_type = Hardware::PC() ? cereal::EncodeIndex::Type::BIG_BOX_LOSSLESS
                                                         : cereal::EncodeIndex::Type::FULL_H_E_V_C;
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
  .publish_name = "roadEncodeData",
  .filename = "fcamera.hevc",
  INIT_ENCODE_FUNCTIONS(RoadEncode),
};

const EncoderInfo main_wide_road_encoder_info = {
  .publish_name = "wideRoadEncodeData",
  .filename = "ecamera.hevc",
  INIT_ENCODE_FUNCTIONS(WideRoadEncode),
};

const EncoderInfo main_driver_encoder_info = {
  .publish_name = "driverEncodeData",
  .filename = "dcamera.hevc",
  .record = Params().getBool("RecordFront"),
  INIT_ENCODE_FUNCTIONS(DriverEncode),
};

const EncoderInfo stream_road_encoder_info = {
  .publish_name = "livestreamRoadEncodeData",
  //.thumbnail_name = "thumbnail",
  .encode_type = cereal::EncodeIndex::Type::QCAMERA_H264,
  .record = false,
  .bitrate = LIVESTREAM_BITRATE,
  INIT_ENCODE_FUNCTIONS(LivestreamRoadEncode),
};

const EncoderInfo stream_wide_road_encoder_info = {
  .publish_name = "livestreamWideRoadEncodeData",
  .encode_type = cereal::EncodeIndex::Type::QCAMERA_H264,
  .record = false,
  .bitrate = LIVESTREAM_BITRATE,
  INIT_ENCODE_FUNCTIONS(LivestreamWideRoadEncode),
};

const EncoderInfo stream_driver_encoder_info = {
  .publish_name = "livestreamDriverEncodeData",
  .encode_type = cereal::EncodeIndex::Type::QCAMERA_H264,
  .record = false,
  .bitrate = LIVESTREAM_BITRATE,
  INIT_ENCODE_FUNCTIONS(LivestreamDriverEncode),
};

const EncoderInfo qcam_encoder_info = {
  .publish_name = "qRoadEncodeData",
  .filename = "qcamera.ts",
  .bitrate = QCAM_BITRATE,
  .encode_type = cereal::EncodeIndex::Type::QCAMERA_H264,
  .frame_width = 526,
  .frame_height = 330,
  INIT_ENCODE_FUNCTIONS(QRoadEncode),
};

const LogCameraInfo road_camera_info{
  .thread_name = "road_cam_encoder",
  .stream_type = VISION_STREAM_ROAD,
  .encoder_infos = {main_road_encoder_info, qcam_encoder_info}
};

const LogCameraInfo wide_road_camera_info{
  .thread_name = "wide_road_cam_encoder",
  .stream_type = VISION_STREAM_WIDE_ROAD,
  .encoder_infos = {main_wide_road_encoder_info}
};

const LogCameraInfo driver_camera_info{
  .thread_name = "driver_cam_encoder",
  .stream_type = VISION_STREAM_DRIVER,
  .encoder_infos = {main_driver_encoder_info}
};

const LogCameraInfo stream_road_camera_info{
  .thread_name = "road_cam_encoder",
  .stream_type = VISION_STREAM_ROAD,
  .encoder_infos = {stream_road_encoder_info}
};

const LogCameraInfo stream_wide_road_camera_info{
  .thread_name = "wide_road_cam_encoder",
  .stream_type = VISION_STREAM_WIDE_ROAD,
  .encoder_infos = {stream_wide_road_encoder_info}
};

const LogCameraInfo stream_driver_camera_info{
  .thread_name = "driver_cam_encoder",
  .stream_type = VISION_STREAM_DRIVER,
  .encoder_infos = {stream_driver_encoder_info}
};

const LogCameraInfo cameras_logged[] = {road_camera_info, wide_road_camera_info, driver_camera_info};
const LogCameraInfo stream_cameras_logged[] = {stream_road_camera_info, stream_wide_road_camera_info, stream_driver_camera_info};
