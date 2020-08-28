#include <string>
#include "common/visionipc.h"
#include "cereal/gen/cpp/log.capnp.h"

#define MAIN_BITRATE 5000000

typedef enum CameraType {
  F_CAMERA,
  D_CAMERA,
  E_CAMERA,
} CameraType;

typedef struct {
  std::string name;
  CameraType type;
  std::string encoder_pkt_name;
  std::string log_file_name_name;
  int bitrate;
  cereal::EncodeIndex::Type encode_type;
  VisionStreamType stream_type;
} Camera;

const Camera fCamera = {
  .name="Rear",
  .type=F_CAMERA,
  .encoder_pkt_name="encodeIdx",
  .log_file_name_name="fcamera.hevc",
  .bitrate=MAIN_BITRATE,
  .encode_type=cereal::EncodeIndex::Type::FULL_H_E_V_C,
  .stream_type=VISION_STREAM_YUV,
};

const Camera dCamera = {
  .name="Front",
  .type=D_CAMERA,
  .encoder_pkt_name="frontEncodeIdx",
  .log_file_name_name="dcamera.hevc",
#ifdef QCOM
  .bitrate=2500000,
  .encode_type=cereal::EncodeIndex::Type::FRONT,
#else
  .bitrate=MAIN_BITRATE,
  .encode_type=cereal::EncodeIndex::Type::FULL_H_E_V_C,
#endif
  .stream_type=VISION_STREAM_YUV_FRONT,
};

const Camera wCamera = {
  .name="Wide",
  .type=E_CAMERA,
  .encoder_pkt_name="frontEncodeIdx",
  .log_file_name_name="ecamera.hevc",
  .bitrate=MAIN_BITRATE,
  .encode_type=cereal::EncodeIndex::Type::FULL_H_E_V_C,
  .stream_type=VISION_STREAM_YUV_WIDE,
};
