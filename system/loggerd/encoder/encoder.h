#pragma once

#include <cassert>
#include <cstdint>
#include <thread>

#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionipc.h"
#include "common/queue.h"
#include "system/camerad/cameras/camera_common.h"
#include "system/loggerd/loggerd.h"

#define V4L2_BUF_FLAG_KEYFRAME 8

class VideoEncoder {
public:
  VideoEncoder(const EncoderInfo &encoder_info, int in_width, int in_height);
  virtual ~VideoEncoder() {};
  virtual int encode_frame(VisionBuf* buf, VisionIpcBufExtra *extra) = 0;
  virtual void encoder_open(const char* path) = 0;
  virtual void encoder_close() = 0;

  static void publisher_publish(VideoEncoder *e, int segment_num, uint32_t idx, VisionIpcBufExtra &extra, unsigned int flags, kj::ArrayPtr<capnp::byte> header, kj::ArrayPtr<capnp::byte> dat);


protected:
  int in_width, in_height;
  const EncoderInfo encoder_info;

private:
  // total frames encoded
  int cnt = 0;
  std::unique_ptr<PubMaster> pm;
};
